"""
Worker de processamento de stream — Versão 3.0
Fluxo determinístico com estado, ROI fixo e janela temporal.

ESTADOS:
  IDLE        → monitoramento leve, sem OCR, extrai tempo a cada 2s
  PRE_TRIGGER → tempo ≥ 85min, fps alto, observa região da tabela
  TRIGGER     → ROI Yellow Card detectado com confiança ≥ 85%
  CONFIRM     → confirma por 2-3 frames consecutivos (anti-transição)
  CAPTURE     → burst de frames
  OCR         → extrai tabela completa do melhor frame
  SAVE        → salva no banco, reseta estado
"""

import cv2
import logging
import time
import signal
import traceback
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pytesseract

from core.capture import StreamCaptureWithReconnect
from detection.detector import TableDetector, GameState
from ocr.extractor import OCRExtractor
from database.manager import DatabaseManager
from utils.logger import setup_logger
from config import settings

logger = logging.getLogger(__name__)


# ─── ROI FIXO DO YELLOW CARD ────────────────────────────────────────────────
# Baseado nas screenshots reais da stream ESB Football.
# Yellow Cards é sempre a última linha da tabela central.
# Coordenadas relativas (0.0 ~ 1.0): (x_inicio, y_inicio, x_fim, y_fim)
ROI_YELLOW_CARD = (0.33, 0.87, 0.67, 0.97)

# Janela temporal válida para disparo
ALERT_MINUTE_START = 85
ALERT_MINUTE_END   = 99   # inclui tempo extra

# Confirmações consecutivas antes de capturar
MIN_CONFIRMATIONS = 2

# Confiança mínima do OCR no Yellow Card
YELLOW_CARD_MIN_CONFIDENCE = 82  # %


class WorkerState(Enum):
    IDLE        = "idle"
    PRE_TRIGGER = "pre_trigger"
    TRIGGER     = "trigger"
    CONFIRM     = "confirm"
    CAPTURE     = "capture"
    OCR         = "ocr"
    SAVE        = "save"
    COOLDOWN    = "cooldown"


class StreamWorker:

    def __init__(self, stream_config: dict):
        self.stream_config = stream_config
        self.stream_id     = stream_config["id"]
        self._setup_components()

        self._state             = WorkerState.IDLE
        self._confirmations     = 0
        self._last_capture_time = 0.0
        self._match_minute      = 0
        self._time_check_counter = 0  # conta frames para verificar tempo

    def _setup_components(self):
        self.capture  = StreamCaptureWithReconnect(
            self.stream_config, settings.DETECTION, settings.RECONNECT)
        self.detector = TableDetector(settings.DETECTION, settings.ROI)
        self.ocr      = OCRExtractor(settings.OCR, settings.ROI)
        self.db       = DatabaseManager(settings.DATABASE)

    # ────────────────────────────────────────────────────────────────────────
    # UTILITÁRIOS
    # ────────────────────────────────────────────────────────────────────────

    def _set_state(self, new_state: WorkerState):
        if new_state != self._state:
            logger.info(f"[{self.stream_id}] STATE: {self._state.value} → {new_state.value}")
            self._state = new_state

    def _log(self, level: str, msg: str, match_id=None):
        getattr(logger, level)(f"[{self.stream_id}] {msg}")
        if level in ("info", "warning", "error"):
            self.db.log_event(self.stream_id, level, msg, match_id)

    def _save_frame(self, frame, label: str = "") -> Optional[str]:
        if not settings.FRAMES.get("save_captures", True):
            return None
        try:
            d = Path(settings.FRAMES["frames_dir"]) / self.stream_id
            d.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            p  = d / f"{ts}_{label}.jpg"
            existing = sorted(d.glob("*.jpg"))
            if len(existing) >= settings.FRAMES.get("max_frames_per_match", 20):
                existing[0].unlink()
            cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return str(p)
        except Exception as e:
            logger.warning(f"[{self.stream_id}] Erro ao salvar frame: {e}")
            return None

    def _cooldown_active(self) -> bool:
        return (time.time() - self._last_capture_time) < \
               settings.DETECTION.get("capture_cooldown_seconds", 30)

    # ────────────────────────────────────────────────────────────────────────
    # EXTRAÇÃO DE TEMPO — só no IDLE, a cada N frames
    # ────────────────────────────────────────────────────────────────────────

    def _extract_minute(self, frame) -> Optional[int]:
        """
        Extrai minuto do jogo do header da stream.
        Usa OCR leve restrito a dígitos e ':'.
        """
        try:
            h, w = frame.shape[:2]
            # ROI do header: parte central superior
            roi = frame[int(h*0.00):int(h*0.15), int(w*0.35):int(w*0.65)]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            enlarged = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(enlarged, 150, 255, cv2.THRESH_BINARY)
            text = pytesseract.image_to_string(
                thresh,
                config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:"
            )
            m = re.search(r'(\d{1,3}):\d{2}', text)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return None

    # ────────────────────────────────────────────────────────────────────────
    # ROI FIXO — YELLOW CARD
    # ────────────────────────────────────────────────────────────────────────

    def _crop_yellow_card_roi(self, frame) -> np.ndarray:
        """Recorta a região exata onde aparece 'Yellow Cards' na tabela."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = ROI_YELLOW_CARD
        return frame[int(h*y1):int(h*y2), int(w*x1):int(w*x2)]

    def _check_yellow_card_roi(self, frame) -> Tuple[bool, float]:
        """
        OCR na ROI fixa do Yellow Card.
        Retorna (encontrado, confiança_media).
        """
        try:
            crop = self._crop_yellow_card_roi(frame)

            # Pré-processamento para maximizar leitura
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            scaled = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR com dados de confiança
            data = pytesseract.image_to_data(
                thresh,
                config="--oem 3 --psm 7",
                output_type=pytesseract.Output.DICT
            )

            confidences = []
            found_yellow = False
            found_card   = False

            for i, word in enumerate(data["text"]):
                word_clean = word.strip().lower()
                conf = int(data["conf"][i]) if data["conf"][i] != "-1" else 0
                if not word_clean:
                    continue
                confidences.append(conf)
                if "yellow" in word_clean:
                    found_yellow = True
                if "card" in word_clean:
                    found_card = True

            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            found = (found_yellow or found_card) and avg_conf >= YELLOW_CARD_MIN_CONFIDENCE

            if found_yellow or found_card:
                logger.debug(
                    f"[{self.stream_id}] Yellow Cards ROI: "
                    f"yellow={found_yellow} card={found_card} conf={avg_conf:.1f}%"
                )

            return found, avg_conf

        except Exception as e:
            logger.debug(f"[{self.stream_id}] Erro OCR Yellow Card: {e}")
            return False, 0.0

    # ────────────────────────────────────────────────────────────────────────
    # MÁQUINA DE ESTADOS
    # ────────────────────────────────────────────────────────────────────────

    def _state_idle(self, frame):
        """
        Monitoramento leve.
        Verifica tempo a cada 4 frames (sem OCR pesado).
        Quando ≥ 85min → PRE_TRIGGER.
        """
        self._time_check_counter += 1
        if self._time_check_counter % 4 != 0:
            return  # só checa a cada 4 frames

        minute = self._extract_minute(frame)
        if minute is not None:
            self._match_minute = minute
            logger.debug(f"[{self.stream_id}] Tempo detectado: {minute}'")

            if ALERT_MINUTE_START <= minute <= ALERT_MINUTE_END:
                self._log("info", f"⚡ Minuto {minute}' — entrando em PRE_TRIGGER")
                self.capture.set_fps(settings.DETECTION["fps_alert"])
                self._set_state(WorkerState.PRE_TRIGGER)

    def _state_pre_trigger(self, frame):
        """
        Alta frequência de análise.
        Verifica estrutura visual da tabela.
        Quando tabela visível → checa ROI Yellow Card → TRIGGER.
        Timeout de 15 min sem detectar → volta para IDLE.
        """
        _, score = self.detector.is_stats_table_visible(frame)

        # Timeout: passou do minuto 99 sem capturar
        minute = self._extract_minute(frame)
        if minute is not None:
            self._match_minute = minute
            if minute > ALERT_MINUTE_END:
                self._log("info", f"Timeout PRE_TRIGGER (minuto {minute}'), voltando para IDLE")
                self.capture.set_fps(settings.DETECTION["fps_idle"])
                self._set_state(WorkerState.IDLE)
                return

        if score < 0.40:
            return  # tabela ainda não visível

        logger.debug(f"[{self.stream_id}] Tabela visível (score={score:.2f}), checando Yellow Card ROI...")

        found, conf = self._check_yellow_card_roi(frame)
        if found:
            self._log("info", f"🟡 Yellow Card detectado! conf={conf:.1f}% score={score:.2f} — iniciando confirmação")
            self._confirmations = 1
            self._set_state(WorkerState.CONFIRM)

    def _state_confirm(self, frame):
        """
        Anti-falso-positivo: confirma Yellow Card por MIN_CONFIRMATIONS frames consecutivos.
        Se perder a detecção, volta para PRE_TRIGGER.
        """
        found, conf = self._check_yellow_card_roi(frame)

        if found:
            self._confirmations += 1
            logger.debug(f"[{self.stream_id}] Confirmação {self._confirmations}/{MIN_CONFIRMATIONS} | conf={conf:.1f}%")
            if self._confirmations >= MIN_CONFIRMATIONS:
                self._log("info", f"✅ Tabela confirmada ({self._confirmations} frames) — disparando captura!")
                self._set_state(WorkerState.CAPTURE)
        else:
            logger.debug(f"[{self.stream_id}] Yellow Card perdido, voltando para PRE_TRIGGER")
            self._confirmations = 0
            self._set_state(WorkerState.PRE_TRIGGER)

    def _state_capture(self, first_frame) -> Optional[list]:
        """
        Burst capture: coleta frames rápidos para escolher o mais nítido.
        """
        if self._cooldown_active():
            self._log("info", "Cooldown ativo, pulando captura")
            self._set_state(WorkerState.COOLDOWN)
            return None

        burst_duration = settings.DETECTION.get("burst_window_seconds", 2.0)
        fps_burst      = settings.DETECTION.get("fps_capture", 15.0)

        self._log("info", f"📸 Burst capture — {burst_duration}s a {fps_burst}fps")
        frames = self.capture.get_burst_frames(burst_duration, fps_burst)
        frames = [first_frame] + (frames or [])

        if not frames:
            self._log("warning", "Burst vazio, abortando")
            self._set_state(WorkerState.PRE_TRIGGER)
            return None

        return frames

    def _state_ocr(self, frames: list) -> Optional[Dict]:
        """
        Seleciona melhor frame do burst e roda OCR completo.
        """
        best_frame, best_score = self.detector.select_best_frame(frames)
        if best_frame is None:
            self._log("warning", "Nenhum frame válido no burst")
            return None

        frame_path = self._save_frame(best_frame, f"capture_score{best_score:.2f}")
        self._log("info", f"🔍 OCR no melhor frame (score={best_score:.3f})")

        data = self.ocr.extract_from_frame(best_frame)
        data["frame_path"]  = frame_path
        data["stream_id"]   = self.stream_id
        data["stream_url"]  = self.stream_config["url"]
        data["match_minute"] = self._match_minute

        if not data.get("_extraction_success"):
            self._log("warning", f"OCR sem dados suficientes | frame={frame_path}")
            return None

        return data

    def _state_save(self, data: Dict) -> Optional[int]:
        """
        Persiste partida no banco e reseta estado.
        """
        clean = {k: v for k, v in data.items() if not k.startswith("_")}
        match_id = self.db.insert_match(clean)

        if match_id:
            self._last_capture_time = time.time()
            th   = clean.get("team_home", "?")
            ta   = clean.get("team_away", "?")
            gh   = clean.get("goals_home", "?")
            ga   = clean.get("goals_away", "?")
            summary = f"✅ Partida #{match_id} salva: {th} {gh}x{ga} {ta}"
            self._log("info", summary, match_id)
            self.db.update_stream_status(
                self.stream_id, self.stream_config["url"],
                is_online=True, increment_matches=True
            )
            return match_id
        else:
            self._log("warning", "Falha ao salvar partida no banco")
            return None

    # ────────────────────────────────────────────────────────────────────────
    # LOOP PRINCIPAL
    # ────────────────────────────────────────────────────────────────────────

    def run(self):
        setup_logger(self.stream_id)
        self._log("info", "═══ Worker iniciado ═══")
        self._log("info", f"URL: {self.stream_config['url']}")

        if not self.capture.start():
            self._log("error", "Falha ao iniciar captura")
            return

        self.detector.set_state(GameState.IN_GAME)
        self.capture.set_fps(settings.DETECTION["fps_idle"])
        self.db.update_stream_status(self.stream_id, self.stream_config["url"], is_online=True)

        frame_count       = 0
        errors_consecutive = 0
        MAX_ERRORS        = 10

        while True:
            try:
                frame = self.capture.get_frame(timeout=5.0)

                if frame is None:
                    errors_consecutive += 1
                    if errors_consecutive >= MAX_ERRORS:
                        self._log("error", "Muitos erros consecutivos, reconectando...")
                        self.db.update_stream_status(self.stream_id, self.stream_config["url"], is_online=False)
                        time.sleep(15)
                        self.capture.start()
                        errors_consecutive = 0
                    continue

                errors_consecutive = 0
                frame_count += 1

                # ── Máquina de estados ──────────────────────────────────────
                if self._state == WorkerState.IDLE:
                    self._state_idle(frame)

                elif self._state == WorkerState.PRE_TRIGGER:
                    self._state_pre_trigger(frame)

                elif self._state == WorkerState.CONFIRM:
                    self._state_confirm(frame)

                elif self._state == WorkerState.CAPTURE:
                    frames = self._state_capture(frame)
                    if frames:
                        self._set_state(WorkerState.OCR)
                        data = self._state_ocr(frames)
                        if data:
                            self._set_state(WorkerState.SAVE)
                            self._state_save(data)

                    # Após captura → cooldown → IDLE
                    post = settings.DETECTION.get("post_capture_cooldown_seconds", 120)
                    self._log("info", f"⏱ Aguardando {post}s para próxima partida...")
                    self.detector.set_state(GameState.POST_CAPTURE)
                    self._set_state(WorkerState.COOLDOWN)
                    time.sleep(post)
                    self.capture.set_fps(settings.DETECTION["fps_idle"])
                    self._set_state(WorkerState.IDLE)
                    self.detector.set_state(GameState.IN_GAME)
                    self._log("info", "▶ Monitorando nova partida")

                elif self._state == WorkerState.COOLDOWN:
                    pass  # aguardando sleep acima

                # ── Log de status periódico ─────────────────────────────────
                if frame_count % 200 == 0:
                    self._log("info",
                        f"Status: state={self._state.value} "
                        f"frames={frame_count} minuto={self._match_minute}'"
                    )

            except KeyboardInterrupt:
                self._log("info", "Interrompido pelo usuário")
                break
            except Exception as e:
                self._log("error", f"Erro no loop: {e}")
                logger.debug(traceback.format_exc())
                errors_consecutive += 1
                time.sleep(1)

        self.capture.stop()
        self._log("info", "═══ Worker encerrado ═══")


def run_worker_process(stream_config: dict):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    worker = StreamWorker(stream_config)
    worker.run()
