"""
Stream Worker v4 — Fluxo determinístico em 3 camadas:
  1. PRÉ-GATILHO  → tempo ≥ 85min
  2. GATILHO      → Yellow Cards detectado no burst
  3. EXTRAÇÃO     → full screen + subROIs → OCR → salva
"""

import cv2
import re
import time
import signal
import logging
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pytesseract

from core.capture import StreamCaptureWithReconnect
from detection.detector import TableDetector, GameState
from ocr.extractor import OCRExtractor
from database.manager import DatabaseManager
from utils.logger import setup_logger
from config import settings

logger = logging.getLogger(__name__)


# ─── ESTADOS ────────────────────────────────────────────────────────────────────
class WorkerState(Enum):
    IDLE        = "idle"         # monitoramento leve, lê tempo
    PRE_TRIGGER = "pre_trigger"  # ≥85min, aguarda 10s
    BURST       = "burst"        # captura N frames, busca Yellow Card
    EXTRACT     = "extract"      # full screen + subROIs + OCR
    COOLDOWN    = "cooldown"     # aguarda antes da próxima


# ─── ROIs FIXAS ─────────────────────────────────────────────────────────────────
# Coordenadas relativas (x1, y1, x2, y2) baseadas nas screenshots reais
ROIS = {
    "yellow_card": (0.33, 0.87, 0.67, 0.97),  # gatilho — última linha da tabela
    "placar":      (0.25, 0.00, 0.75, 0.13),  # times + gols + tempo
    "tabela":      (0.30, 0.15, 0.70, 0.98),  # stats centrais (Possession, Shots...)
    "circulo_l":   (0.00, 0.15, 0.28, 0.98),  # métricas % home
    "circulo_r":   (0.72, 0.15, 1.00, 0.98),  # métricas % away
}

ALERT_MINUTE     = 85   # pré-gatilho
TIMEOUT_MINUTE   = 99   # desiste se passar disso
PRE_WAIT_SECONDS = 10   # aguarda após 85min antes do burst
BURST_FRAMES     = 20   # frames por burst
BURST_FPS        = 10   # fps do burst
YELLOW_MIN_CONF  = 75   # confiança OCR mínima para Yellow Card (%)


class StreamWorker:

    def __init__(self, stream_config: dict):
        self.stream_config  = stream_config
        self.stream_id      = stream_config["id"]
        self._setup_components()
        self._state             = WorkerState.IDLE
        self._match_minute      = 0
        self._last_capture_time = 0.0
        self._frame_counter     = 0

    def _setup_components(self):
        self.capture  = StreamCaptureWithReconnect(
            self.stream_config, settings.DETECTION, settings.RECONNECT)
        self.detector = TableDetector(settings.DETECTION, settings.ROI)
        self.ocr      = OCRExtractor(settings.OCR, settings.ROI)
        self.db       = DatabaseManager(settings.DATABASE)

    # ─── UTILIDADES ─────────────────────────────────────────────────────────────

    def _set_state(self, s: WorkerState):
        if s != self._state:
            logger.info(f"[{self.stream_id}] ► {self._state.value} → {s.value}")
            self._state = s

    def _crop(self, frame, roi_key: str) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = ROIS[roi_key]
        return frame[int(h*y1):int(h*y2), int(w*x1):int(w*x2)]

    def _preprocess(self, crop, scale=2.5) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def _save_full_screen(self, frame, label="capture") -> str:
        d = Path(settings.FRAMES["frames_dir"]) / self.stream_id
        d.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p  = d / f"{ts}_{label}.jpg"
        cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"[{self.stream_id}] 💾 Full screen salvo: {p.name}")
        return str(p)

    # ─── CAMADA 1: PRÉ-GATILHO — TEMPO ──────────────────────────────────────────

    def _extract_minute(self, frame) -> Optional[int]:
        """OCR leve no header para extrair minuto do jogo."""
        try:
            crop = self._crop(frame, "placar")
            thresh = self._preprocess(crop, scale=2.0)
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

    def _phase_idle(self, frame):
        """Monitoramento leve — checa tempo a cada 4 frames."""
        self._frame_counter += 1
        if self._frame_counter % 4 != 0:
            return

        minute = self._extract_minute(frame)
        if minute is not None:
            self._match_minute = minute
            logger.debug(f"[{self.stream_id}] Tempo: {minute}'")

            if minute >= ALERT_MINUTE:
                logger.info(f"[{self.stream_id}] ⚡ {minute}' — pré-gatilho ativado, aguardando {PRE_WAIT_SECONDS}s...")
                self._set_state(WorkerState.PRE_TRIGGER)

    def _phase_pre_trigger(self):
        """Aguarda 10s após 85min para tabela começar a aparecer."""
        time.sleep(PRE_WAIT_SECONDS)
        logger.info(f"[{self.stream_id}] ✅ Aguardo concluído — iniciando burst")
        self.capture.set_fps(BURST_FPS)
        self._set_state(WorkerState.BURST)

    # ─── CAMADA 2: GATILHO — YELLOW CARD ────────────────────────────────────────

    def _check_yellow_card(self, frame) -> Tuple[bool, float]:
        """OCR na ROI fixa do Yellow Card. Retorna (encontrado, confiança)."""
        try:
            crop   = self._crop(frame, "yellow_card")
            thresh = self._preprocess(crop, scale=3.0)
            data   = pytesseract.image_to_data(
                thresh,
                config="--oem 3 --psm 7",
                output_type=pytesseract.Output.DICT
            )
            confs, found = [], False
            for i, word in enumerate(data["text"]):
                w = word.strip().lower()
                c = int(data["conf"][i]) if str(data["conf"][i]) != "-1" else 0
                if not w:
                    continue
                confs.append(c)
                if "yellow" in w or "card" in w:
                    found = True

            avg = sum(confs) / len(confs) if confs else 0
            return found and avg >= YELLOW_MIN_CONF, avg
        except Exception as e:
            logger.debug(f"[{self.stream_id}] Yellow Card check erro: {e}")
            return False, 0.0

    def _phase_burst(self) -> Optional[np.ndarray]:
        """
        Captura N frames rápidos.
        Filtra os que têm tabela visível.
        Nos filtrados, verifica Yellow Card.
        Retorna o melhor frame com Yellow Card confirmado, ou None.
        """
        logger.info(f"[{self.stream_id}] 📸 Iniciando burst ({BURST_FRAMES} frames)...")
        frames = self.capture.get_burst_frames(
            BURST_FRAMES / BURST_FPS, BURST_FPS
        )

        if not frames:
            logger.warning(f"[{self.stream_id}] Burst vazio")
            return None

        # Filtra frames com tabela visível
        table_frames = []
        for f in frames:
            _, score = self.detector.is_stats_table_visible(f)
            if score > 0.40:
                table_frames.append((f, score))

        logger.info(f"[{self.stream_id}] {len(table_frames)}/{len(frames)} frames com tabela visível")

        if not table_frames:
            return None

        # Nos frames filtrados, busca Yellow Card
        for frame, score in sorted(table_frames, key=lambda x: x[1], reverse=True):
            found, conf = self._check_yellow_card(frame)
            if found:
                logger.info(f"[{self.stream_id}] 🟡 Yellow Card confirmado! score={score:.2f} conf={conf:.1f}%")
                return frame

        logger.info(f"[{self.stream_id}] Yellow Card não encontrado neste burst")
        return None

    # ─── CAMADA 3: EXTRAÇÃO — FULL SCREEN + SUBROIs ──────────────────────────────

    def _ocr_roi(self, frame, roi_key: str) -> str:
        """OCR em uma ROI específica. Retorna texto bruto."""
        try:
            crop   = self._crop(frame, roi_key)
            thresh = self._preprocess(crop, scale=3.5)
            return pytesseract.image_to_string(thresh, config="--oem 3 --psm 6")
        except Exception as e:
            logger.warning(f"[{self.stream_id}] OCR erro em {roi_key}: {e}")
            return ""

    def _phase_extract(self, frame) -> Optional[Dict]:
        """
        Full screen salvo + OCR em cada subROI separadamente.
        Monta objeto completo com todos os dados.
        """
        # Salva full screen
        frame_path = self._save_full_screen(frame, "fullscreen")

        # OCR por subROI
        logger.info(f"[{self.stream_id}] 🔍 Extraindo subROIs...")
        raw = {
            "placar":    self._ocr_roi(frame, "placar"),
            "tabela":    self._ocr_roi(frame, "tabela"),
            "circulo_l": self._ocr_roi(frame, "circulo_l"),
            "circulo_r": self._ocr_roi(frame, "circulo_r"),
        }

        for roi, text in raw.items():
            logger.debug(f"[{self.stream_id}] ROI {roi}: {text[:80].strip()}")

        # OCR completo via extractor existente (monta objeto estruturado)
        data = self.ocr.extract_from_frame(frame)
        data["frame_path"]   = frame_path
        data["stream_id"]    = self.stream_id
        data["stream_url"]   = self.stream_config["url"]
        data["match_minute"] = self._match_minute
        data["raw_rois"]     = raw  # texto bruto de cada ROI para debug

        if not data.get("_extraction_success"):
            logger.warning(f"[{self.stream_id}] Extração insuficiente")
            return None

        return data

    def _phase_save(self, data: Dict) -> Optional[int]:
        clean    = {k: v for k, v in data.items() if not k.startswith("_")}
        match_id = self.db.insert_match(clean)
        if match_id:
            self._last_capture_time = time.time()
            th = clean.get("team_home", "?")
            ta = clean.get("team_away", "?")
            gh = clean.get("goals_home", "?")
            ga = clean.get("goals_away", "?")
            logger.info(f"[{self.stream_id}] ✅ #{match_id} salvo: {th} {gh}x{ga} {ta}")
            self.db.update_stream_status(
                self.stream_id, self.stream_config["url"],
                is_online=True, increment_matches=True
            )
        else:
            logger.warning(f"[{self.stream_id}] Falha ao salvar no banco")
        return match_id

    # ─── LOOP PRINCIPAL ──────────────────────────────────────────────────────────

    def run(self):
        setup_logger(self.stream_id)
        logger.info(f"[{self.stream_id}] ═══ Worker v4 iniciado ═══")

        if not self.capture.start():
            logger.error(f"[{self.stream_id}] Falha ao iniciar captura")
            return

        self.capture.set_fps(settings.DETECTION["fps_idle"])
        self.db.update_stream_status(self.stream_id, self.stream_config["url"], is_online=True)

        errors = 0
        burst_attempts = 0

        while True:
            try:
                # ── IDLE: lê frames leves ───────────────────────────────────
                if self._state == WorkerState.IDLE:
                    frame = self.capture.get_frame(timeout=5.0)
                    if frame is None:
                        errors += 1
                    else:
                        errors = 0
                        self._phase_idle(frame)

                # ── PRÉ-GATILHO: aguarda 10s ────────────────────────────────
                elif self._state == WorkerState.PRE_TRIGGER:
                    self._phase_pre_trigger()

                # ── BURST: captura + filtra + Yellow Card ───────────────────
                elif self._state == WorkerState.BURST:
                    best_frame = self._phase_burst()
                    burst_attempts += 1

                    if best_frame is not None:
                        # Yellow Card confirmado → extração
                        self._set_state(WorkerState.EXTRACT)
                        data = self._phase_extract(best_frame)
                        if data:
                            self._phase_save(data)

                        # Cooldown
                        post = settings.DETECTION.get("post_capture_cooldown_seconds", 120)
                        logger.info(f"[{self.stream_id}] ⏱ Cooldown {post}s...")
                        self._set_state(WorkerState.COOLDOWN)
                        time.sleep(post)
                        burst_attempts = 0
                        self.capture.set_fps(settings.DETECTION["fps_idle"])
                        self._set_state(WorkerState.IDLE)
                        logger.info(f"[{self.stream_id}] ▶ Monitorando nova partida")

                    else:
                        # Verifica timeout (minuto > 99)
                        frame = self.capture.get_frame(timeout=2.0)
                        if frame is not None:
                            minute = self._extract_minute(frame)
                            if minute and minute > TIMEOUT_MINUTE:
                                logger.warning(f"[{self.stream_id}] Timeout {minute}' — voltando para IDLE")
                                burst_attempts = 0
                                self.capture.set_fps(settings.DETECTION["fps_idle"])
                                self._set_state(WorkerState.IDLE)
                            else:
                                logger.info(f"[{self.stream_id}] Repetindo burst ({burst_attempts})...")

                # ── Reconexão por erros ─────────────────────────────────────
                if errors >= 10:
                    logger.error(f"[{self.stream_id}] Reconectando...")
                    self.db.update_stream_status(self.stream_id, self.stream_config["url"], is_online=False)
                    time.sleep(15)
                    self.capture.start()
                    errors = 0

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"[{self.stream_id}] Erro: {e}")
                logger.debug(traceback.format_exc())
                errors += 1
                time.sleep(1)

        self.capture.stop()
        logger.info(f"[{self.stream_id}] ═══ Worker encerrado ═══")


def run_worker_process(stream_config: dict):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    StreamWorker(stream_config).run()
