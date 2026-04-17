"""
Worker de processamento de stream.
Cada stream roda como processo independente com multiprocessing.

Lógica de disparo:
  - Monitora em fps_alert continuamente (sem idle)
  - A cada frame: checa score visual + Yellow Cards simultaneamente
  - Se score > 0.45 E Yellow Cards visível → dispara burst capture imediatamente
  - Janela de 500ms~2s: o burst começa no mesmo frame que confirmou
"""

import cv2
import logging
import time
import signal
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import pytesseract

from core.capture import StreamCaptureWithReconnect
from detection.detector import TableDetector, GameState
from ocr.extractor import OCRExtractor
from database.manager import DatabaseManager
from utils.logger import setup_logger
from config import settings


logger = logging.getLogger(__name__)


class StreamWorker:
    def __init__(self, stream_config: dict):
        self.stream_config = stream_config
        self.stream_id = stream_config["id"]
        self._setup_components()
        self._last_capture_time = 0.0

    def _setup_components(self):
        self.capture = StreamCaptureWithReconnect(
            self.stream_config,
            settings.DETECTION,
            settings.RECONNECT,
        )
        self.detector = TableDetector(settings.DETECTION, settings.ROI)
        self.ocr = OCRExtractor(settings.OCR, settings.ROI)
        self.db = DatabaseManager(settings.DATABASE)

    def _save_frame(self, frame, label: str = "") -> Optional[str]:
        if not settings.FRAMES.get("save_captures", True):
            return None
        try:
            frames_dir = Path(settings.FRAMES["frames_dir"]) / self.stream_id
            frames_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = frames_dir / f"{ts}_{label}.jpg"
            existing = sorted(frames_dir.glob("*.jpg"))
            max_frames = settings.FRAMES.get("max_frames_per_match", 20)
            if len(existing) >= max_frames:
                existing[0].unlink()
            cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return str(path)
        except Exception as e:
            logger.warning(f"[{self.stream_id}] Erro ao salvar frame: {e}")
            return None

    def _is_cooldown_active(self) -> bool:
        elapsed = time.time() - self._last_capture_time
        cooldown = settings.DETECTION.get("capture_cooldown_seconds", 30)
        return elapsed < cooldown

    def _log_event(self, event_type: str, message: str, match_id=None):
        self.db.log_event(self.stream_id, event_type, message, match_id)
        logger.info(f"[{self.stream_id}] [{event_type}] {message}")

    def _format_match_summary(self, data: Dict) -> str:
        th = data.get("team_home", "?")
        ta = data.get("team_away", "?")
        gh = data.get("goals_home", "?")
        ga = data.get("goals_away", "?")
        ph = data.get("player_home", "")
        pa = data.get("player_away", "")
        player_str = f" ({ph} vs {pa})" if ph or pa else ""
        xg_h = data.get("expected_goals_home", 0) or 0
        xg_a = data.get("expected_goals_away", 0) or 0
        return (
            f"MATCH: {th}{player_str} {gh} x {ga} {ta} | "
            f"XG: {xg_h:.2f} vs {xg_a:.2f} | "
            f"Posse: {data.get('possession_home','?')}% vs {data.get('possession_away','?')}%"
        )

    def _has_yellow_cards(self, frame) -> bool:
        """
        OCR rápido só na faixa inferior da tabela (onde fica Yellow Cards).
        Retorna True se 'yellow' ou 'cards' aparecer no texto.
        """
        try:
            h, w = frame.shape[:2]
            y1 = int(h * 0.88)
            y2 = int(h * 1.00)
            x1 = int(w * 0.30)
            x2 = int(w * 0.70)
            crop = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            text = pytesseract.image_to_string(
                thresh, config="--oem 3 --psm 7"
            ).lower()
            found = "yellow" in text or "cards" in text
            if found:
                logger.info(f"[{self.stream_id}] 🟡 Yellow Cards detectado!")
            return found
        except Exception as e:
            logger.debug(f"[{self.stream_id}] Erro no check Yellow Cards: {e}")
            return False

    def _should_capture(self, frame) -> bool:
        """
        Gatilho: score visual > 0.45 E Yellow Cards visível no mesmo frame.
        """
        _, score = self.detector.is_stats_table_visible(frame)
        if score < 0.45:
            return False
        logger.debug(f"[{self.stream_id}] Score visual OK ({score:.2f}), checando Yellow Cards...")
        return self._has_yellow_cards(frame)

    def _phase_capture(self, first_frame) -> Optional[Dict]:
        if self._is_cooldown_active():
            logger.info(f"[{self.stream_id}] Em cooldown, pulando captura")
            return None
        self._log_event("capture", "Iniciando burst capture")
        burst_duration = settings.DETECTION.get("burst_window_seconds", 3.0)
        fps_burst = settings.DETECTION.get("fps_capture", 15.0)
        frames = self.capture.get_burst_frames(burst_duration, fps_burst)
        frames = [first_frame] + frames
        best_frame, best_score = self.detector.select_best_frame(frames)
        if best_frame is None:
            logger.warning(f"[{self.stream_id}] Burst vazio, abortando")
            return None
        frame_path = self._save_frame(best_frame, f"capture_score{best_score:.2f}")
        logger.info(f"[{self.stream_id}] Rodando OCR no melhor frame (score={best_score:.3f})")
        data = self.ocr.extract_from_frame(best_frame)
        data["frame_path"] = frame_path
        data["stream_id"] = self.stream_id
        data["stream_url"] = self.stream_config["url"]
        return data if data.get("_extraction_success") else None

    def _phase_save(self, data: Dict) -> Optional[int]:
        clean_data = {k: v for k, v in data.items() if not k.startswith("_")}
        match_id = self.db.insert_match(clean_data)
        if match_id:
            self._last_capture_time = time.time()
            summary = self._format_match_summary(clean_data)
            self._log_event("ocr_success", summary, match_id)
            self.db.update_stream_status(
                self.stream_id, self.stream_config["url"],
                is_online=True, increment_matches=True
            )
            return match_id
        else:
            self._log_event("ocr_fail", "Falha ao salvar no banco de dados")
            return None

    def run(self):
        setup_logger(self.stream_id)
        logger.info(f"[{self.stream_id}] ═══ Worker iniciado ═══")
        logger.info(f"[{self.stream_id}] URL: {self.stream_config['url']}")

        if not self.capture.start():
            logger.error(f"[{self.stream_id}] Falha ao iniciar captura")
            return

        self.detector.set_state(GameState.IN_GAME)
        # Sempre em fps_alert — não tem idle, não perde a janela de 500ms~2s
        self.capture.set_fps(settings.DETECTION["fps_alert"])
        self.db.update_stream_status(self.stream_id, self.stream_config["url"], is_online=True)

        frame_count = 0
        errors_consecutive = 0
        MAX_ERRORS = 10

        while True:
            try:
                frame = self.capture.get_frame(timeout=5.0)

                if frame is None:
                    errors_consecutive += 1
                    if errors_consecutive >= MAX_ERRORS:
                        logger.error(f"[{self.stream_id}] Muitos erros consecutivos, reiniciando...")
                        self.db.update_stream_status(self.stream_id, self.stream_config["url"], is_online=False)
                        time.sleep(15)
                        self.capture.start()
                        errors_consecutive = 0
                    continue

                errors_consecutive = 0
                frame_count += 1

                if self._should_capture(frame):
                    self.detector.set_state(GameState.TABLE_DETECTED)
                    data = self._phase_capture(frame)

                    if data:
                        match_id = self._phase_save(data)
                        if match_id:
                            logger.info(f"[{self.stream_id}] ✅ Partida #{match_id} salva com sucesso!")
                    else:
                        logger.warning(f"[{self.stream_id}] OCR falhou, frame descartado")
                        self._log_event("ocr_fail", "Extração sem dados suficientes")

                    post_cooldown = settings.DETECTION.get("post_capture_cooldown_seconds", 120)
                    logger.info(f"[{self.stream_id}] ⏱ Aguardando {post_cooldown}s para próxima partida...")
                    self.detector.set_state(GameState.POST_CAPTURE)
                    time.sleep(post_cooldown)
                    self.detector.set_state(GameState.IN_GAME)
                    logger.info(f"[{self.stream_id}] ▶ Monitorando nova partida")

                if frame_count % 100 == 0:
                    logger.info(f"[{self.stream_id}] Status: frames={frame_count}")

            except KeyboardInterrupt:
                logger.info(f"[{self.stream_id}] Interrompido pelo usuário")
                break
            except Exception as e:
                logger.error(f"[{self.stream_id}] Erro no loop: {e}")
                logger.debug(traceback.format_exc())
                errors_consecutive += 1
                time.sleep(2)

        self.capture.stop()
        logger.info(f"[{self.stream_id}] ═══ Worker encerrado ═══")


def run_worker_process(stream_config: dict):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    worker = StreamWorker(stream_config)
    worker.run()
