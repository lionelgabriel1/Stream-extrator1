"""
Stream Worker v5 — Plano B
Sem pré-gatilho de tempo. Burst a cada 2s, Yellow Card é o único gatilho.

FLUXO:
  IDLE → burst a cada 2s → filtra tabela visível → Yellow Card? → full screen + ROIs → salva → cooldown → IDLE
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


class WorkerState(Enum):
    IDLE     = "idle"
    BURST    = "burst"
    EXTRACT  = "extract"
    COOLDOWN = "cooldown"


# ROIs fixas baseadas nas screenshots reais da ESB Football
ROIS = {
    "yellow_card": (0.33, 0.87, 0.67, 0.97),
    "placar":      (0.25, 0.00, 0.75, 0.13),
    "tabela":      (0.30, 0.15, 0.70, 0.98),
    "circulo_l":   (0.00, 0.15, 0.28, 0.98),
    "circulo_r":   (0.72, 0.15, 1.00, 0.98),
}

BURST_INTERVAL   = 2.0   # segundos entre bursts
BURST_FRAMES     = 20
BURST_FPS        = 10
YELLOW_MIN_CONF  = 75
TABLE_MIN_SCORE  = 0.40


class StreamWorker:

    def __init__(self, stream_config: dict):
        self.stream_config      = stream_config
        self.stream_id          = stream_config["id"]
        self._setup_components()
        self._state             = WorkerState.IDLE
        self._last_burst_time   = 0.0
        self._last_capture_time = 0.0

    def _setup_components(self):
        self.capture  = StreamCaptureWithReconnect(
            self.stream_config, settings.DETECTION, settings.RECONNECT)
        self.detector = TableDetector(settings.DETECTION, settings.ROI)
        self.ocr      = OCRExtractor(settings.OCR, settings.ROI)
        self.db       = DatabaseManager(settings.DATABASE)

    def _set_state(self, s: WorkerState):
        if s != self._state:
            logger.info(f"[{self.stream_id}] ► {self._state.value} → {s.value}")
            self._state = s

    def _crop(self, frame, roi_key: str) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = ROIS[roi_key]
        return frame[int(h*y1):int(h*y2), int(w*x1):int(w*x2)]

    def _preprocess(self, crop, scale=2.5) -> np.ndarray:
        gray   = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _, th  = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

    def _save_full_screen(self, frame, label="capture") -> str:
        d = Path(settings.FRAMES["frames_dir"]) / self.stream_id
        d.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        p  = d / f"{ts}_{label}.jpg"
        cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.info(f"[{self.stream_id}] 💾 Full screen: {p.name}")
        return str(p)

    def _check_yellow_card(self, frame) -> Tuple[bool, float]:
        try:
            crop  = self._crop(frame, "yellow_card")
            thresh = self._preprocess(crop, scale=3.0)
            data  = pytesseract.image_to_data(
                thresh, config="--oem 3 --psm 7",
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
            logger.debug(f"[{self.stream_id}] Yellow Card erro: {e}")
            return False, 0.0

    def _phase_burst(self) -> Optional[np.ndarray]:
        frames = self.capture.get_burst_frames(BURST_FRAMES / BURST_FPS, BURST_FPS)
        if not frames:
            return None

        # Filtra frames com tabela visível
        table_frames = []
        for f in frames:
            _, score = self.detector.is_stats_table_visible(f)
            if score > TABLE_MIN_SCORE:
                table_frames.append((f, score))

        logger.info(f"[{self.stream_id}] Burst: {len(table_frames)}/{len(frames)} com tabela visível")

        if not table_frames:
            return None

        # Checa Yellow Card nos frames filtrados
        for frame, score in sorted(table_frames, key=lambda x: x[1], reverse=True):
            found, conf = self._check_yellow_card(frame)
            if found:
                logger.info(f"[{self.stream_id}] 🟡 Yellow Card! score={score:.2f} conf={conf:.1f}%")
                return frame

        return None

    def _phase_extract(self, frame) -> Optional[Dict]:
        frame_path = self._save_full_screen(frame)

        logger.info(f"[{self.stream_id}] 🔍 Extraindo ROIs...")
        raw = {
            "placar":    self._ocr_roi(frame, "placar"),
            "tabela":    self._ocr_roi(frame, "tabela"),
            "circulo_l": self._ocr_roi(frame, "circulo_l"),
            "circulo_r": self._ocr_roi(frame, "circulo_r"),
        }

        data = self.ocr.extract_from_frame(frame)
        data["frame_path"] = frame_path
        data["stream_id"]  = self.stream_id
        data["stream_url"] = self.stream_config["url"]
        data["raw_rois"]   = raw

        if not data.get("_extraction_success"):
            logger.warning(f"[{self.stream_id}] Extração insuficiente")
            return None

        return data

    def _ocr_roi(self, frame, roi_key: str) -> str:
        try:
            crop = self._crop(frame, roi_key)
            th   = self._preprocess(crop, scale=3.5)
            return pytesseract.image_to_string(th, config="--oem 3 --psm 6")
        except Exception as e:
            logger.warning(f"[{self.stream_id}] OCR {roi_key} erro: {e}")
            return ""

    def _phase_save(self, data: Dict) -> Optional[int]:
        clean    = {k: v for k, v in data.items() if not k.startswith("_")}
        match_id = self.db.insert_match(clean)
        if match_id:
            self._last_capture_time = time.time()
            th = clean.get("team_home", "?")
            ta = clean.get("team_away", "?")
            gh = clean.get("goals_home", "?")
            ga = clean.get("goals_away", "?")
            logger.info(f"[{self.stream_id}] ✅ #{match_id}: {th} {gh}x{ga} {ta}")
            self.db.update_stream_status(
                self.stream_id, self.stream_config["url"],
                is_online=True, increment_matches=True
            )
        else:
            logger.warning(f"[{self.stream_id}] Falha ao salvar no banco")
        return match_id

    def run(self):
        setup_logger(self.stream_id)
        logger.info(f"[{self.stream_id}] ═══ Worker v5 (Plano B) iniciado ═══")

        if not self.capture.start():
            logger.error(f"[{self.stream_id}] Falha ao iniciar captura")
            return

        self.capture.set_fps(settings.DETECTION["fps_alert"])
        self.db.update_stream_status(self.stream_id, self.stream_config["url"], is_online=True)

        errors = 0

        while True:
            try:
                now = time.time()

                if self._state == WorkerState.IDLE:
                    # Burst a cada 2s
                    if now - self._last_burst_time >= BURST_INTERVAL:
                        self._last_burst_time = now
                        self._set_state(WorkerState.BURST)
                        best_frame = self._phase_burst()

                        if best_frame is not None:
                            self._set_state(WorkerState.EXTRACT)
                            data = self._phase_extract(best_frame)
                            if data:
                                self._phase_save(data)

                            post = settings.DETECTION.get("post_capture_cooldown_seconds", 120)
                            logger.info(f"[{self.stream_id}] ⏱ Cooldown {post}s...")
                            self._set_state(WorkerState.COOLDOWN)
                            time.sleep(post)
                            logger.info(f"[{self.stream_id}] ▶ Monitorando...")

                        self._set_state(WorkerState.IDLE)
                    else:
                        time.sleep(0.1)

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
