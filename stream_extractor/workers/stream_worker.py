"""
Stream Worker v6 — Pré-gatilho por tempo (6 layouts de placar) + gatilho Yellow Card.

FLUXO:
  IDLE       → testa 6 layouts até achar tempo → ≥85min → PRE_TRIGGER
  PRE_TRIGGER → aguarda 10s → BURST
  BURST      → OCR na ROI Yellow Card frame a frame → encontrou → EXTRACT
  EXTRACT    → full screen + 4 subROIs → OCR → SAVE → COOLDOWN → IDLE
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
    IDLE        = "idle"
    PRE_TRIGGER = "pre_trigger"
    BURST       = "burst"
    EXTRACT     = "extract"
    COOLDOWN    = "cooldown"


# ─── 6 LAYOUTS DE PLACAR (coordenadas relativas 1920x1080) ──────────────────
# Onde aparece o tempo de jogo em cada modelo de placar
SCORE_LAYOUTS = {
    "LAYOUT_1": (0.0611, 0.0972, 0.0828, 0.1385),
    "LAYOUT_2": (0.1028, 0.1259, 0.0839, 0.1240),
    "LAYOUT_3": (0.0815, 0.1157, 0.0875, 0.1422),
    "LAYOUT_4": (0.1139, 0.1417, 0.1292, 0.1688),
    "LAYOUT_5": (0.0519, 0.0843, 0.0578, 0.1083),
    "LAYOUT_6": (0.1046, 0.1306, 0.0474, 0.0818),
}

# ─── ROIs DA TABELA DE STATS ─────────────────────────────────────────────────
ROIS = {
    "yellow_card": (0.33, 0.87, 0.67, 0.97),
    "placar":      (0.25, 0.00, 0.75, 0.13),
    "tabela":      (0.30, 0.15, 0.70, 0.98),
    "circulo_l":   (0.00, 0.15, 0.28, 0.98),
    "circulo_r":   (0.72, 0.15, 1.00, 0.98),
}

ALERT_MINUTE     = 85
TIMEOUT_MINUTE   = 99
PRE_WAIT_SECONDS = 10
BURST_FPS        = 10
BURST_SECONDS    = 2.0
YELLOW_MIN_CONF  = 75
CHECK_INTERVAL   = 4   # checa tempo a cada N frames no IDLE


class StreamWorker:

    def __init__(self, stream_config: dict):
        self.stream_config      = stream_config
        self.stream_id          = stream_config["id"]
        self._setup_components()
        self._state             = WorkerState.IDLE
        self._match_minute      = 0
        self._last_capture_time = 0.0
        self._frame_counter     = 0
        self._active_layout     = None

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

    def _crop_abs(self, frame, y1r, y2r, x1r, x2r) -> np.ndarray:
        h, w = frame.shape[:2]
        return frame[int(h*y1r):int(h*y2r), int(w*x1r):int(w*x2r)]

    def _crop_roi(self, frame, roi_key: str) -> np.ndarray:
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

    # ─── EXTRAÇÃO DE TEMPO — testa os 6 layouts ─────────────────────────────

    def _extract_minute(self, frame) -> Tuple[Optional[int], Optional[str]]:
        """
        Testa os 6 layouts de placar em sequência.
        Retorna (minuto, layout_name) quando encontrar tempo válido.
        """
        # Se já identificou o layout ativo, testa ele primeiro
        layouts = list(SCORE_LAYOUTS.items())
        if self._active_layout:
            layouts = [(self._active_layout, SCORE_LAYOUTS[self._active_layout])] + \
                      [(k, v) for k, v in layouts if k != self._active_layout]

        for name, (y1, y2, x1, x2) in layouts:
            try:
                crop   = self._crop_abs(frame, y1, y2, x1, x2)
                thresh = self._preprocess(crop, scale=3.0)
                text   = pytesseract.image_to_string(
                    thresh,
                    config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:+"
                )
                m = re.search(r'(\d{1,3}):\d{2}', text)
                if m:
                    minute = int(m.group(1))
                    if 0 < minute <= 120:
                        return minute, name
            except Exception:
                continue

        return None, None

    # ─── GATILHO YELLOW CARD ─────────────────────────────────────────────────

    def _check_yellow_card(self, frame) -> Tuple[bool, float]:
        try:
            crop  = self._crop_roi(frame, "yellow_card")
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

    # ─── EXTRAÇÃO FULL SCREEN + SUBROIs ──────────────────────────────────────

    def _ocr_roi(self, frame, roi_key: str) -> str:
        try:
            crop = self._crop_roi(frame, roi_key)
            th   = self._preprocess(crop, scale=3.5)
            return pytesseract.image_to_string(th, config="--oem 3 --psm 6")
        except Exception as e:
            logger.warning(f"[{self.stream_id}] OCR {roi_key} erro: {e}")
            return ""

    def _phase_extract(self, frame) -> Optional[Dict]:
        frame_path = self._save_full_screen(frame)

        logger.info(f"[{self.stream_id}] 🔍 Extraindo subROIs...")
        raw = {
            "placar":    self._ocr_roi(frame, "placar"),
            "tabela":    self._ocr_roi(frame, "tabela"),
            "circulo_l": self._ocr_roi(frame, "circulo_l"),
            "circulo_r": self._ocr_roi(frame, "circulo_r"),
        }

        data = self.ocr.extract_from_frame(frame)
        data["frame_path"]    = frame_path
        data["stream_id"]     = self.stream_id
        data["stream_url"]    = self.stream_config["url"]
        data["match_minute"]  = self._match_minute
        data["active_layout"] = self._active_layout
        data["raw_rois"]      = raw

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
            logger.info(f"[{self.stream_id}] ✅ #{match_id}: {th} {gh}x{ga} {ta}")
            self.db.update_stream_status(
                self.stream_id, self.stream_config["url"],
                is_online=True, increment_matches=True
            )
        else:
            logger.warning(f"[{self.stream_id}] Falha ao salvar no banco")
        return match_id

    # ─── LOOP PRINCIPAL ──────────────────────────────────────────────────────

    def run(self):
        setup_logger(self.stream_id)
        logger.info(f"[{self.stream_id}] ═══ Worker v6 iniciado ═══")

        if not self.capture.start():
            logger.error(f"[{self.stream_id}] Falha ao iniciar captura")
            return

        self.capture.set_fps(settings.DETECTION["fps_idle"])
        self.db.update_stream_status(self.stream_id, self.stream_config["url"], is_online=True)

        errors = 0

        while True:
            try:
                frame = self.capture.get_frame(timeout=5.0)
                if frame is None:
                    errors += 1
                    if errors >= 10:
                        logger.error(f"[{self.stream_id}] Reconectando...")
                        self.db.update_stream_status(self.stream_id, self.stream_config["url"], is_online=False)
                        time.sleep(15)
                        self.capture.start()
                        errors = 0
                    continue

                errors = 0
                self._frame_counter += 1

                # ── IDLE: lê tempo a cada N frames ──────────────────────────
                if self._state == WorkerState.IDLE:
                    if self._frame_counter % CHECK_INTERVAL != 0:
                        continue

                    minute, layout = self._extract_minute(frame)
                    if minute is not None:
                        self._match_minute  = minute
                        self._active_layout = layout
                        logger.info(f"[{self.stream_id}] ⏱ {minute}' [{layout}]")

                        if minute >= ALERT_MINUTE:
                            logger.info(f"[{self.stream_id}] ⚡ Pré-gatilho: {minute}' — aguardando {PRE_WAIT_SECONDS}s...")
                            self._set_state(WorkerState.PRE_TRIGGER)

                # ── PRÉ-GATILHO: aguarda 10s ─────────────────────────────────
                elif self._state == WorkerState.PRE_TRIGGER:
                    time.sleep(PRE_WAIT_SECONDS)
                    self.capture.set_fps(settings.DETECTION["fps_alert"])
                    logger.info(f"[{self.stream_id}] 📸 Iniciando burst...")
                    self._set_state(WorkerState.BURST)

                # ── BURST: checa Yellow Card frame a frame ───────────────────
                elif self._state == WorkerState.BURST:
                    # Verifica timeout
                    minute, _ = self._extract_minute(frame)
                    if minute and minute > TIMEOUT_MINUTE:
                        logger.warning(f"[{self.stream_id}] Timeout {minute}' — voltando para IDLE")
                        self.capture.set_fps(settings.DETECTION["fps_idle"])
                        self._set_state(WorkerState.IDLE)
                        continue

                    found, conf = self._check_yellow_card(frame)
                    if found:
                        logger.info(f"[{self.stream_id}] 🟡 Yellow Card! conf={conf:.1f}%")
                        self._set_state(WorkerState.EXTRACT)

                        data = self._phase_extract(frame)
                        if data:
                            self._phase_save(data)

                        post = settings.DETECTION.get("post_capture_cooldown_seconds", 120)
                        logger.info(f"[{self.stream_id}] ⏱ Cooldown {post}s...")
                        self._set_state(WorkerState.COOLDOWN)
                        time.sleep(post)
                        self.capture.set_fps(settings.DETECTION["fps_idle"])
                        self._frame_counter = 0
                        self._set_state(WorkerState.IDLE)
                        logger.info(f"[{self.stream_id}] ▶ Monitorando nova partida")

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
