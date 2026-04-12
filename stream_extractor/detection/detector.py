"""
Detecção da tabela de estatísticas via OpenCV.
Usa análise de ROI, detecção de elementos visuais e OCR leve para confirmação.
"""

import cv2
import numpy as np
import logging
import re
import time
from typing import Optional, Tuple, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class GameState(Enum):
    """Estado atual da partida detectada na stream."""
    UNKNOWN = "unknown"
    IN_GAME = "in_game"
    ALERT = "alert"         # Tempo > 85min, atenção máxima
    TABLE_DETECTED = "table_detected"
    POST_CAPTURE = "post_capture"   # Aguardando próxima partida


class TableDetector:
    """
    Detecta a tabela de estatísticas no frame da stream.
    
    Estratégia em camadas:
    1. Análise rápida de cor/estrutura (sem OCR)
    2. Detecção de elementos visuais característicos
    3. Confirmação via OCR leve (só quando necessário)
    """

    def __init__(self, detection_config: dict, roi_config: dict):
        self.config = detection_config
        self.roi = roi_config
        self.threshold = detection_config.get("detection_threshold", 0.65)
        self._state = GameState.UNKNOWN
        self._last_state_change = time.time()
        self._detection_count = 0

    # ─── Utilidades ROI ─────────────────────────────────────────────────────────

    def _extract_roi(self, frame: np.ndarray, roi_key: str) -> np.ndarray:
        """Recorta região de interesse relativa (0.0 a 1.0)."""
        h, w = frame.shape[:2]
        roi = self.roi.get(roi_key, {"x": 0, "y": 0, "w": 1, "h": 1})
        x1 = int(roi["x"] * w)
        y1 = int(roi["y"] * h)
        x2 = int((roi["x"] + roi["w"]) * w)
        y2 = int((roi["y"] + roi["h"]) * h)
        return frame[y1:y2, x1:x2]

    # ─── Análise de cor: detecta overlay escuro translúcido da tela de stats ────

    def _has_dark_overlay(self, frame: np.ndarray) -> bool:
        """
        A tela de estatísticas tem um overlay escuro semi-transparente.
        Detecta redução significativa de brilho médio na tela.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        # Tela de stats: fundo escuro com texto branco -> brilho médio baixo
        return mean_brightness < 80

    # ─── Detecta linhas horizontais da tabela central ───────────────────────────

    def _detect_table_lines(self, frame: np.ndarray) -> float:
        """
        Detecta linhas horizontais características da tabela central.
        Retorna score 0.0-1.0.
        """
        # Recorta área central onde fica a tabela
        center_roi = self._extract_roi(frame, "stats_table")
        gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)

        # Detecta bordas
        edges = cv2.Canny(gray, 50, 150)

        # Detecta linhas horizontais via Hough
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180,
            threshold=80,
            minLineLength=center_roi.shape[1] * 0.3,  # >= 30% da largura
            maxLineGap=20
        )

        if lines is None:
            return 0.0

        # Conta linhas horizontais (angle ≈ 0°)
        horizontal = [l for l in lines if abs(l[0][1] - l[0][3]) < 5]
        score = min(len(horizontal) / 8.0, 1.0)  # normaliza para 0-1
        return score

    # ─── Detecta barra colorida lateral (verde/laranja) da tabela ───────────────

    def _detect_colored_bars(self, frame: np.ndarray) -> float:
        """
        A tabela tem barras coloridas (verde/laranja) indicando vantagem.
        Detecta presença dessas cores na área central.
        """
        center_roi = self._extract_roi(frame, "stats_table")
        hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)

        # Verde (barra de vantagem home)
        green_mask = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))
        # Laranja/amarelo (barra de vantagem away)
        orange_mask = cv2.inRange(hsv, np.array([10, 100, 100]), np.array([30, 255, 255]))

        green_ratio = np.sum(green_mask > 0) / green_mask.size
        orange_ratio = np.sum(orange_mask > 0) / orange_mask.size

        return min((green_ratio + orange_ratio) * 20, 1.0)

    # ─── Detecta texto branco sobre fundo escuro (padrão da tabela) ─────────────

    def _detect_white_text_pattern(self, frame: np.ndarray) -> float:
        """
        A tabela tem texto branco sobre fundo escuro.
        Detecta esse padrão na área central da tabela.
        """
        center_roi = self._extract_roi(frame, "stats_table")
        gray = cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY)

        # Threshold: pixels muito claros (texto branco)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        white_ratio = np.sum(thresh > 0) / thresh.size

        # Esperado: entre 5% e 35% de pixels brancos (texto + números)
        if 0.05 < white_ratio < 0.35:
            return 0.8
        elif 0.03 < white_ratio < 0.50:
            return 0.4
        return 0.0

    # ─── Detecta os círculos de métricas (Dribble, Shot, Pass) ─────────────────

    def _detect_metric_circles(self, frame: np.ndarray) -> float:
        """
        Detecta os círculos de métricas nos cantos esquerdo/direito.
        São círculos grandes com contorno verde e texto de %.
        """
        score = 0.0
        for side in ["circles_left", "circles_right"]:
            roi = self._extract_roi(frame, side)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)

            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=40,
                param1=50,
                param2=30,
                minRadius=30,
                maxRadius=120
            )

            if circles is not None:
                score += min(len(circles[0]) / 3.0, 0.5)  # até 0.5 por lado

        return min(score, 1.0)

    # ─── Extrai tempo de jogo do header ─────────────────────────────────────────

    def extract_match_time(self, frame: np.ndarray) -> Optional[str]:
        """
        Extrai o tempo de jogo do header (ex: "91:43").
        Usa OCR leve só na ROI do placar.
        """
        try:
            # Import aqui para não forçar dependência se não usar
            import pytesseract
            
            header_roi = self._extract_roi(frame, "score_header")
            gray = cv2.cvtColor(header_roi, cv2.COLOR_BGR2GRAY)
            
            # Upscale para melhor OCR
            scale = 2.0
            enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
            # Threshold
            _, thresh = cv2.threshold(enlarged, 150, 255, cv2.THRESH_BINARY)
            
            text = pytesseract.image_to_string(
                thresh,
                config="--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789:"
            )
            
            # Busca padrão MM:SS
            match = re.search(r'(\d{1,3}):(\d{2})', text)
            if match:
                return match.group(0)
        except Exception:
            pass
        return None

    # ─── Score composto ─────────────────────────────────────────────────────────

    def compute_detection_score(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Computa score de detecção combinando múltiplos indicadores.
        Retorna dict com scores individuais e score final.
        """
        scores = {}

        scores["dark_overlay"] = 0.8 if self._has_dark_overlay(frame) else 0.0
        scores["table_lines"] = self._detect_table_lines(frame)
        scores["colored_bars"] = self._detect_colored_bars(frame)
        scores["white_text"] = self._detect_white_text_pattern(frame)
        scores["circles"] = self._detect_metric_circles(frame)

        # Pesos para cada indicador
        weights = {
            "dark_overlay": 0.15,
            "table_lines": 0.25,
            "colored_bars": 0.25,
            "white_text": 0.20,
            "circles": 0.15,
        }

        final = sum(scores[k] * weights[k] for k in scores)
        scores["final"] = final

        return scores

    # ─── API principal ───────────────────────────────────────────────────────────

    def is_stats_table_visible(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Verifica se a tabela de estatísticas está visível no frame.
        Retorna (detectado, score).
        """
        scores = self.compute_detection_score(frame)
        detected = scores["final"] >= self.threshold

        if detected:
            self._detection_count += 1
            logger.debug(
                f"[DETECT] Score={scores['final']:.3f} | "
                f"lines={scores['table_lines']:.2f} bars={scores['colored_bars']:.2f} "
                f"text={scores['white_text']:.2f} circles={scores['circles']:.2f}"
            )

        return detected, scores["final"]

    def select_best_frame(self, frames: list) -> Tuple[Optional[np.ndarray], float]:
        """
        De uma lista de frames (burst), seleciona o mais nítido
        e com maior score de detecção.
        """
        if not frames:
            return None, 0.0

        best_frame = None
        best_score = -1.0

        for frame in frames:
            _, score = self.is_stats_table_visible(frame)
            # Também considera nitidez (Laplacian variance)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() / 10000
            combined = score * 0.7 + min(sharpness, 1.0) * 0.3

            if combined > best_score:
                best_score = combined
                best_frame = frame

        logger.info(f"[DETECT] Melhor frame selecionado | score={best_score:.3f} | total={len(frames)} frames")
        return best_frame, best_score

    def get_state(self) -> GameState:
        return self._state

    def set_state(self, state: GameState):
        if state != self._state:
            logger.info(f"[STATE] {self._state.value} → {state.value}")
            self._state = state
            self._last_state_change = time.time()

    def time_in_state(self) -> float:
        return time.time() - self._last_state_change
