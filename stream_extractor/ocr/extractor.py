"""
Extração de dados via OCR — calibrado para esbfootball (layout fixo).

Solução final:
  - Máscara dos pixels verdes (barra lateral da tabela) antes do threshold
  - Detecção automática de barras pretas laterais (suporte 16:9 e 21:9)
  - HOME: thresh=120 + máscara verde, scale=3.5, psm=6
  - AWAY: thresh=100, scale=3.5, psm=6
  - Alinhamento Y por proximidade (tolerância=0.055)
"""

import cv2
import numpy as np
import logging
import re
import time
from typing import Optional, Dict, Any, Tuple, List

logger = logging.getLogger(__name__)

# ─── ROI base (relativos ao conteúdo útil, sem barras pretas) ───────────────────
# Estes valores são aplicados APÓS detecção e remoção das barras laterais
ROI_HEADER    = (0.250, 0.000, 0.750, 0.135)
ROI_HOME_NUMS = (0.295, 0.185, 0.365, 0.985)
ROI_AWAY_NUMS = (0.635, 0.185, 0.705, 0.985)
ROI_LABELS    = (0.370, 0.185, 0.630, 0.985)
ROI_CIRCLES_L = (0.000, 0.185, 0.280, 0.985)
ROI_CIRCLES_R = (0.720, 0.185, 1.000, 0.985)

# ─── Parâmetros OCR ──────────────────────────────────────────────────────────────
THRESH_HOME  = 120   # com máscara verde
THRESH_AWAY  = 100
SCALE_NUMS   = 3.5
SCALE_LABELS = 3.0
SCALE_HEADER = 2.5
Y_TOLERANCE  = 0.055

# Máscara HSV para pixels verdes (barra lateral da tabela)
GREEN_HSV_LOW  = np.array([40,  80,  80])
GREEN_HSV_HIGH = np.array([90, 255, 255])

# ─── Posições Y dos labels (relativas ao crop 0.185h–0.985h) ────────────────────
STAT_Y = [
    (0.081, 'possession'),
    (0.137, 'ball_recovery'),
    (0.196, 'shots'),
    (0.250, 'expected_goals'),
    (0.310, 'passes'),
    (0.364, 'tackles'),
    (0.423, 'tackles_won'),
    (0.477, 'interceptions'),
    (0.534, 'saves'),
    (0.591, 'fouls_committed'),
    (0.650, 'offsides'),
    (0.706, 'corners'),
    (0.763, 'free_kicks'),
    (0.820, 'penalty_kicks'),
    (0.874, 'yellow_cards'),
]


# ─── Detecção de barras pretas laterais ─────────────────────────────────────────

def detect_content_bounds(frame: np.ndarray) -> Tuple[int, int]:
    """
    Detecta os limites horizontais do conteúdo útil.
    Necessário para streams 21:9 com barras pretas laterais.
    Retorna (x_start, x_end) em pixels.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    col_means = gray.mean(axis=0)
    content = [i for i, m in enumerate(col_means) if m > 15]
    if not content:
        return 0, w
    return content[0], content[-1]


def adapt_roi(roi: tuple, x_start: int, x_end: int, frame_w: int) -> tuple:
    """Adapta ROI base para frame com barras laterais."""
    cw = x_end - x_start
    x1 = x_start / frame_w + roi[0] * cw / frame_w
    x2 = x_start / frame_w + roi[2] * cw / frame_w
    return (x1, roi[1], x2, roi[3])


# ─── Helpers de imagem ───────────────────────────────────────────────────────────

def _crop(frame: np.ndarray, roi: tuple) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1 = int(roi[0]*w), int(roi[1]*h)
    x2, y2 = int(roi[2]*w), int(roi[3]*h)
    return frame[y1:y2, x1:x2]


def _mask_green(crop: np.ndarray) -> np.ndarray:
    """Remove pixels verdes da barra lateral (pinta-os de preto)."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_HSV_LOW, GREEN_HSV_HIGH)
    result = crop.copy()
    result[mask > 0] = [0, 0, 0]
    return result


def _to_thresh(crop_bgr: np.ndarray, scale: float,
               thresh_val: Optional[int] = None,
               mask_green: bool = False) -> np.ndarray:
    """Converte crop para imagem binária pronta para OCR."""
    img = _mask_green(crop_bgr) if mask_green else crop_bgr
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    en = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if thresh_val is None:
        _, th = cv2.threshold(en, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, th = cv2.threshold(en, thresh_val, 255, cv2.THRESH_BINARY)
    return th


# ─── Extracção de números ────────────────────────────────────────────────────────

def _extract_nums_with_y(thresh: np.ndarray) -> List[Tuple[float, float]]:
    """Extrai (y_rel, valor) de todos os tokens numéricos."""
    import pytesseract
    th_h = thresh.shape[0]
    data = pytesseract.image_to_data(
        thresh,
        config="--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789. ",
        output_type=pytesseract.Output.DICT
    )
    results = []
    for i in range(len(data['text'])):
        txt = data['text'][i].strip()
        if not txt or not re.match(r'^\d+\.?\d*$', txt) or txt == '.':
            continue
        y_mid = (data['top'][i] + data['height'][i] / 2) / th_h
        try:
            results.append((y_mid, float(txt)))
        except ValueError:
            pass
    return sorted(results)


def _match_nums_to_stats(nums: List[Tuple[float, float]]) -> Dict[str, float]:
    """Associa cada número à stat mais próxima por Y."""
    matched: Dict[str, float] = {}
    used: set = set()
    for y_num, val in nums:
        best_key, best_dist = None, 999.0
        for y_stat, key in STAT_Y:
            if key in used:
                continue
            dist = abs(y_num - y_stat)
            if dist < best_dist and dist < Y_TOLERANCE:
                best_dist = dist
                best_key = key
        if best_key:
            matched[best_key] = val
            used.add(best_key)
    return matched


# ─── Header ──────────────────────────────────────────────────────────────────────

def _extract_header(frame: np.ndarray, roi: tuple) -> Dict[str, Any]:
    """Extrai times, placar e tempo de jogo."""
    import pytesseract
    crop = _crop(frame, roi)
    th = _to_thresh(crop, SCALE_HEADER)
    text = pytesseract.image_to_string(th, config="--oem 3 --psm 6")
    result: Dict[str, Any] = {}

    m = re.search(r'(\d{1,3}:\d{2})', text)
    if m:
        result['match_time'] = m.group(1)

    IGNORE = {'TIME','GOL','GOLA','GOLH','BACK','TOGGLE','SCROLL',
               'SUMMARY','POSSESSION','SHOOTING','PASSING','DEFENDING','EVENTS'}
    teams = [t.strip() for t in re.findall(r'\b([A-Z]{3,}(?:\s+[A-Z]{2,})?)\b', text)
             if t.strip().upper() not in IGNORE and len(t.strip()) >= 3]
    if len(teams) >= 2:
        result['team_home'] = teams[0]
        result['team_away'] = teams[-1]
    elif len(teams) == 1:
        result['team_home'] = teams[0]

    for pat in [r'\b(\d)\s*[|\[\]]\s*(\d)\b', r'(?<!\d)(\d)\s*:\s*(\d)(?!\d)']:
        m = re.search(pat, text)
        if m:
            try:
                result['goals_home'] = int(m.group(1))
                result['goals_away'] = int(m.group(2))
                break
            except ValueError:
                pass

    players = [p for p in re.findall(r'\b([a-z][a-z0-9_]{2,15})\b', text)
               if p not in ('back','scroll','toggle','left','right')]
    if len(players) >= 2:
        result['player_home'] = players[0]
        result['player_away'] = players[-1]

    return result


# ─── Círculos ────────────────────────────────────────────────────────────────────

def _extract_circles(frame: np.ndarray, side: str, roi: tuple) -> Dict[str, float]:
    """Extrai Dribble, Shot e Pass Accuracy dos círculos laterais."""
    import pytesseract
    crop = _crop(frame, roi)
    th = _to_thresh(crop, 2.5, thresh_val=160)
    text = pytesseract.image_to_string(
        th,
        config="--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789% "
    )
    suffix = 'home' if side == 'home' else 'away'
    result: Dict[str, float] = {}

    patterns = {
        f'dribble_success_{suffix}': r'dribble.*?(\d+)\s*%',
        f'shot_accuracy_{suffix}':   r'shot.*?(\d+)\s*%',
        f'pass_accuracy_{suffix}':   r'pass.*?(\d+)\s*%',
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            result[key] = float(m.group(1))

    pcts = re.findall(r'(\d+)\s*%', text)
    keys_order = [f'dribble_success_{suffix}', f'shot_accuracy_{suffix}', f'pass_accuracy_{suffix}']
    for i, pct in enumerate(pcts[:3]):
        if keys_order[i] not in result:
            result[keys_order[i]] = float(pct)

    return result


# ─── Confirmação rápida da tabela ────────────────────────────────────────────────

def confirm_table_visible(frame: np.ndarray, roi_labels: tuple) -> Tuple[bool, int]:
    """Verifica se a tabela está visível (OCR leve nos labels)."""
    import pytesseract
    crop = _crop(frame, roi_labels)
    th = _to_thresh(crop, SCALE_LABELS)
    text = pytesseract.image_to_string(th, config="--oem 3 --psm 6").lower()
    keywords = ['possession','shots','passes','tackles',
                'interceptions','saves','fouls','offsides','corners','yellow']
    found = sum(1 for kw in keywords if kw in text)
    return found >= 4, found


# ─── API pública ─────────────────────────────────────────────────────────────────

class OCRExtractor:
    """Extrai dados completos da tabela de estatísticas de um frame."""

    def __init__(self, config: dict = None, roi_config: dict = None):
        self.config = config or {}

    def confirm_table_visible(self, frame: np.ndarray) -> Tuple[bool, int]:
        h, w = frame.shape[:2]
        xs, xe = detect_content_bounds(frame)
        roi = adapt_roi(ROI_LABELS, xs, xe, w)
        return confirm_table_visible(frame, roi)

    def extract_from_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        start = time.time()
        result: Dict[str, Any] = {'_extraction_success': False}

        try:
            h, w = frame.shape[:2]

            # Detecta barras laterais e adapta ROIs
            xs, xe = detect_content_bounds(frame)
            roi_header    = adapt_roi(ROI_HEADER,    xs, xe, w)
            roi_home_nums = adapt_roi(ROI_HOME_NUMS, xs, xe, w)
            roi_away_nums = adapt_roi(ROI_AWAY_NUMS, xs, xe, w)
            roi_circles_l = adapt_roi(ROI_CIRCLES_L, xs, xe, w)
            roi_circles_r = adapt_roi(ROI_CIRCLES_R, xs, xe, w)

            # 1. Header
            result.update(_extract_header(frame, roi_header))

            # 2. Coluna HOME (com máscara verde)
            home_th   = _to_thresh(_crop(frame, roi_home_nums), SCALE_NUMS,
                                   THRESH_HOME, mask_green=True)
            home_nums = _extract_nums_with_y(home_th)
            home_data = _match_nums_to_stats(home_nums)

            # 3. Coluna AWAY (sem máscara — barra fica à direita dos números)
            away_th   = _to_thresh(_crop(frame, roi_away_nums), SCALE_NUMS,
                                   THRESH_AWAY, mask_green=False)
            away_nums = _extract_nums_with_y(away_th)
            away_data = _match_nums_to_stats(away_nums)

            for key, val in home_data.items():
                result[f"{key}_home"] = val
            for key, val in away_data.items():
                result[f"{key}_away"] = val

            # 4. Default 0 para campos de cartões/penaltis não detectados
            for zero_field in ['penalty_kicks', 'yellow_cards', 'red_cards']:
                if f"{zero_field}_home" not in result:
                    result[f"{zero_field}_home"] = 0.0
                if f"{zero_field}_away" not in result:
                    result[f"{zero_field}_away"] = 0.0

            # 5. Círculos
            result.update(_extract_circles(frame, 'home', roi_circles_l))
            result.update(_extract_circles(frame, 'away', roi_circles_r))

            # 6. Campos calculados
            xg_h = result.get('expected_goals_home') or 0
            xg_a = result.get('expected_goals_away') or 0
            result['xg_total'] = round(xg_h + xg_a, 2)
            result['xg_diff']  = round(xg_h - xg_a, 2)

            # 7. Metadados
            ms = int((time.time() - start) * 1000)
            result['ocr_engine']             = 'tesseract'
            result['extraction_duration_ms'] = ms
            result['_home_fields']           = len(home_data)
            result['_away_fields']           = len(away_data)
            result['_content_bounds']        = (xs, xe)
            result['_extraction_success']    = len(home_data) >= 5 and len(away_data) >= 5

            logger.info(
                f"[OCR] {ms}ms | home={len(home_data)}/15 away={len(away_data)}/15 | "
                f"{result.get('team_home','?')} {result.get('goals_home','?')}x"
                f"{result.get('goals_away','?')} {result.get('team_away','?')} | "
                f"bounds=({xs},{xe})"
            )

        except ImportError:
            result['_error'] = 'pytesseract não instalado'
            logger.error("[OCR] pytesseract não instalado")
        except Exception as e:
            result['_error'] = str(e)
            logger.error(f"[OCR] Erro: {e}", exc_info=True)

        return result
