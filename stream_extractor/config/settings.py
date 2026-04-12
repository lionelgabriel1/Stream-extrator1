"""
Configurações centrais do sistema.
As credenciais são lidas do ficheiro .env (criado pelo setup_db.py).
"""

import os
from pathlib import Path

# Carrega o ficheiro .env se existir
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith('#') and '=' in _line:
            _key, _, _val = _line.partition('=')
            os.environ.setdefault(_key.strip(), _val.strip())

# ─── STREAMS ────────────────────────────────────────────────────────────────────
STREAMS = [
    {
        "id":       "esbfootball",
        "name":     "ESB Football",
        "url":      os.environ.get("STREAM_URL", "https://www.twitch.tv/esbfootball"),
        "platform": "twitch",
        "quality":  os.environ.get("STREAM_QUALITY", "720p"),
        "enabled":  True,
    },
]

# ─── BANCO DE DADOS ──────────────────────────────────────────────────────────────
DATABASE = {
    "type":     "postgresql",
    "host":     os.environ.get("DB_HOST",     "localhost"),
    "port":     int(os.environ.get("DB_PORT", "5432")),
    "name":     os.environ.get("DB_NAME",     "stream_stats"),
    "user":     os.environ.get("DB_USER",     "postgres"),
    "password": os.environ.get("DB_PASSWORD", ""),

    # SQLite (fallback se type="sqlite")
    "sqlite_path": "data/stats.db",
}

# ─── DETECÇÃO ────────────────────────────────────────────────────────────────────
DETECTION = {
    "fps_idle":                      0.5,
    "fps_alert":                     5.0,
    "fps_capture":                   15.0,
    "alert_minute":                  85,
    "burst_window_seconds":          3.0,
    "detection_threshold":           0.65,
    "capture_cooldown_seconds":      30,
    "post_capture_cooldown_seconds": 120,
}

# ─── ROI (mantido por compatibilidade com detector.py) ───────────────────────────
ROI = {
    "score_header": {
        "x": 0.25, "y": 0.00, "w": 0.50, "h": 0.15,
    },
    "stats_table": {
        "x": 0.30, "y": 0.15, "w": 0.40, "h": 0.85,
    },
    "full_stats_screen": {
        "x": 0.20, "y": 0.00, "w": 0.60, "h": 1.00,
    },
    "circles_left": {
        "x": 0.00, "y": 0.15, "w": 0.28, "h": 0.85,
    },
    "circles_right": {
        "x": 0.72, "y": 0.15, "w": 0.28, "h": 0.85,
    },
}

# ─── OCR ─────────────────────────────────────────────────────────────────────────
OCR = {
    "primary":  "tesseract",
    "fallback": "easyocr",
    "language": "eng",
    "min_confidence": 0,
    "preprocessing": {
        "scale_factor":      3.5,
        "grayscale":         True,
        "denoise":           False,
        "threshold_method":  "fixed",
        "contrast_enhance":  1.0,
        "sharpness_enhance": 1.0,
    },
}

# ─── RECONEXÃO ───────────────────────────────────────────────────────────────────
RECONNECT = {
    "max_attempts":      9999,
    "delay_seconds":     15,
    "backoff_multiplier": 1.5,
    "max_delay_seconds": 120,
}

# ─── LOGGING ─────────────────────────────────────────────────────────────────────
LOGGING = {
    "level":          "INFO",
    "log_dir":        "logs",
    "max_file_mb":    50,
    "backup_count":   5,
    "console_output": True,
}

# ─── FRAMES ──────────────────────────────────────────────────────────────────────
FRAMES = {
    "save_captures":        True,
    "frames_dir":           "frames",
    "max_frames_per_match": 20,
    "cleanup_after_days":   7,
}

# ─── KEYWORDS para detecção da tabela ────────────────────────────────────────────
TABLE_KEYWORDS = [
    "Yellow Cards", "Possession", "Summary", "Shots",
    "Passes", "Tackles", "Interceptions", "Free Kicks",
    "Penalty Kicks", "Expected Goals",
]
MIN_KEYWORDS_TO_CONFIRM = 3
