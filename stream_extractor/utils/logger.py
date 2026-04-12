"""
Configuração de logging com rotação de arquivos.
"""

import logging
import logging.handlers
from pathlib import Path
from config import settings


def setup_logger(name: str = "root") -> logging.Logger:
    """
    Configura logger com output para console e arquivo com rotação.
    """
    log_config = settings.LOGGING
    log_dir = Path(log_config.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    level_str = log_config.get("level", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Evita handlers duplicados
    if root_logger.handlers:
        return root_logger

    # Console
    if log_config.get("console_output", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

    # Arquivo com rotação
    log_file = log_dir / f"{name}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=log_config.get("max_file_mb", 50) * 1024 * 1024,
        backupCount=log_config.get("backup_count", 5),
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)

    return root_logger
