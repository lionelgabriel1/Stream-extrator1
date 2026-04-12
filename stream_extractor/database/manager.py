"""
Gerenciador de banco de dados.
Suporta PostgreSQL e SQLite com criação automática de tabelas.
"""

import sqlite3
import logging
import json
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── SQL: Criação das tabelas ────────────────────────────────────────────────────

CREATE_MATCHES_TABLE = """
CREATE TABLE IF NOT EXISTS matches (
    id                      SERIAL PRIMARY KEY,
    stream_id               VARCHAR(100) NOT NULL,
    stream_url              VARCHAR(255),

    -- Times e placar
    team_home               VARCHAR(100),
    team_away               VARCHAR(100),
    player_home             VARCHAR(100),
    player_away             VARCHAR(100),
    goals_home              INTEGER DEFAULT 0,
    goals_away              INTEGER DEFAULT 0,
    match_time              VARCHAR(10),

    -- Estatísticas principais
    possession_home         DECIMAL(5,1),
    possession_away         DECIMAL(5,1),
    ball_recovery_home      DECIMAL(5,1),
    ball_recovery_away      DECIMAL(5,1),
    shots_home              DECIMAL(5,1),
    shots_away              DECIMAL(5,1),
    expected_goals_home     DECIMAL(5,2),
    expected_goals_away     DECIMAL(5,2),
    passes_home             DECIMAL(6,1),
    passes_away             DECIMAL(6,1),
    tackles_home            DECIMAL(5,1),
    tackles_away            DECIMAL(5,1),
    tackles_won_home        DECIMAL(5,1),
    tackles_won_away        DECIMAL(5,1),
    interceptions_home      DECIMAL(5,1),
    interceptions_away      DECIMAL(5,1),
    saves_home              DECIMAL(5,1),
    saves_away              DECIMAL(5,1),
    fouls_committed_home    DECIMAL(5,1),
    fouls_committed_away    DECIMAL(5,1),
    offsides_home           DECIMAL(5,1),
    offsides_away           DECIMAL(5,1),
    corners_home            DECIMAL(5,1),
    corners_away            DECIMAL(5,1),
    free_kicks_home         DECIMAL(5,1),
    free_kicks_away         DECIMAL(5,1),
    penalty_kicks_home      DECIMAL(5,1),
    penalty_kicks_away      DECIMAL(5,1),
    yellow_cards_home       DECIMAL(5,1),
    yellow_cards_away       DECIMAL(5,1),

    -- Métricas circulares
    dribble_success_home    DECIMAL(5,1),
    dribble_success_away    DECIMAL(5,1),
    shot_accuracy_home      DECIMAL(5,1),
    shot_accuracy_away      DECIMAL(5,1),
    pass_accuracy_home      DECIMAL(5,1),
    pass_accuracy_away      DECIMAL(5,1),

    -- Campos calculados
    xg_total                DECIMAL(5,2),
    xg_diff                 DECIMAL(5,2),

    -- Metadados
    captured_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    frame_path              VARCHAR(500),
    ocr_engine              VARCHAR(50),
    ocr_raw_text            TEXT,
    confidence_score        DECIMAL(4,2),
    extraction_duration_ms  INTEGER,

    -- Status
    validated               BOOLEAN DEFAULT FALSE,
    notes                   TEXT
);
"""

CREATE_EXTRACTION_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS extraction_log (
    id              SERIAL PRIMARY KEY,
    stream_id       VARCHAR(100),
    event_type      VARCHAR(50),    -- 'detection', 'capture', 'ocr_success', 'ocr_fail', 'reconnect', 'error'
    message         TEXT,
    match_id        INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_STREAM_STATUS_TABLE = """
CREATE TABLE IF NOT EXISTS stream_status (
    stream_id       VARCHAR(100) PRIMARY KEY,
    stream_url      VARCHAR(255),
    is_online       BOOLEAN DEFAULT FALSE,
    last_seen       TIMESTAMP,
    last_match_at   TIMESTAMP,
    total_matches   INTEGER DEFAULT 0,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# SQLite usa autoincrement diferente
CREATE_MATCHES_TABLE_SQLITE = CREATE_MATCHES_TABLE.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
CREATE_EXTRACTION_LOG_TABLE_SQLITE = CREATE_EXTRACTION_LOG_TABLE.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")


# ─── Database Manager ────────────────────────────────────────────────────────────

class DatabaseManager:
    def __init__(self, config: Dict):
        self.config = config
        self.db_type = config.get("type", "sqlite")
        self._connection = None
        self._init_db()

    def _init_db(self):
        """Inicializa banco e cria tabelas se não existirem."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if self.db_type == "sqlite":
                    cursor.execute(CREATE_MATCHES_TABLE_SQLITE)
                    cursor.execute(CREATE_EXTRACTION_LOG_TABLE_SQLITE)
                    cursor.execute(CREATE_STREAM_STATUS_TABLE)
                else:
                    cursor.execute(CREATE_MATCHES_TABLE)
                    cursor.execute(CREATE_EXTRACTION_LOG_TABLE)
                    cursor.execute(CREATE_STREAM_STATUS_TABLE)
                conn.commit()
                logger.info(f"[DB] Banco inicializado ({self.db_type})")
        except Exception as e:
            logger.error(f"[DB] Erro ao inicializar banco: {e}")
            raise

    @contextmanager
    def _get_connection(self):
        """Context manager para conexão com o banco."""
        conn = None
        try:
            if self.db_type == "sqlite":
                Path(self.config.get("sqlite_path", "data/stats.db")).parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(
                    self.config.get("sqlite_path", "data/stats.db"),
                    timeout=30,
                    check_same_thread=False
                )
                conn.row_factory = sqlite3.Row
            else:
                import psycopg2
                conn = psycopg2.connect(
                    host=self.config["host"],
                    port=self.config["port"],
                    dbname=self.config["name"],
                    user=self.config["user"],
                    password=self.config["password"],
                    connect_timeout=10
                )
            yield conn
        finally:
            if conn:
                conn.close()

    def insert_match(self, data: Dict[str, Any]) -> Optional[int]:
        """
        Insere uma partida no banco.
        Retorna o ID inserido ou None em caso de erro.
        """
        fields = [
            "stream_id", "stream_url",
            "team_home", "team_away", "player_home", "player_away",
            "goals_home", "goals_away", "match_time",
            "possession_home", "possession_away",
            "ball_recovery_home", "ball_recovery_away",
            "shots_home", "shots_away",
            "expected_goals_home", "expected_goals_away",
            "passes_home", "passes_away",
            "tackles_home", "tackles_away",
            "tackles_won_home", "tackles_won_away",
            "interceptions_home", "interceptions_away",
            "saves_home", "saves_away",
            "fouls_committed_home", "fouls_committed_away",
            "offsides_home", "offsides_away",
            "corners_home", "corners_away",
            "free_kicks_home", "free_kicks_away",
            "penalty_kicks_home", "penalty_kicks_away",
            "yellow_cards_home", "yellow_cards_away",
            "dribble_success_home", "dribble_success_away",
            "shot_accuracy_home", "shot_accuracy_away",
            "pass_accuracy_home", "pass_accuracy_away",
            "xg_total", "xg_diff",
            "frame_path", "ocr_engine", "ocr_raw_text",
            "confidence_score", "extraction_duration_ms",
        ]

        # Calcula campos derivados
        xg_h = data.get("expected_goals_home") or 0
        xg_a = data.get("expected_goals_away") or 0
        data["xg_total"] = round(xg_h + xg_a, 2)
        data["xg_diff"] = round(xg_h - xg_a, 2)

        available = {f: data.get(f) for f in fields if f in data or data.get(f) is not None}
        cols = ", ".join(available.keys())
        
        if self.db_type == "sqlite":
            placeholders = ", ".join(["?" for _ in available])
        else:
            placeholders = ", ".join([f"%s" for _ in available])

        sql = f"INSERT INTO matches ({cols}) VALUES ({placeholders}) RETURNING id"
        if self.db_type == "sqlite":
            sql = f"INSERT INTO matches ({cols}) VALUES ({placeholders})"

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, list(available.values()))
                if self.db_type == "sqlite":
                    match_id = cursor.lastrowid
                else:
                    match_id = cursor.fetchone()[0]
                conn.commit()
                logger.info(f"[DB] Partida salva | ID={match_id} | {data.get('team_home')} {data.get('goals_home')}x{data.get('goals_away')} {data.get('team_away')}")
                return match_id
        except Exception as e:
            logger.error(f"[DB] Erro ao inserir partida: {e}")
            return None

    def log_event(self, stream_id: str, event_type: str, message: str, match_id: Optional[int] = None):
        """Registra evento no log de extração."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if self.db_type == "sqlite":
                    cursor.execute(
                        "INSERT INTO extraction_log (stream_id, event_type, message, match_id) VALUES (?, ?, ?, ?)",
                        (stream_id, event_type, message, match_id)
                    )
                else:
                    cursor.execute(
                        "INSERT INTO extraction_log (stream_id, event_type, message, match_id) VALUES (%s, %s, %s, %s)",
                        (stream_id, event_type, message, match_id)
                    )
                conn.commit()
        except Exception as e:
            logger.warning(f"[DB] Erro ao salvar log: {e}")

    def update_stream_status(self, stream_id: str, url: str, is_online: bool, increment_matches: bool = False):
        """Atualiza status da stream."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                now = datetime.now().isoformat()
                if self.db_type == "sqlite":
                    cursor.execute("""
                        INSERT INTO stream_status (stream_id, stream_url, is_online, last_seen, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(stream_id) DO UPDATE SET
                            is_online=excluded.is_online,
                            last_seen=excluded.last_seen,
                            updated_at=excluded.updated_at,
                            total_matches = CASE WHEN ? THEN total_matches+1 ELSE total_matches END
                    """, (stream_id, url, is_online, now, now, increment_matches))
                else:
                    cursor.execute("""
                        INSERT INTO stream_status (stream_id, stream_url, is_online, last_seen, updated_at)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT(stream_id) DO UPDATE SET
                            is_online=EXCLUDED.is_online,
                            last_seen=EXCLUDED.last_seen,
                            updated_at=EXCLUDED.updated_at,
                            total_matches = CASE WHEN %s THEN stream_status.total_matches+1 ELSE stream_status.total_matches END
                    """, (stream_id, url, is_online, now, now, increment_matches))
                conn.commit()
        except Exception as e:
            logger.warning(f"[DB] Erro ao atualizar status stream: {e}")

    def get_last_match(self, stream_id: str) -> Optional[Dict]:
        """Retorna a última partida capturada de uma stream."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if self.db_type == "sqlite":
                    cursor.execute(
                        "SELECT * FROM matches WHERE stream_id=? ORDER BY captured_at DESC LIMIT 1",
                        (stream_id,)
                    )
                else:
                    cursor.execute(
                        "SELECT * FROM matches WHERE stream_id=%s ORDER BY captured_at DESC LIMIT 1",
                        (stream_id,)
                    )
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.warning(f"[DB] Erro ao buscar última partida: {e}")
            return None
