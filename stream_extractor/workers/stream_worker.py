"""
Worker de processamento de stream.
Cada stream roda como processo independente com multiprocessing.
Loop principal: captura → detecção → OCR → banco de dados.
"""

import cv2
import logging
import time
import os
import signal
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from multiprocessing import Process, Queue, Event

from core.capture import StreamCaptureWithReconnect
from detection.detector import TableDetector, GameState
from ocr.extractor import OCRExtractor
from database.manager import DatabaseManager
from utils.logger import setup_logger
from config import settings


logger = logging.getLogger(__name__)


class StreamWorker:
    """
    Worker que processa uma única stream continuamente.
    
    Máquina de estados:
    UNKNOWN → IN_GAME → ALERT (>85min) → TABLE_DETECTED → POST_CAPTURE → IN_GAME → ...
    """

    def __init__(self, stream_config: dict):
        self.stream_config = stream_config
        self.stream_id = stream_config["id"]
        self._setup_components()
        self._last_capture_time = 0.0
        self._consecutive_detections = 0
        self._min_confirmations = 2  # Frames consecutivos para confirmar tabela

    def _setup_components(self):
        """Inicializa todos os componentes do worker."""
        self.capture = StreamCaptureWithReconnect(
            self.stream_config,
            settings.DETECTION,
            settings.RECONNECT,
        )
        self.detector = TableDetector(settings.DETECTION, settings.ROI)
        self.ocr = OCRExtractor(settings.OCR, settings.ROI)
        self.db = DatabaseManager(settings.DATABASE)

    # ─── Helpers ────────────────────────────────────────────────────────────────

    def _save_frame(self, frame, label: str = "") -> Optional[str]:
        """Salva frame em disco para debugging."""
        if not settings.FRAMES.get("save_captures", True):
            return None
        try:
            frames_dir = Path(settings.FRAMES["frames_dir"]) / self.stream_id
            frames_dir.mkdir(parents=True, exist_ok=True)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ts}_{label}.jpg"
            path = frames_dir / filename

            # Limita número de frames salvo
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
        """Verifica se está em cooldown após última captura."""
        elapsed = time.time() - self._last_capture_time
        cooldown = settings.DETECTION.get("capture_cooldown_seconds", 30)
        return elapsed < cooldown

    def _log_event(self, event_type: str, message: str, match_id=None):
        """Log no banco e no logger."""
        self.db.log_event(self.stream_id, event_type, message, match_id)
        logger.info(f"[{self.stream_id}] [{event_type}] {message}")

    def _format_match_summary(self, data: Dict) -> str:
        """Formata resumo da partida para log."""
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

    # ─── Fases do processamento ──────────────────────────────────────────────────

    def _phase_idle(self, frame) -> bool:
        """
        Fase idle: processa frame leve, verifica se deve subir para alerta.
        Retorna True se detectou algo.
        """
        # Tenta extrair tempo de jogo
        match_time = self.detector.extract_match_time(frame)
        
        if match_time:
            # Parseia minuto
            try:
                minutes = int(match_time.split(":")[0])
                alert_minute = settings.DETECTION.get("alert_minute", 85)
                
                if minutes >= alert_minute:
                    self.detector.set_state(GameState.ALERT)
                    self.capture.set_fps(settings.DETECTION["fps_alert"])
                    logger.info(f"[{self.stream_id}] ⚡ Modo ALERTA ativado | tempo={match_time}")
                    self._log_event("detection", f"Modo alerta ativado, tempo={match_time}")
                    return True
            except (ValueError, IndexError):
                pass

        # Verificação rápida de estrutura visual mesmo sem tempo
        _, score = self.detector.is_stats_table_visible(frame)
        if score > 0.5:
            logger.debug(f"[{self.stream_id}] Possível tabela detectada (score={score:.2f}), subindo para ALERTA")
            self.detector.set_state(GameState.ALERT)
            self.capture.set_fps(settings.DETECTION["fps_alert"])

        return False

    def _phase_alert(self, frame) -> bool:
        """
        Fase alerta: verifica frames com mais frequência.
        Retorna True se tabela detectada.
        """
        detected, score = self.detector.is_stats_table_visible(frame)
        
        if detected:
            self._consecutive_detections += 1
            logger.debug(f"[{self.stream_id}] Detecção consecutiva: {self._consecutive_detections} | score={score:.3f}")
            
            if self._consecutive_detections >= self._min_confirmations:
                self.detector.set_state(GameState.TABLE_DETECTED)
                self._log_event("detection", f"Tabela confirmada após {self._consecutive_detections} frames | score={score:.3f}")
                return True
        else:
            self._consecutive_detections = 0

        # Timeout: se ficou muito tempo no alerta sem detectar tabela, volta para idle
        if self.detector.time_in_state() > 300:  # 5 minutos
            logger.info(f"[{self.stream_id}] Timeout no modo alerta, voltando para idle")
            self.detector.set_state(GameState.IN_GAME)
            self.capture.set_fps(settings.DETECTION["fps_idle"])
            self._consecutive_detections = 0

        return False

    def _phase_capture(self, first_frame) -> Optional[Dict]:
        """
        Fase de captura: faz burst de frames e processa o melhor.
        Retorna dados extraídos ou None.
        """
        if self._is_cooldown_active():
            logger.info(f"[{self.stream_id}] Em cooldown, pulando captura")
            self.detector.set_state(GameState.IN_GAME)
            return None

        self._log_event("capture", "Iniciando burst capture")

        # Burst: captura vários frames pela janela de tempo
        burst_duration = settings.DETECTION.get("burst_window_seconds", 3.0)
        fps_burst = settings.DETECTION.get("fps_capture", 15.0)
        
        frames = self.capture.get_burst_frames(burst_duration, fps_burst)
        
        # Adiciona o primeiro frame que disparou a detecção
        frames = [first_frame] + frames

        # Seleciona o melhor frame
        best_frame, best_score = self.detector.select_best_frame(frames)
        
        if best_frame is None:
            logger.warning(f"[{self.stream_id}] Burst vazio, abortando")
            return None

        # Salva frame para debugging
        frame_path = self._save_frame(best_frame, f"capture_score{best_score:.2f}")
        
        # OCR
        logger.info(f"[{self.stream_id}] Rodando OCR no melhor frame (score={best_score:.3f})")
        data = self.ocr.extract_from_frame(best_frame)
        data["frame_path"] = frame_path
        data["stream_id"] = self.stream_id
        data["stream_url"] = self.stream_config["url"]

        return data if data.get("_extraction_success") else None

    def _phase_save(self, data: Dict) -> Optional[int]:
        """Salva dados no banco de dados."""
        # Remove campos internos antes de salvar
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

    # ─── Loop principal ──────────────────────────────────────────────────────────

    def run(self):
        """Loop principal do worker. Roda continuamente."""
        setup_logger(self.stream_id)
        logger.info(f"[{self.stream_id}] ═══ Worker iniciado ═══")
        logger.info(f"[{self.stream_id}] URL: {self.stream_config['url']}")

        # Inicia captura com reconexão automática
        if not self.capture.start():
            logger.error(f"[{self.stream_id}] Falha ao iniciar captura")
            return

        self.detector.set_state(GameState.IN_GAME)
        self.capture.set_fps(settings.DETECTION["fps_idle"])
        self.db.update_stream_status(self.stream_id, self.stream_config["url"], is_online=True)

        frame_count = 0
        errors_consecutive = 0
        MAX_ERRORS = 10

        while True:
            try:
                # Obtém frame
                frame = self.capture.get_frame(timeout=5.0)
                
                if frame is None:
                    errors_consecutive += 1
                    if errors_consecutive >= MAX_ERRORS:
                        logger.error(f"[{self.stream_id}] Muitos erros consecutivos, reiniciando...")
                        self.db.update_stream_status(self.stream_id, self.stream_config["url"], is_online=False)
                        time.sleep(15)
                        self.capture.start()  # Tenta reconectar
                        errors_consecutive = 0
                    continue

                errors_consecutive = 0
                frame_count += 1
                state = self.detector.get_state()

                # ── Máquina de estados ──────────────────────────────────────────
                if state == GameState.IN_GAME:
                    self._phase_idle(frame)

                elif state == GameState.ALERT:
                    detected = self._phase_alert(frame)
                    
                    if detected:
                        # Entra na fase de captura
                        data = self._phase_capture(frame)
                        
                        if data:
                            match_id = self._phase_save(data)
                            if match_id:
                                logger.info(f"[{self.stream_id}] ✅ Partida #{match_id} salva com sucesso!")
                        else:
                            logger.warning(f"[{self.stream_id}] OCR falhou, frame descartado")
                            self._log_event("ocr_fail", "Extração sem dados suficientes")

                        # Entra em cooldown post-captura
                        post_cooldown = settings.DETECTION.get("post_capture_cooldown_seconds", 120)
                        logger.info(f"[{self.stream_id}] ⏱ Aguardando {post_cooldown}s para próxima partida...")
                        
                        self.detector.set_state(GameState.POST_CAPTURE)
                        self.capture.set_fps(settings.DETECTION["fps_idle"])
                        self._consecutive_detections = 0
                        time.sleep(post_cooldown)
                        
                        self.detector.set_state(GameState.IN_GAME)
                        logger.info(f"[{self.stream_id}] ▶ Monitorando nova partida")

                elif state == GameState.POST_CAPTURE:
                    # Não deveria chegar aqui (sleep acima), mas safety net
                    time.sleep(1)

                # Log periódico
                if frame_count % 100 == 0:
                    logger.info(f"[{self.stream_id}] Status: estado={state.value} | frames={frame_count}")

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


# ─── Função para rodar em processo separado ──────────────────────────────────────

def run_worker_process(stream_config: dict):
    """Entry point para multiprocessing."""
    # Ignora SIGINT no processo filho (deixa o pai gerenciar)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    worker = StreamWorker(stream_config)
    worker.run()
