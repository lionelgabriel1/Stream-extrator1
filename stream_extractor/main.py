"""
Orquestrador principal do sistema.
Gerencia processos de cada stream com supervisão e reinício automático.
"""

import time
import signal
import logging
import sys
from multiprocessing import Process
from typing import Dict, List

from workers.stream_worker import run_worker_process
from utils.logger import setup_logger
from config import settings

logger = logging.getLogger(__name__)


class StreamOrchestrator:
    """
    Orquestra múltiplos workers de stream como processos independentes.
    Supervisiona e reinicia automaticamente em caso de falha.
    """

    def __init__(self):
        self._processes: Dict[str, Process] = {}
        self._running = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Captura CTRL+C para shutdown gracioso."""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        logger.info("═══ Shutdown solicitado ═══")
        self._running = False

    def _start_worker(self, stream_config: dict) -> Process:
        """Cria e inicia processo para uma stream."""
        stream_id = stream_config["id"]
        p = Process(
            target=run_worker_process,
            args=(stream_config,),
            name=f"worker-{stream_id}",
            daemon=False,
        )
        p.start()
        logger.info(f"[ORCHESTRATOR] ▶ Worker iniciado: {stream_id} (PID={p.pid})")
        return p

    def run(self):
        """Loop de supervisão principal."""
        setup_logger("orchestrator")
        self._running = True

        enabled_streams = [s for s in settings.STREAMS if s.get("enabled", True)]

        if not enabled_streams:
            logger.error("[ORCHESTRATOR] Nenhuma stream habilitada em config/settings.py")
            return

        logger.info(f"[ORCHESTRATOR] ═══ Iniciando {len(enabled_streams)} stream(s) ═══")

        # Inicia todos os workers
        for stream_config in enabled_streams:
            p = self._start_worker(stream_config)
            self._processes[stream_config["id"]] = p

        # Loop de supervisão
        while self._running:
            time.sleep(10)

            for stream_config in enabled_streams:
                stream_id = stream_config["id"]
                p = self._processes.get(stream_id)

                if p and not p.is_alive():
                    exit_code = p.exitcode
                    logger.warning(
                        f"[ORCHESTRATOR] Worker '{stream_id}' morreu "
                        f"(exit={exit_code}). Reiniciando em 15s..."
                    )
                    time.sleep(15)
                    new_p = self._start_worker(stream_config)
                    self._processes[stream_id] = new_p

        # Shutdown: encerra todos os processos
        logger.info("[ORCHESTRATOR] Encerrando workers...")
        for stream_id, p in self._processes.items():
            if p.is_alive():
                logger.info(f"[ORCHESTRATOR] Encerrando {stream_id} (PID={p.pid})...")
                p.terminate()
                p.join(timeout=10)
                if p.is_alive():
                    p.kill()

        logger.info("[ORCHESTRATOR] ═══ Sistema encerrado ═══")


def main():
    print("""
╔══════════════════════════════════════════════════════╗
║     STREAM STATS EXTRACTOR - Sistema de Captura      ║
║     Monitoramento 24h de streams esportivas          ║
╚══════════════════════════════════════════════════════╝
    """)

    orchestrator = StreamOrchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
