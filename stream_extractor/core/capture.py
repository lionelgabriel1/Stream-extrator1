"""
Captura de frames da stream via streamlink + ffmpeg.
Sem navegador, direto ao vídeo. Suporta Twitch e YouTube.
"""

import cv2
import subprocess
import numpy as np
import logging
import threading
import time
import re
from typing import Optional, Tuple, Generator
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class StreamCapture:
    """
    Captura frames de uma stream Twitch/YouTube via streamlink.
    Usa pipe ffmpeg para leitura direta de frames sem decodificar tudo.
    """

    def __init__(self, stream_config: dict, capture_config: dict):
        self.stream_id = stream_config["id"]
        self.stream_url = stream_config["url"]
        self.quality = stream_config.get("quality", "720p")
        self.capture_config = capture_config

        self._process: Optional[subprocess.Popen] = None
        self._streamlink_process: Optional[subprocess.Popen] = None
        self._running = False
        self._frame_queue: Queue = Queue(maxsize=30)

        self._width = 1280
        self._height = 720
        self._fps_target = capture_config.get("fps_idle", 0.5)
        self._last_frame_time = 0.0

        self._lock = threading.Lock()

    # ─── Resolução ──────────────────────────────────────────────────────────────

    def _get_dimensions(self) -> Tuple[int, int]:
        """Mapeia qualidade para dimensões."""
        quality_map = {
            "1080p": (1920, 1080),
            "720p":  (1280, 720),
            "480p":  (854, 480),
            "360p":  (640, 360),
            "best":  (1280, 720),
            "worst": (640, 360),
        }
        return quality_map.get(self.quality, (1280, 720))

    # ─── Obter URL do stream via streamlink ─────────────────────────────────────

    def _get_stream_url(self) -> Optional[str]:
        """
        Usa streamlink para obter a URL direta do vídeo.
        Suporta Twitch e YouTube.
        """
        try:
            qualities_to_try = [self.quality, "best", "720p", "480p", "worst"]
            
            for q in qualities_to_try:
                cmd = ["streamlink", "--stream-url", self.stream_url, q]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0 and result.stdout.strip():
                    url = result.stdout.strip()
                    logger.info(f"[CAPTURE:{self.stream_id}] URL obtida ({q}): {url[:80]}...")
                    return url
                    
        except subprocess.TimeoutExpired:
            logger.error(f"[CAPTURE:{self.stream_id}] Timeout ao obter URL da stream")
        except FileNotFoundError:
            logger.error("[CAPTURE] streamlink não instalado. Execute: pip install streamlink")
        except Exception as e:
            logger.error(f"[CAPTURE:{self.stream_id}] Erro ao obter URL: {e}")
        return None

    # ─── Iniciar processo ffmpeg ─────────────────────────────────────────────────

    def _start_ffmpeg(self, stream_url: str) -> Optional[subprocess.Popen]:
        """Inicia processo ffmpeg que decodifica frames para pipe."""
        self._width, self._height = self._get_dimensions()

        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-i", stream_url,

            # Performance: decodifica só o necessário
            "-an",                              # Sem áudio
            "-vf", f"scale={self._width}:{self._height},fps=30",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",                # Formato OpenCV
            "-",                                 # Output para pipe
        ]

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=10**8
            )
            logger.info(f"[CAPTURE:{self.stream_id}] ffmpeg iniciado | {self._width}x{self._height}")
            return process
        except FileNotFoundError:
            logger.error("[CAPTURE] ffmpeg não instalado. Execute: apt install ffmpeg")
            return None
        except Exception as e:
            logger.error(f"[CAPTURE:{self.stream_id}] Erro ao iniciar ffmpeg: {e}")
            return None

    # ─── Reader de frames em thread separada ────────────────────────────────────

    def _frame_reader_thread(self):
        """Thread que lê frames do pipe ffmpeg e coloca na fila."""
        frame_size = self._width * self._height * 3  # bgr24

        while self._running and self._process:
            try:
                # Controle de FPS
                now = time.time()
                interval = 1.0 / max(self._fps_target, 0.1)
                if now - self._last_frame_time < interval:
                    time.sleep(0.05)
                    continue

                # Ler frame do pipe
                raw = self._process.stdout.read(frame_size)
                if len(raw) < frame_size:
                    logger.warning(f"[CAPTURE:{self.stream_id}] Pipe encerrado")
                    break

                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    (self._height, self._width, 3)
                )

                self._last_frame_time = time.time()

                # Não bloqueia se fila cheia (descarta frame antigo)
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except Empty:
                        pass

                self._frame_queue.put(frame.copy())

            except Exception as e:
                if self._running:
                    logger.error(f"[CAPTURE:{self.stream_id}] Erro no reader: {e}")
                break

        logger.info(f"[CAPTURE:{self.stream_id}] Frame reader encerrado")

    # ─── API pública ────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Conecta à stream e inicia captura de frames."""
        with self._lock:
            if self._running:
                return True

            stream_url = self._get_stream_url()
            if not stream_url:
                return False

            process = self._start_ffmpeg(stream_url)
            if not process:
                return False

            self._process = process
            self._running = True

            # Inicia thread leitora
            t = threading.Thread(target=self._frame_reader_thread, daemon=True)
            t.start()

            return True

    def stop(self):
        """Para a captura e encerra processos."""
        self._running = False
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None
        logger.info(f"[CAPTURE:{self.stream_id}] Captura encerrada")

    def get_frame(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        """Obtém próximo frame disponível."""
        try:
            return self._frame_queue.get(timeout=timeout)
        except Empty:
            return None

    def set_fps(self, fps: float):
        """Altera dinamicamente o FPS alvo de captura."""
        old = self._fps_target
        self._fps_target = fps
        if abs(old - fps) > 0.1:
            logger.debug(f"[CAPTURE:{self.stream_id}] FPS: {old:.1f} → {fps:.1f}")

    def is_running(self) -> bool:
        """Verifica se a captura está ativa."""
        if not self._running or not self._process:
            return False
        return self._process.poll() is None

    def get_burst_frames(self, duration: float = 3.0, fps: float = 15.0) -> list:
        """
        Captura burst de frames por `duration` segundos a `fps` FPS.
        Usado quando tabela é detectada.
        """
        self.set_fps(fps)
        frames = []
        deadline = time.time() + duration

        while time.time() < deadline:
            frame = self.get_frame(timeout=0.5)
            if frame is not None:
                frames.append(frame)

        logger.info(f"[CAPTURE:{self.stream_id}] Burst: {len(frames)} frames em {duration}s")
        return frames


class StreamCaptureWithReconnect:
    """
    Wrapper com reconexão automática sobre StreamCapture.
    Tenta reconectar indefinidamente se a stream cair.
    """

    def __init__(self, stream_config: dict, capture_config: dict, reconnect_config: dict):
        self.stream_config = stream_config
        self.capture_config = capture_config
        self.reconnect_config = reconnect_config
        self.stream_id = stream_config["id"]

        self._capture: Optional[StreamCapture] = None
        self._running = False
        self._reconnect_count = 0

    def _create_capture(self) -> StreamCapture:
        return StreamCapture(self.stream_config, self.capture_config)

    def start(self) -> bool:
        """Inicia com reconexão automática."""
        self._running = True
        return self._connect()

    def _connect(self) -> bool:
        delay = self.reconnect_config.get("delay_seconds", 15)
        max_delay = self.reconnect_config.get("max_delay_seconds", 120)
        backoff = self.reconnect_config.get("backoff_multiplier", 1.5)
        max_attempts = self.reconnect_config.get("max_attempts", 9999)

        for attempt in range(max_attempts):
            if not self._running:
                return False

            logger.info(f"[RECONNECT:{self.stream_id}] Tentativa {attempt + 1}/{max_attempts}")

            if self._capture:
                self._capture.stop()

            self._capture = self._create_capture()

            if self._capture.start():
                self._reconnect_count = 0
                logger.info(f"[RECONNECT:{self.stream_id}] ✓ Conectado")
                return True

            wait = min(delay * (backoff ** attempt), max_delay)
            logger.warning(f"[RECONNECT:{self.stream_id}] Falhou. Aguardando {wait:.0f}s...")
            time.sleep(wait)

        return False

    def get_frame(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        """Obtém frame, reconectando se necessário."""
        if not self._capture or not self._capture.is_running():
            logger.warning(f"[RECONNECT:{self.stream_id}] Stream caiu, reconectando...")
            self._connect()
            return None

        frame = self._capture.get_frame(timeout=timeout)

        # Se nenhum frame por muito tempo, pode ter caído
        if frame is None and not self._capture.is_running():
            self._connect()

        return frame

    def set_fps(self, fps: float):
        if self._capture:
            self._capture.set_fps(fps)

    def get_burst_frames(self, duration: float = 3.0, fps: float = 15.0) -> list:
        if self._capture:
            return self._capture.get_burst_frames(duration, fps)
        return []

    def stop(self):
        self._running = False
        if self._capture:
            self._capture.stop()
