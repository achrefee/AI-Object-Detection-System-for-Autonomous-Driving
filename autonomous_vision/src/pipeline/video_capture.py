"""
video_capture.py — Multi-threaded video capture with ring buffer.

Runs camera/video capture in a dedicated thread so inference
doesn't block frame acquisition.
"""

import threading
import time
from collections import deque

import cv2
import numpy as np


class VideoCapture:
    """
    Thread-safe video capture with buffering.

    Usage:
        cap = VideoCapture("video.mp4")  # or 0 for webcam
        cap.start()
        frame = cap.read()
        cap.stop()
    """

    def __init__(
        self,
        source: str | int = 0,
        buffer_size: int = 2,
        target_fps: float | None = None,
        resize: tuple[int, int] | None = None,
    ):
        """
        Args:
            source: Video file path or camera index.
            buffer_size: Number of frames to buffer.
            target_fps: Limit input FPS (None = no limit).
            resize: Target (width, height) or None.
        """
        self.source = source
        self.buffer_size = buffer_size
        self.target_fps = target_fps
        self.resize = resize

        self._cap: cv2.VideoCapture | None = None
        self._buffer: deque = deque(maxlen=buffer_size)
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()

        # Metadata
        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.frame_count = 0
        self.total_frames = 0

    def start(self) -> "VideoCapture":
        """Open the video source and start the capture thread."""
        self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.source}")

        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        return self

    def stop(self):
        """Stop capture and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()

    def read(self) -> np.ndarray | None:
        """
        Read the latest frame from the buffer.

        Returns:
            BGR frame or None if no frame available.
        """
        with self._lock:
            if len(self._buffer) > 0:
                return self._buffer[-1]  # Latest frame
        return None

    def read_wait(self, timeout: float = 1.0) -> np.ndarray | None:
        """
        Wait for a frame to become available.

        Args:
            timeout: Max seconds to wait.

        Returns:
            BGR frame or None if timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            frame = self.read()
            if frame is not None:
                return frame
            time.sleep(0.001)
        return None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_video_file(self) -> bool:
        return isinstance(self.source, str)

    def _capture_loop(self):
        """Background thread: continuously capture frames."""
        frame_interval = 1.0 / self.target_fps if self.target_fps else 0

        while self._running:
            t0 = time.time()

            ret, frame = self._cap.read()
            if not ret:
                if self.is_video_file:
                    # End of video
                    self._running = False
                    break
                else:
                    # Camera glitch — retry
                    time.sleep(0.1)
                    continue

            self.frame_count += 1

            # Resize if needed
            if self.resize is not None:
                frame = cv2.resize(frame, self.resize)

            with self._lock:
                self._buffer.append(frame)

            # FPS limiting
            if frame_interval > 0:
                elapsed = time.time() - t0
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def __del__(self):
        self.stop()
