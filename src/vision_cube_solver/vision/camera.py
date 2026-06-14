"""OpenCV camera access with explicit lifecycle management."""

from __future__ import annotations

import cv2
import numpy as np

from vision_cube_solver.vision.types import CameraDevice


class CameraError(RuntimeError):
    pass


class CameraService:
    def __init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None

    def list_devices(self, maximum_devices: int = 8) -> list[CameraDevice]:
        devices: list[CameraDevice] = []
        for device_id in range(maximum_devices):
            capture = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
            try:
                if capture.isOpened():
                    devices.append(CameraDevice(device_id, f"Camera {device_id}"))
            finally:
                capture.release()
        return devices

    @property
    def running(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def start(self, device_id: int) -> None:
        self.stop()
        capture = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(device_id)
        if not capture.isOpened():
            capture.release()
            raise CameraError(f"Unable to open camera {device_id}")
        self._capture = capture

    def read_frame(self) -> np.ndarray:
        if not self.running:
            raise CameraError("Camera is not running")
        assert self._capture is not None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("Unable to read a frame from the camera")
        return frame

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
