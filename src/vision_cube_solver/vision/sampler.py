"""Extract stable color statistics from a guided 3x3 image region."""

from __future__ import annotations

import cv2
import numpy as np

from vision_cube_solver.vision.types import PatchStats, Rect


class GridSampler:
    def __init__(self, inner_ratio: float = 0.56) -> None:
        if not 0.2 <= inner_ratio <= 0.9:
            raise ValueError("inner_ratio must be between 0.2 and 0.9")
        self.inner_ratio = inner_ratio

    def guide_rect(self, frame: np.ndarray, size_ratio: float = 0.62) -> Rect:
        height, width = frame.shape[:2]
        size = int(min(width, height) * size_ratio)
        return Rect((width - size) // 2, (height - size) // 2, size, size)

    def extract_patches(self, frame: np.ndarray, guide_rect: Rect) -> tuple[PatchStats, ...]:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a BGR image with three channels")
        self._validate_rect(frame, guide_rect)
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cell_width = guide_rect.width / 3
        cell_height = guide_rect.height / 3
        patches: list[PatchStats] = []

        for row in range(3):
            for column in range(3):
                center_x = guide_rect.x + (column + 0.5) * cell_width
                center_y = guide_rect.y + (row + 0.5) * cell_height
                patch_width = max(2, int(cell_width * self.inner_ratio))
                patch_height = max(2, int(cell_height * self.inner_ratio))
                x = int(center_x - patch_width / 2)
                y = int(center_y - patch_height / 2)
                bounds = Rect(x, y, patch_width, patch_height)
                slices = np.s_[y : y + patch_height, x : x + patch_width]
                bgr = np.median(frame[slices], axis=(0, 1))
                lab = np.median(lab_frame[slices], axis=(0, 1))
                hsv = np.median(hsv_frame[slices], axis=(0, 1))
                patches.append(PatchStats.from_arrays(bgr, lab, hsv, bounds))
        return tuple(patches)

    def draw_guide(self, frame: np.ndarray, guide_rect: Rect) -> np.ndarray:
        result = frame.copy()
        for offset in range(4):
            x = round(guide_rect.x + guide_rect.width * offset / 3)
            y = round(guide_rect.y + guide_rect.height * offset / 3)
            cv2.line(
                result,
                (x, guide_rect.y),
                (x, guide_rect.y + guide_rect.height),
                (255, 255, 255),
                2,
            )
            cv2.line(
                result,
                (guide_rect.x, y),
                (guide_rect.x + guide_rect.width, y),
                (255, 255, 255),
                2,
            )
        return result

    @staticmethod
    def _validate_rect(frame: np.ndarray, rect: Rect) -> None:
        height, width = frame.shape[:2]
        if (
            rect.width <= 0
            or rect.height <= 0
            or rect.x < 0
            or rect.y < 0
            or rect.x + rect.width > width
            or rect.y + rect.height > height
        ):
            raise ValueError("Guide rectangle falls outside the image")

