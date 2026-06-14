"""Data structures shared by camera and recognition services."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision_cube_solver.cube.model import FaceGrid


@dataclass(frozen=True)
class CameraDevice:
    device_id: int
    name: str


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class PatchStats:
    bgr: tuple[float, float, float]
    lab: tuple[float, float, float]
    hsv: tuple[float, float, float]
    bounds: Rect

    @classmethod
    def from_arrays(
        cls, bgr: np.ndarray, lab: np.ndarray, hsv: np.ndarray, bounds: Rect
    ) -> PatchStats:
        return cls(
            tuple(float(value) for value in bgr),
            tuple(float(value) for value in lab),
            tuple(float(value) for value in hsv),
            bounds,
        )


@dataclass(frozen=True)
class FaceRecognitionResult:
    grid: FaceGrid
    patches: tuple[PatchStats, ...]
    low_confidence_indices: tuple[int, ...]

    @property
    def minimum_confidence(self) -> float:
        return min(self.grid.confidences)

