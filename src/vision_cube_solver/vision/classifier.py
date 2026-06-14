"""Center-calibrated cube sticker color classifier."""

from __future__ import annotations

from math import exp

import numpy as np

from vision_cube_solver.cube.enums import CubeColor
from vision_cube_solver.cube.model import FaceGrid
from vision_cube_solver.vision.types import FaceRecognitionResult, PatchStats

# OpenCV Lab values. They are fallback references until center samples are available.
DEFAULT_LAB: dict[CubeColor, tuple[float, float, float]] = {
    CubeColor.WHITE: (242.0, 128.0, 128.0),
    CubeColor.RED: (136.0, 208.0, 195.0),
    CubeColor.GREEN: (154.0, 74.0, 172.0),
    CubeColor.YELLOW: (225.0, 112.0, 210.0),
    CubeColor.ORANGE: (180.0, 166.0, 205.0),
    CubeColor.BLUE: (105.0, 185.0, 72.0),
}


class ColorClassifier:
    def __init__(self, low_confidence_threshold: float = 0.30) -> None:
        self._centers: dict[CubeColor, PatchStats] = {}
        self.low_confidence_threshold = low_confidence_threshold

    @property
    def calibrated_colors(self) -> frozenset[CubeColor]:
        return frozenset(self._centers)

    def register_center_sample(self, color: CubeColor, sample: PatchStats) -> None:
        self._centers[color] = sample

    def clear_center_sample(self, color: CubeColor) -> None:
        self._centers.pop(color, None)

    def classify(
        self,
        patches: tuple[PatchStats, ...],
        forced_center: CubeColor | None = None,
    ) -> FaceRecognitionResult:
        if len(patches) != 9:
            raise ValueError("Exactly nine patches are required")
        colors: list[CubeColor] = []
        confidences: list[float] = []
        for patch in patches:
            color, confidence = self.classify_patch(patch)
            colors.append(color)
            confidences.append(confidence)
        if forced_center is not None:
            colors[4] = forced_center
            confidences[4] = 1.0
        low = tuple(
            index
            for index, confidence in enumerate(confidences)
            if confidence < self.low_confidence_threshold
        )
        return FaceRecognitionResult(
            FaceGrid(tuple(colors), tuple(confidences)),
            patches,
            low,
        )

    def classify_patch(self, patch: PatchStats) -> tuple[CubeColor, float]:
        distances = sorted(
            (self._distance(patch, color), color)
            for color in CubeColor
        )
        best_distance, best_color = distances[0]
        second_distance = distances[1][0]
        separation = max(0.0, second_distance - best_distance)
        confidence = (1.0 - exp(-separation / 22.0)) * exp(-best_distance / 180.0)
        return best_color, min(1.0, max(0.0, confidence))

    def _distance(self, patch: PatchStats, color: CubeColor) -> float:
        reference = self._centers.get(color)
        target_lab = reference.lab if reference is not None else DEFAULT_LAB[color]
        lab_delta = np.asarray(patch.lab) - np.asarray(target_lab)
        # Lightness varies strongly with illumination, so chroma is weighted higher.
        weighted_lab = np.asarray((lab_delta[0] * 0.45, lab_delta[1], lab_delta[2]))
        distance = float(np.linalg.norm(weighted_lab))
        if color == CubeColor.WHITE:
            saturation = patch.hsv[1]
            distance += max(0.0, saturation - 65.0) * 0.35
        return distance
