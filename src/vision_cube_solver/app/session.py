"""Application state for captured cube faces."""

from __future__ import annotations

from dataclasses import dataclass

from vision_cube_solver.cube.enums import COLOR_FACES, FACE_ORDER, CubeColor, Face
from vision_cube_solver.cube.model import FaceGrid, FaceletCube, ValidationResult
from vision_cube_solver.vision.classifier import ColorClassifier
from vision_cube_solver.vision.types import FaceRecognitionResult, PatchStats


@dataclass(frozen=True)
class CapturedFace:
    face: Face
    grid: FaceGrid
    patches: tuple[PatchStats, ...]


class CubeSession:
    def __init__(self, classifier: ColorClassifier | None = None) -> None:
        self.classifier = classifier or ColorClassifier()
        self._faces: dict[Face, CapturedFace] = {}

    @property
    def faces(self) -> dict[Face, FaceGrid]:
        return {face: capture.grid for face, capture in self._faces.items()}

    @property
    def complete(self) -> bool:
        return all(face in self._faces for face in FACE_ORDER)

    def capture(
        self,
        center_color: CubeColor,
        patches: tuple[PatchStats, ...],
    ) -> FaceRecognitionResult:
        face = COLOR_FACES[center_color]
        self.classifier.register_center_sample(center_color, patches[4])
        result = self.classifier.classify(patches, forced_center=center_color)
        self._faces[face] = CapturedFace(face, result.grid, patches)
        if len(self.classifier.calibrated_colors) == len(CubeColor):
            self._reclassify_all()
            result = FaceRecognitionResult(
                self._faces[face].grid,
                patches,
                tuple(
                    index
                    for index, confidence in enumerate(self._faces[face].grid.confidences)
                    if confidence < self.classifier.low_confidence_threshold
                ),
            )
        return result

    def remove(self, face: Face) -> None:
        capture = self._faces.pop(face, None)
        if capture is not None:
            self.classifier.clear_center_sample(capture.grid.colors[4])

    def set_sticker(self, face: Face, index: int, color: CubeColor) -> None:
        if face not in self._faces:
            raise ValueError(f"Face {face.value} has not been captured")
        capture = self._faces[face]
        if index == 4 and color != capture.grid.colors[4]:
            raise ValueError("Center colors cannot be edited")
        self._faces[face] = CapturedFace(
            face,
            capture.grid.replace(index, color),
            capture.patches,
        )

    def validate(self) -> ValidationResult:
        if not self.complete:
            missing = ", ".join(face.value for face in FACE_ORDER if face not in self._faces)
            return ValidationResult(False, (f"Missing faces: {missing}",))
        return self.to_facelet_cube().validate()

    def to_facelet_cube(self) -> FaceletCube:
        return FaceletCube.from_faces(self.faces)

    def _reclassify_all(self) -> None:
        for face, capture in tuple(self._faces.items()):
            center_color = capture.grid.colors[4]
            result = self.classifier.classify(capture.patches, forced_center=center_color)
            self._faces[face] = CapturedFace(face, result.grid, capture.patches)
