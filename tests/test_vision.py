import cv2
import numpy as np

from vision_cube_solver.cube.enums import CubeColor
from vision_cube_solver.vision.classifier import ColorClassifier
from vision_cube_solver.vision.sampler import GridSampler
from vision_cube_solver.vision.types import Rect


def test_grid_sampler_extracts_nine_patch_medians() -> None:
    frame = np.zeros((300, 300, 3), dtype=np.uint8)
    expected = []
    for row in range(3):
        for column in range(3):
            color = np.array((20 + column * 60, 30 + row * 60, 100), dtype=np.uint8)
            frame[row * 100 : (row + 1) * 100, column * 100 : (column + 1) * 100] = color
            expected.append(tuple(float(value) for value in color))

    patches = GridSampler().extract_patches(frame, Rect(0, 0, 300, 300))

    assert len(patches) == 9
    assert [patch.bgr for patch in patches] == expected


def test_center_calibration_classifies_matching_patch() -> None:
    frame = np.full((120, 120, 3), (30, 180, 30), dtype=np.uint8)
    patch = GridSampler().extract_patches(frame, Rect(0, 0, 120, 120))[0]
    classifier = ColorClassifier()
    classifier.register_center_sample(CubeColor.GREEN, patch)

    color, _confidence = classifier.classify_patch(patch)

    assert color == CubeColor.GREEN


def test_draw_guide_does_not_modify_input() -> None:
    frame = np.zeros((90, 90, 3), dtype=np.uint8)

    shown = GridSampler().draw_guide(frame, Rect(0, 0, 90, 90))

    assert not np.array_equal(shown, frame)
    assert np.count_nonzero(frame) == 0
    assert cv2.countNonZero(cv2.cvtColor(shown, cv2.COLOR_BGR2GRAY)) > 0
