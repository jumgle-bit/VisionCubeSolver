"""Facelet and cubie representations of a standard 3x3 cube."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from vision_cube_solver.cube.enums import (
    COLOR_FACES,
    FACE_COLORS,
    FACE_ORDER,
    CubeColor,
    Face,
    Move,
)

CORNER_FACELETS: tuple[tuple[int, int, int], ...] = (
    (8, 9, 20),  # URF
    (6, 18, 38),  # UFL
    (0, 36, 47),  # ULB
    (2, 45, 11),  # UBR
    (29, 26, 15),  # DFR
    (27, 44, 24),  # DLF
    (33, 53, 42),  # DBL
    (35, 17, 51),  # DRB
)

CORNER_COLORS: tuple[tuple[CubeColor, CubeColor, CubeColor], ...] = (
    (CubeColor.WHITE, CubeColor.RED, CubeColor.GREEN),
    (CubeColor.WHITE, CubeColor.GREEN, CubeColor.ORANGE),
    (CubeColor.WHITE, CubeColor.ORANGE, CubeColor.BLUE),
    (CubeColor.WHITE, CubeColor.BLUE, CubeColor.RED),
    (CubeColor.YELLOW, CubeColor.GREEN, CubeColor.RED),
    (CubeColor.YELLOW, CubeColor.ORANGE, CubeColor.GREEN),
    (CubeColor.YELLOW, CubeColor.BLUE, CubeColor.ORANGE),
    (CubeColor.YELLOW, CubeColor.RED, CubeColor.BLUE),
)

EDGE_FACELETS: tuple[tuple[int, int], ...] = (
    (5, 10),  # UR
    (7, 19),  # UF
    (3, 37),  # UL
    (1, 46),  # UB
    (32, 16),  # DR
    (28, 25),  # DF
    (30, 43),  # DL
    (34, 52),  # DB
    (23, 12),  # FR
    (21, 41),  # FL
    (50, 39),  # BL
    (48, 14),  # BR
)

EDGE_COLORS: tuple[tuple[CubeColor, CubeColor], ...] = (
    (CubeColor.WHITE, CubeColor.RED),
    (CubeColor.WHITE, CubeColor.GREEN),
    (CubeColor.WHITE, CubeColor.ORANGE),
    (CubeColor.WHITE, CubeColor.BLUE),
    (CubeColor.YELLOW, CubeColor.RED),
    (CubeColor.YELLOW, CubeColor.GREEN),
    (CubeColor.YELLOW, CubeColor.ORANGE),
    (CubeColor.YELLOW, CubeColor.BLUE),
    (CubeColor.GREEN, CubeColor.RED),
    (CubeColor.GREEN, CubeColor.ORANGE),
    (CubeColor.BLUE, CubeColor.ORANGE),
    (CubeColor.BLUE, CubeColor.RED),
)


@dataclass(frozen=True)
class FaceGrid:
    """Nine recognized colors for a face, stored row-major."""

    colors: tuple[CubeColor, ...]
    confidences: tuple[float, ...] = (1.0,) * 9

    def __post_init__(self) -> None:
        if len(self.colors) != 9:
            raise ValueError("A face grid must contain exactly nine colors")
        if len(self.confidences) != 9:
            raise ValueError("A face grid must contain exactly nine confidence values")

    @classmethod
    def solved(cls, face: Face) -> FaceGrid:
        return cls((FACE_COLORS[face],) * 9)

    def replace(self, index: int, color: CubeColor, confidence: float = 1.0) -> FaceGrid:
        if not 0 <= index < 9:
            raise IndexError(index)
        colors = list(self.colors)
        confidences = list(self.confidences)
        colors[index] = color
        confidences[index] = confidence
        return FaceGrid(tuple(colors), tuple(confidences))


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class CubieCube:
    """Cubie-level representation used by the solver."""

    corner_permutation: tuple[int, ...] = tuple(range(8))
    corner_orientation: tuple[int, ...] = (0,) * 8
    edge_permutation: tuple[int, ...] = tuple(range(12))
    edge_orientation: tuple[int, ...] = (0,) * 12

    def __post_init__(self) -> None:
        if len(self.corner_permutation) != 8 or len(self.corner_orientation) != 8:
            raise ValueError("A cubie cube must contain eight corners")
        if len(self.edge_permutation) != 12 or len(self.edge_orientation) != 12:
            raise ValueError("A cubie cube must contain twelve edges")

    @classmethod
    def solved(cls) -> CubieCube:
        return cls()

    @property
    def is_solved(self) -> bool:
        return self == CubieCube.solved()

    def apply_move(self, move: Move) -> CubieCube:
        result = self
        quarter_turn = MOVE_CUBES[move.face_index]
        for _ in range(move.amount):
            result = result.multiply(quarter_turn)
        return result

    def apply_moves(self, moves: Iterable[Move]) -> CubieCube:
        result = self
        for move in moves:
            result = result.apply_move(move)
        return result

    def multiply(self, other: CubieCube) -> CubieCube:
        """Return this state followed by ``other``."""

        cp = tuple(self.corner_permutation[other.corner_permutation[i]] for i in range(8))
        co = tuple(
            (
                self.corner_orientation[other.corner_permutation[i]]
                + other.corner_orientation[i]
            )
            % 3
            for i in range(8)
        )
        ep = tuple(self.edge_permutation[other.edge_permutation[i]] for i in range(12))
        eo = tuple(
            (
                self.edge_orientation[other.edge_permutation[i]]
                + other.edge_orientation[i]
            )
            % 2
            for i in range(12)
        )
        return CubieCube(cp, co, ep, eo)

    def validate(self) -> ValidationResult:
        errors: list[str] = []
        if sorted(self.corner_permutation) != list(range(8)):
            errors.append("Corner cubies are missing or duplicated")
        if sorted(self.edge_permutation) != list(range(12)):
            errors.append("Edge cubies are missing or duplicated")
        if any(value not in (0, 1, 2) for value in self.corner_orientation):
            errors.append("A corner has an invalid orientation")
        if any(value not in (0, 1) for value in self.edge_orientation):
            errors.append("An edge has an invalid orientation")
        if sum(self.corner_orientation) % 3:
            errors.append("Corner twist sum is invalid")
        if sum(self.edge_orientation) % 2:
            errors.append("Edge flip sum is invalid")
        if _permutation_parity(self.corner_permutation) != _permutation_parity(
            self.edge_permutation
        ):
            errors.append("Corner and edge permutation parity does not match")
        return ValidationResult(not errors, tuple(errors))

    def to_facelets(self) -> tuple[CubeColor, ...]:
        facelets = [FACE_COLORS[face] for face in FACE_ORDER for _ in range(9)]
        for position, cubie in enumerate(self.corner_permutation):
            orientation = self.corner_orientation[position]
            for color_index in range(3):
                target = CORNER_FACELETS[position][(color_index + orientation) % 3]
                facelets[target] = CORNER_COLORS[cubie][color_index]
        for position, cubie in enumerate(self.edge_permutation):
            orientation = self.edge_orientation[position]
            for color_index in range(2):
                target = EDGE_FACELETS[position][(color_index + orientation) % 2]
                facelets[target] = EDGE_COLORS[cubie][color_index]
        return tuple(facelets)


@dataclass(frozen=True)
class FaceletCube:
    """A cube represented by 54 colors in URFDLB order."""

    facelets: tuple[CubeColor, ...]

    def __post_init__(self) -> None:
        if len(self.facelets) != 54:
            raise ValueError("A facelet cube must contain exactly 54 colors")

    @classmethod
    def solved(cls) -> FaceletCube:
        return cls(tuple(FACE_COLORS[face] for face in FACE_ORDER for _ in range(9)))

    @classmethod
    def from_faces(cls, faces: dict[Face, FaceGrid]) -> FaceletCube:
        missing = [face.value for face in FACE_ORDER if face not in faces]
        if missing:
            raise ValueError(f"Missing faces: {', '.join(missing)}")
        return cls(tuple(color for face in FACE_ORDER for color in faces[face].colors))

    @classmethod
    def from_cubie_cube(cls, cube: CubieCube) -> FaceletCube:
        return cls(cube.to_facelets())

    def to_faces(self) -> dict[Face, FaceGrid]:
        return {
            face: FaceGrid(self.facelets[index * 9 : index * 9 + 9])
            for index, face in enumerate(FACE_ORDER)
        }

    def validate(self) -> ValidationResult:
        errors: list[str] = []
        counts = Counter(self.facelets)
        for color in CubeColor:
            if counts[color] != 9:
                errors.append(
                    f"{color.value.title()} must appear 9 times, found {counts[color]}"
                )
        for index, face in enumerate(FACE_ORDER):
            expected = FACE_COLORS[face]
            actual = self.facelets[index * 9 + 4]
            if actual != expected:
                errors.append(
                    f"{face.value} center must be {expected.value}, found {actual.value}"
                )
        if errors:
            return ValidationResult(False, tuple(errors))
        try:
            cubie = self.to_cubie_cube()
        except ValueError as exc:
            return ValidationResult(False, (str(exc),))
        return cubie.validate()

    def to_cubie_cube(self) -> CubieCube:
        cp = [-1] * 8
        co = [0] * 8
        ep = [-1] * 12
        eo = [0] * 12

        for position, indices in enumerate(CORNER_FACELETS):
            orientation = next(
                (
                    ori
                    for ori in range(3)
                    if self.facelets[indices[ori]]
                    in (CubeColor.WHITE, CubeColor.YELLOW)
                ),
                None,
            )
            if orientation is None:
                raise ValueError(f"Corner at position {position} has no white/yellow sticker")
            color1 = self.facelets[indices[(orientation + 1) % 3]]
            color2 = self.facelets[indices[(orientation + 2) % 3]]
            cubie = next(
                (
                    index
                    for index, colors in enumerate(CORNER_COLORS)
                    if colors[1] == color1 and colors[2] == color2
                ),
                None,
            )
            if cubie is None:
                raise ValueError(f"Corner at position {position} has an invalid color combination")
            cp[position] = cubie
            co[position] = orientation % 3

        for position, indices in enumerate(EDGE_FACELETS):
            colors = (self.facelets[indices[0]], self.facelets[indices[1]])
            match = next(
                (
                    (cubie, orientation)
                    for cubie, expected in enumerate(EDGE_COLORS)
                    for orientation in range(2)
                    if colors[0] == expected[orientation]
                    and colors[1] == expected[(orientation + 1) % 2]
                ),
                None,
            )
            if match is None:
                raise ValueError(f"Edge at position {position} has an invalid color combination")
            ep[position], eo[position] = match

        cube = CubieCube(tuple(cp), tuple(co), tuple(ep), tuple(eo))
        validation = cube.validate()
        if not validation.valid:
            raise ValueError("; ".join(validation.errors))
        return cube


def _permutation_parity(permutation: Sequence[int]) -> int:
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    return inversions % 2


def face_for_center(color: CubeColor) -> Face:
    return COLOR_FACES[color]


MOVE_CUBES: tuple[CubieCube, ...] = (
    CubieCube(
        (3, 0, 1, 2, 4, 5, 6, 7),
        (0,) * 8,
        (3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11),
        (0,) * 12,
    ),
    CubieCube(
        (4, 1, 2, 0, 7, 5, 6, 3),
        (2, 0, 0, 1, 1, 0, 0, 2),
        (8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0),
        (0,) * 12,
    ),
    CubieCube(
        (1, 5, 2, 3, 0, 4, 6, 7),
        (1, 2, 0, 0, 2, 1, 0, 0),
        (0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11),
        (0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0),
    ),
    CubieCube(
        (0, 1, 2, 3, 5, 6, 7, 4),
        (0,) * 8,
        (0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11),
        (0,) * 12,
    ),
    CubieCube(
        (0, 2, 6, 3, 4, 1, 5, 7),
        (0, 1, 2, 0, 0, 2, 1, 0),
        (0, 1, 10, 3, 4, 5, 9, 7, 8, 2, 6, 11),
        (0,) * 12,
    ),
    CubieCube(
        (0, 1, 3, 7, 4, 5, 2, 6),
        (0, 0, 1, 2, 0, 0, 2, 1),
        (0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7),
        (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1),
    ),
)
