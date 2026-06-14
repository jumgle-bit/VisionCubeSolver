"""Coordinate encodings and move-table generation for two-phase search."""

from __future__ import annotations

from array import array
from collections.abc import Callable
from itertools import combinations
from math import comb

from vision_cube_solver.cube.enums import ALL_MOVES, PHASE2_MOVES, Move
from vision_cube_solver.cube.model import CubieCube

TWIST_COUNT = 3**7
FLIP_COUNT = 2**11
SLICE_COMBINATION_COUNT = comb(12, 4)
PERMUTATION_8_COUNT = 40320
SLICE_PERMUTATION_COUNT = 24
MOVE_COUNT = len(ALL_MOVES)

ProgressCallback = Callable[[str, int, int], None]


def get_twist(cube: CubieCube) -> int:
    coordinate = 0
    for orientation in cube.corner_orientation[:7]:
        coordinate = coordinate * 3 + orientation
    return coordinate


def set_twist(coordinate: int) -> CubieCube:
    orientations = [0] * 8
    total = 0
    for index in range(6, -1, -1):
        orientations[index] = coordinate % 3
        total += orientations[index]
        coordinate //= 3
    orientations[7] = (-total) % 3
    return CubieCube(corner_orientation=tuple(orientations))


def get_flip(cube: CubieCube) -> int:
    coordinate = 0
    for orientation in cube.edge_orientation[:11]:
        coordinate = coordinate * 2 + orientation
    return coordinate


def set_flip(coordinate: int) -> CubieCube:
    orientations = [0] * 12
    total = 0
    for index in range(10, -1, -1):
        orientations[index] = coordinate % 2
        total += orientations[index]
        coordinate //= 2
    orientations[11] = total % 2
    return CubieCube(edge_orientation=tuple(orientations))


def _slice_coordinate_for_positions(positions: tuple[int, ...]) -> int:
    selected = set(positions)
    coordinate = 0
    found = 0
    for position in range(11, -1, -1):
        if position in selected:
            found += 1
            coordinate += comb(11 - position, found)
    return coordinate


_SLICE_POSITIONS: list[tuple[int, ...] | None] = [None] * SLICE_COMBINATION_COUNT
for _positions in combinations(range(12), 4):
    _SLICE_POSITIONS[_slice_coordinate_for_positions(_positions)] = _positions
assert all(value is not None for value in _SLICE_POSITIONS)


def get_slice_combination(cube: CubieCube) -> int:
    positions = tuple(
        position for position, cubie in enumerate(cube.edge_permutation) if cubie >= 8
    )
    return _slice_coordinate_for_positions(positions)


def set_slice_combination(coordinate: int) -> CubieCube:
    positions = _SLICE_POSITIONS[coordinate]
    assert positions is not None
    selected = set(positions)
    regular = iter(range(8))
    slice_edges = iter(range(8, 12))
    permutation = tuple(
        next(slice_edges) if position in selected else next(regular)
        for position in range(12)
    )
    return CubieCube(edge_permutation=permutation)


def permutation_rank(permutation: tuple[int, ...]) -> int:
    remaining = list(range(len(permutation)))
    rank = 0
    for value in permutation:
        index = remaining.index(value)
        rank = rank * len(remaining) + index
        remaining.pop(index)
    return rank


def permutation_unrank(rank: int, size: int) -> tuple[int, ...]:
    digits = [0] * size
    for radix in range(1, size + 1):
        digits[size - radix] = rank % radix
        rank //= radix
    remaining = list(range(size))
    return tuple(remaining.pop(digit) for digit in digits)


def get_corner_permutation(cube: CubieCube) -> int:
    return permutation_rank(cube.corner_permutation)


def set_corner_permutation(coordinate: int) -> CubieCube:
    return CubieCube(corner_permutation=permutation_unrank(coordinate, 8))


def get_ud_edge_permutation(cube: CubieCube) -> int:
    permutation = cube.edge_permutation[:8]
    if any(value >= 8 for value in permutation):
        raise ValueError("UD edge permutation is only defined inside the phase-two subgroup")
    return permutation_rank(permutation)


def set_ud_edge_permutation(coordinate: int) -> CubieCube:
    permutation = permutation_unrank(coordinate, 8) + tuple(range(8, 12))
    return CubieCube(edge_permutation=permutation)


def get_slice_permutation(cube: CubieCube) -> int:
    permutation = tuple(value - 8 for value in cube.edge_permutation[8:12])
    if any(value not in range(4) for value in permutation):
        raise ValueError("Slice permutation is only defined inside the phase-two subgroup")
    return permutation_rank(permutation)


def set_slice_permutation(coordinate: int) -> CubieCube:
    slice_part = tuple(value + 8 for value in permutation_unrank(coordinate, 4))
    return CubieCube(edge_permutation=tuple(range(8)) + slice_part)


class CoordinateMoveTables:
    """Flattened coordinate transition tables indexed by coordinate * 18 + move."""

    def __init__(
        self,
        twist: array,
        flip: array,
        slice_combination: array,
        corner_permutation: array,
        ud_edge_permutation: array,
        slice_permutation: array,
    ) -> None:
        self.twist = twist
        self.flip = flip
        self.slice_combination = slice_combination
        self.corner_permutation = corner_permutation
        self.ud_edge_permutation = ud_edge_permutation
        self.slice_permutation = slice_permutation

    @classmethod
    def build(cls, progress: ProgressCallback | None = None) -> CoordinateMoveTables:
        def build_table(
            label: str,
            count: int,
            setter: Callable[[int], CubieCube],
            getter: Callable[[CubieCube], int],
            moves: tuple[Move, ...],
        ) -> array:
            table = array("H", [0]) * (count * MOVE_COUNT)
            for coordinate in range(count):
                cube = setter(coordinate)
                for move in moves:
                    table[coordinate * MOVE_COUNT + int(move)] = getter(cube.apply_move(move))
                if progress and coordinate % max(1, count // 100) == 0:
                    progress(label, coordinate, count)
            if progress:
                progress(label, count, count)
            return table

        return cls(
            build_table("twist moves", TWIST_COUNT, set_twist, get_twist, ALL_MOVES),
            build_table("flip moves", FLIP_COUNT, set_flip, get_flip, ALL_MOVES),
            build_table(
                "slice combination moves",
                SLICE_COMBINATION_COUNT,
                set_slice_combination,
                get_slice_combination,
                ALL_MOVES,
            ),
            build_table(
                "corner permutation moves",
                PERMUTATION_8_COUNT,
                set_corner_permutation,
                get_corner_permutation,
                PHASE2_MOVES,
            ),
            build_table(
                "UD edge permutation moves",
                PERMUTATION_8_COUNT,
                set_ud_edge_permutation,
                get_ud_edge_permutation,
                PHASE2_MOVES,
            ),
            build_table(
                "slice permutation moves",
                SLICE_PERMUTATION_COUNT,
                set_slice_permutation,
                get_slice_permutation,
                PHASE2_MOVES,
            ),
        )

    @staticmethod
    def next(table: array, coordinate: int, move: Move) -> int:
        return table[coordinate * MOVE_COUNT + int(move)]
