from vision_cube_solver.cube.enums import Move
from vision_cube_solver.cube.model import CubieCube
from vision_cube_solver.solver.coordinates import (
    FLIP_COUNT,
    PERMUTATION_8_COUNT,
    SLICE_COMBINATION_COUNT,
    TWIST_COUNT,
    get_corner_permutation,
    get_flip,
    get_slice_combination,
    get_twist,
    permutation_rank,
    permutation_unrank,
    set_corner_permutation,
    set_flip,
    set_slice_combination,
    set_twist,
)


def test_coordinate_zero_is_solved() -> None:
    cube = CubieCube.solved()

    assert get_twist(cube) == 0
    assert get_flip(cube) == 0
    assert get_slice_combination(cube) == 0
    assert get_corner_permutation(cube) == 0


def test_coordinate_setters_round_trip() -> None:
    for coordinate in (0, 1, TWIST_COUNT // 2, TWIST_COUNT - 1):
        assert get_twist(set_twist(coordinate)) == coordinate
    for coordinate in (0, 1, FLIP_COUNT // 2, FLIP_COUNT - 1):
        assert get_flip(set_flip(coordinate)) == coordinate
    for coordinate in (0, 1, SLICE_COMBINATION_COUNT // 2, SLICE_COMBINATION_COUNT - 1):
        assert get_slice_combination(set_slice_combination(coordinate)) == coordinate
    for coordinate in (0, 1, PERMUTATION_8_COUNT // 2, PERMUTATION_8_COUNT - 1):
        assert get_corner_permutation(set_corner_permutation(coordinate)) == coordinate


def test_permutation_rank_unrank() -> None:
    for rank in (0, 1, 17, 719, 40319):
        permutation = permutation_unrank(rank, 8)
        assert permutation_rank(permutation) == rank


def test_phase_one_coordinates_detect_scramble() -> None:
    cube = CubieCube.solved().apply_move(Move.F)

    assert get_flip(cube) != 0

