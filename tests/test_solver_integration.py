from pathlib import Path

import pytest

from vision_cube_solver.cube.enums import Move
from vision_cube_solver.cube.model import CubieCube
from vision_cube_solver.solver.pattern_database import PatternDatabaseManager
from vision_cube_solver.solver.two_phase import TwoPhaseIDASolver

CACHE = Path("data/pdb/two_phase_v1.pkl")


@pytest.mark.integration
@pytest.mark.skipif(not CACHE.exists(), reason="pattern database cache has not been generated")
def test_known_scramble_is_solved_with_cached_database() -> None:
    scramble = tuple(
        Move.from_notation(notation)
        for notation in ("R", "U", "R'", "U'", "F2", "D", "L2", "B2")
    )
    cube = CubieCube.solved().apply_moves(scramble)

    result = TwoPhaseIDASolver(PatternDatabaseManager(CACHE)).solve(cube)

    assert cube.apply_moves(result.moves).is_solved
    assert result.moves
