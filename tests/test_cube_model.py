from vision_cube_solver.cube.enums import FACE_ORDER, Move
from vision_cube_solver.cube.model import CubieCube, FaceGrid, FaceletCube


def test_solved_facelet_cube_is_valid() -> None:
    cube = FaceletCube.solved()

    assert cube.validate().valid
    assert cube.to_cubie_cube().is_solved


def test_face_and_cubie_round_trip_after_moves() -> None:
    moves = tuple(Move.from_notation(value) for value in ("R", "U", "F2", "L'", "D2", "B"))
    cubie = CubieCube.solved().apply_moves(moves)

    reconstructed = FaceletCube.from_cubie_cube(cubie).to_cubie_cube()

    assert reconstructed == cubie


def test_each_face_returns_to_solved_after_four_turns() -> None:
    for move in (Move.U, Move.R, Move.F, Move.D, Move.L, Move.B):
        cube = CubieCube.solved().apply_moves((move,) * 4)
        assert cube.is_solved


def test_invalid_flipped_edge_is_rejected() -> None:
    cube = CubieCube(edge_orientation=(1,) + (0,) * 11)

    result = cube.validate()

    assert not result.valid
    assert "Edge flip sum is invalid" in result.errors


def test_from_faces_requires_every_face() -> None:
    faces = {face: FaceGrid.solved(face) for face in FACE_ORDER[:-1]}

    try:
        FaceletCube.from_faces(faces)
    except ValueError as exc:
        assert "Missing faces" in str(exc)
    else:
        raise AssertionError("Expected missing face error")

