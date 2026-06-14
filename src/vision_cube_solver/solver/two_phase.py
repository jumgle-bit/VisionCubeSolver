"""Two-phase IDA* Rubik's Cube solver."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event

from vision_cube_solver.cube.enums import ALL_MOVES, PHASE2_MOVES, Move
from vision_cube_solver.cube.model import CubieCube
from vision_cube_solver.solver.coordinates import (
    MOVE_COUNT,
    SLICE_COMBINATION_COUNT,
    SLICE_PERMUTATION_COUNT,
    CoordinateMoveTables,
    get_corner_permutation,
    get_flip,
    get_slice_combination,
    get_slice_permutation,
    get_twist,
    get_ud_edge_permutation,
)
from vision_cube_solver.solver.pattern_database import PatternDatabaseManager, PatternDatabases


class SolverError(RuntimeError):
    pass


class SolverTimeout(SolverError):
    pass


class SolverCancelled(SolverError):
    pass


@dataclass(frozen=True)
class SolveResult:
    moves: tuple[Move, ...]
    phase1_length: int
    elapsed_seconds: float
    nodes_searched: int

    @property
    def notation(self) -> str:
        return " ".join(move.notation for move in self.moves)


class TwoPhaseIDASolver:
    def __init__(self, database_manager: PatternDatabaseManager | None = None) -> None:
        self.database_manager = database_manager or PatternDatabaseManager()
        self._deadline = float("inf")
        self._cancel_event: Event | None = None
        self._nodes = 0

    def solve(
        self,
        cube: CubieCube,
        timeout_seconds: float = 30,
        cancel_event: Event | None = None,
    ) -> SolveResult:
        validation = cube.validate()
        if not validation.valid:
            raise SolverError("; ".join(validation.errors))
        if cube.is_solved:
            return SolveResult((), 0, 0.0, 0)

        self.database_manager.ensure_ready()
        started = time.monotonic()
        self._deadline = started + timeout_seconds
        self._cancel_event = cancel_event
        self._nodes = 0

        phase1 = self.search_phase1(cube)
        intermediate = cube.apply_moves(phase1)
        phase2 = self.search_phase2(intermediate)
        moves, phase1_length = _join_phases(phase1, phase2)
        if not cube.apply_moves(moves).is_solved:
            raise SolverError("Internal solver verification failed")
        return SolveResult(moves, phase1_length, time.monotonic() - started, self._nodes)

    def search_phase1(self, cube: CubieCube) -> list[Move]:
        databases = self.database_manager.load()
        moves = databases.moves
        start = (get_twist(cube), get_flip(cube), get_slice_combination(cube))
        bound = self._phase1_heuristic(databases, start)
        path: list[Move] = []
        while bound <= 12:
            if self._dfs_phase1(databases, moves, start, 0, bound, None, path):
                return path.copy()
            bound += 1
        raise SolverError("Phase-one search exceeded the supported depth")

    def search_phase2(self, cube: CubieCube) -> list[Move]:
        databases = self.database_manager.load()
        moves = databases.moves
        try:
            start = (
                get_corner_permutation(cube),
                get_ud_edge_permutation(cube),
                get_slice_permutation(cube),
            )
        except ValueError as exc:
            raise SolverError("Cube is not inside the phase-two subgroup") from exc
        bound = self._phase2_heuristic(databases, start)
        path: list[Move] = []
        while bound <= 18:
            if self._dfs_phase2(databases, moves, start, 0, bound, None, path):
                return path.copy()
            bound += 1
        raise SolverError("Phase-two search exceeded the supported depth")

    def _dfs_phase1(
        self,
        databases: PatternDatabases,
        moves: CoordinateMoveTables,
        state: tuple[int, int, int],
        depth: int,
        bound: int,
        previous: Move | None,
        path: list[Move],
    ) -> bool:
        self._check_limits()
        heuristic = self._phase1_heuristic(databases, state)
        if depth + heuristic > bound:
            return False
        if state == (0, 0, 0):
            return True
        if depth == bound:
            return False

        twist, flip, slice_combination = state
        for move in ALL_MOVES:
            if _should_prune(previous, move):
                continue
            next_state = (
                moves.twist[twist * MOVE_COUNT + int(move)],
                moves.flip[flip * MOVE_COUNT + int(move)],
                moves.slice_combination[slice_combination * MOVE_COUNT + int(move)],
            )
            path.append(move)
            if self._dfs_phase1(
                databases, moves, next_state, depth + 1, bound, move, path
            ):
                return True
            path.pop()
        return False

    def _dfs_phase2(
        self,
        databases: PatternDatabases,
        moves: CoordinateMoveTables,
        state: tuple[int, int, int],
        depth: int,
        bound: int,
        previous: Move | None,
        path: list[Move],
    ) -> bool:
        self._check_limits()
        heuristic = self._phase2_heuristic(databases, state)
        if depth + heuristic > bound:
            return False
        if state == (0, 0, 0):
            return True
        if depth == bound:
            return False

        corner, edge, slice_permutation = state
        for move in PHASE2_MOVES:
            if _should_prune(previous, move):
                continue
            next_state = (
                moves.corner_permutation[corner * MOVE_COUNT + int(move)],
                moves.ud_edge_permutation[edge * MOVE_COUNT + int(move)],
                moves.slice_permutation[slice_permutation * MOVE_COUNT + int(move)],
            )
            path.append(move)
            if self._dfs_phase2(
                databases, moves, next_state, depth + 1, bound, move, path
            ):
                return True
            path.pop()
        return False

    @staticmethod
    def _phase1_heuristic(
        databases: PatternDatabases, state: tuple[int, int, int]
    ) -> int:
        twist, flip, slice_combination = state
        return max(
            databases.phase1_twist_slice[twist * SLICE_COMBINATION_COUNT + slice_combination],
            databases.phase1_flip_slice[flip * SLICE_COMBINATION_COUNT + slice_combination],
        )

    @staticmethod
    def _phase2_heuristic(
        databases: PatternDatabases, state: tuple[int, int, int]
    ) -> int:
        corner, edge, slice_permutation = state
        return max(
            databases.phase2_corner_slice[
                corner * SLICE_PERMUTATION_COUNT + slice_permutation
            ],
            databases.phase2_edge_slice[
                edge * SLICE_PERMUTATION_COUNT + slice_permutation
            ],
        )

    def _check_limits(self) -> None:
        self._nodes += 1
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise SolverCancelled("Solve cancelled")
        if time.monotonic() > self._deadline:
            raise SolverTimeout("Solve timed out")


def _should_prune(previous: Move | None, candidate: Move) -> bool:
    if previous is None:
        return False
    previous_face = previous.face_index
    candidate_face = candidate.face_index
    if previous_face == candidate_face:
        return True
    return previous_face % 3 == candidate_face % 3 and previous_face > candidate_face


def _join_phases(phase1: list[Move], phase2: list[Move]) -> tuple[tuple[Move, ...], int]:
    """Combine a redundant same-face pair at the phase boundary."""

    first = phase1.copy()
    second = phase2.copy()
    if first and second and first[-1].face_index == second[0].face_index:
        left = first.pop()
        right = second.pop(0)
        combined_amount = (left.amount + right.amount) % 4
        if combined_amount:
            first.append(Move(left.face_index * 3 + combined_amount - 1))
    return tuple(first + second), len(first)
