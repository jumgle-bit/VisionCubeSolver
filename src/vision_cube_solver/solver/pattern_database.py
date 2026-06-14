"""Persistent move tables and admissible two-phase pruning tables."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vision_cube_solver.cube.enums import ALL_MOVES, PHASE2_MOVES, Move
from vision_cube_solver.solver.coordinates import (
    FLIP_COUNT,
    MOVE_COUNT,
    PERMUTATION_8_COUNT,
    SLICE_COMBINATION_COUNT,
    SLICE_PERMUTATION_COUNT,
    TWIST_COUNT,
    CoordinateMoveTables,
)

PDB_VERSION = 1
ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class PatternDatabases:
    moves: CoordinateMoveTables
    phase1_twist_slice: bytearray
    phase1_flip_slice: bytearray
    phase2_corner_slice: bytearray
    phase2_edge_slice: bytearray


class PatternDatabaseManager:
    def __init__(self, cache_path: Path | str = "data/pdb/two_phase_v1.pkl") -> None:
        self.cache_path = Path(cache_path)
        self._loaded: PatternDatabases | None = None

    def ensure_ready(self, progress_callback: ProgressCallback | None = None) -> None:
        if self._loaded is not None:
            return
        try:
            self._loaded = self._load_from_disk()
        except (
            OSError,
            EOFError,
            ValueError,
            pickle.PickleError,
            AttributeError,
            ImportError,
            TypeError,
        ):
            self.rebuild(progress_callback)

    def load(self) -> PatternDatabases:
        self.ensure_ready()
        assert self._loaded is not None
        return self._loaded

    def rebuild(self, progress_callback: ProgressCallback | None = None) -> None:
        moves = CoordinateMoveTables.build(progress_callback)
        databases = PatternDatabases(
            moves=moves,
            phase1_twist_slice=_build_pruning_table(
                "phase 1 twist/slice",
                moves.twist,
                TWIST_COUNT,
                moves.slice_combination,
                SLICE_COMBINATION_COUNT,
                ALL_MOVES,
                progress_callback,
            ),
            phase1_flip_slice=_build_pruning_table(
                "phase 1 flip/slice",
                moves.flip,
                FLIP_COUNT,
                moves.slice_combination,
                SLICE_COMBINATION_COUNT,
                ALL_MOVES,
                progress_callback,
            ),
            phase2_corner_slice=_build_pruning_table(
                "phase 2 corner/slice",
                moves.corner_permutation,
                PERMUTATION_8_COUNT,
                moves.slice_permutation,
                SLICE_PERMUTATION_COUNT,
                PHASE2_MOVES,
                progress_callback,
            ),
            phase2_edge_slice=_build_pruning_table(
                "phase 2 edge/slice",
                moves.ud_edge_permutation,
                PERMUTATION_8_COUNT,
                moves.slice_permutation,
                SLICE_PERMUTATION_COUNT,
                PHASE2_MOVES,
                progress_callback,
            ),
        )
        self._validate(databases)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(".tmp")
        with temporary.open("wb") as output:
            pickle.dump(
                {"version": PDB_VERSION, "databases": databases},
                output,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        temporary.replace(self.cache_path)
        self._loaded = databases

    def _load_from_disk(self) -> PatternDatabases:
        with self.cache_path.open("rb") as source:
            payload = pickle.load(source)
        if not isinstance(payload, dict) or payload.get("version") != PDB_VERSION:
            raise ValueError("Pattern database version mismatch")
        databases = payload.get("databases")
        if not isinstance(databases, PatternDatabases):
            raise ValueError("Invalid pattern database payload")
        self._validate(databases)
        return databases

    @staticmethod
    def _validate(databases: PatternDatabases) -> None:
        expected = (
            (databases.phase1_twist_slice, TWIST_COUNT * SLICE_COMBINATION_COUNT),
            (databases.phase1_flip_slice, FLIP_COUNT * SLICE_COMBINATION_COUNT),
            (
                databases.phase2_corner_slice,
                PERMUTATION_8_COUNT * SLICE_PERMUTATION_COUNT,
            ),
            (databases.phase2_edge_slice, PERMUTATION_8_COUNT * SLICE_PERMUTATION_COUNT),
        )
        for table, size in expected:
            if not isinstance(table, bytearray) or len(table) != size or table[0] != 0:
                raise ValueError("Pattern database failed validation")


def _build_pruning_table(
    label: str,
    first_moves: object,
    first_count: int,
    second_moves: object,
    second_count: int,
    allowed_moves: tuple[Move, ...],
    progress: ProgressCallback | None,
) -> bytearray:
    size = first_count * second_count
    distances = bytearray([255]) * size
    distances[0] = 0
    frontier = [0]
    visited = 1
    depth = 0

    while frontier:
        next_frontier: list[int] = []
        for coordinate in frontier:
            first, second = divmod(coordinate, second_count)
            first_base = first * MOVE_COUNT
            second_base = second * MOVE_COUNT
            for move in allowed_moves:
                move_index = int(move)
                target = (
                    first_moves[first_base + move_index] * second_count
                    + second_moves[second_base + move_index]
                )
                if distances[target] == 255:
                    distances[target] = depth + 1
                    next_frontier.append(target)
        frontier = next_frontier
        visited += len(frontier)
        depth += 1
        if progress:
            progress(label, visited, size)

    if visited != size:
        raise ValueError(f"{label} is incomplete: reached {visited} of {size} states")
    return distances
