"""Shared cube enumerations and orientation conventions."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class CubeColor(StrEnum):
    WHITE = "white"
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    BLUE = "blue"


class Face(StrEnum):
    U = "U"
    R = "R"
    F = "F"
    D = "D"
    L = "L"
    B = "B"


FACE_ORDER: tuple[Face, ...] = (Face.U, Face.R, Face.F, Face.D, Face.L, Face.B)

FACE_COLORS: dict[Face, CubeColor] = {
    Face.U: CubeColor.WHITE,
    Face.R: CubeColor.RED,
    Face.F: CubeColor.GREEN,
    Face.D: CubeColor.YELLOW,
    Face.L: CubeColor.ORANGE,
    Face.B: CubeColor.BLUE,
}

COLOR_FACES: dict[CubeColor, Face] = {color: face for face, color in FACE_COLORS.items()}

TOP_REFERENCE: dict[CubeColor, CubeColor] = {
    CubeColor.GREEN: CubeColor.WHITE,
    CubeColor.RED: CubeColor.WHITE,
    CubeColor.BLUE: CubeColor.WHITE,
    CubeColor.ORANGE: CubeColor.WHITE,
    CubeColor.WHITE: CubeColor.BLUE,
    CubeColor.YELLOW: CubeColor.GREEN,
}

DISPLAY_COLORS: dict[CubeColor, str] = {
    CubeColor.WHITE: "#f5f5f5",
    CubeColor.RED: "#d32f2f",
    CubeColor.GREEN: "#2eaa58",
    CubeColor.YELLOW: "#f5d442",
    CubeColor.ORANGE: "#f57c00",
    CubeColor.BLUE: "#1976d2",
}


class Move(IntEnum):
    U = 0
    U2 = 1
    U_PRIME = 2
    R = 3
    R2 = 4
    R_PRIME = 5
    F = 6
    F2 = 7
    F_PRIME = 8
    D = 9
    D2 = 10
    D_PRIME = 11
    L = 12
    L2 = 13
    L_PRIME = 14
    B = 15
    B2 = 16
    B_PRIME = 17

    @property
    def face_index(self) -> int:
        return int(self) // 3

    @property
    def amount(self) -> int:
        return int(self) % 3 + 1

    @property
    def notation(self) -> str:
        face = "URFDLB"[self.face_index]
        suffix = ("", "2", "'")[int(self) % 3]
        return face + suffix

    @classmethod
    def from_notation(cls, notation: str) -> Move:
        notation = notation.strip()
        if not notation or notation[0] not in "URFDLB":
            raise ValueError(f"Invalid move notation: {notation!r}")
        suffix = notation[1:]
        amount_index = {"": 0, "2": 1, "'": 2}.get(suffix)
        if amount_index is None:
            raise ValueError(f"Invalid move notation: {notation!r}")
        return cls("URFDLB".index(notation[0]) * 3 + amount_index)


ALL_MOVES: tuple[Move, ...] = tuple(Move)
PHASE2_MOVES: tuple[Move, ...] = (
    Move.U,
    Move.U2,
    Move.U_PRIME,
    Move.D,
    Move.D2,
    Move.D_PRIME,
    Move.R2,
    Move.L2,
    Move.F2,
    Move.B2,
)

