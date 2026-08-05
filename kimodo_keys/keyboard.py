"""A virtual keyboard in the motion model's world space.

The board is a rigid object: 52 white keys x 23.5 mm (standard), black keys in the rear
half of the key bed. It is PLACED relative to the generated character (in front of the
body, at the height of its hands), then never resized or reshaped — hand targets are
expressed through it.
"""
from __future__ import annotations

import numpy as np

WHITE_W = 0.0235          # standard white key pitch (m)
MIDI_LOW = 21             # A0
WHITE_SEMIS = {0, 2, 4, 5, 7, 9, 11}
DEPTH = 0.145             # key bed depth (m)
WHITE_TOP = 0.032         # z of white key top surface, board-local (m)
BLACK_TOP = 0.044


def is_white(n: int) -> bool:
    return (n % 12) in WHITE_SEMIS


def white_index(n: int) -> int:
    return sum(1 for k in range(MIDI_LOW, n + 1) if is_white(k))


N_WHITE = white_index(108)
BOARD_W = N_WHITE * WHITE_W


def key_local(n: int) -> np.ndarray:
    """Board-local centre of a key's playing surface (x along the board, y into it)."""
    if is_white(n):
        return np.array([(white_index(n) - 0.5) * WHITE_W - BOARD_W / 2, 0.0, 0.0])
    return np.array([white_index(n - 1) * WHITE_W - BOARD_W / 2, 0.088, 0.0])


def key_top_local(n: int) -> np.ndarray:
    v = key_local(n).copy()
    v[2] = BLACK_TOP if not is_white(n) else WHITE_TOP
    return v


class PlacedKeyboard:
    """A keyboard placed in world space: world = local @ R.T + T."""

    def __init__(self, R: np.ndarray, T: np.ndarray):
        self.R, self.T = R, T

    def key_point(self, note: int) -> np.ndarray:
        return key_top_local(note) @ self.R.T + self.T

    def key_centre(self, note: int) -> np.ndarray:
        return key_local(note) @ self.R.T + self.T

    @classmethod
    def place_from_take(cls, joints: np.ndarray, names: list[str], plan,
                        up_axis: int = 1) -> "PlacedKeyboard":
        """Block the set from a generated take: board in front of the character at hand
        height, the PLAYED key zone centred on the body.

        joints: [T, J, 3] posed joints of the base take. names: skeleton joint names.
        up_axis: which world axis is up in the model's convention (Kimodo BVH: Y=1).
        """
        ix = {n: i for i, n in enumerate(names)}
        lh, rh = joints[:, ix["LeftHand"]], joints[:, ix["RightHand"]]
        hips = joints[:, ix["Hips"]]
        mid = (lh.mean(0) + rh.mean(0)) / 2

        # facing: hips -> average hand direction, flattened out of the up axis
        fwd = mid - hips.mean(0)
        fwd[up_axis] = 0.0
        n = np.linalg.norm(fwd)
        fwd = fwd / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
        up = np.zeros(3)
        up[up_axis] = 1.0
        left = np.cross(up, fwd)

        played = [key_local(q.note)[0] for q in plan]
        zone_mid = (min(played) + max(played)) / 2 if played else 0.0

        centre = mid + fwd * 0.06 - up * 0.02 + (-left) * zone_mid
        # board local +x = high notes = player's RIGHT = -left
        bx = -left / max(1e-9, np.linalg.norm(left))
        by = -fwd                       # local +y (rear of key bed) points away from player
        bz = up
        R = np.column_stack([bx, np.cross(bz, bx), bz])   # orthonormal, z up
        # re-express so that local y maps to -fwd:
        R[:, 1] = np.cross(bz, bx)
        if np.dot(R[:, 1], -fwd) < 0:
            R[:, 1] = -R[:, 1]
        return cls(R, centre)
