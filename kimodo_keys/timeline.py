"""Hand location timelines: where each WRIST must be, at every moment.

Authored from the press plan before any generation: hold on the key during a press
(with a small dip), glide between key zones during gaps, arrive ~120 ms before each
onset. One control point per CHORD (simultaneous presses move together); the wrist is
offset so the pressing finger — not the palm centre — meets the key.
"""
from __future__ import annotations

import numpy as np

from .keyboard import PlacedKeyboard
from .plan import Press, chord_groups

# lateral fingertip offset from the wrist along the key axis (m, + toward pinky side)
FINGER_LATERAL = {1: -0.015, 2: 0.005, 3: 0.022, 4: 0.038, 5: 0.052}
ANTICIPATION = 0.12
HOVER, DIP = 0.021, 0.018
WRIST_BACK = 0.075         # wrist sits behind the key contact point


def _wrist_for_press(kb: PlacedKeyboard, q: Press) -> np.ndarray:
    lat = FINGER_LATERAL.get(q.finger or 3, 0.02)
    if q.hand == "Left":
        lat = -lat
    local = kb.R.T @ (kb.key_point(q.note) - kb.T)
    local = local.copy()
    local[0] -= lat
    local[1] -= WRIST_BACK
    return local @ kb.R.T + kb.T


def _smoothstep(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


class HandTimeline:
    def __init__(self, kb: PlacedKeyboard, plan: list[Press], hand: str, up: np.ndarray):
        self.pts: list[tuple[float, np.ndarray]] = []
        for grp in chord_groups(plan, hand):
            w = np.mean([_wrist_for_press(kb, q) for q in grp], axis=0)
            t_on = min(q.t_on for q in grp)
            t_off = max(q.t_off for q in grp)
            self.pts += [
                (max(0.0, t_on - ANTICIPATION), w + up * HOVER),
                (t_on, w + up * DIP),
                (t_off, w + up * DIP),
                (t_off + 0.06, w + up * HOVER),
            ]
        self.pts.sort(key=lambda x: x[0])

    def __bool__(self):
        return bool(self.pts)

    def at(self, t: float) -> np.ndarray | None:
        if not self.pts:
            return None
        if t <= self.pts[0][0]:
            return self.pts[0][1]
        if t >= self.pts[-1][0]:
            return self.pts[-1][1]
        for (t0, p0), (t1, p1) in zip(self.pts[:-1], self.pts[1:]):
            if t0 <= t <= t1:
                u = _smoothstep((t - t0) / max(1e-4, t1 - t0))
                return p0 + (p1 - p0) * u
        return self.pts[-1][1]

    def sample(self, n_frames: int, fps: float) -> np.ndarray:
        return np.stack([self.at(i / fps) for i in range(n_frames)])
