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

# Lateral fingertip offset from the wrist along the key axis (m, + toward pinky side).
#
# CALIBRATED 2026-08-06. Two distinct numbers were being confused here, so both are named:
#   RESTING SPREAD  where the fingers naturally sit, thumb to pinky. This is what belongs
#                   in this table, because it places the wrist for an ordinary press.
#   MAXIMUM SPAN    the stretched thumb-to-pinky reach. Larger, and only relevant to how
#                   wide an interval the hand can take without moving the arm.
# Boyle, Boyle & Booker 2015 (APPCA, n=473) give MAXIMUM span: 22.6 cm male, 20.1 cm
# female. pianoplayer's size-M model pairs a 17.2 cm maximum with resting offsets spanning
# 10.3 cm, so scaling those offsets by 22.6/17.2 gives a 13.5 cm resting spread for an
# adult male hand. The previous 6.7 cm was roughly half an anatomical hand.
_REST_CM = (-5.74, -2.30, 0.0, +2.30, +4.59)      # pianoplayer size M, from middle finger
_SPAN_SCALE = 22.6 / 17.2                          # size M maximum -> adult male maximum
_MIDDLE_AT = 0.022                                 # middle fingertip sits ahead of wrist
FINGER_LATERAL = {i + 1: round(v / 100.0 * _SPAN_SCALE + _MIDDLE_AT, 4)
                  for i, v in enumerate(_REST_CM)}
HAND_SPAN_M = 0.226        # MAXIMUM span, male; 0.201 female (scale this table by 0.889)

# Anticipation is NOT a fixed duration. Dalla Bella & Palmer 2011 (PLOS ONE 6:e20518) found
# the hand leads by roughly one inter-onset interval at every tempo tested (100-123% of IOI,
# from 500 ms down to 120 ms). A constant lead is right only near IOI 240 ms; on a fast run
# it reaches twice as early as a human, which reads as floaty and pre-emptive.
ANTICIPATION_IOI = 1.05
ANTICIPATION = 0.12        # fallback when the plan is too sparse to measure an IOI

# Vertical geometry, Dalla Bella & Palmer 2011: fingertips swing 21.7-26.1 mm above the
# pressed key, and the key surface sits ~10 mm above that, leaving ~13 mm of clearance.
HOVER, DIP = 0.013, 0.010  # DIP = standard grand key dip (only 6-7 mm used at soft levels)
WRIST_BACK = 0.075         # wrist sits behind the key contact point


def measure_anticipation(plan: list[Press], hand: str) -> float:
    """Lead time for this hand, scaled to the music's own inter-onset interval."""
    onsets = sorted({q.t_on for q in plan if q.hand == hand})
    gaps = [b - a for a, b in zip(onsets, onsets[1:]) if 1e-3 < b - a < 2.0]
    if not gaps:
        return ANTICIPATION
    gaps.sort()
    return max(0.06, min(0.60, ANTICIPATION_IOI * gaps[len(gaps) // 2]))


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
        lead = measure_anticipation(plan, hand)
        for grp in chord_groups(plan, hand):
            w = np.mean([_wrist_for_press(kb, q) for q in grp], axis=0)
            t_on = min(q.t_on for q in grp)
            t_off = max(q.t_off for q in grp)
            self.pts += [
                (max(0.0, t_on - lead), w + up * HOVER),
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
