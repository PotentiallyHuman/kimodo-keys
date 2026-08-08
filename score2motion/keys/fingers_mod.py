"""Inject MIDI finger motion into the exported skeleton's finger channels.

Kimodo's export skeleton (somaskel77) carries full finger chains, but the generative
model leaves them at the relaxed rest pose. This module writes real playing motion into
those channels, from the same press plan that drives the hands.

Anatomy (from the clinical ROM literature): MCP flexes and abducts, PIP/DIP are hinges;
a press is mostly MCP flexion with slight PIP/DIP counter-motion; travelling fingers
lift. Angles are deltas on TOP of the model's relaxed rest pose, so hands keep their
natural curl between presses.
"""
from __future__ import annotations

import math

import torch

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

STRIKE = {"1": 22.0, "2": -7.0, "3": -5.0}   # segment -> extra flexion (deg) during press
TRAVEL = {"1": -7.0, "2": 4.0, "3": 2.0}     # lift while moving to the next note
ATTACK, RELEASE = 0.06, 0.09                  # s


def _flex_state(plan, hand, finger, t):
    """(weight of strike pose 0..1, weight of travel pose 0..1) for this finger at t."""
    strike = 0.0
    nxt_gap = None
    for q in plan:
        if q.hand != hand or q.finger != finger:
            continue
        if q.t_on <= t <= q.t_off:
            strike = 1.0
            break
        if q.t_on - ATTACK <= t < q.t_on:
            strike = max(strike, (t - (q.t_on - ATTACK)) / ATTACK)
        elif q.t_off < t <= q.t_off + RELEASE:
            strike = max(strike, 1.0 - (t - q.t_off) / RELEASE)
        if t < q.t_on and (nxt_gap is None or q.t_on - t < nxt_gap):
            nxt_gap = q.t_on - t
    travel = 1.0 if (strike < 0.05 and nxt_gap is not None and nxt_gap < 0.6) else 0.0
    return strike, travel


def inject_fingers(
    local_rot_mats: torch.Tensor,     # [T, 77, 3, 3] — somaskel77 local rotations
    skeleton77,
    plan,
    *,
    fps: float,
    flex_axis: int = 0,               # local axis a finger joint bends around (X for SOMA)
) -> torch.Tensor:
    """Return a copy with finger channels animated from the press plan."""
    out = local_rot_mats.clone()
    ix = {n: i for i, n in enumerate(skeleton77.bone_order_names)}
    T = out.shape[0]

    def rot(axis: int, deg: float, dtype, device):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        m = torch.eye(3, dtype=dtype, device=device)
        i, j = [(1, 2), (0, 2), (0, 1)][axis]
        m[i, i] = c; m[j, j] = c
        m[i, j] = -s if axis != 1 else s
        m[j, i] = s if axis != 1 else -s
        return m

    for f in range(T):
        t = f / fps
        for side, hand in (("Left", "Left"), ("Right", "Right")):
            for fi, fname in enumerate(FINGER_NAMES, start=1):
                strike, travel = _flex_state(plan, hand, fi, t)
                if strike < 1e-3 and travel < 1e-3:
                    continue
                for seg in ("1", "2", "3"):
                    jname = f"{side}Hand{fname}{seg}"
                    if jname not in ix:
                        continue
                    deg = STRIKE[seg] * strike + TRAVEL[seg] * travel
                    j = ix[jname]
                    m = rot(flex_axis, deg, out.dtype, out.device)
                    out[f, j] = out[f, j] @ m
    return out
