"""Bridge to `pianoplayer` (MIT) — the ergonomic fingering engine.

pianoplayer implements the cost model from the piano-pedagogy literature (Parncutt-style
ergonomics): candidate fingerings are scored on hand-span stretch against hand size,
per-finger strength weights (thumb/index cheap, ring expensive), and a multi-note
lookahead that plans thumb-unders before runs arrive. This is what makes the output
finger like a pianist rather than type with one finger.
"""
from __future__ import annotations

from .plan import Press

OCTAVE_CM = 16.5


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def assign_with_pianoplayer(right, left, *, hand_size: str = "M") -> list[Press]:
    """right/left: [(t_on, t_off, midi)] already split per hand -> fingered Press list."""
    from pianoplayer.hand import Hand
    from pianoplayer.models import INote

    out: list[Press] = []
    for hand_side, sub in (("Left", left), ("Right", right)):
        sub = sorted(sub, key=lambda n: (n[0], n[2]))
        if not sub:
            continue
        seq = []
        for i, (t_on, t_off, pitch) in enumerate(sub):
            n = INote()
            n.pitch = pitch
            n.octave = pitch // 12 - 1
            n.time = t_on
            n.duration = max(0.05, t_off - t_on)
            n.isBlack = (pitch % 12) in {1, 3, 6, 8, 10}
            n.x = OCTAVE_CM * (pitch // 12) + (pitch % 12) * (OCTAVE_CM / 7.0)
            n.noteID = i
            seq.append(n)
        h = Hand(seq, "left" if hand_side == "Left" else "right", hand_size)
        h.noteseq = seq
        h.generate(start_measure=0, nmeasures=10 ** 5)
        for n, (t_on, t_off, pitch) in zip(seq, sub):
            f = int(n.fingering) if str(n.fingering).isdigit() else 3
            out.append(Press(round(t_on, 3), round(t_off, 3), pitch, hand_side,
                             _clamp(f, 1, 5)))
    if not out:
        raise RuntimeError("no notes fingered")
    return sorted(out, key=lambda p: p.t_on)
