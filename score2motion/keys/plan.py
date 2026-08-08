"""MIDI -> per-hand press plan.

A press = one finger putting one key down for an interval. The plan is the only musical
input the rest of the library needs: everything downstream (hand paths, constraints,
finger animation) is derived from it.

Fingering: uses `pianoplayer` (MIT) when installed — real lookahead fingering with
thumb-unders. Falls back to a pitch-order heuristic so the library works without it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Press:
    t_on: float
    t_off: float
    note: int          # MIDI pitch
    hand: str          # "Left" | "Right"
    finger: int        # 1=thumb .. 5=pinky

    def to_dict(self):
        return asdict(self)


def _fallback_fingering(notes, hand):
    """Pitch-order heuristic: spread simultaneous notes across fingers, else alternate."""
    out = []
    notes = sorted(notes, key=lambda n: (n[0], n[2]))
    i = 0
    while i < len(notes):
        chord = [notes[i]]
        while i + 1 < len(notes) and notes[i + 1][0] - notes[i][0] < 0.04:
            i += 1
            chord.append(notes[i])
        chord.sort(key=lambda n: n[2])
        if hand == "Left":       # low note -> pinky
            fingers = [5, 3, 1, 2, 4][: len(chord)]
        else:                    # low note -> thumb
            fingers = [1, 3, 5, 2, 4][: len(chord)]
        for (t0, t1, p), f in zip(chord, fingers):
            out.append(Press(round(t0, 3), round(t1, 3), p, hand, f))
        i += 1
    return out


def extract_presses(
    midi_path: str,
    start: float,
    end: float,
    *,
    solo_track: Optional[int] = None,
    chord_track: Optional[int] = None,
    split: int = 60,
    hand_size: str = "M",
) -> list[Press]:
    """Read a MIDI file and produce the press plan for a window [start, end) seconds.

    With solo_track/chord_track given, that split is used (solo -> Right, chords below
    `split` -> Left). Without, all notes are split by pitch at `split`.
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(midi_path)

    def window(inst):
        return [(max(0.0, n.start - start), min(end - start, n.end - start), n.pitch)
                for n in inst.notes if n.start < end and n.end > start]

    if solo_track is not None:
        right = window(pm.instruments[solo_track])
        left = []
        if chord_track is not None:
            left = [n for n in window(pm.instruments[chord_track]) if n[2] < split]
    else:
        allnotes = [n for i in pm.instruments if not i.is_drum for n in window(i)]
        right = [n for n in allnotes if n[2] >= split]
        left = [n for n in allnotes if n[2] < split]

    # dedup doubled voices (same onset+pitch)
    def dedup(ns):
        seen, out = set(), []
        for t0, t1, p in sorted(ns):
            k = (round(t0, 2), p)
            if k not in seen:
                seen.add(k)
                out.append((t0, t1, p))
        return out

    right, left = dedup(right), dedup(left)

    try:
        from .fingering_pp import assign_with_pianoplayer
        return assign_with_pianoplayer(right, left, hand_size=hand_size)
    except Exception:
        return _fallback_fingering(left, "Left") + _fallback_fingering(right, "Right")


def chord_groups(plan: list[Press], hand: str, window_s: float = 0.04) -> list[list[Press]]:
    """Presses grouped into chords: same hand, onsets within `window_s` play together."""
    ps = sorted([q for q in plan if q.hand == hand and q.finger], key=lambda q: q.t_on)
    groups: list[list[Press]] = []
    for q in ps:
        if groups and q.t_on - groups[-1][0].t_on < window_s:
            groups[-1].append(q)
        else:
            groups.append([q])
    return groups
