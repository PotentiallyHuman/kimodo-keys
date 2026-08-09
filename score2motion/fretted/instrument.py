"""What a fretted instrument is, as far as the solver is concerned.

Everything downstream (fingering, hand placement, marker display) is derived
from these numbers, so a new instrument is a new entry here and nothing else.

Fret positions come from the rule of 18 -- the twelfth fret sits at exactly
half the scale length, and each fret is 2**(1/12) closer than the last. That
is a fact of physics, not a modelling choice, so the geometry a canon .blend
carries must agree with it; `check_against` tells you whether it does.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Instrument:
    name: str
    open_notes: List[int]            # MIDI note of each open string, lowest first
    scale_length_in: float           # nut to bridge
    nut_width_mm: float
    max_fret: int = 12

    @property
    def n_strings(self) -> int:
        return len(self.open_notes)

    @property
    def string_spacing_mm(self) -> float:
        # strings are laid out across ~85% of the nut, leaving the edge margin
        # a real neck has so the outer strings do not fall off the fretboard
        return (self.nut_width_mm * 0.85) / (self.n_strings - 1)

    def fret_distance_in(self, fret: int) -> float:
        """Distance from the nut to `fret`, in inches. Fret 0 is the nut."""
        return self.scale_length_in * (1 - 2 ** (-fret / 12.0))

    def note_at(self, string: int, fret: int) -> int:
        return self.open_notes[string] + fret

    def positions_for(self, pitch: int):
        """Every (string, fret) that sounds this pitch, low fret first."""
        out = []
        for s, o in enumerate(self.open_notes):
            f = pitch - o
            if 0 <= f <= self.max_fret:
                out.append((s, f))
        return sorted(out, key=lambda p: p[1])

    def range_midi(self):
        return self.open_notes[0], self.open_notes[-1] + self.max_fret

    def check_against(self, fret_x_cm: dict, tol_mm: float = 3.0):
        """Compare a canon's modelled fret positions to the rule of 18.

        Returns (ok, worst_mm, detail). A canon that fails this is not a canon
        -- the fingers would be told to press where no fret is.
        """
        if 0 not in fret_x_cm or self.max_fret not in fret_x_cm:
            return False, float("inf"), "no fret 0 / max fret in the geometry"
        nut = fret_x_cm[0]
        span = abs(fret_x_cm[self.max_fret] - nut)
        ideal_span = self.fret_distance_in(self.max_fret) * 2.54
        k = span / ideal_span if ideal_span else 1.0        # model units per real cm
        worst, where = 0.0, None
        for f, x in sorted(fret_x_cm.items()):
            want = self.fret_distance_in(f) * 2.54 * k
            got = abs(x - nut)
            err = abs(want - got) * 10
            if err > worst:
                worst, where = err, f
        return worst <= tol_mm, worst, f"worst at fret {where}"


# 24 frets. The default of 12 is a ukulele's worth of neck, and it dropped real
# notes: a part asking for anything above the 12th fret simply lost them. A modern
# electric has 22 to 24, and 24 puts the top note two full octaves above the open
# string, which is what the arrangement expects.
#
# This caps the PLANNER, not just the geometry. Left at 12 the planner never even
# offers a position above the twelfth, so a part gets re-voiced down or loses
# notes before it ever reaches the neck -- and the neck looks innocent, because
# nothing it was asked to play was ever above the twelfth.
BASS4 = Instrument("bass", [28, 33, 38, 43], 34.0, 38.0,
                   max_fret=24)     # E1 A1 D2 G2
GUITAR6 = Instrument("guitar", [40, 45, 50, 55, 59, 64], 25.5, 43.0,
                     max_fret=24)   # E2 A2 D3 G3 B3 E4
BASS5 = Instrument("bass5", [23, 28, 33, 38, 43], 34.0, 45.0,
                   max_fret=24)     # B0 on top
UKULELE = Instrument("ukulele", [67, 60, 64, 69], 17.0, 35.0)     # G C E A, reentrant

PRESETS = {i.name: i for i in (BASS4, GUITAR6, BASS5, UKULELE)}


def get(name: str) -> Instrument:
    if name not in PRESETS:
        raise SystemExit(f"unknown instrument {name!r}; have: {', '.join(sorted(PRESETS))}")
    return PRESETS[name]
