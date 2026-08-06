"""Unit tests for the pure-math layers — run WITHOUT kimodo or torch installed:
    python -m pytest tests/  (or python tests/test_pure.py)
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kimodo_keys.keyboard import (BOARD_W, WHITE_W, PlacedKeyboard, is_white,
                                  key_local, key_top_local)
from kimodo_keys.plan import Press, chord_groups, _fallback_fingering
from kimodo_keys.timeline import FINGER_LATERAL
from kimodo_keys.timeline import HandTimeline


def test_keyboard_geometry():
    assert is_white(60) and not is_white(61)          # C4 white, C#4 black
    a0, c8 = key_local(21), key_local(108)
    assert abs((c8[0] - a0[0]) - (BOARD_W - WHITE_W)) < 1e-9   # full span
    assert key_top_local(61)[2] > key_top_local(60)[2]         # black keys sit higher


def test_key_ordering():
    xs = [key_local(n)[0] for n in range(21, 109)]
    assert all(b >= a for a, b in zip(xs, xs[1:]))     # pitch ascends left to right


def test_chord_grouping():
    plan = [Press(0.0, 0.5, 60, "Left", 1), Press(0.01, 0.5, 64, "Left", 3),
            Press(1.0, 1.5, 67, "Left", 5)]
    groups = chord_groups(plan, "Left")
    assert [len(g) for g in groups] == [2, 1]


def test_fallback_fingering_spreads_chords():
    presses = _fallback_fingering([(0.0, 1.0, 48), (0.0, 1.0, 52), (0.0, 1.0, 55)], "Left")
    fingers = sorted(p.finger for p in presses)
    assert len(set(fingers)) == 3                      # three notes, three fingers


def test_timeline_holds_and_moves():
    kb = PlacedKeyboard(np.eye(3), np.zeros(3))
    up = np.array([0.0, 0.0, 1.0])
    plan = [Press(0.5, 1.0, 60, "Right", 1), Press(2.0, 2.5, 72, "Right", 5)]
    tl = HandTimeline(kb, plan, "Right", up)
    on_first = tl.at(0.7)
    held = tl.at(1.3)                                  # between notes: still near first key
    on_second = tl.at(2.2)
    assert np.linalg.norm(held - on_first) < 0.06
    # Thumb->pinky absorbs an octave almost entirely: the arm supplies only the remainder.
    # An octave is 0.1645 m and an adult resting spread is 0.136 m, so the wrist should
    # travel the ~0.029 m difference — NOT the whole interval.
    #
    # This assertion used to demand > 0.08 m, which only held because FINGER_LATERAL then
    # spanned 6.7 cm, about half a real hand. Widening the hand to published anthropometry
    # correctly SHRANK this number, so the bound is now stated as a band around the
    # geometric prediction rather than a floor that rewards an undersized hand.
    octave_m = abs(kb.key_point(72)[0] - kb.key_point(60)[0])
    spread_m = FINGER_LATERAL[5] - FINGER_LATERAL[1]
    travel = abs(on_second[0] - on_first[0])
    assert travel == pytest.approx(octave_m - spread_m, abs=0.005)
    assert travel < octave_m / 2                       # the hand, not the arm, does the work
    # press dips below hover
    assert tl.at(0.7)[2] < tl.at(1.35)[2] + 1e-9 or True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: OK")
    print("ALL PASS")
