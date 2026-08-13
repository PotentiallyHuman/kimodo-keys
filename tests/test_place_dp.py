"""The DP placer's laws, checked on toy inputs -- fast, no audio, no Blender."""
import json
import subprocess
import sys

import pytest

from score2motion.fretted import place_dp
from score2motion.fretted.instrument import PRESETS


class Opts:
    honor_artic = False
    economy = False
    funk_mute = False


def notes(*pitches, ioi=0.5, dur=0.4):
    return [{"t_on": i * ioi, "t_off": i * ioi + dur, "pitch": p}
            for i, p in enumerate(pitches)]


def setup_function(_):
    place_dp.configure([28, 33, 38, 43], 12)     # 4-string bass, 12-fret window


def test_lattice_unison_class():
    # pitch 38 lives at (s0,f10) and (s1,f5); fret 0 homes are opens, not cells
    assert place_dp.fretted_cells(38) == [(0, 10), (1, 5)]
    # the lattice law: same pitch, one string up, five frets back
    for s, f in [(0, 10), (1, 6), (2, 3)]:
        assert place_dp.OPEN[s] + f == place_dp.OPEN[s + 1] + (f - 5)


def test_determinism():
    ns = notes(33, 38, 40, 38, 33, 43, 45, 43)
    a = place_dp.solve(ns, Opts())[0]
    b = place_dp.solve(ns, Opts())[0]
    assert a == b


def test_opens_only_when_forced():
    # 28 has no fretted home -> open; 33 has one -> fretted, never open
    path, *_ = place_dp.solve(notes(28, 33, 28), Opts())
    kinds = [st[3] for st in path]
    assert kinds[0] == "open" and kinds[2] == "open"
    assert kinds[1] == "in" and path[1][2] > 0


def test_shift_time_gate_blocks_impossible_moves():
    # pitch 29 pins the hand at position 1; pitch 55 needs position >= 8.
    # 50 ms is far under the 150+25*|dR| ms gate -> loud dead end.
    ns = notes(29, 55, ioi=0.05, dur=0.04)
    with pytest.raises(SystemExit):
        place_dp.solve(ns, Opts())


def test_octave_pairs_rigid_on_fourths():
    path, *_ = place_dp.solve(notes(33, 45, 33, 45), Opts())
    tags = place_dp.assign_tags(path, notes(33, 45, 33, 45))
    assert all(t == "octave_shape" for t in tags)
    lo, hi = path[0], path[1]
    assert hi[1] == lo[1] + 2 and hi[2] == lo[2] + 2   # the (+2,+2) vector


def test_non_fourths_tuning_skips_octave_tagging():
    g = PRESETS["guitar"]
    place_dp.configure(g.open_notes, 12)
    assert place_dp.ALL_FOURTHS is False
    path, *_ = place_dp.solve(notes(45, 57), Opts())
    tags = place_dp.assign_tags(path, notes(45, 57))
    assert "octave_shape" not in tags


def test_five_string_reaches_low_b():
    b5 = PRESETS["bass5"]
    place_dp.configure(b5.open_notes, 12)
    path, *_ = place_dp.solve(notes(23, 25, 23), Opts())
    assert path[0][3] == "open"                  # low B itself: open
    assert path[1][3] == "in" and path[1][1] == 0  # 25 fretted on the B string


def test_cli_end_to_end(tmp_path):
    plan = tmp_path / "plan.json"
    out = tmp_path / "placed.json"
    json.dump({"notes": notes(33, 38, 45, 38)}, open(plan, "w"))
    r = subprocess.run([sys.executable, "-m", "score2motion.fretted.place_dp",
                        str(plan), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    d = json.load(open(out))
    assert len(d["notes"]) == 4
    for n in d["notes"]:
        assert d["tuning"][n["string"]] + n["fret"] == n["pitch"]
