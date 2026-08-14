"""Are the fingers actually on the strings -- on the instrument as it is NOW?

Contact is the one thing a performance cannot fake, and it is easy to check
wrongly: measure against where the instrument was at bake time and a hand
that has drifted 10 cm still reports perfect. So every distance here is taken
against the LIVE fret and string objects, evaluated at the frame being judged.

For each planned note, at the middle of the frames it sounds, this prints how
far the pressing fingertip is from the middle of its fret cell on its string,
and how many notes miss the 6 mm gate.

    blender -b --factory-startup SCENE.blend --python audit_contact.py -- \
        PLACED.json [PREFIX]

PREFIX is for scenes holding several players, where objects are named
BASS_INST_neck, GUITAR_INST_fret5 and so on.
"""
import json
import sys

import bpy
from mathutils import Vector

GATE_MM = 6.0
FPS = 30
# fingers, low to high: 1 index .. 4 pinky, on the MPFB/MakeHuman naming
FINGERS = {1: "finger2", 2: "finger3", 3: "finger4", 4: "finger5"}


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    plan, pfx = args[0], (args[1] if len(args) > 1 else "")
    notes = json.load(open(plan))["notes"]
    rig = bpy.data.objects[f"{pfx}PLAYER.rig"]
    pb, Wm = rig.pose.bones, rig.matrix_world
    sc = bpy.context.scene

    frets, strings = {}, {}
    for o in bpy.data.objects:
        if o.name.startswith(f"{pfx}INST_fret") and o.name[len(pfx) + 9:].isdigit():
            frets[int(o.name[len(pfx) + 9:])] = o
        if o.name.startswith(f"{pfx}INST_string"):
            strings[int(o.name[len(pfx) + 11:])] = o
    if not frets or not strings:
        raise SystemExit(f"no {pfx}INST_fret*/{pfx}INST_string* in this scene")

    def centre(o):
        return sum((o.matrix_world @ v.co for v in o.data.vertices),
                   Vector()) / len(o.data.vertices)

    def cell(fret, string):
        """Press point on the live board: mid-cell along the live string."""
        hi = centre(frets[fret]) if fret in frets else centre(frets[max(frets)])
        x = (centre(frets[fret - 1]) + hi) / 2 if (fret - 1) in frets else hi
        so = strings[string]
        sv = [so.matrix_world @ v.co for v in so.data.vertices]
        ax = sv[-1] - sv[0]
        t = max(0.0, min(1.0, (x - sv[0]).dot(ax) / max(ax.length_squared, 1e-12)))
        return sv[0] + ax * t

    errs, worst = [], (0.0, None)
    for n in notes:
        if not n.get("fret") or not n.get("finger"):
            continue                      # open strings press nothing
        f0 = round(n["t_on"] * FPS) + 1
        f1 = max(f0 + 1, round(n["t_off"] * FPS))
        mid = min((f0 + f1) // 2, int(sc.frame_end))
        if f0 > int(sc.frame_end):
            break
        sc.frame_set(mid)
        tip = Wm @ pb[f"{FINGERS[n['finger']]}-3.L"].tail
        e = (tip - cell(n["fret"], n["string"])).length * 1000
        errs.append(e)
        if e > worst[0]:
            worst = (e, (mid, n["pitch"], n["string"], n["fret"]))
    if not errs:
        print("no fretted notes to audit")
        return
    errs_sorted = sorted(errs)
    n_ = len(errs_sorted)
    over = sum(1 for e in errs_sorted if e > GATE_MM)
    print(f"contact over {n_} pressed notes (live instrument): "
          f"median {errs_sorted[n_ // 2]:.1f}mm  "
          f"p90 {errs_sorted[int(n_ * 0.9)]:.1f}mm  worst {worst[0]:.1f}mm")
    print(f"worst at frame {worst[1][0]}, pitch {worst[1][1]}, "
          f"string {worst[1][2]} fret {worst[1][3]}")
    print(f"{over}/{n_} notes over the {GATE_MM:.0f}mm gate")


if __name__ == "__main__":
    main()
