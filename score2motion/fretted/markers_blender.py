"""A red square on the fret being played, with the finger number printed on it.

Driven by the SAME plan that drives the hands, so the marker is the claim and
the hand is the proof: if the fingertip lands anywhere but inside the red
square, the miss is visible immediately -- no log reading.

  red square    sits on the board in the fret cell being pressed, on the
                string's line, for exactly as long as the note sounds
  black digit   which finger the plan says presses (1 index .. 4 pinky)
  open string   the square sits over the nut and carries no digit
  carried slide the square PARKS on the start fret for the first half of the
                note, then travels at constant speed and lands on the target
                cell exactly as the note ends -- the digit rides with it

Requires the scene to follow NAMING_CONTRACT.md: INST_neck, INST_fretN,
INST_stringN. A carried slide = detected glide covering >= 0.8 of the interval
with a small same-string move (the performer's law; see slides.py).

  blender scene.blend -b --python markers_blender.py -- \
      IN.blend PLAN.json OUT.blend [SLIDES.json]
"""
import json
import os
import sys

import bpy
from mathutils import Matrix, Vector

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
if len(args) < 3:
    raise SystemExit(__doc__)
SRC, PLAN, OUT = args[0], args[1], args[2]
SLIDES = args[3] if len(args) > 3 else None
FPS = 30
RED = (0.90, 0.04, 0.04, 1.0)
CARRY_COVERED, CARRY_MAX_FRETS = 0.8, 4

bpy.ops.wm.open_mainfile(filepath=SRC)
sc = bpy.context.scene
bpy.context.view_layer.update()
notes = json.load(open(PLAN))["notes"]
neck = bpy.data.objects["INST_neck"]
N, Ni = neck.matrix_world, neck.matrix_world.inverted()

# which note-pairs are CARRIED slides -- same thresholds as the hand itself,
# so the dot glides exactly when the finger does
glide_next = {}
if SLIDES and os.path.exists(SLIDES):
    _idx = {round(n["t_on"], 3): i for i, n in enumerate(notes)}
    for s_ in json.load(open(SLIDES)).get("slides", []):
        i = _idx.get(round(s_["from_t"], 3))
        if i is None or i + 1 >= len(notes):
            continue
        a, b = notes[i], notes[i + 1]
        if (s_.get("covered", 0) >= CARRY_COVERED
                and a["string"] == b["string"]
                and 0 < abs(b["fret"] - a["fret"]) <= CARRY_MAX_FRETS):
            glide_next[i] = True
    print(f"marker glides through {len(glide_next)} carried slide(s)")

# the board, in its own frame: frets along x, strings along y, out of board z
fx = {}
for o in bpy.data.objects:
    if o.name.startswith("INST_fret") and o.name[9:].isdigit():
        v = [Ni @ (o.matrix_world @ p.co) for p in o.data.vertices]
        fx[int(o.name[9:])] = sum(q.x for q in v) / len(v)
sy = {}
for o in bpy.data.objects:
    if o.name.startswith("INST_string"):
        v = [Ni @ (o.matrix_world @ p.co) for p in o.data.vertices]
        sy[int(o.name[11:])] = sum(q.y for q in v) / len(v)
TOP = max(v.co.z for v in neck.data.vertices)
HIGH = max(k for k in fx if k > 0)
NUT_SIGN = 1 if fx[1] > fx[HIGH] else -1      # which way is toward the nut
gap = {s: min(abs(sy[s] - sy[t]) for t in sy if t != s) for s in sy}
print(f"board read: {len(fx)} frets, {len(sy)} strings, top at {TOP:.4f}")


def cell(fret, string):
    """The middle of the fret cell on that string, in the neck's own frame."""
    if fret <= 0:                              # open: over the nut
        x = fx[0] + NUT_SIGN * 0.012
        w = 0.018
    else:
        lo, hi = fx.get(fret - 1, fx[fret] + NUT_SIGN * 0.03), fx[fret]
        x, w = (lo + hi) / 2.0, abs(lo - hi) * 0.72
    return Vector((x, sy[string], TOP + 0.004)), w, gap[string] * 0.85


# one red square, teleported note to note; four digits sharing its ride
mat = bpy.data.materials.new("NOTE_RED")
mat.diffuse_color = RED            # the workbench renderer paints THIS
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = RED
bsdf.inputs["Emission Color"].default_value = RED
bsdf.inputs["Emission Strength"].default_value = 3.0
ink = bpy.data.materials.new("NOTE_INK")
ink.diffuse_color = (0.0, 0.0, 0.0, 1.0)
ink.use_nodes = True
ib = ink.node_tree.nodes["Principled BSDF"]
ib.inputs["Base Color"].default_value = (0.0, 0.0, 0.0, 1.0)
ib.inputs["Emission Color"].default_value = (0.0, 0.0, 0.0, 1.0)
ib.inputs["Emission Strength"].default_value = 0.0

bpy.ops.mesh.primitive_plane_add(size=1.0)
mark = bpy.context.active_object
mark.name = "NOTE_MARK"
mark.data.materials.append(mat)
mark.parent = neck
mark.matrix_parent_inverse = Matrix.Identity(4)

digits = {}
for f in (1, 2, 3, 4):
    curve = bpy.data.curves.new(f"DIGIT_{f}", type='FONT')
    curve.body = str(f)
    curve.align_x, curve.align_y = 'CENTER', 'CENTER'
    d = bpy.data.objects.new(f"DIGIT_{f}", curve)
    sc.collection.objects.link(d)
    d.data.materials.append(ink)
    d.parent = neck
    d.matrix_parent_inverse = Matrix.Identity(4)
    digits[f] = d


def key_all(obj, frame):
    obj.keyframe_insert("location", frame=frame)
    obj.keyframe_insert("scale", frame=frame)
    obj.keyframe_insert("hide_viewport", frame=frame)
    obj.keyframe_insert("hide_render", frame=frame)


def hide(obj, yes, frame):
    obj.hide_viewport = yes
    obj.hide_render = yes
    key_all(obj, frame)


# everything hidden until the first note
for o in [mark] + list(digits.values()):
    o.scale = (0.001, 0.001, 0.001)
    hide(o, True, 0)

GLIDE_LIN = []            # (frame whose key goes LINEAR, finger riding along)
shown = 0
for ni, n in enumerate(notes):
    f0 = round(n["t_on"] * FPS) + 1
    f1 = max(f0 + 2, round(n["t_off"] * FPS))
    c, w, h = cell(n["fret"], n["string"])
    mark.location = c
    mark.scale = (w, h, 1.0)
    hide(mark, False, f0)
    fing = n.get("finger") or 0
    glide = False
    if glide_next.get(ni):
        nxt = notes[ni + 1]
        nf0 = round(nxt["t_on"] * FPS) + 1
        if nf0 - f0 >= 3:                      # room for park + travel
            glide = True
            mid = max(f0 + 1, min(nf0 - 1, (f0 + nf0) // 2))
            key_all(mark, mid)                 # parked half, start fret
            c2, w2, h2 = cell(nxt["fret"], nxt["string"])
            mark.location = c2
            mark.scale = (w2, h2, 1.0)
            key_all(mark, nf0)                 # lands EXACTLY at note end
            GLIDE_LIN.append((mid, fing))
    if not glide:
        hide(mark, True, f1)                   # normal note: vanish at off
    for f, d in digits.items():
        if f == fing and n["fret"]:
            d.location = c + Vector((0, 0, 0.0015))
            d.scale = (h * 0.9, h * 0.9, h * 0.9)
            hide(d, False, f0)
            if glide:
                key_all(d, mid)                # digit rides the same glide
                d.location = c2 + Vector((0, 0, 0.0015))
                d.scale = (h2 * 0.9, h2 * 0.9, h2 * 0.9)
                key_all(d, nf0)
            else:
                hide(d, True, f1)
        elif not d.hide_render:
            hide(d, True, f0)
    shown += 1

# a marker must JUMP between notes -- EXCEPT through a carried slide, where
# the key opening the glide span goes LINEAR so the dot travels at constant
# speed into the target
for o in [mark] + list(digits.values()):
    if o.animation_data and o.animation_data.action:
        for fc in o.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'CONSTANT'
for mid, fing in GLIDE_LIN:
    for o in [mark] + ([digits[fing]] if fing in digits else []):
        if not (o.animation_data and o.animation_data.action):
            continue
        for fc in o.animation_data.action.fcurves:
            if fc.data_path in ("location", "scale"):
                for kp in fc.keyframe_points:
                    if abs(kp.co.x - mid) < 0.5:
                        kp.interpolation = 'LINEAR'
print(f"marked {shown} notes; {len(GLIDE_LIN)} glide(s) keyed LINEAR mid-note")
bpy.ops.wm.save_as_mainfile(filepath=OUT)
print(f"saved {OUT}")
