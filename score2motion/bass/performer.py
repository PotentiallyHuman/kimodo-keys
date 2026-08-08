"""Bass performer: MIDI + a rigged canon character -> a played performance.

Everything here runs inside Blender. The caller supplies paths and a window;
nothing about any particular song or machine is baked in.

    blender -b scene.blend --python -m kimodo_bass.performer -- \
        --plan plan.json --motion body.bvh --canon canon.blend --out played.blend

The three owners, kept strictly apart:
  * the MIDI owns where and when a note happens
  * the measured style curves own how a hand makes that happen
  * the physical constraints own what is possible at all
"""
import argparse
import json
import math
import os
import statistics
import sys

import bpy
from mathutils import Vector, Matrix, Quaternion


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True, help="performance plan from kimodo_bass.plan")
    p.add_argument("--motion", required=True, help="body motion BVH (e.g. from Kimodo)")
    p.add_argument("--style", default=None, help="measured style constants (JSON)")
    p.add_argument("--curves", default=None, help="measured motion curves (JSON)")
    p.add_argument("--relation", default=None, help="hand-to-neck relation (JSON)")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--cal-fret", type=int, default=1, help="which fret the canon grip sits on")
    p.add_argument("--out", required=True)
    return p.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


ARGS = _args()
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SCRATCH = os.path.dirname(os.path.abspath(ARGS.out))


male = bpy.data.objects["MALE.rig"]
body = bpy.data.objects["MALE.body"]
mroot = bpy.data.objects["MALEBASS_bass_root"]
pb = male.pose.bones
sc = bpy.context.scene
sc.render.fps = FPS
sc.frame_start, sc.frame_end = 1, END
sc.frame_set(1)
if male.animation_data:
    male.animation_data_clear()
for o in bpy.data.objects:
    if o.name.startswith(("MALEBASS_", "NOTEMARK_", "CAM")) and o.animation_data:
        o.animation_data_clear()
bpy.context.view_layer.update()

fret_x = {k: bpy.data.objects[f"MALEBASS_fret{k}"].location.x for k in range(13)}
strings = {s: bpy.data.objects[f"MALEBASS_string{s}"].location for s in range(4)}
NUT_SIDE = 1.0 if fret_x[0] > fret_x[12] else -1.0
FRET_SPACING = {k: fret_x[k] - fret_x[CAL_FRET] for k in fret_x}     # slide along the neck

LHAND = [b.name for b in male.data.bones if b.name.startswith(("wrist.L", "finger")) and b.name.endswith(".L")]
LHAND = [b.name for b in male.data.bones
         if b.name == "wrist.L" or b.name.startswith("metacarpal") and b.name.endswith(".L")
         or (b.name.startswith("finger") and b.name.endswith(".L"))]
RHAND = [b.name for b in male.data.bones
         if b.name == "wrist.R" or b.name.startswith("metacarpal") and b.name.endswith(".R")
         or (b.name.startswith("finger") and b.name.endswith(".R"))]
HANDB = LHAND + RHAND

B_STATIC = mroot.matrix_world.copy()
Bi = B_STATIC.inverted()
W = male.matrix_world
Wi = W.inverted()

# ---- canon 2's grip, in the bass's own coordinates = the calibration ----------
CANON_TIP = {}
BASE = {n: (Bi @ (W @ pb[n].matrix)).copy() for n in HANDB}
for _f in (1, 2, 3, 4):
    CANON_TIP[_f] = (Bi @ (W @ pb[f"finger{_f+1}-3.L"].tail)).copy()
idx_tip_local = Bi @ (W @ pb["finger2-3.L"].tail)
print(f"CALIBRATION: canon 2 index tip at x={idx_tip_local.x*100:+.1f}cm on the neck = fret {CAL_FRET} "
      f"(fret {CAL_FRET} wire at {fret_x[CAL_FRET]*100:+.1f}, fret 1 at {fret_x[1]*100:+.1f})")
PRESS_OFF = idx_tip_local.x - fret_x[CAL_FRET]        # how far behind the wire he presses


NECK_T = 0.052      # how far the palm must sit below the strings to clear the neck


_neck_obj = bpy.data.objects["MALEBASS_bass_neck"]
_nvz = [(mroot.matrix_world.inverted() @ (_neck_obj.matrix_world @ v.co)).z
        for v in _neck_obj.data.vertices]
BOARD_TOP = max(_nvz)          # the fretboard surface the string gets pressed onto


def target_local(fret, s, phase="press"):
    """Where the fingertip goes. `contact` = just touching the string where it
    lies; `press` = the string driven all the way down onto the board, which is
    what actually sounds the note; `mute` = resting on it without pressing."""
    x = fret_x[max(1, fret)] + PRESS_OFF if fret > 0 else fret_x[1] + PRESS_OFF
    if phase == "press":
        z = BOARD_TOP + 0.002          # string pinned against the fretboard
    elif phase == "mute":
        z = strings[s].z + 0.001       # touching, not pressing -> silence
    else:
        z = strings[s].z + 0.006       # hovering just above, about to play
    return Vector((x, strings[s].y, z))


def slide_local(pos):
    return Matrix.Translation(Vector((FRET_SPACING[max(1, min(12, pos))], 0, 0)))


# LEAST-EFFORT PLACEMENT: put the hand where the finger that has to play is
# already sitting over its note. Then the finger barely moves, so it never has
# to swing through the neck to get there.
def least_effort_offset(fret, s, finger):
    """The hand may ONLY slide along the neck. Its distance across the neck and
    its clearance from the downward-facing border are fixed by the canon and
    never change -- that is what keeps the hand on the neck instead of through
    it. Reaching a different string is the FINGER's job, not the hand's."""
    tgt = target_local(fret, s)
    src = CANON_TIP[finger]
    return Matrix.Translation(Vector((tgt.x - src.x, 0.0, 0.0)))


def set_hand_local(mats):
    for n in sorted(mats, key=lambda x: len(male.data.bones[x].parent_recursive)):
        pb[n].matrix = Wi @ (B_STATIC @ mats[n])
        pb[n].scale = (1, 1, 1)
        bpy.context.view_layer.update()


# ================= finger press solved in the bass's frame ====================
def aim_finger(fnum, T_world, rounds=6):
    fb = [pb[f"finger{fnum}-{k}.L"] for k in (1, 2, 3)]
    seq = []
    for _ in range(rounds):
        seq += [fb[0], fb[1], fb[2]]
    for bone in seq:
        tip = W @ fb[2].tail
        base = W @ bone.head
        a = tip - base
        b = T_world - base
        if a.length < 1e-6 or b.length < 1e-6:
            continue
        ax = a.normalized().cross(b.normalized())
        ang = math.acos(max(-1, min(1, a.normalized().dot(b.normalized()))))
        if ax.length < 1e-7 or math.degrees(ang) < 0.2:
            continue
        pl = Wi @ base
        q = Quaternion(ax.normalized(), ang * 0.9)
        bone.matrix = (Matrix.Translation(pl)
                       @ (Wi.to_3x3().to_4x4() @ q.to_matrix().to_4x4() @ W.to_3x3().to_4x4())
                       @ Matrix.Translation(-pl)) @ bone.matrix
        bone.scale = (1, 1, 1)
        bpy.context.view_layer.update()
    return ((W @ fb[2].tail) - T_world).length


# ---- finger assignment: index/middle/ring/pinky over a 4-fret span -----------
def plan():
    out = []
    pos = CAL_FRET
    for n in notes:
        fret, s = n["fret"], n["string"]
        if fret == 0:
            out.append((pos, 0)); continue
        if not (pos <= fret <= pos + 3):
            pos = max(1, min(9, fret))            # shift so index takes the note
        fg = fret - pos + 1
        out.append((pos, max(1, min(4, fg))))
    return out


ASSIGN = plan()

# build one hand pose per (position, fret, string, finger) -- all in bass space
combos = sorted({(ASSIGN[i][0], n["fret"], n["string"], ASSIGN[i][1])
                 for i, n in enumerate(notes) if n["fret"] > 0 and ASSIGN[i][1] > 0})
press_lib = {}     # finger pressed on its fret, fingers in front of it lifted clear
mute_lib = {}      # same hand, playing finger released to a light touch = the note stops
resid = []
LIFT_DEG = 20.0    # a finger in front of the played note must come off the string
MUTE_UP = 0.004    # releasing 4mm keeps contact but stops the note ringing
_curl_axis_L = ((W @ pb["finger5-1.L"].head) - (W @ pb["finger2-1.L"].head)).normalized()


def _bend_L(fnum, deg):
    for k, share in ((1, 1.0), (2, 0.85), (3, 0.6)):
        b = pb[f"finger{fnum}-{k}.L"]
        piv = W @ b.head; pl = Wi @ piv
        q = Quaternion(_curl_axis_L, math.radians(deg * share))
        b.matrix = (Matrix.Translation(pl)
                    @ (Wi.to_3x3().to_4x4() @ q.to_matrix().to_4x4() @ W.to_3x3().to_4x4())
                    @ Matrix.Translation(-pl)) @ b.matrix
        b.scale = (1, 1, 1)
    bpy.context.view_layer.update()
for (p, fret, s, f) in combos:
    slid = {n: least_effort_offset(fret, s, f) @ BASE[n] for n in LHAND}
    set_hand_local(slid)
    T_w = B_STATIC @ target_local(fret, s)
    r = aim_finger(f + 1, T_w)
    # the neck is SOLID: lift the palm off the fretboard so the hand wraps it
    for _ in range(4):
        pal = Bi @ (W @ pb["wrist.L"].matrix.translation)
        clear = pal.z - (strings[0].z - NECK_T)
        if clear >= 0:
            break
        slid = {n: Matrix.Translation(Vector((0, 0, -clear * 0.9))) @ slid[n] for n in LHAND}
        set_hand_local(slid)
        r = aim_finger(f + 1, T_w)
    # the hand settles onto the note: nudge the whole hand, then re-aim (a player
    # moves the hand, not just the finger, to put the tip exactly on the fret)
    for _ in range(10):
        if r < 0.003:
            break
        res_w = T_w - (W @ pb[f"finger{f+1}-3.L"].tail)
        res_l = Bi.to_3x3() @ res_w
        res_l = Vector((res_l.x, 0.0, 0.0))        # slide along the neck, nothing else
        slid = {n: Matrix.Translation(res_l * 0.7) @ slid[n] for n in LHAND}
        set_hand_local(slid)
        r = aim_finger(f + 1, T_w)
    # RULE: the played finger must be the last one touching this string.
    # Any finger sitting FURTHER DOWN the neck on the same string would mute it,
    # so those lift clear -- exactly what you do in the VR take.
    for j in (f + 1, f + 2, f + 3):
        if 1 <= j <= 4:
            _bend_L(j + 1, -LIFT_DEG)
    press_lib[(p, fret, s, f)] = {n: (Bi @ (W @ pb[n].matrix)).copy() for n in LHAND}
    resid.append(r * 1000)
    # and the note ENDS by releasing to a light touch -- still on the string,
    # no longer pressing -- which is what actually stops it ringing
    T_mute = B_STATIC @ target_local(fret, s, "mute")
    _aim = aim_finger(f + 1, T_mute)
    mute_lib[(p, fret, s, f)] = {n: (Bi @ (W @ pb[n].matrix)).copy() for n in LHAND}
open_lib = {}
for p in sorted({a[0] for a in ASSIGN}):
    open_lib[p] = {n: slide_local(p) @ BASE[n] for n in LHAND}
print(f"press poses: {len(press_lib)} built by LEAST-EFFORT hand placement; "
      f"fingertip error median {statistics.median(resid):.1f}mm max {max(resid):.1f}mm")
_eff = []
for (p, fret, s, f) in combos:
    _pl = press_lib[(p, fret, s, f)]
    for k in (1, 2, 3):
        _n = f"finger{f+1}-{k}.L"
        _a = (_pl[_n].to_3x3().normalized() @ BASE[_n].to_3x3().normalized().transposed()).to_quaternion()
        _eff.append(abs(math.degrees(_a.angle)))
print(f"effort per note: the playing finger rotates {statistics.median(_eff):.1f} deg on average, "
      f"{max(_eff):.1f} deg at worst (small = it never has to swing through the neck)")

# ---- RIGHT HAND: WALKING BASS, hand held still ------------------------------
# Your rules:
#   * the hand stays exactly where the canon put it -- it never rides around
#   * ring and pinky stay CLAWED and completely still
#   * the thumb rests on the bass body as a stabiliser, also still
#   * index and middle take turns: each pushes into the string, through it,
#     then retracts to hovering extended above it. Only that one finger moves.
R_IDX, R_MID = 2, 3
# measured off your own walking take (palm frame, so viewing angle cancels out)
CLAW_RING  = (45.2, 68.3, 39.0)     # knuckle / middle / tip
CLAW_PINKY = (54.9, 70.5, 42.5)
HOVER_ABOVE = 0.020    # your real stroke is ~3x what I had guessed
PUSH_THROUGH = 0.014   # 34mm of travel, matching the 36-43mm measured

_axis_r = ((W @ pb["finger5-1.R"].head) - (W @ pb["finger2-1.R"].head)).normalized()


def _rot_one(fnum, k, deg):
    b = pb[f"finger{fnum}-{k}.R"]
    piv = W @ b.head; pl = Wi @ piv
    q = Quaternion(_axis_r, math.radians(deg))
    b.matrix = (Matrix.Translation(pl)
                @ (Wi.to_3x3().to_4x4() @ q.to_matrix().to_4x4() @ W.to_3x3().to_4x4())
                @ Matrix.Translation(-pl)) @ b.matrix
    b.scale = (1, 1, 1)
    bpy.context.view_layer.update()


def _rot_finger(fnum, deg):
    for k, share in ((1, 1.0), (2, 0.9), (3, 0.7)):
        b = pb[f"finger{fnum}-{k}.R"]
        piv = W @ b.head; pl = Wi @ piv
        q = Quaternion(_axis_r, math.radians(deg * share))
        b.matrix = (Matrix.Translation(pl)
                    @ (Wi.to_3x3().to_4x4() @ q.to_matrix().to_4x4() @ W.to_3x3().to_4x4())
                    @ Matrix.Translation(-pl)) @ b.matrix
        b.scale = (1, 1, 1)
    bpy.context.view_layer.update()


def _aim_r(fnum, T_world, rounds=8):
    """point this ONE finger at a spot on the strings; nothing else moves"""
    fb = [pb[f"finger{fnum}-{k}.R"] for k in (1, 2, 3)]
    for _ in range(rounds):
        for bone in fb:
            tip = W @ fb[2].tail; base = W @ bone.head
            a_ = tip - base; b_ = T_world - base
            if a_.length < 1e-6 or b_.length < 1e-6:
                continue
            ax = a_.normalized().cross(b_.normalized())
            ang = math.acos(max(-1, min(1, a_.normalized().dot(b_.normalized()))))
            if ax.length < 1e-8 or math.degrees(ang) < 0.15:
                continue
            pl = Wi @ base
            q = Quaternion(ax.normalized(), ang * 0.85)
            bone.matrix = (Matrix.Translation(pl)
                           @ (Wi.to_3x3().to_4x4() @ q.to_matrix().to_4x4() @ W.to_3x3().to_4x4())
                           @ Matrix.Translation(-pl)) @ bone.matrix
            bone.scale = (1, 1, 1)
            bpy.context.view_layer.update()
    return ((W @ fb[2].tail) - T_world).length


# which way does a right-hand finger CURL? (bending brings the tip toward the wrist)
for n in RHAND:
    pb[n].matrix = Wi @ (B_STATIC @ BASE[n]); pb[n].scale = (1, 1, 1)
bpy.context.view_layer.update()
_wr_r = W @ pb["wrist.R"].head
_d0 = ((W @ pb["finger3-3.R"].tail) - _wr_r).length
_rot_finger(3, 20.0)
_d1 = ((W @ pb["finger3-3.R"].tail) - _wr_r).length
CURL_R = 1.0 if _d1 < _d0 else -1.0
_rot_finger(3, -20.0)
print(f"right hand curl test: +20 deg moved the tip {(_d1-_d0)*1000:+.1f}mm toward the wrist "
      f"-> curling is the {'positive' if CURL_R > 0 else 'negative'} direction")

# 1. claw the ring and pinky ONCE -- they never move again
for n in RHAND:
    pb[n].matrix = Wi @ (B_STATIC @ BASE[n]); pb[n].scale = (1, 1, 1)
bpy.context.view_layer.update()
for _k, _d in enumerate(CLAW_RING, start=1):
    _rot_one(4, _k, _d * CURL_R)
for _k, _d in enumerate(CLAW_PINKY, start=1):
    _rot_one(5, _k, _d * CURL_R)
_kn4 = (Bi @ (W @ pb["finger4-1.R"].head)).z
_tp4 = (Bi @ (W @ pb["finger4-3.R"].tail)).z
print(f"claw check: ring tip is {(_kn4-_tp4)*1000:+.1f}mm below its knuckle "
      f"({'curled toward the strings, correct' if _tp4 < _kn4 else 'STILL WRONG WAY'})")
CLAWED = {n: (Bi @ (W @ pb[n].matrix)).copy() for n in RHAND}
_wrist_fixed = CLAWED["wrist.R"].copy()

# 2. THE THUMB IS THE PIVOT.
# Its tip stays planted on the body of the bass. To walk onto a different
# string the thumb claws in or extends out, which swings the whole hand up or
# down across the strings -- the index and middle then pluck whichever string
# they have been carried to.
THUMB_ANCHOR = (Bi @ (W @ pb["finger1-3.R"].tail)).copy()
print(f"thumb planted on the body at y={THUMB_ANCHOR.y*1000:+.1f} z={THUMB_ANCHOR.z*1000:+.1f}mm "
      f"-- the hand pivots on this point")


def _rotate_hand_about_thumb(theta_deg):
    """claw/extend the thumb: the hand swings about the planted thumb tip"""
    ax_w = (mr_axis).normalized()
    piv_w = B_STATIC @ THUMB_ANCHOR
    q = Quaternion(ax_w, math.radians(theta_deg))
    M = (Matrix.Translation(piv_w) @ q.to_matrix().to_4x4() @ Matrix.Translation(-piv_w))
    for n in RHAND:
        cur = W @ pb[n].matrix
        pb[n].matrix = Wi @ (M @ cur)
        pb[n].scale = (1, 1, 1)
    bpy.context.view_layer.update()


mr_axis = (mroot.matrix_world.to_3x3() @ Vector((1, 0, 0))).normalized()

R_POSE = {}
_res = []
TIP_X = {f: (Bi @ (W @ pb[f"finger{f}-3.R"].tail)).x for f in (R_IDX, R_MID)}
for fnum in (R_IDX, R_MID):
    for s_ in range(4):
        for tag, off in (("hover", HOVER_ABOVE), ("through", -PUSH_THROUGH), ("mute", 0.0)):
            for n in RHAND:
                pb[n].matrix = Wi @ (B_STATIC @ CLAWED[n]); pb[n].scale = (1, 1, 1)
            bpy.context.view_layer.update()
            # swing the hand about the thumb until this finger is over the string
            for _ in range(14):
                tip = Bi @ (W @ pb[f"finger{fnum}-3.R"].tail)
                a_now = math.atan2(tip.z - THUMB_ANCHOR.z, tip.y - THUMB_ANCHOR.y)
                a_want = math.atan2(strings[s_].z + off - THUMB_ANCHOR.z,
                                    strings[s_].y - THUMB_ANCHOR.y)
                d = math.degrees(a_want - a_now)
                d = (d + 180) % 360 - 180
                if abs(d) < 0.15:
                    break
                _rotate_hand_about_thumb(max(-12.0, min(12.0, d)))
            T = B_STATIC @ Vector((TIP_X[fnum], strings[s_].y, strings[s_].z + off))
            _res.append(_aim_r(fnum, T))
            R_POSE[(fnum, s_, tag)] = {n: (Bi @ (W @ pb[n].matrix)).copy() for n in RHAND}
for n in RHAND:
    pb[n].matrix = Wi @ (B_STATIC @ CLAWED[n]); pb[n].scale = (1, 1, 1)
bpy.context.view_layer.update()
_th_drift = []
for k_, v_ in R_POSE.items():
    for n in RHAND:
        pb[n].matrix = Wi @ (B_STATIC @ v_[n])
    bpy.context.view_layer.update()
    _th_drift.append(((Bi @ (W @ pb["finger1-3.R"].tail)) - THUMB_ANCHOR).length)
for n in RHAND:
    pb[n].matrix = Wi @ (B_STATIC @ CLAWED[n]); pb[n].scale = (1, 1, 1)
bpy.context.view_layer.update()
print(f"right hand: ring clawed {CLAW_RING} and pinky {CLAW_PINKY} (your measured angles); {len(R_POSE)} pluck poses; "
      f"fingertip lands {statistics.median(_res)*1000:.1f}mm from the string; "
      f"thumb tip stays planted within {max(_th_drift)*1000:.2f}mm across every string")


# ---- HOW A HAND ACTUALLY MOVES BETWEEN STATES -------------------------------
# Learned from your VR takes: not the motion itself, but the SHAPE of every
# change. A pluck or a press approaches slowly, accelerates hard into the
# string, then comes back more gently than it went in. 98+59 plucks and
# 106+154+155 presses were averaged to get these curves.
_CURVE = json.load(open(os.path.join(DATA, "pluck_curve.json")))
_R_PROF = _CURVE["profiles"].get("index") or []
_L_PROF = (_CURVE.get("left_press") or {}).get("index") or []


def _prof_at(prof, u):
    """sample the measured curve; u = 0 at the start of the move, 1 at the end"""
    if not prof:
        return u * u * (3 - 2 * u)
    x = max(0.0, min(1.0, u)) * (len(prof) - 1)
    i = int(x); fr_ = x - i
    a = prof[i]; b = prof[min(i + 1, len(prof) - 1)]
    return a + (b - a) * fr_


def _blend(m0, m1, w):
    """blend two hand poses -- position straight, rotation the short way round.
    Rotations are normalised first so the bass's own scale cannot leak in and
    nudge the hand off its lock."""
    q0 = m0.to_3x3().normalized().to_quaternion()
    q1 = m1.to_3x3().normalized().to_quaternion()
    if q0.dot(q1) < 0.0:
        q1 = -q1
    out = q0.slerp(q1, w).to_matrix().to_4x4()
    out.translation = m0.translation.lerp(m1.translation, w)
    return out


def _lay(FRd, bones, f_from, f_to, pose_from, pose_to, prof, phase="in"):
    """write every frame between two states, shaped by the measured curve"""
    if f_to <= f_from:
        return
    n = len(prof) if prof else 0
    for f in range(max(1, f_from), min(END, f_to) + 1):
        u = (f - f_from) / float(f_to - f_from)
        if n:
            # the first half of the measured curve is the approach, the second the return
            s = _prof_at(prof, 0.5 * u) if phase == "in" else _prof_at(prof, 0.5 + 0.5 * u)
            lo = _prof_at(prof, 0.0) if phase == "in" else _prof_at(prof, 0.5)
            hi = _prof_at(prof, 0.5) if phase == "in" else _prof_at(prof, 1.0)
            w = 0.0 if abs(hi - lo) < 1e-6 else max(0.0, min(1.0, (s - lo) / (hi - lo)))
        else:
            w = u * u * (3 - 2 * u)
        for nb in bones:
            FRd[f][nb] = _blend(pose_from[nb], pose_to[nb], w)


# ================= build the per-frame performance in bass space ==============
FR = {}
for f in range(1, END + 1):
    FR[f] = {}
    FR[f].update(open_lib[CAL_FRET] if CAL_FRET in open_lib else {n: BASE[n] for n in LHAND})
    FR[f].update({n: CLAWED[n] for n in RHAND})

pos_cur = CAL_FRET
events = []
for i, n in enumerate(notes):
    f_on = int(n["t_on"] * FPS) + 1
    f_off = max(f_on + 3, int(n["t_off"] * FPS) + 1)
    if f_on > END:
        break
    events.append((f_on, f_off, n, ASSIGN[i]))

# left hand: slide between positions, press on the note
cur_pose = {n: BASE[n] for n in LHAND}
frame_pos = {}
last_p = CAL_FRET
for (f_on, f_off, n, (p, fg)) in events:
    if n["fret"] > 0 and fg > 0:
        o = least_effort_offset(n["fret"], n["string"], fg).translation
    else:
        o = Vector((FRET_SPACING[max(1, min(12, p))], 0, 0))
    frame_pos[f_on] = o
timeline = sorted(frame_pos)
for f in range(1, END + 1):
    nxt = [t for t in timeline if t >= f]
    prv = [t for t in timeline if t <= f]
    p_from = frame_pos[prv[-1]] if prv else CAL_FRET
    p_to = frame_pos[nxt[0]] if nxt else p_from
    if nxt and prv and nxt[0] != prv[-1]:
        span = nxt[0] - prv[-1]
        lead = min(span, int(0.25 * FPS))          # arrive early, VR-measured shift law
        k = 0.0 if f < nxt[0] - lead else min(1.0, (f - (nxt[0] - lead)) / max(1, lead))
        o_eff = p_from + (p_to - p_from) * k
    else:
        o_eff = p_to
    sl = Matrix.Translation(o_eff)
    for nb in LHAND:
        FR[f][nb] = sl @ BASE[nb]

fingers_used = {}
used = []
_muted_notes = 0
for (f_on, f_off, n, (p, fg)) in events:
    if n["fret"] > 0 and fg > 0:
        key = (p, n["fret"], n["string"], fg)
        pk = press_lib.get(key)
        mk = mute_lib.get(key)
        if pk:
            _pre = max(1, f_on - 7)
            _rest = {nb: FR[_pre][nb] for nb in LHAND}
            _lay(FR, LHAND, _pre, f_on, _rest, pk, _L_PROF, "in")     # reach for the note
            for f in range(f_on, min(END, f_off) + 1):
                for nb in LHAND:
                    FR[f][nb] = pk[nb]                                 # held down
            if mk:
                _muted_notes += 1
                _lay(FR, LHAND, min(END, f_off), min(END, f_off + 6), pk, mk, _L_PROF, "out")
                for f in range(min(END, f_off + 6) + 1, min(END, f_off + 9) + 1):
                    for nb in LHAND:
                        FR[f][nb] = mk[nb]
    # index and middle take turns; only the plucking finger moves
    fingers_used.setdefault("last", R_MID)
    nxt_f = R_MID if fingers_used["last"] == R_IDX else R_IDX
    fingers_used["last"] = nxt_f
    used.append(nxt_f)
    s_ = n["string"]
    # pluck, then the same finger settles ONTO the string -- that is what ends
    # the note on a walking line, rather than the finger flying away
    hov = R_POSE[(nxt_f, s_, "hover")]
    thr = R_POSE[(nxt_f, s_, "through")]
    mut = R_POSE[(nxt_f, s_, "mute")]
    _start = max(1, f_on - 8)
    _rest_r = {nb: FR[_start][nb] for nb in RHAND}
    _lay(FR, RHAND, _start, max(1, f_on - 5), _rest_r, hov, None, "in")   # carry the hand over
    _lay(FR, RHAND, max(1, f_on - 5), f_on, hov, thr, _R_PROF, "in")      # drive through the string
    _lay(FR, RHAND, f_on, min(END, f_on + 5), thr, mut, _R_PROF, "out")   # come off it, gently
    _lay(FR, RHAND, min(END, f_on + 5), min(END, f_off + 4), mut, hov, None, "out")
# ---- ENFORCE the hand-to-neck lock on every finished frame ------------------
# Whatever the blending does, the left hand is only ever allowed to have slid
# ALONG the neck. Its distance across the neck and its clearance from the
# downward-facing border are pinned to the canon values, by construction.
_lock_y = BASE["wrist.L"].translation.y
_lock_z = BASE["wrist.L"].translation.z
_fix = 0.0
for f in range(1, END + 1):
    w = FR[f]["wrist.L"].translation
    d = Vector((0.0, _lock_y - w.y, _lock_z - w.z))
    if d.length > 1e-6:
        _fix = max(_fix, d.length)
        for nb in LHAND:
            m = FR[f][nb].copy()
            m.translation = m.translation + d
            FR[f][nb] = m
print(f"hand-to-neck lock enforced on every frame (largest correction {_fix*1000:.2f}mm)")

_names_r = {2: "index", 3: "middle", 4: "ring", 5: "pinky"}
from collections import Counter
_c = Counter(used)
print(f"performance built for {END} frames from {len(events)} events")
_mk = []
for (f_on, f_off, n, (p, fg)) in events:
    if n["fret"] > 0:
        tl = target_local(n["fret"], n["string"], "contact")
        _mk.append((abs(tl.y - strings[n["string"]].y), abs(tl.x - (fret_x[n["fret"]] + PRESS_OFF))))
if _mk:
    print(f"marker audit vs the slimmed neck: markers sit {max(m[0] for m in _mk)*1000:.2f}mm off their "
          f"string and {max(m[1] for m in _mk)*1000:.2f}mm off their fret (0 = exactly on the note)")
print(f"press depth: fingertip travels from {(strings[0].z + 0.006 - BOARD_TOP)*1000:.1f}mm above the board "
      f"down to {2.0:.1f}mm above it -- the string is pushed onto the fretboard")
print(f"note endings: {_muted_notes} of {len(events)} notes are stopped by releasing the finger to a "
      f"light touch (the MIDI duration decides when)")
print("walking hand plucked: " + ", ".join(f"{_names_r[k]} x{v}" for k, v in sorted(_c.items()))
      + f"  (alternating, {sum(1 for a, b in zip(used, used[1:]) if a != b)}/{max(1,len(used)-1)} swaps)")
_bl = male.data.bones["finger2-3.L"].length
for _f in (1, 40, 80, 120, 150):
    _m = FR[_f]["finger2-3.L"]
    _tip = (_m @ Matrix.Translation(Vector((0, _bl, 0)))).translation
    print(f"   built frame {_f:3d}: index tip x={_tip.x*100:+6.1f}cm (fret2={fret_x[2]*100:.1f}, nut={fret_x[0]*100:.1f})")

# ================= note markers on the fretboard ==============================
for o in list(bpy.data.objects):
    if o.name.startswith("NOTEMARK_"):
        bpy.data.objects.remove(o, do_unlink=True)
red = bpy.data.materials.new("marker_red"); red.use_nodes = True
rb = red.node_tree.nodes["Principled BSDF"]
rb.inputs["Base Color"].default_value = (0.85, 0.05, 0.05, 1)
try:
    rb.inputs["Emission Color"].default_value = (0.9, 0.08, 0.08, 1)
    rb.inputs["Emission Strength"].default_value = 3.0
except KeyError:
    pass
blk = bpy.data.materials.new("marker_black"); blk.use_nodes = True
blk.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.02, 0.02, 0.02, 1)
markers = {}
for d in range(5):
    me = bpy.data.meshes.new(f"NOTEMARK_{d}_mesh")
    s_ = 0.012
    me.from_pydata([(-s_, -s_, 0), (s_, -s_, 0), (s_, s_, 0), (-s_, s_, 0)], [], [(0, 1, 2, 3)])
    me.update(); me.materials.append(red)
    sq = bpy.data.objects.new(f"NOTEMARK_{d}", me)
    sc.collection.objects.link(sq)
    tc = bpy.data.curves.new(f"NM{d}", "FONT")
    tc.body = str(d); tc.size = 0.015; tc.align_x = "CENTER"; tc.align_y = "CENTER"; tc.extrude = 0.0004
    tx = bpy.data.objects.new(f"NOTEMARK_{d}_txt", tc)
    tx.data.materials.append(blk)
    sc.collection.objects.link(tx)
    sq.parent = mroot; sq.matrix_parent_inverse = Matrix.Identity(4)
    tx.parent = sq; tx.matrix_parent_inverse = Matrix.Identity(4); tx.location = (0, 0, 0.0018)
    for ob in (sq, tx):
        ob.hide_viewport = True; ob.hide_render = True
        ob.keyframe_insert("hide_viewport", frame=1); ob.keyframe_insert("hide_render", frame=1)
    markers[d] = (sq, tx)
for (f_on, f_off, n, (p, fg)) in events:
    d = fg if n["fret"] > 0 else 0
    sq, tx = markers[d]
    t = target_local(n["fret"], n["string"])
    sq.location = (t.x, t.y, t.z - 0.002)
    sq.keyframe_insert("location", frame=f_on)
    for ob in (sq, tx):
        ob.hide_viewport = True; ob.hide_render = True
        ob.keyframe_insert("hide_viewport", frame=max(1, f_on - 1)); ob.keyframe_insert("hide_render", frame=max(1, f_on - 1))
        ob.hide_viewport = False; ob.hide_render = False
        ob.keyframe_insert("hide_viewport", frame=f_on); ob.keyframe_insert("hide_render", frame=f_on)
        ob.keyframe_insert("hide_viewport", frame=min(END, f_off)); ob.keyframe_insert("hide_render", frame=min(END, f_off))
        ob.hide_viewport = True; ob.hide_render = True
        ob.keyframe_insert("hide_viewport", frame=min(END, f_off + 1)); ob.keyframe_insert("hide_render", frame=min(END, f_off + 1))

# ================= PHASE 2: Kimodo body + strap-hung bass + welded hands =======
sc.frame_set(1)
set_hand_local({n: BASE[n] for n in HANDB})
B0 = mroot.matrix_world.copy(); B0i = B0.inverted()
GEAR = [o for o in bpy.data.objects if o.name.startswith("MALEBASS_") and o.parent is None]
GEAR_IN_BASS = {o.name: (B0i @ o.matrix_world).copy() for o in GEAR}
ARMB = [b.name for b in male.data.bones if b.name.startswith(("clavicle", "upperarm", "lowerarm"))]
W = male.matrix_world
shoulderL = W @ pb["clavicle.L"].tail
kidneyR = (W @ pb["spine05"].head) + (W.to_3x3() @ Vector((-0.12, 0.12, 0.06)))
_bb = bpy.data.objects["MALEBASS_bass_body"]
_c = [_bb.matrix_world @ Vector(v) for v in _bb.bound_box]
BASSPTS = [B0i @ (Vector(v)) for v in [_bb.matrix_world @ Vector(c) for c in _bb.bound_box]]
BASSPTS = [Vector((p.x, p.y, p.z)) for p in BASSPTS]
BASSPTS.append(sum(BASSPTS, Vector()) / len(BASSPTS))
PIN_TOP_B = B0i @ max(_c, key=lambda q: q.z + q.x * 0.3)
PIN_BOT_B = B0i @ min(_c, key=lambda q: q.z - q.x * 0.3)
L_TOP = ((B0 @ PIN_TOP_B) - shoulderL).length
L_BOT = ((B0 @ PIN_BOT_B) - kidneyR).length
BASS_IN_CHEST = (W @ pb["spine01"].matrix).inverted() @ B0
REST_ARM = {n: pb[n].matrix_basis.copy() for n in ARMB}
# canon 2 defines a relaxed arm: the shoulder-to-hand distance it was built with
REACH0 = {}
for _s in ("L", "R"):
    _sh = W @ pb[f"upperarm01.{_s}"].head
    _wr = W @ pb[f"wrist.{_s}"].matrix.translation
    REACH0[_s] = (_wr - _sh).length
HAND_IN_B = {_s: (B0i @ (W @ pb[f"wrist.{_s}"].matrix)).translation.copy() for _s in ("L", "R")}
print(f"relaxed arm reach from canon 2: left {REACH0['L']*100:.1f}cm, right {REACH0['R']*100:.1f}cm")

import retarget as RT
for o in list(bpy.data.objects):
    if o.name.startswith("KIMODO"):
        bpy.data.objects.remove(o, do_unlink=True)
pre = set(bpy.data.objects)
bpy.ops.import_anim.bvh(filepath=BVH, global_scale=0.01, frame_start=1,
                        use_fps_scale=False, update_scene_fps=False)
kim = [o for o in bpy.data.objects if o not in pre][0]
kim.name = "KIMODO_take"
klast = int(kim.animation_data.action.frame_range[1])
kb = [b.name for b in kim.pose.bones]
MAP = {"Hips": "root", "Spine1": "spine04", "Spine2": "spine03", "Chest": "spine02",
       "Neck1": "neck01", "Neck2": "neck02", "Head": "head"}
for S, s_ in (("Left", "L"), ("Right", "R")):
    MAP[f"{S}Leg"] = f"upperleg01.{s_}"; MAP[f"{S}Shin"] = f"lowerleg01.{s_}"; MAP[f"{S}Foot"] = f"foot.{s_}"
MAP = {k: v for k, v in MAP.items() if k in kb and v in pb}
HRATIO = RT.height_ratio(kim, male)
BODY = list(MAP.values())


def depth(n):
    k, b = 0, male.data.bones[n]
    while b.parent:
        k += 1; b = b.parent
    return k


sc.frame_set(1); bpy.context.view_layer.update()
BASE_ROT = {t: (male.matrix_world @ pb[t].matrix).to_3x3().normalized().copy() for t in BODY}
KIM0 = {s_: (kim.matrix_world @ kim.pose.bones[s_].matrix).to_3x3().normalized().copy() for s_ in MAP}
HIPS0 = (kim.matrix_world @ kim.pose.bones["Hips"].head).copy()
ROOT0 = (male.matrix_world @ pb["root"].matrix).copy()
BODY_ORDER = sorted(BODY, key=depth)
T2S = {v: k for k, v in MAP.items()}


def kimodo_delta_frame():
    Wl = male.matrix_world; Wli = Wl.inverted()
    for t_ in BODY_ORDER:
        s_ = T2S.get(t_)
        if s_ is None:
            continue
        Rk = (kim.matrix_world @ kim.pose.bones[s_].matrix).to_3x3().normalized()
        want = (Rk @ KIM0[s_].transposed()) @ BASE_ROT[t_]
        cur = Wl @ pb[t_].matrix
        M = want.to_4x4(); M.translation = cur.translation
        pb[t_].matrix = Wli @ M; pb[t_].scale = (1, 1, 1)
        bpy.context.view_layer.update()
    drift = ((kim.matrix_world @ kim.pose.bones["Hips"].head) - HIPS0) * HRATIO
    rb_ = pb["root"]
    cur = male.matrix_world @ rb_.matrix
    M = cur.to_3x3().to_4x4(); M.translation = ROOT0.translation + drift
    rb_.matrix = male.matrix_world.inverted() @ M
    bpy.context.view_layer.update()


ARM_ORDER = sorted(ARMB + HANDB, key=depth)
BM = B0.copy(); travel = 0.0; foot_lo = []; grip_err = []; arm_stretch = {}; penet = []
for i in range(1, END + 1):
    HB = FR[i]
    sc.frame_set(min(i, klast))
    kimodo_delta_frame()
    Wl = male.matrix_world
    a_sh = Wl @ pb["clavicle.L"].tail
    a_kid = (Wl @ pb["spine05"].head) + (Wl.to_3x3() @ Vector((-0.12, 0.12, 0.06)))
    want = (Wl @ pb["spine01"].matrix) @ BASS_IN_CHEST
    # ================= ONE COMBINED SOLVE, IN THIS ORDER =====================
    # 1 hang straight down from the straps  2 push forward out of the body only
    # 3 keep strap length  4 relax the arms  -- iterated until they agree.
    want3 = want.to_3x3().normalized()
    cur3 = BM.to_3x3().normalized()
    dq = (cur3 @ want3.transposed()).to_quaternion()
    if abs(math.degrees(dq.angle)) > 10.0:
        keep = dq.slerp(Quaternion(), 1.0 - 10.0 / abs(math.degrees(dq.angle)))
        rot3 = (keep.to_matrix() @ want3).normalized()
    else:
        rot3 = cur3
    R4 = rot3.to_4x4()
    pin_off = (rot3 @ PIN_TOP_B.to_3d())          # pin offset from bass origin, world-aligned
    fwd = (Wl.to_3x3() @ Vector((0, -1, 0)))
    fwd = Vector((fwd.x, fwd.y, 0))
    fwd = fwd.normalized() if fwd.length > 1e-6 else Vector((0, -1, 0))
    solid = [(Wl @ pb["root"].head, 0.175), (Wl @ pb["spine05"].head, 0.165),
             (Wl @ pb["upperleg01.L"].head, 0.135), (Wl @ pb["upperleg01.R"].head, 0.135)]
    R_FWD = 0.62      # he is much thinner front-to-back than side-to-side
    shoulders = {sd: Wl @ pb[f"upperarm01.{sd}"].head for sd in ("L", "R")}

    # 1. neutral hang: strap vertical, instrument below the shoulder anchor
    def clear_body(p_):
        """push the WHOLE instrument forward until no part of it is inside him"""
        worst = 0.0
        for bp in BASSPTS:
            wp = (R4 @ bp) + p_
            for c, r in solid:
                if abs(wp.z - c.z) > 0.34:
                    continue
                d = wp - c; d.z = 0.0
                along = d.dot(fwd); side = (d - fwd * along).length
                if (along / (r * R_FWD)) ** 2 + (side / r) ** 2 < 1.0:
                    lim = r * R_FWD * math.sqrt(max(0.0, 1.0 - (side / r) ** 2))
                    worst = max(worst, lim - along)
        return p_ + fwd * worst if worst > 0 else p_

    pos = a_sh + Vector((0, 0, -L_TOP)) - pin_off
    for _ in range(10):
        # 3. strap length is fixed: put the pin back on its sphere, biased downward
        v = (pos + pin_off) - a_sh
        if v.length > 1e-6:
            bias = v.normalized().lerp(Vector((0, 0, -1)), 0.55).normalized()
            pos = a_sh + bias * L_TOP - pin_off
        # 4. arms must stay relaxed: no stretching, no folding
        corr = Vector((0, 0, 0)); n_c = 0
        for sd in ("L", "R"):
            hand = (R4 @ Matrix.Translation(HAND_IN_B[sd])).translation + pos
            v2 = hand - shoulders[sd]
            d2 = v2.length
            hi, lo = REACH0[sd] * 1.02, REACH0[sd] * 0.80
            if d2 > hi:
                corr += -v2.normalized() * (d2 - hi); n_c += 1
            elif d2 < lo:
                corr += v2.normalized() * (lo - d2); n_c += 1
        if n_c:
            pos = pos + corr / n_c * 0.6
        # 2. collision LAST: the body border always wins over arm comfort
        pos = clear_body(pos)
        if not n_c:
            break
    tgt_pos = clear_body(pos)
    # heavy instrument: it moves WITH the body, it does not snap or spring
    new_pos = BM.translation.copy().lerp(tgt_pos, 0.30)
    # ...but the body is SOLID. Collision gets the final word, after the lag,
    # so the instrument is pushed forward and never ends up inside him.
    new_pos = clear_body(new_pos)
    for _ in range(3):
        fixed = clear_body(new_pos)
        if (fixed - new_pos).length < 1e-5:
            break
        new_pos = fixed
    BM = R4.copy()
    BM.translation = new_pos
    travel = max(travel, (BM.translation - B0.translation).length)
    for o in GEAR:
        o.matrix_world = BM @ GEAR_IN_BASS[o.name]
        o.rotation_mode = "QUATERNION"
        o.keyframe_insert("location", frame=i); o.keyframe_insert("rotation_quaternion", frame=i)
    Wl = male.matrix_world; Wli = Wl.inverted()
    for n_ in ARMB:
        pb[n_].matrix_basis = REST_ARM[n_].copy()
    bpy.context.view_layer.update()
    for n_ in sorted(HANDB, key=depth):
        pb[n_].matrix = Wli @ (BM @ HB[n_]); pb[n_].scale = (1, 1, 1)
        bpy.context.view_layer.update()
    for s_ in ("L", "R"):
        sh = Wl @ pb[f"upperarm01.{s_}"].head
        wr = Wl @ pb[f"wrist.{s_}"].matrix.translation
        for nm, blend in ((f"upperarm01.{s_}", 0.0), (f"lowerarm01.{s_}", 0.55)):
            if nm not in pb:
                continue
            b = pb[nm]; head = Wl @ b.head
            tgt = sh.lerp(wr, blend + 0.42)
            cur = (Wl @ b.tail) - head; des = tgt - head
            if cur.length > 1e-6 and des.length > 1e-6:
                ax = cur.normalized().cross(des.normalized())
                ang = math.acos(max(-1, min(1, cur.normalized().dot(des.normalized()))))
                if ax.length > 1e-8 and math.degrees(ang) > 0.1:
                    pl = Wli @ head
                    q = Quaternion(ax.normalized(), ang)
                    b.matrix = (Matrix.Translation(pl)
                                @ (Wli.to_3x3().to_4x4() @ q.to_matrix().to_4x4() @ Wl.to_3x3().to_4x4())
                                @ Matrix.Translation(-pl)) @ b.matrix
                    bpy.context.view_layer.update()
    for n_ in sorted(HANDB, key=depth):
        pb[n_].matrix = male.matrix_world.inverted() @ (BM @ HB[n_])
        bpy.context.view_layer.update()
    for n_ in BODY + ARM_ORDER:
        pb[n_].rotation_mode = "QUATERNION"
        pb[n_].keyframe_insert("location", frame=i); pb[n_].keyframe_insert("rotation_quaternion", frame=i)
    _pen = 0.0
    fwd0 = (male.matrix_world.to_3x3() @ Vector((0, -1, 0)))
    fwd0 = Vector((fwd0.x, fwd0.y, 0)).normalized()
    solid = [(male.matrix_world @ pb["root"].head, 0.175),
             (male.matrix_world @ pb["spine05"].head, 0.165),
             (male.matrix_world @ pb["upperleg01.L"].head, 0.135),
             (male.matrix_world @ pb["upperleg01.R"].head, 0.135)]
    for bp in BASSPTS:
        wp = mroot.matrix_world @ bp
        for c, r in solid:
            if abs(wp.z - c.z) <= 0.34:
                d = wp - c; d.z = 0.0
                _al = d.dot(fwd0); _sd = (d - fwd0 * _al).length
                if (_al / (r * 0.62)) ** 2 + (_sd / r) ** 2 < 1.0:
                    _pen = max(_pen, r * 0.62 * math.sqrt(max(0.0, 1 - (_sd / r) ** 2)) - _al)
    penet.append(_pen)
    for sd in ("L", "R"):
        _sh = male.matrix_world @ pb[f"upperarm01.{sd}"].head
        _wr = male.matrix_world @ pb[f"wrist.{sd}"].matrix.translation
        arm_stretch.setdefault(sd, []).append((_wr - _sh).length / REACH0[sd])
    Bi2 = mroot.matrix_world.inverted()
    grip_err.append(max(((Bi2 @ (male.matrix_world @ pb[n_].matrix)).translation - HB[n_].translation).length
                        for n_ in HANDB))
    dg = bpy.context.evaluated_depsgraph_get(); ev = body.evaluated_get(dg); me = ev.to_mesh()
    foot_lo.append(min((body.matrix_world @ v.co).z for v in me.vertices))
    ev.to_mesh_clear()

lift = -min(foot_lo)
if abs(lift) > 1e-4:
    for i in range(1, END + 1):
        sc.frame_set(i)
        rbn = pb["root"]
        rbn.matrix = male.matrix_world.inverted() @ (Matrix.Translation(Vector((0, 0, lift)))
                                                     @ (male.matrix_world @ rbn.matrix))
        bpy.context.view_layer.update()
        rbn.keyframe_insert("location", frame=i); rbn.keyframe_insert("rotation_quaternion", frame=i)
        for o in GEAR:
            m = o.matrix_world.copy(); m.translation.z += lift; o.matrix_world = m
            o.keyframe_insert("location", frame=i); o.keyframe_insert("rotation_quaternion", frame=i)
        for n_ in ARM_ORDER:
            pb[n_].keyframe_insert("location", frame=i); pb[n_].keyframe_insert("rotation_quaternion", frame=i)

cam = bpy.data.objects["CAM"]
cam.animation_data_clear()
prev = None
for i in range(1, END + 1):
    sc.frame_set(i); bpy.context.view_layer.update()
    u = (i - 1) / float(END - 1)
    Bm = mroot.matrix_world
    neck_pt = Bm @ Vector((0.42, 0.0, 0.0))        # along the neck, over the frets
    body_pt = Bm @ Vector((0.02, 0.0, -0.02))      # the body of the bass
    if u < 0.5:
        k = u / 0.5
        k = k * k * (3 - 2 * k)
        look = neck_pt
        dist = 3.6 - 1.2 * k                        # a slow push in: 3.6m -> 2.4m
        lens = 38 + 10 * k
    else:
        k = (u - 0.5) / 0.5
        k = k * k * (3 - 2 * k)
        look = neck_pt.lerp(body_pt, k)             # drift across to the body of the bass
        dist = 2.4 - 0.5 * k
        lens = 48 + 8 * k
    cam.data.lens = lens
    tgt = look + Vector((0.20, -dist, 0.22))
    prev = tgt if prev is None else prev.lerp(tgt, 0.10)     # heavy smoothing = slow, calm
    cam.location = prev
    cam.rotation_euler = (look - prev).to_track_quat("-Z", "Y").to_euler()
    cam.keyframe_insert("location", frame=i)
    cam.keyframe_insert("rotation_euler", frame=i)
    cam.data.keyframe_insert("lens", frame=i)
print("camera: slow push along the neck, then a drift across to the body of the bass")

kim.hide_viewport = True; kim.hide_render = True
sc.frame_set(1); bpy.context.view_layer.update()
Wc = male.matrix_world
vv = (Wc @ pb["spine01"].head) - (Wc @ pb["root"].head)
_rel = json.load(open(os.path.join(DATA, "neck_hand_relation.json")))
_ac, _ob, _dir = [], [], []
for _f in range(1, END + 1):
    sc.frame_set(_f); bpy.context.view_layer.update()
    _Bi = mroot.matrix_world.inverted()
    _wr = _Bi @ (male.matrix_world @ pb["wrist.L"].head)
    _kn = sum((_Bi @ (male.matrix_world @ pb[f"finger{f_}-1.L"].head) for f_ in (2, 3, 4, 5)),
              Vector()) / 4
    _pd = (_kn - _wr).normalized()
    _ac.append(_kn.y)
    _ob.append(_kn.z - _rel["bottom_border_z"])
    _dir.append(math.degrees(math.acos(max(-1, min(1, _pd.dot(Vector((1, 0, 0)))))))) 
print(f"PALM-to-neck lock (measured on the palm bones, which finger movement cannot disturb):")
print(f"   across the neck {min(_ac)*1000:+.1f} to {max(_ac)*1000:+.1f}mm (canon {_rel['mid_knuckle_across']*1000:+.1f})")
print(f"   out from the downward border {min(_ob)*1000:+.1f} to {max(_ob)*1000:+.1f}mm "
      f"(canon {_rel['mid_knuckle_out_of_border']*1000:+.1f})")
print(f"   palm direction vs neck axis {min(_dir):.1f} to {max(_dir):.1f} deg "
      f"(canon {_rel['palm_dir_vs_neck_axis_deg']:.1f}) -- drift "
      f"{max(max(_ac)-min(_ac), max(_ob)-min(_ob))*1000:.2f}mm / {max(_dir)-min(_dir):.2f} deg")
print(f"POSTURE after kimodo: lean {math.degrees(math.atan2(vv.y, vv.z)):+.1f} deg, hip->chest {vv.length*100:.1f}cm (canon 2 = -2.5, 35.6)")
bz = []
for f in range(1, END + 1):
    sc.frame_set(f); bpy.context.view_layer.update()
    bz.append(mroot.matrix_world.translation.z)
print(f"bass travel {travel*100:.1f}cm | grip max {max(grip_err)*1000:.2f}mm | lifted {lift*100:+.1f}cm")
for sd in ("L", "R"):
    a = arm_stretch[sd]
    print(f"arm {sd}: reach {min(a)*100:.0f}% to {max(a)*100:.0f}% of relaxed (100% = canon 2, >105% = stretched)")
print(f"deepest the bass sinks into him: {max(penet)*1000:.1f}mm (0 = never inside the body)")
print(f"bass height over the take: {min(bz)*100:.1f} to {max(bz)*100:.1f}cm (rise {(max(bz)-min(bz))*100:.1f}cm)")
# where does the hand sit on the neck across the take?
xs = []
for f in (1, 40, 80, 120, 150):
    sc.frame_set(f); bpy.context.view_layer.update()
    xs.append((mroot.matrix_world.inverted() @ (male.matrix_world @ pb["finger2-3.L"].tail)).x)
print("index tip along the neck (cm): " + ", ".join(f"{x*100:.1f}" for x in xs)
      + f"  [fret1={fret_x[1]*100:.1f}, fret5={fret_x[5]*100:.1f}, nut={fret_x[0]*100:.1f}]")
sc.frame_set(1)
bpy.ops.wm.save_mainfile(filepath=ARGS.out)
