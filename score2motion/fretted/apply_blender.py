"""Play a fretted plan on a rigged character inside Blender.

Run it as:  blender scene.blend --python apply_blender.py -- plan.json [PREFIX]

The scene must meet NAMING_CONTRACT.md. Nothing here knows whether it is a
bass or a guitar -- the plan says which string and fret, the scene says where
those are, and the same code plays both.

Two things are worth knowing before changing anything:

1. The hand is dragged, not curled. Bending a finger to reach a string moves
   the palm too, so a curl loop chases its own tail and never converges. The
   finger keeps its calibrated shape and the whole hand is translated until the
   tip touches. Gain 0.5 -- at 1.0 it oscillates instead of settling.

2. The fingertip is aimed at the string *where it lies over that fret*, not at
   the string object's origin. The origin sits mid-neck, which is where a
   243mm error came from the first time.
"""
import json
import math
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

FPS = 30
PRESS_DEPTH = 0.004      # how far past first contact the string is driven down
# A fret stands about a millimetre proud of the board, and a pressed string
# stops on it. That is where a fingertip stops too -- see press_point, which
# used to aim BELOW the board instead and buried every fretting hand in the wood.
FRET_HEIGHT = 0.001
HOVER = 0.013            # measured resting height of an unused finger, metres
GAIN = 0.5
# 40, not 12. Each pass moves the hand at most 6cm, and 12 of those is 72cm of
# travel in the best case -- fine near the nut, where this was tuned. Once the
# bass got its full 24 frets the part moved up to the 22nd, some 50cm further
# along the neck, and the loop ran out of passes mid-reach: a steady 8-9mm short
# on exactly the high notes. Not a hand that could not reach, a hand that was not
# given enough goes.
ITERS = 40
# A SLIDE: two notes on the same string, far apart, close in time. Nobody lifts
# and re-places across that -- the finger stays down and travels, and the pitch
# comes with it. Without this the hand teleports between the two frets and the
# in-between frames are whatever the interpolation invents, usually a lift.
SLIDE_FRETS = 3          # a jump this big is not covered by stretching a finger
SLIDE_GAP = 0.35         # ...and this close in time is one gesture, not two


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if not a:
        raise SystemExit("usage: ... --python apply_blender.py -- plan.json [PREFIX]")
    return a[0], (a[1] if len(a) > 1 else "")


class Scene:
    """The naming contract, resolved once."""

    def __init__(self, p=""):
        g = bpy.data.objects
        self.rig = g[p + "PLAYER.rig"]
        self.root = g[p + "INST_root"]
        self.neck = g[p + "INST_neck"]
        self.pb = self.rig.pose.bones
        self.frets = {}
        for k in range(30):
            o = g.get(f"{p}INST_fret{k}")
            if o:
                self.frets[k] = o
        self.strings = []
        for k in range(12):
            o = g.get(f"{p}INST_string{k}")
            if o:
                self.strings.append(o)
        if len(self.strings) < 2 or len(self.frets) < 2:
            raise SystemExit("scene does not meet NAMING_CONTRACT.md")
        self.prefix = p

    @property
    def Bi(self):
        return self.root.matrix_world.inverted()

    def board_top(self, x=None, y=None):
        """How high the playing surface is -- at a given place on it.

        A single number for the whole neck is only right if the neck is exactly
        level in the instrument's frame. Necks are not: they are angled back, and a
        constant height therefore drifts further below the real surface the further
        up you go. Measured on the band's bass, using the global maximum put the
        target 0.7mm under the wood at the nut and 4.1mm under it by the 12th fret --
        the error grew with the fret number, which is the signature of a tilt being
        ignored rather than an offset being wrong.

        So the top face is fitted as a plane and read at the point asked for. With no
        point given it falls back to the old global maximum, which keeps existing
        callers working.
        """
        Bi = self.Bi
        vs = [Bi @ (self.neck.matrix_world @ v.co) for v in self.neck.data.vertices]
        if x is None or y is None or len(vs) < 4:
            return max(v.z for v in vs)
        top = sorted(vs, key=lambda v: -v.z)[:max(3, len(vs) // 2)]
        # least squares z = a*x + b*y + c over the top face
        n = len(top)
        sx = sum(v.x for v in top)
        sy = sum(v.y for v in top)
        sz = sum(v.z for v in top)
        sxx = sum(v.x * v.x for v in top)
        syy = sum(v.y * v.y for v in top)
        sxy = sum(v.x * v.y for v in top)
        sxz = sum(v.x * v.z for v in top)
        syz = sum(v.y * v.z for v in top)
        A = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, float(n)]]
        b = [sxz, syz, sz]
        # 3x3 solve by elimination; if the face is degenerate, fall back
        for i in range(3):
            p = max(range(i, 3), key=lambda r: abs(A[r][i]))
            if abs(A[p][i]) < 1e-12:
                return max(v.z for v in vs)
            A[i], A[p] = A[p], A[i]
            b[i], b[p] = b[p], b[i]
            for r in range(3):
                if r == i:
                    continue
                f = A[r][i] / A[i][i]
                for c in range(i, 3):
                    A[r][c] -= f * A[i][c]
                b[r] -= f * b[i]
        a_, b_, c_ = (b[i] / A[i][i] for i in range(3))
        return a_ * x + b_ * y + c_

    def press_point(self, string, fret):
        """Where the fingertip must end up, in world space.

        Just behind the fret wire, across at that string's line, and resting ON the
        board -- at the height of the fret wire, which is what actually stops the
        string.

        This used to read `board_top - PRESS_DEPTH`, four millimetres BELOW the top of
        the neck. That is inside the wood. Every hand solved against it was being told
        to put its fingertips through the fretboard, and every contact check passed,
        because the check measured the distance to this target and the target itself
        was the error. Measured on the band's bass it placed fingertips 6 to 10mm deep,
        and the fault showed up identically on the guitarist -- knuckles buried, thumb
        pushed out through the front of the neck -- because both read it from here.

        A pressed string does not go into the board: it is driven down until it lands
        on the fret, and the fingertip stops there. So the target sits a fret's height
        ABOVE the board, not a press depth below it.
        """
        Bi = self.Bi
        nut = self.frets[0].location.x
        toward_nut = 1.0 if nut > self.frets[max(self.frets)].location.x else -1.0
        x = self.frets[fret].location.x + toward_nut * 0.006 if fret else nut
        y = (Bi @ self.strings[string].matrix_world.translation).y
        z = self.board_top(x, y)          # the surface HERE, not the neck's high point
        return self.root.matrix_world @ Vector((x, y, z + FRET_HEIGHT))

    def hover_point(self, string, fret):
        p = self.press_point(string, fret)
        up = (self.root.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
        return p + up * HOVER

    def check_plan(self, notes):
        """Can this neck play this plan at all? Ask before touching the scene.

        `press_point` indexes `self.frets[fret]` directly, so a part that asks for
        a fret the neck does not have dies with a bare KeyError from inside a
        geometry helper -- after markers, materials and font curves have already
        been created in the open file. This says what is wrong instead, and says
        it first.

        Compares SETS, not maxima: the frets are collected with `.get()` per index,
        so a neck can be missing a fret in the middle and still have a high one.
        """
        need = {n["fret"] for n in notes if n.get("fret")} | {0}
        missing = sorted(need - set(self.frets))
        if missing:
            raise SystemExit(
                f"plan needs fret(s) {missing}; this neck has {sorted(self.frets)}"
                f" -- extend the neck, or lower max_fret on the preset in"
                f" instrument.py and re-plan")


def hand_bones(rig, side):
    """Wrist, metacarpals and fingers, PARENTS FIRST.

    The order is load-bearing: writing a child's matrix and then its parent's
    silently undoes the child, because the child is defined relative to it.
    Metacarpals must be in the set too -- they sit between wrist and knuckles
    on this rig, and leaving them out lets the palm twist away from the neck.
    """
    names = [b.name for b in rig.data.bones
             if b.name.endswith("." + side)
             and (b.name.startswith(("wrist", "finger", "metacarpal")) or b.name == "hand." + side)]
    return sorted(names, key=lambda n: len(rig.data.bones[n].parent_recursive))


def drag_hand_to(sc, bones, base, finger, target):
    """Move the whole hand -- keeping its shape -- until `finger`'s tip touches.

    Returns the residual error in metres, so the caller can assert on it rather
    than trust it.
    """
    rig, pb = sc.rig, sc.pb
    W = rig.matrix_world
    Wi = W.inverted()
    tipname = f"finger{finger + 1}-3.L"
    if tipname not in pb:
        tipname = f"finger{finger}-3.L"
    off = Vector((0, 0, 0))
    for _ in range(ITERS):
        for n in bones:
            m = base[n].copy()
            m.translation = m.translation + off
            pb[n].matrix = Wi @ m
            pb[n].scale = (1, 1, 1)
            # the setter converts through the bone's CURRENT evaluated parent,
            # so the parent must be committed before the child is written --
            # hoisting this update out of the loop leaves the hand where it was
            bpy.context.view_layer.update()
        err = target - (W @ pb[tipname].tail)
        if err.length < 0.002:
            break
        step = err * GAIN
        if step.length > 0.06:
            step = step.normalized() * 0.06
        off = off + step
    return (target - (W @ pb[tipname].tail)).length


def tip_name(pb, finger):
    t = f"finger{finger + 1}-3.L"
    return t if t in pb else f"finger{finger}-3.L"


def barre_line(sc, finger, fret, strings):
    """A barre is a LINE, not a point: where the finger must lie, and how long.

    Aiming the tip at one of the barred strings is not enough. Measured on a
    three-string barre at the twelfth fret, the tip landed correctly and the
    other strings were still 4.6mm and 8.2mm off -- and the error grew with
    distance back from the tip, which is the signature of a finger crossing the
    strings DIAGONALLY rather than lying along them.

    An angle cannot be curled out with the hand held still. So the barre is
    described as a line -- from the nearest barred string to the farthest -- and
    the hand is asked to put the finger's KNUCKLE on that line, one finger-length
    back. The finger is then swung onto the line and covers every string on it,
    which is what a barre actually is.

    Returns (tip_target, knuckle_target, direction, aimed_string).
    """
    knuckle_b = f"finger{finger + 1}-1.L"
    knuckle = sc.rig.matrix_world @ sc.pb[knuckle_b].head
    tip = sc.rig.matrix_world @ sc.pb[tip_name(sc.pb, finger)].tail
    length = (tip - knuckle).length
    pts = [(sc.press_point(s, fret), s) for s in strings]
    far, aimed = max(pts, key=lambda p: (p[0] - knuckle).length)
    near, _ = min(pts, key=lambda p: (p[0] - knuckle).length)
    u = (far - near)
    u = u.normalized() if u.length > 1e-6 else (far - knuckle).normalized()
    # the tip goes a little past the last string, the way a real barre overhangs
    return far + u * 0.004, far - u * length, u, aimed


def swing_finger_onto(sc, finger, direction, max_deg=40.0):
    """Turn a finger about its knuckle until it points along `direction`.

    Only the root joint moves; the joints past it keep the calibrated shape, so
    the finger stays a finger and simply lies the other way.
    """
    rig, pb = sc.rig, sc.pb
    W, Wi = rig.matrix_world, rig.matrix_world.inverted()
    root_b = f"finger{finger + 1}-1.L"
    if root_b not in pb:
        return
    p = W @ pb[root_b].head
    have = (W @ pb[tip_name(pb, finger)].tail) - p
    if have.length < 1e-5:
        return
    have.normalize()
    d = max(-1.0, min(1.0, have.dot(direction)))
    ang = min(math.acos(d), math.radians(max_deg))
    ax = have.cross(direction)
    if ax.length < 1e-6 or ang < 1e-4:
        return
    q = Quaternion(ax.normalized(), ang)
    Mw = W @ pb[root_b].matrix
    Mw = (Matrix.Translation(p) @ q.to_matrix().to_4x4()
          @ Matrix.Translation(-p)) @ Mw
    pb[root_b].matrix = Wi @ Mw
    pb[root_b].scale = (1, 1, 1)
    bpy.context.view_layer.update()


def drag_hand_to_many(sc, bones, base, targets):
    """Place the hand once so it serves EVERY finger in a chord.

    `drag_hand_to` moves the whole hand until ONE fingertip touches. Called once
    per note of a chord it runs four times on the same frame and each pass drags
    the hand off the note before it, so only the last note of the chord lands --
    which is exactly what the guitar was doing.

    This moves the hand by the AVERAGE of what its fingers are asking for, so no
    single finger wins. The residual per finger is closed afterwards by curling
    that finger, which is cheap now because the hand is already in the right
    place.

    targets: {finger: (world point, "tip" | "knuckle")}. A barring finger asks for
    its KNUCKLE to be placed, because what a barre needs is the finger lying along
    the strings, and that is decided by where the finger STARTS.

    Returns {finger: residual metres}.
    """
    rig, pb = sc.rig, sc.pb
    W, Wi = rig.matrix_world, rig.matrix_world.inverted()

    def at(f, which):
        b = f"finger{f + 1}-1.L" if which == "knuckle" else tip_name(pb, f)
        return (W @ pb[b].head) if which == "knuckle" else (W @ pb[b].tail)

    off = Vector((0, 0, 0))
    for _ in range(ITERS):
        for n in bones:
            m = base[n].copy()
            m.translation = m.translation + off
            pb[n].matrix = Wi @ m
            pb[n].scale = (1, 1, 1)
            bpy.context.view_layer.update()
        errs = [t - at(f, which) for f, (t, which) in targets.items()]
        mean = sum(errs, Vector()) / len(errs)
        if mean.length < 0.001:
            break
        step = mean * GAIN
        if step.length > 0.06:
            step = step.normalized() * 0.06
        off = off + step
    return {f: (t - at(f, which)).length for f, (t, which) in targets.items()}


def curl_finger_to(sc, finger, target, iters=14, max_deg=12.0):
    """Bend one finger's own joints to close the last few millimetres.

    The engine's rule is that the hand is dragged, not curled -- a curl loop that
    has to cover the whole reach chases its own tail, because bending a finger
    moves the palm too. That still holds. This is the other case: the hand is
    already placed and only this finger is a few millimetres out, so the
    corrections are small enough that they settle instead of oscillating.

    Rotates each joint from knuckle to tip toward the target, a little at a time.
    """
    rig, pb = sc.rig, sc.pb
    W, Wi = rig.matrix_world, rig.matrix_world.inverted()
    chain = [f"finger{finger + 1}-{k}.L" for k in (1, 2, 3)]
    chain = [c for c in chain if c in pb]
    if not chain:
        return None
    tip = chain[-1]
    for _ in range(iters):
        moved = False
        for b in chain:
            p = W @ pb[b].head
            have = (W @ pb[tip].tail) - p
            want = target - p
            if have.length < 1e-5 or want.length < 1e-5:
                continue
            have.normalize()
            want.normalize()
            d = max(-1.0, min(1.0, have.dot(want)))
            ang = math.acos(d)
            if ang < 1e-4:
                continue
            ang = min(ang, math.radians(max_deg))
            ax = have.cross(want)
            if ax.length < 1e-6:
                continue
            q = Quaternion(ax.normalized(), ang)
            Mw = W @ pb[b].matrix
            Mw = (Matrix.Translation(p) @ q.to_matrix().to_4x4()
                  @ Matrix.Translation(-p)) @ Mw
            pb[b].matrix = Wi @ Mw
            pb[b].scale = (1, 1, 1)
            bpy.context.view_layer.update()
            moved = True
        if (target - (W @ pb[tip].tail)).length < 0.002 or not moved:
            break
    return (target - (W @ pb[tip].tail)).length


def make_markers(sc, n=6):
    """A red dot at each pressed fret, carrying the number of the finger on it.

    ONE PER NOTE, not one per finger. Per finger, a barre -- one finger holding
    three strings -- lit a single fret and left the other two dark, and a chord
    could never show more marks than it used distinct fingers. A pool of six
    covers any chord a six-string can sound.

    The number is a CHILD of the dot but Blender does not inherit visibility
    through parenting, so the digits sat on screen permanently while their dots
    blinked. Both are keyed together now; see `key()`.
    """
    made = {}
    for d in range(1, n + 1):
        name = f"{sc.prefix}NOTEMARK_{d}"
        ob = bpy.data.objects.get(name)
        if ob is None:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.008)
            ob = bpy.context.object
            ob.name = name
            mat = bpy.data.materials.new(f"notemark_red_{d}")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes["Principled BSDF"]
            bsdf.inputs["Base Color"].default_value = (1, 0.05, 0.05, 1)
            bsdf.inputs["Emission Color"].default_value = (1, 0.05, 0.05, 1)
            bsdf.inputs["Emission Strength"].default_value = 6.0
            ob.data.materials.append(mat)
            txt = bpy.data.curves.new(f"{name}_c", type="FONT")
            txt.body = str(d)
            txt.align_x = txt.align_y = "CENTER"
            t = bpy.data.objects.new(f"{name}_txt", txt)
            t.scale = (0.03, 0.03, 0.03)
            t.data.materials.append(mat)
            bpy.context.scene.collection.objects.link(t)
            t.parent = ob
            t.location = (0, 0, 0.02)
        made[d] = ob
    return made


def key(ob, frame, hide):
    """Show or hide a marker AND the number riding on it.

    Blender does not inherit visibility through parenting, so hiding the dot left
    its digit hanging in mid-air for the whole take. The children are keyed
    explicitly.
    """
    for o in (ob, *ob.children):
        o.hide_viewport = o.hide_render = hide
        o.keyframe_insert("hide_viewport", frame=frame)
        o.keyframe_insert("hide_render", frame=frame)
    if not hide:
        ob.keyframe_insert("location", frame=frame)


def main():
    plan_path, prefix = args()
    doc = json.load(open(plan_path))
    notes = doc["notes"]
    sc = Scene(prefix)
    sc.check_plan(notes)   # fail before anything is created in the open file
    scene = bpy.context.scene
    scene.render.fps = FPS
    end = int(max(n["t_off"] for n in notes) * FPS) + 10
    scene.frame_start, scene.frame_end = 1, end
    scene.frame_set(1)

    bones = hand_bones(sc.rig, "L")
    W = sc.rig.matrix_world
    base = {n: (W @ sc.pb[n].matrix).copy() for n in bones}
    markers = make_markers(sc)

    print(f"{doc['instrument']}: {len(notes)} notes over {end / FPS:.1f}s, "
          f"{doc.get('hand_shifts', '?')} hand shifts")

    worst = 0.0
    for m in markers.values():
        key(m, 1, True)

    # Notes that land on the same frame are a CHORD and have to be placed
    # together. Handled one at a time, each one drags the whole hand off the one
    # before it and only the last of them ends up on its string.
    frames = {}
    for n in notes:
        if n["finger"] == 0:          # open string: nothing to fret
            continue
        frames.setdefault(max(1, int(n["t_on"] * FPS)), []).append(n)

    n_chords = sum(1 for g in frames.values() if len(g) > 1)
    if n_chords:
        print(f"  {n_chords} of {len(frames)} frames are chords")

    for f_on in sorted(frames):
        group = frames[f_on]

        # one target per FINGER, not per note -- a barring finger holds several
        # strings down with the same finger and must not be aimed twice
        by_finger = {}
        for n in group:
            by_finger.setdefault((n["finger"], n["fret"]), []).append(n["string"])
        targets, barres = {}, {}
        for (finger, fret), strings in by_finger.items():
            if len(strings) > 1:
                tip_t, knuckle_t, u, aimed = barre_line(sc, finger, fret, strings)
                barres[finger] = (fret, strings, aimed, u, tip_t)
                targets[finger] = (knuckle_t, "knuckle")
            else:
                targets[finger] = (sc.press_point(strings[0], fret), "tip")

        if len(targets) == 1 and not barres:
            # the single-note path, unchanged -- this is what the bass uses and
            # what was measured at 1.7mm, so it stays exactly as it was
            finger, (tgt, _) = next(iter(targets.items()))
            errs = {finger: drag_hand_to(sc, bones, base, finger, tgt)}
        else:
            errs = drag_hand_to_many(sc, bones, base, targets)
            # lay each barring finger down along its strings, then let the
            # ordinary fingers close their own last millimetres
            for finger, (_, strings, _, u, tip_t) in barres.items():
                swing_finger_onto(sc, finger, u)
                errs[finger] = curl_finger_to(sc, finger, tip_t) or errs[finger]
            for finger, (tgt, which) in targets.items():
                if which == "tip" and errs[finger] > 0.002:
                    r = curl_finger_to(sc, finger, tgt)
                    if r is not None:
                        errs[finger] = r

        for finger, e in errs.items():
            worst = max(worst, e)
            if e > 0.006:
                b = barres.get(finger)
                extra = (f" (barring strings {b[1]}, tip aimed at {b[2]})" if b else "")
                print(f"    frame {f_on:4d} finger {finger} still {e * 1000:.1f}mm "
                      f"short{extra}")

        for b in bones:
            sc.pb[b].keyframe_insert("location", frame=f_on)
            sc.pb[b].keyframe_insert("rotation_quaternion", frame=f_on)

        # one marker per SOUNDING NOTE, so every fret the chord holds lights up
        for slot, n in enumerate(group, start=1):
            if slot not in markers:
                break
            f_off = max(f_on + 1, int(n["t_off"] * FPS))
            mk = markers[slot]
            mk.location = sc.press_point(n["string"], n["fret"])
            for ch in mk.children:                 # the digit says WHICH finger
                if ch.type == "FONT":
                    ch.data.body = str(n["finger"])
            key(mk, f_on - 1, True)
            key(mk, f_on, False)
            key(mk, f_off, False)
            key(mk, f_off + 1, True)

    # ---- slides ----
    # Done after the notes are placed, so it only fills the frames BETWEEN two
    # notes and never disturbs the frames the notes themselves own. The finger
    # that started the slide is the one that travels -- that is what makes it a
    # slide rather than a jump.
    seq = sorted([n for n in notes if n["finger"] > 0], key=lambda n: n["t_on"])
    slid = 0
    for a, b in zip(seq, seq[1:]):
        if a["string"] != b["string"] or not a["fret"] or not b["fret"]:
            continue
        # A slide is ONE finger travelling. If the plan changes finger between the
        # two notes it is a re-place, and dragging the first finger to the second
        # note's fret sends a finger somewhere the plan never asked for.
        if a["finger"] != b["finger"]:
            continue
        if abs(b["fret"] - a["fret"]) < SLIDE_FRETS:
            continue
        if b["t_on"] - a["t_off"] > SLIDE_GAP:
            continue
        fa = max(1, int(a["t_on"] * FPS))
        fb = max(1, int(b["t_on"] * FPS))
        if fb - fa < 2:
            continue                      # no room between them to travel through
        pa = sc.press_point(a["string"], a["fret"])
        pb_ = sc.press_point(b["string"], b["fret"])
        for f in range(fa + 1, fb):
            u = (f - fa) / float(fb - fa)
            drag_hand_to(sc, bones, base, a["finger"], pa.lerp(pb_, u))
            for bn in bones:
                sc.pb[bn].keyframe_insert("location", frame=f)
                sc.pb[bn].keyframe_insert("rotation_quaternion", frame=f)
        slid += 1
        way = "up" if b["fret"] > a["fret"] else "down"
        print(f"  slide {way} string {a['string']}: fret {a['fret']} -> {b['fret']} "
              f"over frames {fa}..{fb}")
    if slid:
        print(f"  {slid} slide(s) animated -- finger stays down and travels")

    print(f"worst fingertip-to-string error across the part: {worst * 1000:.1f}mm")
    if worst > 0.006:
        print("WARNING: over 6mm somewhere -- the hand could not reach; check the "
              "grip calibration or the neck geometry")
    return worst


if __name__ == "__main__":
    main()
