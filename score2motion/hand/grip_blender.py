"""Put a rigged hand on a round object, inside Blender. Run it directly.

    blender -b scene.blend --python grip_blender.py -- --object Handle
    blender -b scene.blend --python grip_blender.py -- --object Neck --side L
    blender -b scene.blend --python grip_blender.py -- --object Pole --around 120 --along -1

The grip is not derived. It is a measured table -- one real hand, placed by hand
on a real object -- read back in the frame of whatever object you name.

That choice of frame is the whole design. Written in room coordinates the same
grip needs cross products, tangents and closing directions, and every mirrored or
reversed case turns into a hunt for the sign that flipped. Three attempts died
that way: a palm-side test that is degenerate on a flat hand, a distance passed
where a penetration was expected, and two different orientations of one axis used
in the same pass. In the object's own frame there is nothing to flip. The axis IS
z; a bone sits so many radii out, so many degrees round, so many radii along.

    the other end of it   negate `along` and `round` -- a half turn about x
    the other hand        mirror across one plane, and rebuild each bone on the
                          axes its own side of the rig was built with
    another approach      add a constant to `round`
    another object        the numbers are in radii, so they already fit

and no other line of code changes.

Bone names follow the MakeHuman convention (wrist, metacarpal1..4,
finger1-1..finger5-3, .L/.R). Pass --prefix if yours are decorated.

Prior art: AutoGrip (github.com/Jetpack-Crow/autogrip, MIT) does this with
shrinkwrap constraints on added control bones. It handles any mesh, not only
round ones. It also adds bones, and its IK carries no pole target, so nothing in
it knows which way a fingernail ends up facing.
"""
import json
import math
import os
import sys

try:
    import bpy
    from mathutils import Matrix, Quaternion, Vector
except ImportError:                                # importable without Blender
    bpy = None

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from score2motion.hand.grip import first_contact, fit_cylinder  # noqa: E402

TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "grip_table.json")
FINGERS = {"index": ["finger2-1", "finger2-2", "finger2-3"],
           "middle": ["finger3-1", "finger3-2", "finger3-3"],
           "ring": ["finger4-1", "finger4-2", "finger4-3"],
           "pinky": ["finger5-1", "finger5-2", "finger5-3"],
           "thumb": ["finger1-1", "finger1-2", "finger1-3"]}
MC = {"index": "metacarpal1", "middle": "metacarpal2",
      "ring": "metacarpal3", "pinky": "metacarpal4"}
ORDER = ["wrist"] + list(MC.values()) + [b for f in FINGERS.values() for b in f]


def parse(argv):
    a = argv[argv.index("--") + 1:] if "--" in argv else []

    def opt(n, d=None):
        return a[a.index(n) + 1] if n in a else d
    return {"side": opt("--side", "R").upper(),
            "object": opt("--object"),
            "part_near": opt("--part-near"),
            "grip_at": float(opt("--grip-at", "0.5")),
            "press": float(opt("--press", "0.001")),
            "around": float(opt("--around", "0")),
            "along": int(opt("--along", "1")),
            "prefix": opt("--prefix", ""),
            "rig": opt("--rig"),
            "table": opt("--table", TABLE),
            "save": opt("--save")}


def object_frame(ob, part_near=None, station=0.5):
    """The object's own axes: z along it, origin on the centre line."""
    vs = [ob.matrix_world @ v.co for v in ob.data.vertices]
    if part_near is not None:
        import bmesh
        bm = bmesh.new()
        bm.from_mesh(ob.data)
        bm.verts.ensure_lookup_table()
        seen, comps = set(), []
        for v in bm.verts:
            if v.index in seen:
                continue
            stack, comp = [v], []
            while stack:
                x = stack.pop()
                if x.index in seen:
                    continue
                seen.add(x.index)
                comp.append(x)
                for e in x.link_edges:
                    stack.append(e.other_vert(x))
            comps.append(comp)
        nv = [part_near.matrix_world @ v.co for v in part_near.data.vertices]
        nc = sum(nv, Vector()) / len(nv)
        pick = min(comps, key=lambda c: min(
            ((ob.matrix_world @ x.co) - nc).length for x in c))
        vs = [ob.matrix_world @ x.co for x in pick]
        bm.free()
    cyl = fit_cylinder([(v.x, v.y, v.z) for v in vs], station=station)
    C, a = Vector(cyl.centre), Vector(cyl.axis)
    x = a.orthogonal().normalized()
    y = a.cross(x).normalized()
    return Matrix(((x.x, y.x, a.x, C.x), (x.y, y.y, a.y, C.y),
                   (x.z, y.z, a.z, C.z), (0.0, 0.0, 0.0, 1.0))), cyl


# A mirror is a whole transform applied on both sides, never a sign edited into
# one component. But the two sides are not the same matrix and are not even in
# the same space. On the left is where the bone GOES, in the object's frame. On
# the right is how the bone is BUILT: a bone matrix's columns are that bone's own
# x, y, z, and the .L bones of a mirrored rig do not carry the mirrored .R axes,
# they carry them with x negated -- measured here, 0.000mm and the same map on
# all twenty hand bones:
#
#     rest(.L) = mirror_across_x @ rest(.R) @ diag(-1, 1, 1)
#
# Repeating the object-frame plane on the right instead, which is the tidy-
# looking thing to write, negates the bone's OWN y -- its length -- and differs
# from the truth by half a turn about each bone's z. The wrist still parks on the
# target to five microns because that one bone is placed outright, and every
# finger below it swings 136mm off the object. That is the whole left-hand bug.
#
# A flip to the other end takes no right-hand side at all. It is a half turn, a
# rotation, det +1; it moves the hand without reflecting it, and a hand that is
# not reflected keeps the axes it was built with. Conjugating by it -- also tidy-
# looking -- rerolls every bone 180 degrees and stands the fingers 6 to 11mm off.
MIRROR = Matrix.Diagonal((1.0, -1.0, 1.0, 1.0))     # object frame: where it goes
LOCAL = Matrix.Diagonal((-1.0, 1.0, 1.0, 1.0))      # bone's own axes: how it is built
FLIP = Matrix.Diagonal((1.0, -1.0, -1.0, 1.0))      # half turn, both ends of it


def placed(entry, radius, around, along, mirror):
    """One bone's transform in the object's frame, after the asked-for changes."""
    th = math.radians(entry["degrees_round"])
    r = entry["radius_in_radii"] * radius
    M = Matrix.LocRotScale(
        Vector((r * math.cos(th), r * math.sin(th),
                entry["along_in_radii"] * radius)),
        Quaternion(entry["quat"]), None)
    if along < 0:
        M = FLIP @ M
    if mirror:
        M = MIRROR @ M @ LOCAL
    if abs(around) > 1e-9:
        M = Matrix.Rotation(math.radians(around), 4, 'Z') @ M
    return M


def run(cfg):
    bpy.context.view_layer.update()
    rig = (bpy.data.objects[cfg["rig"]] if cfg["rig"]
           else next(o for o in bpy.data.objects if o.type == 'ARMATURE'))
    pb = rig.pose.bones
    P, S = cfg["prefix"], cfg["side"]
    tbl = json.load(open(cfg["table"]))
    mirror = (S != tbl.get("hand", "R"))

    def bn(n):
        return f"{P}{n}.{S}"

    ob = bpy.data.objects[cfg["object"]]
    near = bpy.data.objects[cfg["part_near"]] if cfg["part_near"] else None
    OBJ, cyl = object_frame(ob, near, cfg["grip_at"])
    R = cyl.radius
    print(f"{cfg['object']}: radius {R*1000:.1f}mm at {cfg['grip_at']*100:.0f}% "
          f"along (table measured at {tbl['reference_radius_mm']:.1f}mm)")
    print(f"  {S} hand{' mirrored' if mirror else ''}, {cfg['around']:+.0f} deg "
          f"round it, {'other end' if cfg['along'] < 0 else 'same end'}")

    def targets(frame, radius):
        return {bn(n): frame @ placed(tbl["bones"][n], radius, cfg["around"],
                                      cfg["along"], mirror)
                for n in ORDER if n in tbl["bones"] and bn(n) in pb}

    want = targets(OBJ, R)

    # carry the body so the wrist lands where the table puts it, then set the
    # hand. The arm is not solved here -- rebuild it after, if the body must
    # stay where it is.
    #
    # The object is held still across that carry, because it may be hanging off
    # the very rig being carried -- a microphone stand parented to the singer is
    # exactly that -- and then it runs from the hand at the speed the hand chases
    # it. No move of the rig can ever close that gap: the hand's position
    # relative to a prop the rig carries is fixed by the POSE, not by where the
    # rig stands. Left alone it does not fail loudly, it fails silently, because
    # the grip is then graded in the frame the object occupied BEFORE the carry:
    # every fingertip reads a tidy -0.8mm while the real stand is 2.7 metres and
    # 55 degrees away, and the hand closes on nothing at all. A prop stays in the
    # room; the body is what walks to it.
    wr = bn("wrist")
    if wr in want:
        held = [(o, o.matrix_world.copy()) for o in (ob, near) if o is not None]
        rig.matrix_world = (want[wr]
                            @ (rig.matrix_world @ pb[wr].matrix).inverted()
                            @ rig.matrix_world)
        bpy.context.view_layer.update()
        for o, M in held:
            o.matrix_world = M
        bpy.context.view_layer.update()
        # and measure it again where it now IS, so that any object that did move
        # is gripped and graded where it ended up rather than where it began
        OBJ, cyl = object_frame(ob, near, cfg["grip_at"])
        R = cyl.radius
        want = targets(OBJ, R)
    for n in ORDER:
        if bn(n) in want:
            pb[bn(n)].matrix = rig.matrix_world.inverted() @ want[bn(n)]
            bpy.context.view_layer.update()

    # ------------------------------------------------ close for THIS radius
    mesh = next((o for o in bpy.data.objects if o.type == 'MESH'
                 and any(m.type == 'ARMATURE' and m.object == rig
                         for m in o.modifiers)), None)
    inv = OBJ.inverted()

    def skin(names, weight=0.5):
        if not mesh:
            return []
        gi = {g.name: g.index for g in mesh.vertex_groups}
        want_g = {gi[n] for n in names if n in gi}
        if not want_g:
            return []
        ev = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
        M = ev.matrix_world
        return [M @ v.co for v in ev.data.vertices
                if any(g.group in want_g and g.weight > weight
                       for g in v.groups)]

    def gap(names):
        """How far this skin is FROM the surface. Negative once it is inside.

        Distance, not penetration. They are negatives of each other, and passing
        one where the other is expected reads "38mm outside" as "38mm inside" --
        the search then stops at its first try and reports a grip on a hand that
        never moved.
        """
        pts = [inv @ p for p in skin(names)]
        pts = [q for q in pts if cyl.lo - 0.004 <= q.z <= cyl.hi + 0.004]
        return (min(math.hypot(q.x, q.y) for q in pts) - R) if pts else None

    ratio = R / (tbl["reference_radius_mm"] / 1000.0)
    if abs(ratio - 1.0) > 0.02:
        # A hand does not scale with the object, so the CURL is re-solved. Each
        # finger keeps the shape it was measured in and only closes further or
        # less far along it.
        base = {b.name: b.matrix_basis.copy() for b in pb}
        for fam, chain in FINGERS.items():
            bones = [bn(n) for n in chain if bn(n) in pb]
            if not bones:
                continue

            def dial(t, bones=bones):
                for b in bones:
                    q = base[b].to_quaternion()
                    pb[b].matrix_basis = Matrix.LocRotScale(
                        base[b].translation,
                        Quaternion(q.axis, q.angle * t), None)
                bpy.context.view_layer.update()
                g = gap([bones[-1]])
                return 1.0 if g is None else g

            dial(first_contact(dial, top=1.8, press=cfg["press"]))
        print(f"  object is {ratio:.2f}x the reference radius -- fingers re-closed")

    # --------------------------------------------------------- the check
    # The four fingers are graded against the OBJECT. The thumb is not: it
    # opposes the index, and on the measured grip it sits nearly two radii clear
    # of the surface on purpose. Grading it like a finger fails a correct hand.
    rows = [(f, gap([bn(FINGERS[f][-1])])) for f in
            ("index", "middle", "ring", "pinky")]
    for f, g in rows:
        print(f"  {f:7s} fingertip " + ("hangs past the end of it" if g is None
              else f"skin {g*1000:+.1f}mm from the surface"))
    W = rig.matrix_world
    tt, it = bn(FINGERS["thumb"][-1]), bn(FINGERS["index"][-1])
    thumb_gap = ((W @ pb[tt].tail) - (W @ pb[it].tail)).length * 1000 \
        if tt in pb and it in pb else None
    print(f"  thumb   tip {thumb_gap:.1f}mm from the index tip (it opposes the "
          f"index, not the object)")
    got = [g * 1000 for _, g in rows if g is not None]
    ok = len(got) == 4 and max(got) < 1.0 and min(got) > -4.0
    print("  PASS -- the hand is on it" if ok else
          "  the fingers did not all reach")
    if cfg["save"]:
        bpy.ops.wm.save_as_mainfile(filepath=cfg["save"])
        print(f"saved {cfg['save']}")
    return ok


if __name__ == "__main__" and bpy is not None:
    cfg = parse(sys.argv)
    if not cfg["object"]:
        print(__doc__)
        sys.exit(2)
    run(cfg)
