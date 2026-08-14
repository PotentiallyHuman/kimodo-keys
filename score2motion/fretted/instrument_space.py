"""The hands belong to the INSTRUMENT, not to the character.

A player's hand is not at a place in the room and not at a place on the body:
it is at a place on the instrument. If the instrument moves -- a strap swings
it, the body leans, a physics pass shoves it -- the hands must go with it, and
every fret they were pressing stays pressed.

The way to get that is to stop storing hand poses in world or character space:

    capture(...)  -> per-frame wrist matrices expressed in INSTRUMENT space
    stamp(...)    -> put a wrist back at that matrix on the instrument WHEREVER
                     the instrument now is

Why this module exists: the same performance, evaluated in a scene whose
instrument sits 11 cm from where it sat when the take was baked, has the
fingers playing a ghost instrument -- fingertips 108 mm from the strings while
every "the wrist is exactly where it should be" check still passed, because
the wrist was exactly where it should be *on the character*. Capturing in
instrument space makes that failure impossible to express.

    # bake time, in the scene where the fingers are provably on the strings
    data = capture(rig, inst_root, frames=858)

    # play time, in any scene, with the instrument moving however it likes
    for f in frames:
        inst_root.matrix_world = wherever_physics_put_it(f)
        stamp(rig, "wrist.L", inst_root, data["L"][f - 1])

Arms are solved separately as connectors from the shoulder to the stamped
wrist; fingers ride the wrist and need no correction at all.
"""
from mathutils import Matrix


def capture(rig, inst_root, frames, scene=None, bones=("wrist.L", "wrist.R")):
    """Record each bone's matrix in INSTRUMENT space, frame by frame.

    Returns {bone_key: [Matrix, ...]} where bone_key is the bone name and the
    list is indexed frame-1. Run this in the scene where the performance was
    built, i.e. where the contact is known to be right.
    """
    import bpy
    scene = scene or bpy.context.scene
    Wm = rig.matrix_world
    out = {b: [] for b in bones}
    for f in range(1, frames + 1):
        scene.frame_set(f)
        Bi = inst_root.matrix_world.inverted()
        for b in bones:
            out[b].append((Bi @ (Wm @ rig.pose.bones[b].matrix)).copy())
    return out


def stamp(rig, bone, inst_root, local_matrix, iterations=2):
    """Put `bone` at `local_matrix` of the instrument, as it is RIGHT NOW.

    Written through matrix_basis rather than assigned to pose_bone.matrix:
    on a rig whose parents carry scale or inherit-scale settings, assigning
    .matrix does not land where you asked. Two passes converge to 0.00 mm and
    0.00 deg; the caller can verify with `error(...)` and should.
    """
    import bpy
    pb = rig.pose.bones[bone]
    desired = rig.matrix_world.inverted() @ (inst_root.matrix_world @ local_matrix)
    for _ in range(iterations):
        bpy.context.view_layer.update()
        cur = pb.matrix.copy()
        pb.matrix_basis = pb.matrix_basis @ (cur.inverted() @ desired)
    bpy.context.view_layer.update()


def error(rig, bone, inst_root, local_matrix):
    """(position error in metres, rotation error in radians) after a stamp."""
    got = inst_root.matrix_world.inverted() @ (rig.matrix_world
                                               @ rig.pose.bones[bone].matrix)
    dp = (got.translation - local_matrix.translation).length
    dq = got.to_quaternion().rotation_difference(local_matrix.to_quaternion())
    return dp, dq.angle


def to_json(data):
    """Capture -> plain lists, for saving between Blender sessions."""
    return {k: [[list(r) for r in m] for m in v] for k, v in data.items()}


def from_json(d):
    return {k: [Matrix(m) for m in v] for k, v in d.items()}
