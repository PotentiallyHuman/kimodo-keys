"""Find the frames where a retargeted bone spun about itself instead of moving.

A retargeter fixes a bone's ROLL from a reference direction taken off the source
skeleton. Where that reference goes near-degenerate the solution flips to the
other side. Both answers point the bone in exactly the same direction; only the
rotation about its own length differs, by half a turn.

On a body that is the difference between a hand that follows the arm and a hand
that snaps over and back. Measured on a generated take of a singer: her free
wrist changed orientation by 179.9 degrees between two consecutive frames while
the bone itself moved 0.9 degrees. Six times in five seconds. The fingers hang
off the wrist, so the whole hand went with it.

The test is deliberately narrow, and both halves matter:

    the ORIENTATION jumped        more than `jump` degrees
    the DIRECTION did not         less than `still` degrees

A real turn of a wrist moves the bone's direction as well, and is left alone.
Checking only the orientation would flag every fast movement.

    from score2motion.motion.roll_flips import find_flips

    bad = find_flips([(direction_xyz, quat_wxyz), ...])

Returns the indices where the roll BREAKS -- not every frame that is on the
wrong side. If frames 30 to 32 are all flipped the same way, nothing changes
between them; the discontinuities are at 30 and at 33, and those are what comes
back. That is the honest answer for detection, and it is not the list to hand
to a repair.

A repair walks the take in order and compares each frame against the frame it
has ALREADY repaired, so a run of flipped frames is undone one after another:
fix 30, and 31 is now half a turn from a corrected neighbour, so it is caught
in its turn. Feeding it only the indices above would unflip the first frame of
each run and leave the rest.

Pure python. No rig, no scene, no Blender.
"""
import math

JUMP = 90.0          # degrees of orientation change that is not a movement
STILL = 15.0         # degrees of direction change that still counts as "not moving"


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _angle_between(a, b):
    na = math.sqrt(_dot(a, a))
    nb = math.sqrt(_dot(b, b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    c = max(-1.0, min(1.0, _dot(a, b) / (na * nb)))
    return math.degrees(math.acos(c))


def quat_angle(qa, qb):
    """Shortest rotation between two quaternions, in degrees.

    Both signs of a quaternion are the same rotation, so the dot product is
    taken absolute -- without that, half the frames in any take read as 180
    degrees apart and every one of them looks like a flip.
    """
    d = abs(_dot(qa, qb))
    d = max(-1.0, min(1.0, d))
    return math.degrees(2.0 * math.acos(d))


def find_flips(frames, jump=JUMP, still=STILL):
    """Indices of frames where the bone spun about itself and did not move.

    frames: [(direction, quat), ...] one per frame, in order. direction is the
    bone's world direction as (x, y, z); quat is its world orientation as
    (w, x, y, z). Frame 0 is never returned -- there is nothing before it to
    have flipped away from.
    """
    out = []
    for i in range(1, len(frames)):
        d0, q0 = frames[i - 1]
        d1, q1 = frames[i]
        turned = quat_angle(q0, q1)
        moved = _angle_between(d0, d1)
        if turned > jump and moved < still:
            out.append(i)
    return out


def worst_spin(frames, still=STILL):
    """The largest orientation change on a frame where the bone barely moved.

    The number to report after a repair: it should be a few degrees. If it is
    still near 180 the flips were not what was wrong.
    """
    worst = 0.0
    for i in range(1, len(frames)):
        d0, q0 = frames[i - 1]
        d1, q1 = frames[i]
        if _angle_between(d0, d1) < still:
            worst = max(worst, quat_angle(q0, q1))
    return worst
