"""Cases that a roll-flip detector has to get right, all of them from a real take.

The failure this exists for: a retargeted wrist changing orientation by 179.9
degrees between two frames while the bone moved 0.9. The traps around it are
what make the test narrow rather than a plain "did it turn a lot":

  a fast but REAL turn moves the bone's direction too, and must survive
  a quaternion and its negative are the SAME rotation, so a naive comparison
  calls half the frames of any take a flip
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from score2motion.motion.roll_flips import find_flips, quat_angle, worst_spin  # noqa: E402


def q_axis(axis, deg):
    """Quaternion (w, x, y, z) for a turn about an axis."""
    n = math.sqrt(sum(c * c for c in axis))
    axis = [c / n for c in axis]
    h = math.radians(deg) / 2.0
    s = math.sin(h)
    return (math.cos(h), axis[0] * s, axis[1] * s, axis[2] * s)


def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


DIR = (0.0, 1.0, 0.0)          # a bone pointing along +Y


def test_a_roll_flip_is_found():
    # same direction both frames, orientation half a turn about the bone
    frames = [(DIR, q_axis((0, 1, 0), 0.0)),
              (DIR, q_axis((0, 1, 0), 180.0))]
    assert find_flips(frames) == [1]


def test_a_real_turn_is_left_alone():
    # the bone swings 40 degrees: orientation AND direction both change
    a = (0.0, 1.0, 0.0)
    b = (math.sin(math.radians(40)), math.cos(math.radians(40)), 0.0)
    frames = [(a, q_axis((0, 0, 1), 0.0)), (b, q_axis((0, 0, 1), 40.0))]
    assert find_flips(frames) == []


def test_the_negated_quaternion_is_not_a_flip():
    # q and -q are the same orientation; a detector that misses this calls
    # roughly half the frames of any take a flip
    q = q_axis((0.3, 0.5, 0.8), 37.0)
    neg = tuple(-c for c in q)
    assert quat_angle(q, neg) < 1e-6
    assert find_flips([(DIR, q), (DIR, neg)]) == []


def test_slow_roll_is_not_a_flip():
    frames = [(DIR, q_axis((0, 1, 0), d)) for d in range(0, 60, 5)]
    assert find_flips(frames) == []


def test_several_flips_are_all_found():
    frames = []
    for i in range(10):
        roll = 180.0 if i in (3, 7) else 0.0
        frames.append((DIR, q_axis((0, 1, 0), roll)))
    assert find_flips(frames) == [3, 4, 7, 8]


def test_worst_spin_reports_what_is_left():
    clean = [(DIR, q_axis((0, 1, 0), d)) for d in (0, 3, 6, 9)]
    assert worst_spin(clean) < 5.0
    dirty = [(DIR, q_axis((0, 1, 0), 0.0)), (DIR, q_axis((0, 1, 0), 180.0))]
    assert worst_spin(dirty) > 170.0


def test_empty_and_single_frame_do_not_crash():
    assert find_flips([]) == []
    assert find_flips([(DIR, q_axis((0, 1, 0), 0.0))]) == []


def test_a_run_of_flipped_frames_reports_only_its_edges():
    """Frames 3,4,5 all flipped: nothing changes BETWEEN them.

    Written after expecting [3,4,5,6] from a real take and getting [3,6] --
    the code was right. A detector reports where the roll breaks; a repair
    walks the take in order against frames it has already corrected, which is
    how the middle of a run gets undone.
    """
    frames = []
    for i in range(9):
        roll = 180.0 if i in (3, 4, 5) else 0.0
        frames.append((DIR, q_axis((0, 1, 0), roll)))
    assert find_flips(frames) == [3, 6]
