"""The grip geometry, checked without Blender.

Every case here is a mistake that actually happened while building this, kept as
a test so it cannot happen twice.
"""
import math

from score2motion.hand.grip import (Cylinder, far_side, fit_cylinder,
                                    nail_along_axis, solve_dial, thumb_goal,
                                    wrap_angle)


def ring(centre, axis_pt, radius, length, n=48, m=9):
    """Points on a cylinder of the given radius, along centre -> axis_pt."""
    a = axis_pt
    n0 = (0.0, 0.0, 1.0) if abs(a[2]) < 0.9 else (1.0, 0.0, 0.0)
    u = (a[1] * n0[2] - a[2] * n0[1], a[2] * n0[0] - a[0] * n0[2],
         a[0] * n0[1] - a[1] * n0[0])
    ul = math.sqrt(sum(c * c for c in u))
    u = tuple(c / ul for c in u)
    v = (a[1] * u[2] - a[2] * u[1], a[2] * u[0] - a[0] * u[2],
         a[0] * u[1] - a[1] * u[0])
    pts = []
    for i in range(m):
        t = -length / 2.0 + length * i / (m - 1.0)
        for j in range(n):
            th = 2.0 * math.pi * j / n
            pts.append(tuple(centre[k] + a[k] * t
                             + radius * (math.cos(th) * u[k]
                                         + math.sin(th) * v[k])
                             for k in range(3)))
    return pts


def test_axis_is_the_real_one_not_the_nearest_world_axis():
    # A cylinder leaning 65 degrees, like a mic on a boom. Picking whichever
    # world axis spans most would answer "straight up"; the fit must not.
    a = (0.0, math.sin(math.radians(65)), math.cos(math.radians(65)))
    cyl = fit_cylinder(ring((0.1, 0.2, 1.4), a, 0.018, 0.15))
    assert abs(abs(cyl.axis[0] * a[0] + cyl.axis[1] * a[1]
                   + cyl.axis[2] * a[2]) - 1.0) < 1e-3
    assert abs(cyl.radius - 0.018) < 5e-4


def test_radius_is_measured_at_the_station_not_over_the_whole_object():
    # A thin neck with a fat head on the end: held at the neck, the radius that
    # matters is the neck's. Averaged over everything it comes out far too big.
    neck = ring((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.018, 0.15)
    head = ring((0.0, 0.0, 0.105), (0.0, 0.0, 1.0), 0.036, 0.06)
    at_neck = fit_cylinder(neck + head, station=0.15)
    whole = sum(math.hypot(p[0], p[1]) for p in neck + head) / len(neck + head)
    assert abs(at_neck.radius - 0.018) < 3e-3
    assert at_neck.radius < whole - 0.004


def test_a_point_past_the_end_is_not_inside():
    # A hand is wider than a short object, so a fingertip hangs off the end.
    # Judged against an endless cylinder it reads as deeply buried, and a grip
    # solver will spend itself pulling a finger out of a solid that is not there.
    cyl = fit_cylinder(ring((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.02, 0.08))
    inside = (0.0, 0.0, 0.0)
    past = (0.0, 0.0, 0.30)
    assert cyl.beside(inside)
    assert not cyl.beside(past)
    assert cyl.depth([past]) is None
    assert cyl.depth([inside]) is not None


def test_wrap_lands_the_far_end_on_the_surface():
    cyl = Cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.02, -0.2, 0.2)
    head = (0.05, 0.0, 0.0)
    tail = (0.05, 0.03, 0.0)          # bone running tangentially
    rho = 0.028
    phi, ok = wrap_angle(head, tail, cyl, rho, flex=1.0)
    assert ok
    c, s = math.cos(phi), math.sin(phi)
    d = (tail[0] - head[0], tail[1] - head[1], tail[2] - head[2])
    rot = (d[0] * c - d[1] * s, d[0] * s + d[1] * c, d[2])
    end = (head[0] + rot[0], head[1] + rot[1], head[2] + rot[2])
    assert abs(math.hypot(end[0], end[1]) - rho) < 1e-6


def test_unreachable_bone_points_at_the_centre_line_not_past_it():
    # Far too short to touch. It must aim inward, its closest approach -- not
    # swing by its full joint limit, which throws it past the object entirely.
    cyl = Cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.02, -0.2, 0.2)
    head = (0.20, 0.0, 0.0)
    tail = (0.20, 0.01, 0.0)
    phi, ok = wrap_angle(head, tail, cyl, 0.028, flex=1.0)
    assert not ok
    c, s = math.cos(phi), math.sin(phi)
    d = (0.0, 0.01, 0.0)
    rot = (d[0] * c - d[1] * s, d[0] * s + d[1] * c, d[2])
    end = (head[0] + rot[0], head[1] + rot[1], head[2] + rot[2])
    assert math.hypot(end[0], end[1]) < math.hypot(head[0], head[1])


def test_dial_settles_where_the_skin_is_pressed_in():
    press = 0.001
    # deeper the further it closes, as a real finger is
    def measure(t):
        return -0.05 + 0.06 * t
    t = solve_dial(measure, press)
    assert abs(measure(t) - press) < 5e-4


def test_dial_does_not_stop_at_zero_when_the_finger_starts_touching():
    # The trap: measuring the whole finger instead of its tip. The base is
    # already against the object, so the reading looks satisfied at t=0 and the
    # hand is reported gripping while still wide open.
    def tip_only(t):
        return -0.05 + 0.06 * t
    assert solve_dial(tip_only, 0.001) > 0.5


def test_dial_can_open_a_finger_that_starts_buried():
    def measure(t):
        return 0.02 + 0.06 * t        # already 20mm in at t=0
    t = solve_dial(measure, 0.001)
    assert t < 0.0


def test_dial_reports_the_ceiling_when_nothing_can_be_reached():
    assert solve_dial(lambda t: -0.05, 0.001) == 1.35
    assert solve_dial(lambda t: None, 0.001) == 1.35


# --------------------------------------------------------------- the thumb


def test_thumb_goes_to_the_side_the_fingers_are_not_on():
    cyl = Cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.02, -0.1, 0.1)
    tips = [(0.03, 0.005, 0.0), (0.03, 0.0, 0.0), (0.03, -0.005, 0.0),
            (0.028, -0.01, 0.0)]
    far = far_side(cyl, tips)
    assert far[0] < -0.9                      # fingers on +x, thumb goes to -x


def test_thumb_may_slide_but_never_round_onto_the_fingers_side():
    # Left free it walks back to the fingers and the grip stops opposing:
    # measured, it drifted 93 degrees and shared a side with index and pinky.
    cyl = Cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.02, -0.1, 0.1)
    tips = [(0.03, 0.0, 0.0)]
    goal = thumb_goal(cyl, tips, (0.03, 0.001, 0.0), 0.028, max_off=50.0)
    u = (goal[0], goal[1], 0.0)
    n = math.hypot(u[0], u[1])
    ang = math.degrees(math.acos(max(-1.0, min(1.0, -u[0] / n))))
    assert ang <= 50.5


def test_thumb_goal_lands_on_the_surface_and_beside_the_object():
    cyl = Cylinder((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 0.02, -0.1, 0.1)
    goal = thumb_goal(cyl, [(0.03, 0.0, 0.0)], (-0.05, 0.0, 0.4), 0.028)
    assert abs(cyl.distance(goal) - 0.028) < 1e-9
    assert cyl.beside(goal)                    # clamped back onto the object


def test_thumb_nail_lies_as_near_the_object_as_a_nail_can():
    # A finger's nail points straight out, 90 degrees to the centre line. The
    # thumb's leans toward lying ALONG the object, because gripping something
    # that fits exactly puts thumb tip and index tip together, and that needs
    # their pads facing each other.
    #
    # It cannot literally lie along the axis: a nail is across its own bone, so
    # a thumb bone running near-parallel to the object has no nail direction
    # near-parallel to it either. The contract is the closest one available.
    axis = (0.0, 0.0, 1.0)
    for bone in ((0.3, 0.0, 0.9), (1.0, 0.0, 0.35), (0.6, 0.4, 0.7)):
        nail = nail_along_axis(bone, axis)
        assert abs(sum(nail[i] * bone[i] for i in range(3))) < 1e-9
        best = abs(nail[2])
        for k in range(360):                   # no direction across the bone
            th = math.radians(k)               # gets nearer the centre line
            u = [b / math.sqrt(sum(x * x for x in bone)) for b in bone]
            n0 = (0.0, 0.0, 1.0) if abs(u[2]) < 0.9 else (1.0, 0.0, 0.0)
            e1 = [u[1] * n0[2] - u[2] * n0[1], u[2] * n0[0] - u[0] * n0[2],
                  u[0] * n0[1] - u[1] * n0[0]]
            m = math.sqrt(sum(x * x for x in e1))
            e1 = [x / m for x in e1]
            e2 = [u[1] * e1[2] - u[2] * e1[1], u[2] * e1[0] - u[0] * e1[2],
                  u[0] * e1[1] - u[1] * e1[0]]
            cand = [math.cos(th) * e1[i] + math.sin(th) * e2[i] for i in range(3)]
            assert abs(cand[2]) <= best + 1e-9


def test_thumb_nail_is_undefined_when_the_bone_lies_along_the_axis():
    assert nail_along_axis((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)) is None
