"""The grip geometry, checked without Blender.

Every case here is a mistake that actually happened while building this, kept as
a test so it cannot happen twice.
"""
import math

from score2motion.hand.grip import (Cylinder, far_side, first_contact,
                                    fit_cylinder, solve_dial, wrap_angle)


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
    assert far_side(cyl, tips)[0] < -0.9         # fingers +x, so the thumb -x


def test_first_contact_stops_where_the_thumb_first_meets_something():
    # closes steadily: 40mm short at t=0, into the thing by t=1
    def gap(t):
        return 0.04 - 0.05 * t
    t = first_contact(gap, press=0.001)
    assert abs(gap(t) + 0.001) < 2e-3


def test_first_contact_does_not_miss_a_touch_it_swings_past():
    # THE trap: a thumb can pass through a contact and come out the far side, so
    # the gap is not monotone in t. Bisecting straight away reported "nothing
    # within reach" for a thumb that was 7mm inside the object half way closed.
    def gap(t):
        return abs(t - 0.5) * 0.10 - 0.008       # only touches around t=0.5
    t = first_contact(gap, press=0.001)
    assert gap(t) <= -0.001
    assert 0.3 < t < 0.5                         # the FIRST crossing, not the last


def test_first_contact_leaves_the_thumb_out_when_nothing_is_reachable():
    # An object too big to get a hand round. The thumb should stay extended at
    # its closest approach, not curl into thin air.
    def gap(t):
        return 0.05 - 0.02 * t                   # never reaches zero
    t = first_contact(gap, press=0.001)
    assert t == 1.4                              # as close as it can get


def test_first_contact_returns_the_closest_approach_it_found():
    def gap(t):
        return 0.02 + (t - 0.7) ** 2             # nearest at t=0.7, never touches
    t = first_contact(gap, press=0.001)
    assert abs(t - 0.7) < 0.05


def test_first_contact_digs_a_buried_thumb_out_on_a_fat_object():
    # On an object fatter than the measured one the thumb starts INSIDE, before
    # it has closed at all -- 12.6mm through a 30mm cylinder, measured. Reading
    # that first sample as "contact" stops the search on step one and leaves it
    # buried. Closing is what frees it: the curl that wraps a thin object lifts
    # the thumb off a thick one.
    def gap(t):
        return -0.0126 + 0.025 * t               # inside at t=0, out by t=0.6
    t = first_contact(gap, press=0.001)
    assert abs(gap(t) + 0.001) < 2e-4            # settles AT the press depth
    assert 0.4 < t < 0.5


def test_first_contact_takes_the_least_buried_when_it_never_surfaces():
    # Wedged: no closure gets it out, so take the shallowest rather than the
    # first sample, which is the deepest of them.
    def gap(t):
        return -0.03 + 0.01 * t                  # still 16mm in when fully closed
    t = first_contact(gap, press=0.001)
    assert t == 1.4
