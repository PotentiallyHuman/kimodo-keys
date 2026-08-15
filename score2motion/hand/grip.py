"""Close a skeleton hand around anything round, without losing the roll.

The geometry, with no Blender in it, so it can be tested and argued with.

A grip is three questions:

  1. where is the object's centre line, and how thick is it WHERE it is held
  2. how far must each bone turn for its far end to land on that surface
  3. how far to close the whole finger, given that skin is not where bone is

The one idea worth taking away: every bone is turned about an axis PARALLEL TO
THE OBJECT'S CENTRE LINE, and nothing else. A rotation about that axis cannot
change how a bone is rolled, so a hand whose fingernail directions were
calibrated once, flat, still has them pointing out of the back of each finger
after it closes. Wrap a cylinder and the nails end up radiating out of it. That
is not arranged afterwards; it falls out of the choice of rotation axis, and it
is what makes the closed hand usable rather than merely touching.

Prior art: AutoGrip (github.com/Jetpack-Crow/autogrip, MIT) solves the same
problem for MakeHuman, Rigify and Auto-Rig Pro rigs with shrinkwrap constraints
on extra control bones. It handles arbitrary meshes, which this does not. It
also adds bones to the rig, and its IK constraints carry no pole target, so a
finger's twist is whatever the solver picks -- there is nothing in it that knows
which way a nail faces. The per-finger grip dial here is its idea.
"""
import math

__all__ = ["fit_cylinder", "Cylinder", "wrap_angle", "solve_dial"]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _norm(a):
    n = math.sqrt(_dot(a, a))
    return (0.0, 0.0, 0.0) if n < 1e-12 else (a[0] / n, a[1] / n, a[2] / n)


def _len(a):
    return math.sqrt(_dot(a, a))


class Cylinder(object):
    """A centre line, a direction, a radius, and where the object stops.

    `lo` and `hi` are how far the object reaches along `axis` from `centre`.
    They matter: an object is not an endless cylinder, and a fingertip past the
    end of one is not inside it, however small its distance to the centre line.
    """

    __slots__ = ("centre", "axis", "radius", "lo", "hi")

    def __init__(self, centre, axis, radius, lo, hi):
        self.centre, self.axis, self.radius = centre, axis, radius
        self.lo, self.hi = lo, hi

    def radial(self, p):
        """The part of `p` that is across the centre line, not along it."""
        v = _sub(p, self.centre)
        return _sub(v, _mul(self.axis, _dot(v, self.axis)))

    def distance(self, p):
        return _len(self.radial(p))

    def beside(self, p, slack=0.004):
        """Is this point alongside the object, or past the end of it?"""
        t = _dot(_sub(p, self.centre), self.axis)
        return self.lo - slack <= t <= self.hi + slack

    def depth(self, points, slack=0.004):
        """How far the deepest of these points is INSIDE the surface.

        Points past the ends are ignored. Returns None if none are beside the
        object at all -- which is a real answer, meaning "nothing to touch",
        and is not the same as "touching at zero".
        """
        near = [p for p in points if self.beside(p, slack)]
        if not near:
            return None
        return self.radius - min(self.distance(p) for p in near)


def fit_cylinder(points, station=0.5, band=None):
    """Fit a centre line to `points`, and take the radius AT one station along it.

    `station` is 0..1 along the object; 0.5 is halfway. The radius is averaged
    over a band of points around that station rather than over the whole object,
    because real instruments are not one thickness end to end -- measured over a
    whole microphone the answer is the fat head, not the neck a hand holds.

    The direction is the true principal axis, found by power iteration on the
    covariance. Taking whichever of the object's own X, Y and Z axes spans the
    most is NOT the same thing and quietly disagrees on anything mounted at an
    angle: on a microphone hanging off a boom it returned straight up for an
    object whose real axis was 65 degrees off vertical.
    """
    pts = list(points)
    if len(pts) < 3:
        raise ValueError("need at least 3 points to fit a cylinder")
    n = float(len(pts))
    c = (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n,
         sum(p[2] for p in pts) / n)

    a = _norm((0.3, 0.4, 0.9))
    for _ in range(64):
        acc = (0.0, 0.0, 0.0)
        for p in pts:
            d = _sub(p, c)
            acc = _add(acc, _mul(d, _dot(d, a)))
        if _len(acc) < 1e-14:
            break
        nxt = _norm(acc)
        if 1.0 - abs(_dot(nxt, a)) < 1e-14:
            a = nxt
            break
        a = nxt

    ts = [_dot(_sub(p, c), a) for p in pts]
    lo, hi = min(ts), max(ts)
    at = lo + (hi - lo) * station
    width = band if band is not None else max(0.010, (hi - lo) * 0.15)
    rs = [_len(_sub(_sub(p, c), _mul(a, t)))
          for p, t in zip(pts, ts) if abs(t - at) < width]
    if not rs:
        rs = [_len(_sub(_sub(p, c), _mul(a, t))) for p, t in zip(pts, ts)]
    centre = _add(c, _mul(a, at))
    return Cylinder(centre, a, sum(rs) / len(rs), lo - at, hi - at)


def wrap_angle(head, tail, cyl, rho, flex):
    """Turn about the centre line to land the bone's far end at radius `rho`.

    Returns (angle, reachable). Working in the plane across the centre line, the
    head sits at `h` and the bone spans `d`; turning `d` about the axis by phi
    puts the tail at |h + R(phi)d|, and asking for that to equal rho gives

        cos(psi + phi) = (rho^2 - |h|^2 - |d|^2) / (2 |h| |d|)

    with psi the angle from h to d. Two solutions come back; the one taken is
    the one that CLOSES the hand, which is what `flex` (+1/-1) selects.

    When the bone cannot reach -- the right-hand side outside [-1, 1] -- it is
    turned to point straight at the centre line, its closest approach. Turning
    it by the joint's full limit instead, which is the obvious fallback, swings
    a short finger clean past a small object and leaves it facing backwards: a
    little finger did exactly that and ended up with its nail 171 degrees wrong.
    """
    h = cyl.radial(head)
    d = _sub(cyl.radial(tail), h)
    lh, ld = _len(h), _len(d)
    if lh < 1e-9 or ld < 1e-9:
        return 0.0, False
    psi = math.atan2(_dot(_cross(h, d), cyl.axis), _dot(h, d))
    k = (rho * rho - lh * lh - ld * ld) / 2.0
    den = lh * ld
    if abs(k) > den:
        phi = (math.pi if lh > rho else 0.0) - psi
        return (phi + math.pi) % (2.0 * math.pi) - math.pi, False
    want = math.acos(max(-1.0, min(1.0, k / den)))
    cands = [((c + math.pi) % (2.0 * math.pi) - math.pi)
             for c in (want - psi, -want - psi)]
    closing = [c for c in cands if c * flex > 0]
    return min(closing or cands, key=abs), True


def solve_dial(measure, press, lo=-0.6, hi=1.35, steps=18):
    """Find how far to close one finger, by looking at where its SKIN ends up.

    `measure(t)` closes the finger to t and returns how deep its skin is inside
    the surface, or None if it has nothing to touch. `press` is how far inside
    to settle -- a millimetre or so, because skin and flesh compress and a hand
    that merely grazes an object reads as a hand hovering beside it.

    Three things this shape gets right, each of which was got wrong first:

    * ONE dial for the whole finger, not a correction per bone. Correcting bone
      by bone needs to know which way each one closes, and that test inverts the
      moment a finger passes square -- the correction then drives the fingertips
      deeper instead of backing them out.
    * measure the FINGERTIP. Anything nearer the palm is already against the
      object once the palm is on it, so a reading taken over the whole finger is
      satisfied before the finger has moved and the dial settles at zero with
      the hand still wide open.
    * `lo` is negative: a finger that starts inside the object has to be able to
      open away from it, not only close.

    The ceiling stays near 1. Raised to help thin objects it lets a joint turn
    past square and folds the finger over on itself.
    """
    at_hi = measure(hi)
    if at_hi is None or at_hi < press:
        return hi                      # never reaches, even fully closed
    at_lo = measure(lo)
    if at_lo is not None and at_lo > press:
        return lo                      # buried even fully open
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        d = measure(mid)
        if d is None or d < press:
            lo = mid
        else:
            hi = mid
    return hi
