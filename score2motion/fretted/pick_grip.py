"""Where a held pick is, in one place, so nothing can drift out of agreement.

The standard pick grip: the hand opens, the index CLAWS shut, and the thumb tip
comes down on the SIDE of the index's last joint -- the side facing the thumb.
The pick is trapped at that crossing. Every published account of the grip says
the same thing, and adds the number: only 3-6mm of the point should show.

And it POINTS THE WAY THE PALM FACES -- out of the crossing, away from the palm,
which is what carries it into the strings. Pointing it "along the hand" left the
tip trailing the fingers instead of leading them.

The consequence is the whole reason this file exists: THE INDEX FINGERTIP NEVER
TOUCHES A STRING. It curls past the pick and points back at the palm. Anything
that measures picking from the fingertips is measuring the wrong object.

    clamp   midway between the thumb tip and the index DIP joint
    normal  the thumb->index line: the pick is SQUEEZED along it, so its flat
            faces lie across it
    out     the way the palm faces, across that normal: where the point goes
    tip     clamp + out * clamp_to_tip

Which way the palm faces is never assumed. The fingers curl TOWARDS the palm, so
whichever side the curled middle fingertip is on is the palm side.

`clamp_to_tip` is the grip's one free number -- gripping flesh plus the showing
tip. It defaults to 13mm and is replaced by the measured value as soon as the
PICK ON STRING calibration stage has been recorded (see CAL_PATH).
"""
import json
import os

import numpy as np

# Where the measured calibration lives, if there is one. Resolved from the
# environment rather than hard-coded, so nothing in this file names a machine.
CAL_PATH = os.environ.get("PICK_GRIP_CAL",
                          os.path.join(os.path.expanduser("~"), ".cache",
                                       "score2motion", "pick_grip_cal.json"))
PICK_L, PICK_W = 0.026, 0.022                     # a real pick, life size
# The round end sits AT the crossing and the whole length sticks out past it,
# so the point is one pick-length away. The PICK ON STRING calibration replaces
# this with what a measured hand measures.
DEFAULT_CLAMP_TO_TIP = PICK_L

# the joints the grip is built from, in WebXR names and in canon bone terms.
# (metacarpal1 is the INDEX, 2 the middle, 3 the ring, 4 the pinky -- checked
# against the rig's own parenting, not guessed from the numbering.)
WEBXR = {"thumb_tip": "thumb-tip",
         "index_dip": "index-finger-phalanx-distal",
         "wrist": "wrist",
         "index_mc": "index-finger-metacarpal",
         "pinky_mc": "pinky-finger-metacarpal",
         "mid_mc": "middle-finger-metacarpal",
         "mid_tip": "middle-finger-tip"}
CANON = {"thumb_tip": ("finger1-3.R", "tail"),
         "index_dip": ("finger2-3.R", "head"),
         "wrist": ("wrist.R", "head"),
         "index_mc": ("metacarpal1.R", "head"),
         "pinky_mc": ("metacarpal4.R", "head"),
         "mid_mc": ("metacarpal2.R", "head"),
         "mid_tip": ("finger3-3.R", "tail")}


def clamp_to_tip():
    """The measured protrusion if the calibration has been run, else the default."""
    if os.path.exists(CAL_PATH):
        try:
            v = json.load(open(CAL_PATH)).get("clamp_to_tip_m")
            if v and 0.004 < float(v) < 0.040:
                return float(v)
        except (ValueError, KeyError, json.JSONDecodeError):
            pass
    return DEFAULT_CLAMP_TO_TIP


def _n(v):
    v = np.asarray(v, dtype=float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def palm_facing(g):
    """The direction the palm looks, from the hand itself."""
    w = np.asarray(g["wrist"], dtype=float)
    n = np.cross(np.asarray(g["index_mc"], dtype=float) - w,
                 np.asarray(g["pinky_mc"], dtype=float) - w)
    n = _n(n)
    curl = (np.asarray(g["mid_tip"], dtype=float)
            - np.asarray(g["mid_mc"], dtype=float))
    return -n if float(curl @ n) < 0 else n      # fingers curl palm-wards


def grip_frame(g):
    """clamp point and the pick's own axes, from the tracked joints.

    The pick lies ALONG the palm vector: round end trapped at the crossing,
    point running away from the palm. Squaring it against the thumb->index line
    first kept twisting it -- when the grip closes those two directions nearly
    line up, and what survives the projection is noise.
    """
    tt = np.asarray(g["thumb_tip"], dtype=float)
    dip = np.asarray(g["index_dip"], dtype=float)
    clamp = (tt + dip) / 2.0
    out = palm_facing(g)
    # the flat faces are squeezed between thumb and finger, so the thickness
    # runs along whatever of that line is square to the palm vector
    nrm = tt - dip
    nrm = nrm - out * float(nrm @ out)
    if float(nrm @ nrm) < 1e-12:
        nrm = np.asarray(g["mid_mc"], dtype=float) - np.asarray(g["wrist"], dtype=float)
    nrm = _n(nrm)
    up = _n(np.cross(nrm, out))
    return clamp, out, up, nrm


def pick_tip(g, reach=None):
    """The point of the pick -- the only part of the hand that touches a string."""
    clamp, out, _, _ = grip_frame(g)
    return clamp + out * (clamp_to_tip() if reach is None else reach)


def from_webxr(hand, joint_index):
    """Pull the grip joints out of one logged hand row. None if untracked."""
    got = {}
    for key, name in WEBXR.items():
        e = hand[joint_index[name]]
        if not e:
            return None
        got[key] = np.array(e[:3], dtype=float)
    return got


def from_canon(pb, world):
    """Pull the grip joints off a posed armature."""
    return {k: np.array(world @ (pb[b].tail if e == "tail" else pb[b].head))
            for k, (b, e) in CANON.items()}


def solve_clamp_to_tip(frames, to_string, lo=0.004, hi=0.040):
    """How far the point reaches, from frames where the point rests ON a string.

    The tip is somewhere along `out`; the string it is touching is at a known
    place. So the reach that puts the tip ON the string is just the distance
    that minimises tip-to-string, solved per frame and taken as the median --
    a measurement, where before there was a guess.
    """
    best = []
    rs = np.linspace(lo, hi, 181)
    for g in frames:
        clamp, out, _, _ = grip_frame(g)
        d = [to_string(clamp + out * r) for r in rs]
        best.append(float(rs[int(np.argmin(d))]))
    if not best:
        return None, None
    best = np.array(best)
    return float(np.median(best)), float(np.percentile(np.abs(best - np.median(best)), 68))
