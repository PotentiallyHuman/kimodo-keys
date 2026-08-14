"""The four movable barre shapes, from theory, playable anywhere on the neck.

Capturing chord grips in VR did not work -- lining a hand up against a neck he
can barely see, at an angle that also shows the fingertips, is not something the
headset can give. Theory can: these four shapes are exact, and every chord a
guitarist plays with a barre is one of them slid to a fret.

    E  shape   root on the low E (our string0), all six strings
    Em shape   the same, minus the middle finger -- that is the whole difference
    A  shape   root on the A (our string1), five strings, low E muted
    Am shape   the same, with the middle finger added on the B string

Fingers: 1 index (the barre), 2 middle, 3 ring, 4 pinky.
Strings here are OURS: 0 = low E ... 5 = high E, per the naming contract. A
guitarist counts the other way (their 6 is the low E), so every shape below is
written once in their numbers as a comment and once in ours as data.

FRET SPACING IS NOT UNIFORM -- each fret is about 5.6% shorter than the last, so
a shape at fret 10 is physically much narrower than the same shape at fret 1.
Nothing here stores millimetres; a shape is fret NUMBERS, and the geometry turns
them into positions using the instrument's own fret table.
"""

# offsets are FRETS ABOVE THE BARRE, so a shape works at any root fret
SHAPES = {
    "E": {
        "name": "E shape (major)", "root_string": 0, "lowest_string": 0,
        # theirs: barre 6-1, middle G(3)+1, ring A(5)+2, pinky D(4)+2
        "barre": {"finger": 1, "from_string": 0, "to_string": 5, "fret_off": 0},
        "fingers": [{"finger": 3, "string": 1, "fret_off": 2},
                    {"finger": 4, "string": 2, "fret_off": 2},
                    {"finger": 2, "string": 3, "fret_off": 1}],
        "muted": [],
    },
    "Em": {
        "name": "Em shape (minor)", "root_string": 0, "lowest_string": 0,
        # E shape with the middle finger LIFTED -- that one note is the third
        "barre": {"finger": 1, "from_string": 0, "to_string": 5, "fret_off": 0},
        "fingers": [{"finger": 3, "string": 1, "fret_off": 2},
                    {"finger": 4, "string": 2, "fret_off": 2}],
        "muted": [],
    },
    "A": {
        "name": "A shape (major)", "root_string": 1, "lowest_string": 1,
        # theirs: barre 5-1, then D(4) G(3) B(2) all at +2 -- usually one finger
        "barre": {"finger": 1, "from_string": 1, "to_string": 5, "fret_off": 0},
        "fingers": [{"finger": 3, "string": 2, "fret_off": 2},
                    {"finger": 3, "string": 3, "fret_off": 2},
                    {"finger": 3, "string": 4, "fret_off": 2}],
        "muted": [0],
    },
    "Am": {
        "name": "Am shape (minor)", "root_string": 1, "lowest_string": 1,
        # theirs: barre 5-1, ring D(4)+2, pinky G(3)+2, middle B(2)+1
        "barre": {"finger": 1, "from_string": 1, "to_string": 5, "fret_off": 0},
        "fingers": [{"finger": 3, "string": 2, "fret_off": 2},
                    {"finger": 4, "string": 3, "fret_off": 2},
                    {"finger": 2, "string": 4, "fret_off": 1}],
        "muted": [0],
    },
}
OPEN = [40, 45, 50, 55, 59, 64]          # our string0..string5, standard tuning


def voicing(shape, root_fret):
    """Every (string, fret) this shape sounds when barred at root_fret."""
    sh = SHAPES[shape]
    out = {}
    b = sh["barre"]
    for s in range(b["from_string"], b["to_string"] + 1):
        out[s] = root_fret + b["fret_off"]
    for f in sh["fingers"]:
        out[f["string"]] = root_fret + f["fret_off"]
    return out


def pitches(shape, root_fret):
    return sorted(OPEN[s] + f for s, f in voicing(shape, root_fret).items())


def finger_of(shape, string, root_fret):
    """Which finger holds a given string in this shape."""
    sh = SHAPES[shape]
    for f in sh["fingers"]:
        if f["string"] == string:
            return f["finger"]
    b = sh["barre"]
    if b["from_string"] <= string <= b["to_string"]:
        return b["finger"]
    return None


def match(notes, max_fret=12):
    """Which shape and fret best plays this set of MIDI pitches.

    Scored on the notes it actually has to sound: a shape that covers every one
    of them wins, and among those the lowest position wins, because a guitarist
    plays the nearest voicing rather than reaching up the neck for the same
    chord. Returns None when nothing covers them -- better to say so than to
    return a shape that leaves notes out.
    """
    want = set(notes)
    best = None
    for shape in SHAPES:
        for fret in range(1, max_fret + 1):
            have = set(pitches(shape, fret))
            if not want <= have:
                continue
            extra = len(have - want)
            score = (fret, extra)
            if best is None or score < best[0]:
                best = (score, shape, fret)
    if best is None:
        return None
    _, shape, fret = best
    return {"shape": shape, "root_fret": fret,
            "name": SHAPES[shape]["name"],
            "voicing": voicing(shape, fret),
            "muted": SHAPES[shape]["muted"],
            "covers": sorted(want)}


if __name__ == "__main__":
    import json
    import sys
    print("the four shapes, barred at fret 5, as (our string: fret)")
    for k in SHAPES:
        v = voicing(k, 5)
        print(f"  {SHAPES[k]['name']:20s} {v}  muted {SHAPES[k]['muted']}")
    if len(sys.argv) > 1:
        plan = json.load(open(sys.argv[1]))["notes"]
        groups = {}
        for n in plan:
            groups.setdefault(round(n["t_on"], 3), []).append(n)
        print("\nmatching the chords in that plan:")
        for t in sorted(groups):
            ns = [n["note"] for n in groups[t]]
            m = match(ns)
            if m:
                print(f"  t={t:5.2f} notes {sorted(ns)} -> {m['name']} "
                      f"barred at fret {m['root_fret']}")
            else:
                print(f"  t={t:5.2f} notes {sorted(ns)} -> no barre shape covers this")
