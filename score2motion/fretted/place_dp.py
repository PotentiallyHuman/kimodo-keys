"""Exact Viterbi DP over hand positions: globally optimal string/fret/finger.

Replaces per-note greedy placement with one exact, deterministic dynamic
program over (hand position, cell, kind, finger) states. No RNG anywhere;
every tie-break is defined; same input always gives the same performance.

STATE (per note) = (hand_fret R, cell (string,fret), kind, finger)
  mode is a pure function of R (OFPF_FROM below):
    R < OFPF_FROM   Simandl 1-2-4: in-box d=f-R in {0,1,2} -> finger {1,2,4}
    R >= OFPF_FROM  one-finger-per-fret: in-box d in {0,1,2,3} -> finger d+1
  kind 'in'    in-box cell
  kind 'f4x'   OFPF d=4 pinky stretch (does not exist in Simandl range)
  kind 'pivot' lone +-1-fret excursion that returns; its neighbours must be
               same-R non-pivot states; never the first/last note
  kind 'open'  fret 0, finger 0. DEFAULT MODE: generated ONLY when the pitch
               has no fretted cell (H7) -- an open string reads as a missed
               note on screen unless it is truly forced. R passes through.
  kind 'ghost' articulation ghost/dead: any cell, any R, finger 0

EMISSIONS: every playable (s, f=pitch-OPEN[s], 1<=f<=MAX_FRET) cell of the
note's pitch, expanded to all states above. Pitch fidelity is hard: no
clamps, no per-note octave moves; one global octave shift tried from
[0,+12,-12,+24,-24], first that fits the range wins; none -> loud failure.

COSTS (integers x100; lexicographic objective (total cost, sum of roots) --
the second term prefers low positions as a pure tie-break):
  shift (R change)   100 + 50*(|dR|-1); x0.30 across a musical seam
                     (IOI >= 1.5 x median IOI); x0.70 with a guide finger;
                     HARD GATE: forbidden unless IOI >= 0.150 + 0.025*|dR| s
  string cross       |ds| -> 0/25/60/100 (+40 per step beyond 3 strings);
                     waived for a joint-bar (same fret+finger, adjacent string)
  pivot +50 / f4x enter +80 (high R) or +400 (low R), +30/+150 per note
  finger micro       finger 3 +10, finger 4 +15 (doubled under 100 ms);
                     same-finger crawl +50
  --economy          opens generated for every open-eligible pitch except
                     rigid octave-pair members; an open covers 60 off the
                     following shift
  --honor-artic      slap -> low strings, pop -> high strings, as candidate
                     pruning that yields rather than break pitch or H7

Octave pairs (adjacent +-12 semitone notes) are tagged 'octave_shape' and
kept rigid under one hand position when the tuning is all-fourths, where the
octave is the lattice vector (+2 strings, +2 frets).

Usage:
  python3 -m score2motion.fretted.place_dp plan.json --out placed.json
  python3 -m score2motion.fretted.place_dp plan.json -i bass5 --out placed.json

plan.json = {"notes": [{"t_on", "t_off", "pitch", "vel"?, "artic"?}, ...]}
(the output of score2motion.fretted.cli / plan.build, or any transcription).
"""
import argparse
import json
import statistics
import sys

from .instrument import PRESETS

# configured by configure(); bass defaults so imports stay usable in tests
OPEN = [28, 33, 38, 43]
SNAME = "EADG"
MAX_FRET = 12
PITCH_LO, PITCH_HI = OPEN[0], OPEN[-1] + MAX_FRET
OFPF_FROM = 5                               # measured pose-library fact
SIMANDL_FINGER = {0: 1, 1: 2, 2: 4}
KIND_RANK = {"in": 0, "f4x": 1, "pivot": 2, "open": 3, "ghost": 4}
ALL_FOURTHS = True

NOTE_NAMES = "C C# D D# E F F# G G# A A# B".split()

# integer costs, x100
SHIFT_BASE, SHIFT_PER_EXTRA = 100, 50
SEAM_NUM, SEAM_DEN = 30, 100
GUIDE_NUM, GUIDE_DEN = 70, 100
CROSS = (0, 25, 60, 100)
PIVOT_COST = 50
F4X_ENTER_HI, F4X_NOTE_HI = 80, 30
F4X_ENTER_LO, F4X_NOTE_LO = 400, 150
FINGER3, FINGER4, CRAWL = 10, 15, 50
FAST_NOTE = 0.100
OPEN_FUNK_MUTE = 40
ECON_OPEN_COVER = 60

# physics gate: declared hand-speed cap
TMIN_BASE, TMIN_PER_FRET = 0.150, 0.025     # seconds
SEAM_IOI_FACTOR = 1.5


def configure(open_notes, max_fret):
    """Point the module at an instrument. Everything else derives from this."""
    global OPEN, SNAME, MAX_FRET, PITCH_LO, PITCH_HI, CROSS, ALL_FOURTHS
    OPEN = list(open_notes)
    MAX_FRET = int(max_fret)
    SNAME = "".join(NOTE_NAMES[p % 12][0] for p in OPEN)
    PITCH_LO, PITCH_HI = OPEN[0], OPEN[-1] + MAX_FRET
    base = [0, 25, 60, 100]
    while len(base) < len(OPEN):
        base.append(base[-1] + 40)
    CROSS = tuple(base)
    gaps = [b - a for a, b in zip(OPEN, OPEN[1:])]
    ALL_FOURTHS = all(g == 5 for g in gaps)


def mode_of(root):
    return "simandl124" if root < OFPF_FROM else "ofpf"


def state_key(st):
    root, s, f, kind, fg = st
    return (root, s, f, KIND_RANK[kind], fg)


def fretted_cells(pitch):
    return [(s, pitch - OPEN[s]) for s in range(len(OPEN))
            if 1 <= pitch - OPEN[s] <= MAX_FRET]


def open_cells(pitch):
    return [(s, 0) for s in range(len(OPEN)) if pitch == OPEN[s]]


def states_for(pitch, artic, is_first, is_last, paired, opts):
    """All legal (R, s, f, kind, finger) states for one note, sorted.
    paired = note sits in an adjacent 12-semitone pair: its open states are
    pruned even in economy mode (a rigid octave shape is fretted) unless the
    pitch has no fretted home at all (pitch beats shape)."""
    out = []
    roots = range(1, MAX_FRET + 1)
    if artic in ("ghost", "dead"):
        for (s, f) in fretted_cells(pitch) + open_cells(pitch):
            for root in roots:
                out.append((root, s, f, "ghost", 0))
        return sorted(out, key=state_key)

    fr = fretted_cells(pitch)
    if opts.honor_artic and artic in ("slap", "pop"):
        half = len(OPEN) // 2
        allowed = set(range(half)) if artic == "slap" else \
            set(range(half, len(OPEN)))
        filt = [c for c in fr if c[0] in allowed]
        if filt:
            fr = filt                                    # else artic yields
    for (s, f) in fr:
        for root in roots:
            d = f - root
            if root < OFPF_FROM:                         # Simandl 1-2-4
                if d in SIMANDL_FINGER:
                    out.append((root, s, f, "in", SIMANDL_FINGER[d]))
                elif d == -1 and not (is_first or is_last):
                    out.append((root, s, f, "pivot", 1))
                elif d == 3 and not (is_first or is_last):
                    out.append((root, s, f, "pivot", 4))
            else:                                        # one finger per fret
                if 0 <= d <= 3:
                    out.append((root, s, f, "in", d + 1))
                elif d == 4:
                    out.append((root, s, f, "f4x", 4))
                elif d == -1 and not (is_first or is_last):
                    out.append((root, s, f, "pivot", 1))
    if not fr or (opts.economy and not paired):
        for (s, f) in open_cells(pitch):
            for root in roots:
                out.append((root, s, f, "open", 0))
    return sorted(out, key=state_key)


def emission(st, dur, opts):
    root, s, f, kind, fg = st
    c = 0
    if kind == "pivot":
        c += PIVOT_COST
    elif kind == "f4x":
        c += F4X_NOTE_HI if root >= 7 else F4X_NOTE_LO
    elif kind == "open" and opts.funk_mute:
        c += OPEN_FUNK_MUTE
    fast = dur < FAST_NOTE
    if fg == 3:
        c += FINGER3 * (2 if fast else 1)
    elif fg == 4:
        c += FINGER4 * (2 if fast else 1)
    return c


def transition(a, b, w, seam, opts):
    """Cost of moving state a -> b across an onset gap of w seconds.
    Returns None when the move is forbidden."""
    ra, sa, fa, ka, ga = a
    rb, sb, fb, kb, gb = b
    if ka == "pivot" or kb == "pivot":                   # excursion returns
        if ra != rb or (ka == "pivot" and kb == "pivot"):
            return None
    c = 0
    droot = abs(rb - ra)
    if droot:
        if w < TMIN_BASE + TMIN_PER_FRET * droot:        # hard time gate
            return None
        sc = SHIFT_BASE + SHIFT_PER_EXTRA * (droot - 1)
        if seam:
            sc = sc * SEAM_NUM // SEAM_DEN
        if sa == sb and ga == gb and ga > 0:
            sc = sc * GUIDE_NUM // GUIDE_DEN             # guide finger
        if opts.economy and ka == "open":
            sc = max(0, sc - ECON_OPEN_COVER)
        c += sc
    ds = abs(sb - sa)
    if ds:
        barred = (ds == 1 and droot == 0 and fa == fb and fa > 0
                  and ga == gb and ga > 0)               # joint-bar
        if not barred:
            c += CROSS[ds]
    if kb == "f4x" and not (ka == "f4x" and droot == 0):
        c += F4X_ENTER_HI if rb >= 7 else F4X_ENTER_LO
    if (droot == 0 and sa == sb and ga == gb and ga > 0 and fa != fb
            and ka != "pivot" and kb != "pivot"):
        c += CRAWL                                       # crawling
    return c


def solve(notes, opts):
    n = len(notes)
    if n == 0:
        sys.exit("empty plan: no notes to place")
    if n == 1:
        lay = states_for(notes[0]["pitch"], notes[0].get("artic"),
                         True, True, False, opts)
        if not lay:
            sys.exit(f"note 0 pitch {notes[0]['pitch']} has no playable cell")
        best = min(lay, key=lambda st: (emission(
            st, notes[0]["t_off"] - notes[0]["t_on"], opts), st[0],
            state_key(st)))
        return [best], (0, best[0]), 0.0, [], []
    iois = [notes[i]["t_on"] - notes[i - 1]["t_on"] for i in range(1, n)]
    med = statistics.median(iois)
    seam_at = [w >= SEAM_IOI_FACTOR * med for w in iois]  # edge i-1 -> i
    durs = [iois[i] if i < n - 1 else notes[i]["t_off"] - notes[i]["t_on"]
            for i in range(n)]

    paired = [(i > 0 and abs(notes[i]["pitch"] - notes[i - 1]["pitch"]) == 12)
              or (i < n - 1 and abs(notes[i]["pitch"] - notes[i + 1]["pitch"]) == 12)
              for i in range(n)]
    layers = [states_for(notes[i]["pitch"], notes[i].get("artic"),
                         i == 0, i == n - 1, paired[i], opts)
              for i in range(n)]
    for i, lay in enumerate(layers):
        if not lay:
            sys.exit(f"PITCH FAILURE: note {i} pitch {notes[i]['pitch']} has "
                     f"no playable cell in 0..{MAX_FRET} on {OPEN}")

    # dp[i][j] = (cost, rootsum, pred_index_in_layer_{i-1})
    dp = [[None] * len(lay) for lay in layers]
    for j, st in enumerate(layers[0]):
        dp[0][j] = (emission(st, durs[0], opts), st[0], -1)
    for i in range(1, n):
        w, seam = iois[i - 1], seam_at[i - 1]
        for j, b in enumerate(layers[i]):
            em = emission(b, durs[i], opts)
            best = None
            for k, a in enumerate(layers[i - 1]):        # sorted = tie order
                if dp[i - 1][k] is None:
                    continue
                tc = transition(a, b, w, seam, opts)
                if tc is None:
                    continue
                cand = (dp[i - 1][k][0] + tc + em, dp[i - 1][k][1] + b[0], k)
                if best is None or cand[:2] < best[:2]:  # strict: first wins
                    best = cand
            dp[i][j] = best
        if all(v is None for v in dp[i]):
            sys.exit(f"DEAD END at note {i} (pitch {notes[i]['pitch']}, IOI "
                     f"{w * 1000:.0f}ms): no transition satisfies the shift "
                     f"time gate / pivot rules"
                     + (" -- articulation filters (--honor-artic) may have "
                        "pruned the only reachable cells" if opts.honor_artic
                        else ""))

    end = min((dp[n - 1][j][:2], j) for j, st in enumerate(layers[n - 1])
              if dp[n - 1][j] is not None)
    total, j = end[0], end[1]
    path = [None] * n
    for i in range(n - 1, -1, -1):
        path[i] = layers[i][j]
        j = dp[i][j][2]
    return path, total, med, seam_at, iois


def assign_tags(path, notes):
    n = len(path)
    tags = []
    for i, (root, s, f, kind, fg) in enumerate(path):
        if kind == "open":
            tags.append("open")
        elif kind == "pivot":
            tags.append("pivot")
        elif kind == "f4x":
            tags.append("f4x")
        else:
            tags.append(mode_of(root))
    if not ALL_FOURTHS:
        return tags        # the (+2,+2) octave vector only exists in fourths
    for i in range(n - 1):
        a, b = path[i], path[i + 1]
        pa, pb = notes[i]["pitch"], notes[i + 1]["pitch"]
        if abs(pa - pb) != 12 or a[0] != b[0]:
            continue
        lo, hi = (i, i + 1) if pa < pb else (i + 1, i)
        (rl, sl, fl, kl, _), (rh, sh, fh, kh, _) = path[lo], path[hi]
        if kl == "in" and kh == "in" and sh == sl + 2 and fh == fl + 2:
            for x in (i, i + 1):
                if tags[x] in ("simandl124", "ofpf"):
                    tags[x] = "octave_shape"
    return tags


def lib_label(st):
    root, s, f, kind, fg = st
    if kind == "open":
        return "-"
    if kind == "f4x":
        return f"f4xs{s}"
    return f"f{fg}s{s}" + ("*" if kind == "pivot" else "")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="exact DP placement: plan.json -> string/fret/finger")
    ap.add_argument("plan", help='{"notes":[{t_on,t_off,pitch,...}]}')
    ap.add_argument("--instrument", "-i", default="bass",
                    choices=sorted(PRESETS))
    ap.add_argument("--tuning", help="override: comma MIDI, lowest first")
    ap.add_argument("--max-fret", type=int, default=None)
    ap.add_argument("--honor-artic", action="store_true")
    ap.add_argument("--economy", action="store_true")
    ap.add_argument("--funk-mute", action="store_true")
    ap.add_argument("--out")
    opts = ap.parse_args(argv)

    inst = PRESETS[opts.instrument]
    open_notes = ([int(x) for x in opts.tuning.split(",")] if opts.tuning
                  else inst.open_notes)
    configure(open_notes, opts.max_fret or inst.max_fret)

    notes = json.load(open(opts.plan))["notes"]
    if not notes:
        sys.exit("empty plan: no notes to place")
    n = len(notes)
    lo, hi = min(x["pitch"] for x in notes), max(x["pitch"] for x in notes)
    shift = next((sh for sh in (0, 12, -12, 24, -24)
                  if PITCH_LO <= lo + sh and hi + sh <= PITCH_HI), None)
    if shift is None:                                    # loud, global
        bad = sorted({x["pitch"] for x in notes
                      if not PITCH_LO <= x["pitch"] <= PITCH_HI})
        sys.exit(f"RANGE FAILURE: no single whole-octave shift fits {lo}..{hi} "
                 f"into {PITCH_LO}..{PITCH_HI}; offending pitches {bad}. "
                 f"Consider a lower tuning or a 5-string (--instrument bass5).")
    if shift:
        notes = [dict(x, pitch=x["pitch"] + shift) for x in notes]

    path, (cost, rootsum), med, seam_at, iois = solve(notes, opts)
    tags = assign_tags(path, notes)

    mode = "economy-opens" if opts.economy else "fretted-preferred"
    print(f"place_dp -- exact DP  instrument={opts.instrument} "
          f"tuning={OPEN} max_fret={MAX_FRET} mode={mode} "
          f"honor_artic={opts.honor_artic}")
    print(f"{n} notes, pitch {lo}..{hi}, global octave shift {shift:+d}")

    shifts = [i for i in range(1, n) if path[i][0] != path[i - 1][0]]
    fretted = [(st, i) for i, st in enumerate(path) if st[2] > 0]
    meand = (sum(abs(st[2] - st[0]) for st, _ in fretted) / len(fretted)
             if fretted else 0.0)
    opens = sum(1 for st in path if st[3] == "open")
    print(f"  positions: {len(set(st[0] for st in path))} roots, "
          f"{len(shifts)} shifts, {opens}/{n} open, "
          f"mean in-box distance {meand:.2f}, "
          f"objective ({cost / 100:.2f}, rootsum {rootsum})")

    if opts.out:
        out = [dict(t_on=notes[i]["t_on"], t_off=notes[i]["t_off"],
                    pitch=notes[i]["pitch"], vel=notes[i].get("vel", 90),
                    string=path[i][1], fret=path[i][2], hand_fret=path[i][0],
                    finger=path[i][4], pattern_tag=tags[i],
                    artic=notes[i].get("artic")) for i in range(n)]
        json.dump({"instrument": opts.instrument, "source": "place_dp",
                   "tuning": OPEN, "octave_shift": shift, "mode": mode,
                   "notes": out}, open(opts.out, "w"), indent=1)
        print(f"wrote {opts.out}")
    return path, tags


if __name__ == "__main__":
    main()
