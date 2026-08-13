"""MIDI -> a fretted performance plan: which string, which fret, which finger, when.

The plan is deliberately dumb about bodies. It says *where and when*; how the
hand gets there is the animator's problem (see apply_blender.py), and what is
physically possible is the rig's. Keeping those three apart is the whole trick
-- when they are mixed, a fingering fix breaks the posture and vice versa.

Fingering is chosen by least effort over the whole segment, not note by note.
A greedy choice takes the nearest fret every time and then pays for it with a
huge jump two notes later; a short dynamic program over hand positions costs
nothing and plays like someone who read ahead, because they did.
"""
import json
from dataclasses import dataclass

from .instrument import Instrument, get

HAND_SPAN = 4            # frets covered without moving the hand: index..pinky
SHIFT_COST = 2.5         # a hand shift is this much dearer than a finger stretch
STRING_COST = 0.4        # crossing strings is cheap but not free


@dataclass
class Note:
    t_on: float
    t_off: float
    pitch: int
    vel: int


def load_midi(path, t0=0.0, t1=1e9):
    import mido
    m = mido.MidiFile(path)
    now, on, notes = 0.0, {}, []
    for msg in m:
        now += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            on[msg.note] = (now, msg.velocity)
        elif msg.type in ("note_off", "note_on") and msg.note in on:
            s, v = on.pop(msg.note)
            if t0 <= s < t1:
                notes.append(Note(s - t0, max(s - t0 + 0.08, now - t0), msg.note, v))
    notes.sort(key=lambda n: n.t_on)
    return notes


def fit_octave(notes, inst: Instrument):
    """Transcriptions (Suno's especially) often sit an octave or two above the
    real instrument. Shift in whole octaves until the part fits the neck."""
    lo_i, hi_i = inst.range_midi()
    fits, best, best_out = [], None, 0
    for shift in (0, -12, -24, -36, 12, 24):
        lo = min(n.pitch for n in notes) + shift
        hi = max(n.pitch for n in notes) + shift
        out = sum(1 for n in notes if not (lo_i <= n.pitch + shift <= hi_i))
        if lo >= lo_i and hi <= hi_i:
            fits.append(shift)
        if best is None or out < best_out:
            best, best_out = shift, out
    if fits:
        # PLAY IT LOW. Returning the first shift that merely FITS meant 0 always
        # won, and once the neck was extended to 24 frets a part that used to be
        # forced down an octave suddenly "fit" up at the 12th to 22nd -- the
        # player ended up clinging to the far end of the neck with the body of
        # the instrument in the way. Extending the neck had quietly removed the
        # only thing keeping the part in a sane position.
        #
        # A bassist plays the LOWEST position the part allows, near where the
        # hand rests. So among the shifts that fit, take the one whose notes need
        # the lowest frets.
        def mean_fret(shift):
            fs = []
            for n in notes:
                p = inst.positions_for(n.pitch + shift)
                if p:
                    fs.append(min(f for _s, f in p))
            return sum(fs) / len(fs) if fs else 99

        # But do not transpose a part that is ALREADY in a sane position: picking
        # purely by lowest fret pulled a guitar sitting comfortably at the 2nd
        # fret down another octave, so the character played notes the recording
        # does not contain. Shifting is a repair for a transcription in the wrong
        # register, not a preference. So: keep the written octave when it plays
        # below COMFORT, and only shift when it does not.
        COMFORT = 7.0
        home = [s for s in fits if mean_fret(s) <= COMFORT]
        if home:
            return min(home, key=lambda s: (abs(s), mean_fret(s))), 0
        return min(fits, key=mean_fret), 0
    return best, best_out


def _positions(pitch, inst):
    p = inst.positions_for(pitch)
    # open strings are allowed but a fretted note is preferred when both exist:
    # an open string cannot be muted or vibratoed, so real players avoid it mid-line
    return sorted(p, key=lambda sf: (sf[1] == 0, sf[1]))


def choose_positions(notes, inst: Instrument):
    """Dynamic program: state = the fret the index finger sits at."""
    cand = [_positions(n.pitch, inst) for n in notes]
    for i, c in enumerate(cand):
        if not c:
            raise SystemExit(f"note {notes[i].pitch} is off the {inst.name}'s neck "
                             f"(range {inst.range_midi()}) -- fit_octave first")
    hands = sorted({max(1, f - k) for c in cand for _, f in c for k in range(HAND_SPAN)})
    best = {h: (0.0, None) for h in hands}
    back = []
    prev_pos = {h: None for h in hands}
    for i, c in enumerate(cand):
        cur, choice = {}, {}
        for h in hands:
            reach = [(s, f) for s, f in c if f == 0 or h <= f < h + HAND_SPAN]
            if not reach:
                continue
            for s, f in reach:
                for ph, (pc, _) in best.items():
                    if pc == float("inf"):
                        continue
                    cost = pc + SHIFT_COST * abs(h - ph)
                    if prev_pos[ph] is not None:
                        cost += STRING_COST * abs(s - prev_pos[ph][0])
                    # the pinky is weaker than the index -- FRETTED NOTES ONLY.
                    # An open string uses no finger at all, and f is 0 for one, so
                    # ungated this term became -0.15*h: a discount that grew the
                    # further up the neck the hand sat. A part with open strings in
                    # it was therefore planned high up the board for no musical
                    # reason -- the guitar came out at the 11th fret.
                    if f:
                        cost += 0.15 * (f - h)
                    if h not in cur or cost < cur[h][0]:
                        cur[h] = (cost, ph)
                        choice[h] = (s, f)
        if not cur:
            raise SystemExit(f"no reachable hand position for note {i} (pitch {notes[i].pitch})")
        back.append((choice, {h: v[1] for h, v in cur.items()}))
        best = {h: (cur.get(h, (float("inf"), None))[0], None) for h in hands}
        prev_pos = {h: choice.get(h) for h in hands}

    h = min((h for h in hands if best[h][0] < float("inf")), key=lambda x: best[x][0])
    out = []
    for i in range(len(notes) - 1, -1, -1):
        choice, prev = back[i]
        out.append((h,) + choice[h])
        h = prev[h] if prev[h] is not None else h
    return list(reversed(out))


def finger_for(hand_fret, fret):
    """1 = index ... 4 = pinky. An open string uses no finger."""
    if fret == 0:
        return 0
    return max(1, min(4, fret - hand_fret + 1))


def build(midi_path, inst_name="bass", t0=0.0, t1=1e9, out=None):
    inst = get(inst_name)
    notes = load_midi(midi_path, t0, t1)
    if not notes:
        raise SystemExit(f"no notes between {t0}s and {t1}s in {midi_path}")
    shift, off = fit_octave(notes, inst)
    for n in notes:
        n.pitch += shift
    notes = [n for n in notes if inst.positions_for(n.pitch)]
    placed = choose_positions(notes, inst)

    plan, shifts = [], 0
    for i, (n, (h, s, f)) in enumerate(zip(notes, placed)):
        if i and h != placed[i - 1][0]:
            shifts += 1
        plan.append({"t_on": round(n.t_on, 4), "t_off": round(n.t_off, 4),
                     "pitch": n.pitch, "vel": n.vel,
                     "string": s, "fret": f, "hand_fret": h,
                     "finger": finger_for(h, f)})
    doc = {"instrument": inst.name, "tuning": inst.open_notes,
           "octave_shift": shift, "notes_dropped": off,
           "n_notes": len(plan), "hand_shifts": shifts,
           "shifts_per_note": round(shifts / len(plan), 3), "notes": plan}
    if out:
        json.dump(doc, open(out, "w"), indent=1)
    return doc
