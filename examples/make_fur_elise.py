"""Generate examples/fur_elise.mid — the opening of Beethoven's Bagatelle No. 25 in
A minor, "Für Elise" (1810, public domain). Encoded from the score: right hand carries
the famous sixteenth-note theme, left hand the broken-chord accompaniment."""
import pretty_midi

S = 0.16                    # one sixteenth note (poco moto, slightly stately)
pm = pretty_midi.PrettyMIDI()
inst = pretty_midi.Instrument(program=0, name="piano")


def rh(notes):
    """notes: list of (pitch|None, sixteenths). Advances the right-hand clock."""
    global t_r
    for p, n in notes:
        if p is not None:
            inst.notes.append(pretty_midi.Note(92, p, t_r, t_r + S * n * 0.92))
        t_r += S * n


def lh(start, pitches):
    """Broken chord: three ascending eighths from `start` (seconds)."""
    for i, p in enumerate(pitches):
        s = start + i * S
        inst.notes.append(pretty_midi.Note(68, p, s, s + S * 2.6))


E5, DS5, B4, D5, C5, A4, GS4 = 76, 75, 71, 74, 72, 69, 68
C4, E4, A3_, B3 = 60, 64, 57, 59

t_r = 0.0
theme = [(E5, 1), (DS5, 1), (E5, 1), (DS5, 1), (E5, 1), (B4, 1), (D5, 1), (C5, 1)]

for cycle in range(2):
    rh(theme)
    bar = t_r
    lh(bar, [45, 52, 57])                       # A2 E3 A3 under the A4
    rh([(A4, 2), (None, 1), (C4, 1), (E4, 1), (A4, 1)])
    bar = t_r
    lh(bar, [40, 52, 56])                       # E2 E3 G#3 under the B4
    rh([(B4, 2), (None, 1), (E4, 1), (GS4, 1), (B4, 1)])
    bar = t_r
    lh(bar, [45, 52, 57])
    if cycle == 0:
        rh([(C5, 2), (None, 1), (E4, 1)])       # ...and the theme returns
    else:
        rh([(A4, 4)])                           # final resolution: A held

pm.instruments.append(inst)
pm.write("examples/fur_elise.mid")
print(f"wrote examples/fur_elise.mid  {pm.get_end_time():.1f}s, {len(inst.notes)} notes")
