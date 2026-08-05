"""Generate examples/ode_to_joy.mid — a public-domain melody (Beethoven, 1824) with a
simple left-hand accompaniment, so the repo ships a copyright-clean demo input."""
import pretty_midi

pm = pretty_midi.PrettyMIDI()
inst = pretty_midi.Instrument(program=0, name="keys")

# right hand: the theme in E major-ish C (C major for simplicity), quarter = 0.5 s
melody = [64, 64, 65, 67, 67, 65, 64, 62, 60, 60, 62, 64, 64, 62, 62,
          64, 64, 65, 67, 67, 65, 64, 62, 60, 60, 62, 64, 62, 60, 60]
t = 0.0
for i, p in enumerate(melody):
    dur = 0.75 if i in (13, 27) else (0.25 if i in (14,) else 0.5)
    inst.notes.append(pretty_midi.Note(velocity=90, pitch=p, start=t, end=t + dur * 0.9))
    t += dur

# left hand: root-fifth every bar
bass = [(48, 55), (43, 50), (45, 52), (43, 50)] * 2
bt = 0.0
for lo, hi in bass:
    for p in (lo, hi):
        inst.notes.append(pretty_midi.Note(velocity=70, pitch=p, start=bt, end=bt + 1.8))
    bt += 2.0

pm.instruments.append(inst)
pm.write("examples/ode_to_joy.mid")
print("wrote examples/ode_to_joy.mid", pm.get_end_time(), "s")
