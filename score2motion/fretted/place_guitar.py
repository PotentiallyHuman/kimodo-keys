"""Put every MIDI note of a guitar part on a string and a fret.

Standard tuning, low to high: E2 A2 D3 G3 B3 E4 = 40 45 50 55 59 64, and our
string0 is the low E per the naming contract.

Notes that sound together are a chord, and a chord is played across DIFFERENT
strings -- one note per string, low notes on low strings. Within that, the
voicing is chosen to keep the whole shape inside the smallest fret span a hand
can hold, because a chord whose notes are eight frets apart is not a chord
anyone plays.

    python3 place_guitar_midi.py PART.mid OUT.json [--secs 8] [--chord-ms 35]
"""
import json
import sys
from itertools import product

import mido

OPEN = [40, 45, 50, 55, 59, 64]          # string0 (low E) .. string5 (high E)
MAX_FRET = 15
argv = [a for a in sys.argv[1:] if not a.startswith("--")]
MID, OUT = argv[0], argv[1]
SECS = float(sys.argv[sys.argv.index("--secs") + 1]) if "--secs" in sys.argv else 1e9
CHORD = (float(sys.argv[sys.argv.index("--chord-ms") + 1])
         if "--chord-ms" in sys.argv else 35.0) / 1000.0

m = mido.MidiFile(MID)
tempo, t, live, notes = 500000, 0.0, {}, []
for msg in mido.merge_tracks(m.tracks):
    t += mido.tick2second(msg.time, m.ticks_per_beat, tempo)
    if msg.type == "set_tempo":
        tempo = msg.tempo
    elif msg.type == "note_on" and msg.velocity > 0:
        live[msg.note] = (t, msg.velocity)
    elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
        if msg.note in live:
            on, vel = live.pop(msg.note)
            notes.append({"t_on": on, "t_off": t, "note": msg.note, "vel": vel})
notes = [n for n in notes if n["t_on"] <= SECS]
notes.sort(key=lambda n: (n["t_on"], n["note"]))

# group into chords by onset
groups, cur = [], []
for n in notes:
    if cur and n["t_on"] - cur[0]["t_on"] > CHORD:
        groups.append(cur)
        cur = []
    cur.append(n)
if cur:
    groups.append(cur)

placed, unplayable = [], 0
for g in groups:
    g = sorted(g, key=lambda n: n["note"])
    opts = []
    for n in g:
        o = [(s, n["note"] - OPEN[s]) for s in range(6)
             if 0 <= n["note"] - OPEN[s] <= MAX_FRET]
        opts.append(o or [(0, max(0, n["note"] - OPEN[0]))])
    best, best_cost = None, None
    for combo in product(*opts):
        strings = [c[0] for c in combo]
        if len(set(strings)) != len(strings):
            continue                                  # two notes, one string
        if strings != sorted(strings):
            continue                                  # low notes on low strings
        frets = [c[1] for c in combo if c[1] > 0]
        span = (max(frets) - min(frets)) if frets else 0
        if span > 4:
            continue                                  # wider than a hand
        cost = span * 10 + (min(frets) if frets else 0)
        if best_cost is None or cost < best_cost:
            best_cost, best = cost, combo
    if best is None:                                  # fall back: nearest fit
        best = [o[0] for o in opts]
        unplayable += 1
    for n, (s, fr) in zip(g, best):
        placed.append({"t_on": round(n["t_on"], 4), "t_off": round(n["t_off"], 4),
                       "note": n["note"], "vel": n["vel"],
                       "string": s, "fret": fr})

placed.sort(key=lambda n: (n["t_on"], n["string"]))
json.dump({"source": MID, "notes": placed}, open(OUT, "w"), indent=1)
spans = []
for g in groups:
    fs = [p["fret"] for p in placed
          if abs(p["t_on"] - g[0]["t_on"]) <= CHORD and p["fret"] > 0]
    if fs:
        spans.append(max(fs) - min(fs))
print(f"{len(placed)} notes in {len(groups)} onsets "
      f"({max(len(g) for g in groups)} at once at most)")
print(f"  fret span within a chord: {max(spans) if spans else 0} at worst "
      f"-- a hand covers about 4")
if unplayable:
    print(f"  WARNING {unplayable} chord(s) had no playable voicing")
print(f"wrote {OUT}")
