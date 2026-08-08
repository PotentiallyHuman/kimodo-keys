#!/usr/bin/env python3
"""MIDI -> slap-bass performance plan (string, fret, articulation, timing).

Usage: python3 bass_plan.py <bass.mid> <t0> <t1> <out.json>

Articulation dispatch per SLAP_BASS_DOCTRINE.md (measured 2026-08-06):
slap by default (thumb, low string bias); pops ~12% rhythm-placed on the
higher strings; two quick close notes after a run -> badink pair (one gesture,
middle then ring releasing 100ms apart); ghost picks on long gaps.
"""
import json, sys, random

import mido

# 4-string bass EADG. The Suno transcription sits an octave (or two) above a
# real bass; shift down in whole octaves until the segment fits the neck.
OPEN = [28, 33, 38, 43]          # E1 A1 D2 G2
MAX_FRET = 12


def load_notes(path, t0, t1):
    m = mido.MidiFile(path)
    now, notes = 0.0, []
    on = {}
    for msg in m:
        now += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            on[msg.note] = (now, msg.velocity)
        elif msg.type in ("note_off", "note_on"):
            if msg.note in on:
                s, v = on.pop(msg.note)
                if t0 <= s < t1:
                    notes.append({"t_on": s - t0, "t_off": max(s - t0 + 0.08, now - t0),
                                  "pitch": msg.note, "vel": v})
    notes.sort(key=lambda n: n["t_on"])
    return notes


def fit_octave(notes):
    for shift in (0, -12, -24, -36):
        lo = min(n["pitch"] for n in notes) + shift
        hi = max(n["pitch"] for n in notes) + shift
        if lo >= OPEN[0] and hi <= OPEN[3] + MAX_FRET:
            return shift
    return -24


def assign_string_fret(pitch):
    """Lowest playable position: prefer the thickest string that keeps fret <= MAX.
    Notes forced above fret 7 (no higher string to escape to) drop an octave —
    a real player's choice, and the fret-8 stretches were every gate outlier."""
    def best_for(p):
        best = None
        for s in range(4):
            fret = p - OPEN[s]
            if 0 <= fret <= MAX_FRET:
                cost = fret + s * 0.5
                if best is None or cost < best[0]:
                    best = (cost, s, fret)
        return best
    best = best_for(pitch)
    if best and best[2] > 7:
        lower = best_for(pitch - 12)
        if lower and lower[2] <= 7:
            best = lower
    if best is None:                      # out of range: clamp
        s = 0 if pitch < OPEN[0] else 3
        return s, max(0, min(MAX_FRET, pitch - OPEN[s]))
    return best[1], best[2]


def dispatch(notes):
    """Assign articulations: slap default; badink pairs; pops on accents; ghosts."""
    random.seed(6)
    out = []
    i = 0
    last_badink = -9.0
    while i < len(notes):
        n = notes[i]
        s, f = assign_string_fret(n["pitch"])
        art = "slap"
        # badink: a spice, not the meal — two quick notes on higher strings,
        # min 2s since the last one (doctrine: occasional)
        if (i + 1 < len(notes)
                and notes[i + 1]["t_on"] - n["t_on"] < 0.16
                and n["t_on"] - last_badink > 2.0
                and s >= 1 and assign_string_fret(notes[i + 1]["pitch"])[0] >= s):
            last_badink = n["t_on"]
            n2 = notes[i + 1]
            s2, f2 = assign_string_fret(n2["pitch"])
            out.append({**n, "string": s, "fret": f, "artic": "badink_ba"})
            out.append({**n2, "string": s2, "fret": f2, "artic": "badink_dink"})
            i += 2
            continue
        # pops: ~12%, rhythm-placed — favour offbeat-ish onsets and string >= 2
        offbeat = (n["t_on"] % 0.5) > 0.18
        if s >= 2 and offbeat and random.random() < 0.35:
            art = "pop"
        elif random.random() < 0.04:
            art = "pop"
        out.append({**n, "string": s, "fret": f, "artic": art})
        i += 1
    # ghost picks in gaps > 0.45s (one per gap, on the last string played)
    ghosts = []
    for a, b in zip(out, out[1:]):
        gap = b["t_on"] - a["t_off"]
        if gap > 0.45 and random.random() < 0.6:
            tg = a["t_off"] + gap * 0.55
            ghosts.append({"t_on": tg, "t_off": tg + 0.06, "pitch": a["pitch"],
                           "vel": 30, "string": a["string"], "fret": a["fret"],
                           "artic": "ghost"})
    out += ghosts
    out.sort(key=lambda n: n["t_on"])
    return out


def main():
    midi, t0, t1, dst = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    notes = load_notes(midi, t0, t1)
    shift = fit_octave(notes)
    for n in notes:
        n["pitch"] += shift
    plan = dispatch(notes)
    from collections import Counter
    c = Counter(p["artic"] for p in plan)
    json.dump(plan, open(dst, "w"), indent=1)
    print(f"{len(plan)} events over {t1-t0:.0f}s (octave shift {shift}): {dict(c)}")
    frets = Counter(p["fret"] for p in plan)
    print("fret usage:", dict(sorted(frets.items())))


if __name__ == "__main__":
    main()


def write_plan(midi_path, t0, t1, out_path):
    """MIDI window -> a per-note plan: string, fret, articulation, timing."""
    notes = load_notes(midi_path, t0, t1)
    shift = fit_octave(notes)
    plan = build_plan(notes, shift) if "build_plan" in globals() else notes
    json.dump(plan, open(out_path, "w"), indent=1)
    return plan
