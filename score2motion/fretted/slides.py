"""Find the slides: continuous pitch glides in the stem, note to note.

MIDI cannot say "slide" -- it steps. The stem can: between two consecutive
notes, a slide is a monotonic f0 ramp that BRIDGES the interval (>= 1.5
semitones of it, voiced and continuous), where a fretted change is a step
with a gap or a jump.

Downstream rule (the performer's law): a slide is CARRIED (finger stays
pressed, hand glides, marker travels) only when it covers >= 0.8 of the
interval AND the fret move is small (|dfret| <= 4) on one string; anything
less is a grace -- the placement stands and the hands jump as normal.

Needs the 'audio' extra (librosa):  pip install score2motion[audio]

    python3 -m score2motion.fretted.slides placed.json stem.wav --t0 0 --out slides.json
"""
import argparse
import json

CARRY_COVERED = 0.8      # the carried-slide eligibility law
CARRY_MAX_FRETS = 4


def main(argv=None):
    ap = argparse.ArgumentParser(description="detect note-pair slides")
    ap.add_argument("plan")
    ap.add_argument("stem")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import librosa
    import numpy as np

    notes = json.load(open(a.plan))["notes"]
    if len(notes) < 2:
        json.dump({"slides": []}, open(a.out, "w"), indent=1)
        print("0 slide(s) found (fewer than 2 notes)")
        return
    SR = 16000
    end = max(n["t_off"] for n in notes) + 1.0
    y, sr = librosa.load(a.stem, sr=SR, offset=a.t0, duration=end)
    HOP = 80                                   # 5ms
    f0, _, _ = librosa.pyin(y, fmin=30, fmax=400, sr=sr, hop_length=HOP,
                            frame_length=2048)
    midi = 69 + 12 * np.log2(np.where(np.isnan(f0), 1e-9, f0) / 440.0)
    midi[np.isnan(f0)] = np.nan
    print(f"f0 track: {len(f0)} frames over {end:.1f}s, "
          f"voiced {np.mean(~np.isnan(f0)) * 100:.0f}%")

    slides = []
    for i, (na, nb) in enumerate(zip(notes, notes[1:])):
        dp = nb["pitch"] - na["pitch"]
        if dp == 0:
            continue
        gap = nb["t_on"] - na["t_off"]
        if gap > 0.15:                          # detached notes: no one slide
            continue
        i0 = int((na["t_on"] + 0.6 * (min(na["t_off"], nb["t_on"])
                                      - na["t_on"])) * sr / HOP)
        i1 = int((nb["t_on"] + 0.04) * sr / HOP)
        if i1 <= i0 + 3 or i1 >= len(midi):
            continue
        seg = midi[i0:i1]
        ok = ~np.isnan(seg)
        if ok.mean() < 0.7:                     # a slide RINGS through
            continue
        seg = seg[ok]
        step = np.abs(np.diff(seg))
        if len(step) == 0 or step.max() > 1.0:  # no frame jumps inside
            continue
        travelled = seg[-1] - seg[0]
        frac = travelled / dp if dp else 0.0
        mono = (np.mean(np.sign(np.diff(seg)) == np.sign(dp))
                if len(seg) > 1 else 0)
        if abs(travelled) >= max(1.5, 0.5 * abs(dp)) and frac > 0.5 and mono > 0.6:
            slides.append({
                "from_t": round(na["t_on"], 3), "to_t": round(nb["t_on"], 3),
                "from_pitch": na["pitch"], "to_pitch": nb["pitch"],
                "semitones": round(float(travelled), 2),
                "covered": round(float(frac), 2), "mono": round(float(mono), 2),
                "from_idx": i, "to_idx": i + 1})

    json.dump({"slides": slides}, open(a.out, "w"), indent=1)
    print(f"{len(slides)} slide(s) found:")
    for s in slides:
        carried = (s["covered"] >= CARRY_COVERED)
        print(f"   note {s['from_idx']:3d} -> {s['to_idx']:3d}  "
              f"{s['from_t']:7.2f}s  {s['from_pitch']}->{s['to_pitch']}  "
              f"glide {s['semitones']:+.1f} st, covered {s['covered']:.0%}, "
              f"monotone {s['mono']:.0%}"
              f"  [{'carried candidate' if carried else 'grace'}]")


if __name__ == "__main__":
    main()
