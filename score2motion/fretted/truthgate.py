"""The truth gate: every planned note must be backed by the audio's own pitch.

Transcriptions octave-err quiet notes (a note written an octave high sends the
hand to the far end of the neck for a pitch nobody plays). Before placement,
each note is checked against pyin f0 over its own span in the source stem:

  - if the audio's median f0 matches pitch-12 or pitch+12 better than pitch,
    the note is OCTAVE-CORRECTED to what the audio actually sounds
  - if the span is unvoiced and near-silent, the note is DROPPED (no evidence)
  - a fifth-mismatch (+-7) is flagged loudly but left alone (tracker error)

Needs the 'audio' extra (librosa):  pip install score2motion[audio]

    python3 -m score2motion.fretted.truthgate plan.json stem.wav --t0 0 --out gated.json
"""
import argparse
import json


def main(argv=None):
    ap = argparse.ArgumentParser(description="gate a plan against its stem")
    ap.add_argument("plan")
    ap.add_argument("stem")
    ap.add_argument("--t0", type=float, default=0.0,
                    help="song time of the plan's t=0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import librosa
    import numpy as np

    d = json.load(open(a.plan))
    notes = d["notes"]
    if not notes:
        raise SystemExit("empty plan: nothing to gate")
    end = max(n["t_off"] for n in notes) + 0.5
    y, sr = librosa.load(a.stem, sr=16000, offset=a.t0, duration=end)
    HOP = 80
    f0, _, _ = librosa.pyin(y, fmin=25, fmax=500, sr=sr, hop_length=HOP,
                            frame_length=2048)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    floor = float(np.percentile(rms, 20))

    kept, fixed, dropped, flagged = [], 0, 0, 0
    for n in notes:
        i0 = int(n["t_on"] * sr / HOP)
        i1 = max(int(n["t_off"] * sr / HOP), i0 + 3)
        seg = f0[i0:i1]
        seg = seg[~np.isnan(seg)] if len(seg) else seg
        loud = float(np.median(rms[i0:i1])) if i1 > i0 else 0.0
        if len(seg) < 2 and loud < floor:
            dropped += 1
            print(f"  DROP  t={n['t_on']:7.2f} p{n['pitch']}: "
                  f"unvoiced and near-silent")
            continue
        if len(seg) >= 2:
            heard = 69 + 12 * np.log2(float(np.median(seg)) / 440.0)
            cands = [n["pitch"] - 12, n["pitch"], n["pitch"] + 12]
            best = min(cands, key=lambda c: abs(heard - c))
            if best != n["pitch"] and abs(heard - best) < 1.0:
                print(f"  FIX   t={n['t_on']:7.2f} p{n['pitch']} -> p{best}  "
                      f"(audio {heard:.1f})")
                n["pitch"] = best
                fixed += 1
            elif abs(heard - n["pitch"]) > 1.5:
                near5 = min((n["pitch"] - 7, n["pitch"] + 7),
                            key=lambda c: abs(heard - c))
                if abs(heard - near5) < 1.0:
                    flagged += 1
                    print(f"  FLAG  t={n['t_on']:7.2f} p{n['pitch']}: audio "
                          f"{heard:.1f} looks a fifth off -- left alone")
        kept.append(n)
    d["notes"] = kept
    json.dump(d, open(a.out, "w"), indent=1)
    print(f"gate: {len(notes)} in -> {len(kept)} kept, {fixed} octave-fixed, "
          f"{dropped} dropped, {flagged} flagged")


if __name__ == "__main__":
    main()
