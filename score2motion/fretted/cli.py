"""python3 -m score2motion.fretted.cli <midi> --instrument guitar --out plan.json

Then, inside Blender:
    blender your_scene.blend --python score2motion/fretted/apply_blender.py -- plan.json
"""
import argparse

from .instrument import PRESETS
from .plan import build


def main(argv=None):
    p = argparse.ArgumentParser(description="MIDI -> fretted performance plan")
    p.add_argument("midi")
    p.add_argument("--instrument", "-i", default="bass", choices=sorted(PRESETS))
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=1e9)
    p.add_argument("--out", "-o", default="plan.json")
    a = p.parse_args(argv)

    d = build(a.midi, a.instrument, a.start, a.end, a.out)
    print(f"{d['instrument']}: {d['n_notes']} notes, octave shift {d['octave_shift']:+d}, "
          f"{d['hand_shifts']} hand shifts ({d['shifts_per_note']}/note)")
    if d["notes_dropped"]:
        print(f"  {d['notes_dropped']} notes fall outside the neck and were dropped")
    print(f"  wrote {a.out}")
    return d


if __name__ == "__main__":
    main()
