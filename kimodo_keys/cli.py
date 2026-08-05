"""CLI: kimodo-keys --midi song.mid --start 177 --end 195 --out played.bvh"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a keyboard-playing motion skeleton from a MIDI file, "
                    "using a Kimodo motion model + end-effector constraints.")
    ap.add_argument("--midi", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="window start (s)")
    ap.add_argument("--end", type=float, required=True, help="window end (s)")
    ap.add_argument("--solo-track", type=int, default=None)
    ap.add_argument("--chord-track", type=int, default=None)
    ap.add_argument("--split", type=int, default=60, help="left/right split pitch")
    ap.add_argument("--hand-size", default="M", choices=list("SML") + ["XS", "XL"])
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--model", default="soma")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default="played.bvh")
    ap.add_argument("--plan-out", default=None, help="also write the press plan as JSON")
    args = ap.parse_args()

    from .generate import PROMPT_DEFAULT, generate_player
    from .plan import extract_presses

    plan = extract_presses(args.midi, args.start, args.end,
                           solo_track=args.solo_track, chord_track=args.chord_track,
                           split=args.split, hand_size=args.hand_size)
    print(f"plan: {len(plan)} presses "
          f"(L {sum(1 for p in plan if p.hand == 'Left')} / "
          f"R {sum(1 for p in plan if p.hand == 'Right')})")
    if args.plan_out:
        json.dump([p.to_dict() for p in plan], open(args.plan_out, "w"), indent=1)

    generate_player(plan, seconds=args.end - args.start,
                    prompt=args.prompt or PROMPT_DEFAULT, model_name=args.model,
                    device=args.device, seed=args.seed,
                    diffusion_steps=args.steps, out_bvh=args.out)


if __name__ == "__main__":
    main()
