# kimodo-keys

Generate **keyboard-playing characters** with [Kimodo](https://github.com/NVIDIA/kimodo):
give it a MIDI file, get back a motion skeleton (BVH) whose hands reach the right keys at
the right times and whose fingers press them — ready to retarget onto any rigged
character.

This is a companion library, not a fork: it drives an existing Kimodo install through its
public constraint API. If you can run the Kimodo demo, you can run this.

## How it works

1. **Press plan** — the MIDI window is split into left/right-hand presses and fingered
   with [pianoplayer](https://github.com/marcomusy/pianoplayer) (ergonomic cost model:
   hand-span limits, per-finger strength, lookahead for thumb-unders).
2. **Base take** — Kimodo generates *"a person standing playing a keyboard"*.
3. **Set blocking** — a virtual keyboard (standard 23.5 mm key pitch) is placed in front
   of the character; per-frame **hand location timelines** are authored from the plan
   (hold during a press, glide between keys, arrive ~120 ms early).
4. **Constrained take** — the timelines are fed back through Kimodo's
   `EndEffectorConstraintSet`, and the diffusion re-imagines the whole body so reaching
   those keys looks human. The model owns liveliness; the MIDI owns correctness.
5. **Fingers** — the generative model leaves finger channels at rest; real press/travel
   motion is injected into the exported skeleton from the same plan.

## Usage

```bash
pip install -e .          # alongside a working kimodo install
kimodo-keys --midi song.mid --start 177 --end 195 \
    --solo-track 2 --chord-track 1 --out played.bvh
```

Import `played.bvh` into any DCC and retarget to your character. The keyboard the motion
was authored against is a standard 88-key board; rebuild it from
`kimodo_keys.keyboard.key_local()` if you want the prop to match exactly.

## License

Apache-2.0, same as Kimodo. Not affiliated with NVIDIA.
