# score2motion

Give it a score, get back a musician playing it.

A MIDI file goes in; motion for a character **actually playing that part** comes out —
the right notes, on the right strings or keys, with the right fingers, at the right times.

Three modules so far, all **beta**:

| | what it does | output |
|---|---|---|
| `score2motion.keys` | keyboard player — ergonomic fingering, hands reach the right keys | BVH to retarget |
| `score2motion.bass` | bass player — fretting, walking/slap right hand, strap physics | rendered MP4 |
| `score2motion.fretted` | **any** fretted instrument — bass, guitar, 5-string, ukulele — from one engine | plan JSON + Blender animation |

### Fretted, in two commands

```bash
python3 -m score2motion.fretted.cli song.mid --instrument guitar --out plan.json
blender your_scene.blend --python score2motion/fretted/apply_blender.py -- plan.json
```

Your scene supplies the character and the instrument; it has to meet
[the naming contract](score2motion/fretted/NAMING_CONTRACT.md), which is seven object
names and nothing else. No canon of ours is required or included — bring your own
rigged character.

Measured on two different instruments built to that contract, with the same code and no
per-instrument tuning: every fretted note lands within 6 mm of where the string is
actually stopped — **bass 17/17, guitar 9/9, median 1.2 mm** on both. Adding an instrument
is one entry in `instrument.py`: tuning, scale length, nut width, fret count. Fret spacing
is then derived from the rule of 18, and `check_against()` will tell you whether your
model's frets agree with physics before you waste a day blaming the solver.

It plays more than one note at a time:

- **chords** are placed as one shape. A hand is positioned once to serve every note in the
  chord, then each finger closes its own last millimetres. Placing them note by note makes
  each note drag the hand off the one before it, so only the last one lands.
- **barres** are treated as a line, not a point. One finger across several strings has to
  *lie along* them; aiming its tip at one string leaves the others a string-spacing past
  the end of the finger. The hand puts the knuckle on that line instead.
- **slides** keep the finger down and travelling between two notes on the same string,
  rather than lifting and re-placing.
- `check_plan()` refuses a part your neck cannot play — *"plan needs fret 22; this neck has
  0..12"* — before anything is created in your open file.

## The idea

Three things own three different parts of the problem, and keeping them apart is what
makes it work:

- **the MIDI owns where and when** — every note lands where the score says, deterministically
- **measured human motion owns how** — the *shape* of a press or a pluck, learned from
  motion-capture of a real player: slow approach, hard acceleration into the string,
  gentler return
- **physical constraints own what is possible** — the hand cannot pass through the neck,
  the arm cannot stretch, the instrument cannot float inside the player

Nothing is copied from a recording. Only the shape of a movement is learned, then applied
to whatever the score demands.

## Bass

```bash
pip install -e .
s2m-bass --midi bass.mid --audio song.wav --canon player.blend \
         --start 12.0 --duration 10 --out played.mp4
```

`--canon` is your rigged character already holding the instrument in the grip you want.
`DOCTRINE.md` explains how that grip is calibrated and why it is defined by the palm
rather than the fingers.

What the bass pipeline enforces, and measures on every run:

- the fretting hand may only **slide along the neck** — its distance across the neck and
  its clearance from the underside are pinned; reaching another string is the finger's job
- the hand is placed so the finger that must play is **already over its note**, so a note
  costs a couple of degrees of finger rotation instead of a swing through the instrument
- **muting is modelled**: the played finger must be the last one touching that string, and
  a note ends by releasing to a light touch, timed by the MIDI note-off
- the plucking hand stays still, thumb planted on the body as the pivot, ring and pinky
  clawed, index and middle strictly alternating
- the instrument hangs on straps, and the player's body is **solid** — it pushes the
  instrument forward, never through

Every run prints its own numbers: fingertip error to each target, hand-to-neck drift,
alternation ratio, instrument penetration, arm reach as a percentage of relaxed. If a
claim is not in that output, it is not a claim.

## Keys

```bash
s2m-keys --midi song.mid --start 177 --end 195 --solo-track 2 --chord-track 1 --out played.bvh
```

Fingering comes from [pianoplayer](https://github.com/marcomusy/pianoplayer)'s ergonomic
cost model; hand timelines are fed back through a motion model's constraint API so the
body re-imagines itself around reaching those keys.

## Both hands

The fretting hand chooses the notes; the picking hand decides whether the instrument
sounds like rhythm or like a line. They are **different mechanisms, not one with a size
knob**:

- **strumming** (guitar) sweeps the whole hand across every string in the chord, and the
  strings sound in sequence as it rakes across them rather than all at once. Crucially the
  arm is driven by the **beat**, not by the notes: a strumming arm does not stop on a rest,
  it keeps swinging and misses the strings. Driven from the notes it freezes between
  chords, which is the clearest tell that nobody is really playing.
- **walking fingerstyle** (bass) sounds one note at a time with index and middle strictly
  alternating, the fingers doing the work while the hand stays put. Using one finger for
  everything is the other obvious tell.

Both are measured every run: the pick is on a string it actually sounds at every sounding
moment, and every bass note is plucked by the named finger with the alternation checked.

## Status

Beta, honestly. What works end to end and is measured: bass and guitar, both hands,
chords, barres and slides. What is not proven: no video-model beautification pass has been
shown to preserve finger-string contact, so this is the last step that can promise the
notes are right.

## Hardware

Measured on the machine this was built on — an ARM box with 121 GiB of unified memory:

| step | cost |
|---|---|
| plan a part from MIDI | instant, plain Python, no Blender |
| body motion, 5 s at 100 diffusion steps | ~2.5 min, **CPU-only**; much faster on a GPU |
| bake both hands for one player | ~1–2 min |
| render 150 frames at 960×540, EEVEE | ~1 min |

The floor is lower than it looks: the motion model is the only heavy part, it runs on CPU
if it must, and everything else is geometry. **16 GB of RAM and any GPU Blender supports**
is enough to run the whole thing; the model weights are the largest download. You do not
need the machine above — it is simply what the timings were taken on, so they are real
rather than estimated.

## Motion models

The bass pipeline is model-agnostic — it consumes body motion as BVH, from
[Kimodo](https://github.com/NVIDIA/kimodo) or anything else, and applies it as a delta on
top of the player's own posture. The keys pipeline currently drives Kimodo's constraint API
directly.

## License

Apache-2.0. Not affiliated with NVIDIA.
