# score2motion

Give it a score, get back a musician playing it.

A MIDI file goes in; motion for a character **actually playing that part** comes out —
the right notes, on the right strings or keys, with the right fingers, at the right times.

Two instruments so far, both **beta**:

| | what it does | output |
|---|---|---|
| `score2motion.keys` | keyboard player — ergonomic fingering, hands reach the right keys | BVH to retarget |
| `score2motion.bass` | bass player — fretting, walking/slap right hand, strap physics | rendered MP4 |

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

## Status

Beta, honestly. Both work end to end and are measured, but: the bass hand-to-neck lock
still shows about a centimetre of residual drift at extremes, guitar is not built yet, and
no video-model beautification pass has been proven to preserve finger-string contact.

## Motion models

The bass pipeline is model-agnostic — it consumes body motion as BVH, from
[Kimodo](https://github.com/NVIDIA/kimodo) or anything else, and applies it as a delta on
top of the player's own posture. The keys pipeline currently drives Kimodo's constraint API
directly.

## License

Apache-2.0. Not affiliated with NVIDIA.
