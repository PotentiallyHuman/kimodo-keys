# The naming contract

`apply_blender.py` never guesses. It looks for exactly these objects, and if any
are missing it stops rather than animating something wrong. Meet the contract
and your character plays; you do not have to modify the code.

| object | what it is |
|---|---|
| `PLAYER.rig` | the armature. Needs `wrist.L` and `finger1-3.L` … `finger5-3.L` (thumb first) |
| `PLAYER.body` | the skinned mesh |
| `INST_root` | empty or object at the instrument's origin. **Everything else is parented to it**, and all fret/string maths happens in its local space, so the player can move without breaking the fingering |
| `INST_neck` | the neck mesh. Its highest local Z is taken as the fretboard surface |
| `INST_fret0` … `INST_fretN` | one object per fret wire. `INST_fret0` is the nut |
| `INST_string0` … `INST_stringN` | one per string, **lowest pitch first** |
| `INST_body`, `INST_headstock` | optional, cosmetic |

An optional prefix argument lets several players share one scene:
`... -- plan.json GUITAR_` looks for `GUITAR_PLAYER.rig`, `GUITAR_INST_fret3`, and so on.

## Why the local space matters

Fret and string positions are read in `INST_root`'s local space. That means the
instrument can be strapped to a moving body, tilted, or scaled, and the note
targets follow it exactly. It is also why a guitar is just a bass at 75% — the
same grip lands on the same string because both are expressed relative to the
same root.

## Checking a model before you trust it

```python
from score2motion.fretted.instrument import GUITAR6
ok, worst_mm, detail = GUITAR6.check_against({0: 0.0, 1: 1.43, 12: 19.1})
```

Fret spacing follows the rule of 18 — the twelfth fret is at exactly half the
scale length. That is physics, not style. If your model disagrees by more than
a few millimetres the fingers will be told to press where there is no fret, and
you will spend a day blaming the solver. Check first.

## Two rules the solver depends on

1. **The hand is dragged, not curled.** Bending a finger to reach a string also
   moves the palm, so a curl loop chases its own tail. The finger keeps its
   calibrated shape and the whole hand translates until the tip touches.
   Damping gain is 0.5; at 1.0 it oscillates instead of settling.

2. **Write pose bones parents-first, updating after each one.** Blender converts
   `pose_bone.matrix` through the bone's *current* evaluated parent. Write a
   child before its parent, or batch the depsgraph update to the end of the
   loop, and the hand silently stays where it was. Both mistakes cost a day
   during development; both are one line to avoid.
