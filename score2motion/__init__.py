"""score2motion — turn a score into a musician playing it.

A MIDI file goes in; motion for a character actually playing that part comes out.
Two instruments so far, both beta:

    score2motion.keys  — keyboard player (constraint-guided, exports BVH)
    score2motion.bass  — bass player (full Blender performance + render)

The division of labour is the whole idea:
  * the MIDI owns WHERE and WHEN every note happens
  * measured human motion owns HOW a hand gets there
  * physical constraints own WHAT IS POSSIBLE at all

Nothing is copied from a recording; only the shape of a movement is learned.
"""

__version__ = "0.3.0-beta"
