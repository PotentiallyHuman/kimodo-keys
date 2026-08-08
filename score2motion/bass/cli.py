"""song + bass MIDI in, a rendered bass player out.

    s2m-bass --midi bass.mid --audio song.wav --canon canon.blend \
             --start 12.0 --duration 10 --out played.mp4

Everything is supplied by the caller: no paths, songs or machines are baked in.
The canon .blend is your rigged character already holding the instrument, with
the grip you want; see DOCTRINE.md for how that is calibrated.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def _run(cmd: list[str], what: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-4000:] + r.stderr[-4000:])
        raise SystemExit(f"{what} failed")
    return r.stdout


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--midi", required=True, help="the bass part, as MIDI")
    ap.add_argument("--audio", required=True, help="the song itself (wav/mp3)")
    ap.add_argument("--canon", required=True, help=".blend of the rigged player + instrument")
    ap.add_argument("--motion", help="body motion BVH; omit to stand still")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the song")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--audio-offset", type=float, default=0.0,
                    help="seconds to shift the audio if the MIDI and mix are not aligned")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--blender", default="blender")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    for p, what in ((a.midi, "MIDI"), (a.audio, "audio"), (a.canon, "canon .blend")):
        if not os.path.exists(p):
            raise SystemExit(f"{what} not found: {p}")
    if shutil.which(a.blender) is None:
        raise SystemExit(f"blender not found on PATH (looked for {a.blender!r})")

    work = tempfile.mkdtemp(prefix="s2m-bass-")
    frames = os.path.join(work, "frames")
    os.makedirs(frames, exist_ok=True)
    plan = os.path.join(work, "plan.json")
    baked = os.path.join(work, "played.blend")
    end = a.start + a.duration

    # 1. what is played, and with which finger
    from . import plan as planner
    planner.write_plan(a.midi, a.start, end, plan)
    n = len(json.load(open(plan)))
    print(f"[1/3] planned {n} notes between {a.start:.2f}s and {end:.2f}s")

    # 2. how it is played: hands, physics, body, camera -- inside Blender
    cmd = [a.blender, "-b", a.canon, "--python", os.path.join(HERE, "performer.py"), "--",
           "--plan", plan, "--out", baked,
           "--frames", str(int(a.duration * a.fps)), "--fps", str(a.fps)]
    if a.motion:
        cmd += ["--motion", a.motion]
    print("[2/3] performing")
    print(_run(cmd, "performance")[-1500:])

    # 3. render and lay the song underneath
    print("[3/3] rendering")
    _run([a.blender, "-b", baked, "-o", os.path.join(frames, "f_"), "-F", "PNG",
          "-s", "1", "-e", str(int(a.duration * a.fps)), "-a"], "render")
    _run(["ffmpeg", "-y", "-framerate", str(a.fps),
          "-i", os.path.join(frames, "f_%04d.png"),
          "-ss", str(a.start + a.audio_offset), "-t", str(a.duration), "-i", a.audio,
          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
          "-c:a", "aac", "-b:a", "192k", "-shortest", a.out], "mux")

    if not a.keep_frames:
        shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
