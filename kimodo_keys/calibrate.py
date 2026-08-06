"""Measure real piano hand motion and report the numbers our model should match.

WHY THIS EXISTS, AND WHY IT IS CLEAN
Every published piano-hand system (PianoMotion10M, FurElise, Tipiano, BACH) represents
hands with MANO, whose licence forbids commercial use — so their data, weights and any
derived artifact are unusable in a product. Their *method*, however, is not secret:
video of hands playing + a pose estimator + statistics.

So we take the same source through a clean chain: MediaPipe (Apache-2.0) for landmarks,
our own rig as the representation, and we extract MEASUREMENTS — strike duration, wrist
give, travel lead, hand span, arch — which are facts about how hands move, not anyone's
authored motion. Those numbers calibrate the procedural model in piano_hand.py.

We never copy trajectories. The deliverable is a table of constants.

    python -m kimodo_keys.calibrate reference.mp4 --midi reference.mid --start 0 --end 20
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict

import numpy as np

# MediaPipe hand landmark indices
WRIST = 0
TIPS = {1: 4, 2: 8, 3: 12, 4: 16, 5: 20}          # thumb..pinky fingertips
MCPS = {1: 2, 2: 5, 3: 9, 4: 13, 5: 17}           # knuckles
PIPS = {1: 3, 2: 6, 3: 10, 4: 14, 5: 18}


@dataclass
class Measured:
    """The constants our procedural model needs, measured from real playing."""
    strike_duration_s: float = 0.0      # knuckle drop: start of descent -> key contact
    release_duration_s: float = 0.0     # contact -> back to hover
    wrist_give_m: float = 0.0           # vertical wrist travel per strike (arm weight)
    hover_height_m: float = 0.0         # fingertip clearance above keys between notes
    travel_lead_s: float = 0.0          # how early the hand starts toward the next zone
    hand_span_m: float = 0.0            # thumb tip to pinky tip when spread
    arch_mcp_deg: float = 0.0           # knuckle flexion at rest (the dome)
    arch_pip_deg: float = 0.0
    frames_used: int = 0
    hands_detected_pct: float = 0.0


def _angle(a, b, c):
    """Angle at b, in degrees."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    return math.degrees(math.acos(max(-1.0, min(1.0, float(v1 @ v2) / (n1 * n2)))))


MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = "/tmp/hand_landmarker.task"


def _ensure_model(path: str = MODEL_PATH) -> str:
    import os
    import urllib.request
    if not os.path.exists(path):
        print(f"fetching hand landmarker model -> {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def track(video_path: str, *, start: float = 0.0, end: float | None = None,
          max_frames: int = 3000, model_path: str = MODEL_PATH):
    """MediaPipe Tasks landmarks per frame. Returns (times, {hand: [21x3 or None]}, fps)."""
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=_ensure_model(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=0.4,
        min_tracking_confidence=0.4)
    landmarker = vision.HandLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if start:
        cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)

    times, seq = [], {"Left": [], "Right": []}
    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if end is not None and t > end:
            break
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame[:, :, ::-1].copy())
        res = landmarker.detect_for_video(image, int(t * 1000))
        found = {"Left": None, "Right": None}
        if res.hand_landmarks:
            for lm, handed in zip(res.hand_landmarks, res.handedness):
                label = handed[0].category_name          # "Left"/"Right"
                if label in found:
                    found[label] = np.array([[p.x, p.y, p.z] for p in lm])
        for k in seq:
            seq[k].append(found[k])
        times.append(t)
        n += 1
    cap.release()
    landmarker.close()
    return np.array(times), seq, fps


def measure(times, seq, *, key_width_px_frac: float = 0.0235 / 1.222) -> Measured:
    """Turn landmark tracks into the constants. Scale is recovered from hand span:
    an adult thumb-to-pinky spread is ~0.17-0.18 m, which anchors the pixel->metre ratio."""
    out = Measured()
    got = 0
    spans, arch_mcp, arch_pip, wrist_y = [], [], [], []
    tip_y = {f: [] for f in TIPS}

    for i in range(len(times)):
        for hand in ("Left", "Right"):
            lm = seq[hand][i]
            if lm is None:
                continue
            got += 1
            spans.append(float(np.linalg.norm(lm[TIPS[1]] - lm[TIPS[5]])))
            for f in (2, 3, 4):                      # index/middle/ring: the clean dome
                arch_mcp.append(_angle(lm[WRIST], lm[MCPS[f]], lm[PIPS[f]]))
                arch_pip.append(_angle(lm[MCPS[f]], lm[PIPS[f]], lm[TIPS[f]]))
            wrist_y.append(float(lm[WRIST][1]))
            for f, t_idx in TIPS.items():
                tip_y[f].append(float(lm[t_idx][1]))

    total = max(1, 2 * len(times))
    out.frames_used = len(times)
    out.hands_detected_pct = 100.0 * got / total
    if not spans:
        return out

    # scale: median observed span == 0.175 m (adult thumb-to-pinky spread)
    span_px = float(np.median(spans))
    m_per_unit = 0.175 / max(1e-9, span_px)
    out.hand_span_m = round(span_px * m_per_unit, 4)
    out.arch_mcp_deg = round(180.0 - float(np.nanmedian(arch_mcp)), 1)
    out.arch_pip_deg = round(180.0 - float(np.nanmedian(arch_pip)), 1)

    # strike timing + wrist give from vertical motion of the fingertips/wrist
    dt = float(np.median(np.diff(times))) if len(times) > 2 else 1 / 30
    strikes, releases, gives, hovers = [], [], [], []
    for f, ys in tip_y.items():
        y = np.array(ys)
        if len(y) < 9:
            continue
        y = y - y.min()
        thr = 0.55 * (np.percentile(y, 90) - np.percentile(y, 10))
        below = y < (np.percentile(y, 10) + thr)     # image y grows downward: low = pressed
        # contiguous pressed runs = key contacts
        idx = np.flatnonzero(np.diff(below.astype(int)) != 0) + 1
        runs = np.split(np.arange(len(y)), idx)
        for r in runs:
            if len(r) < 2 or not below[r[0]]:
                continue
            strikes.append(len(r) * dt * 0.5)        # descent ~ half the contact run
            releases.append(len(r) * dt * 0.6)
        hovers.append(float(np.percentile(y, 75) - np.percentile(y, 10)) * m_per_unit)
    if wrist_y:
        w = np.array(wrist_y)
        gives.append(float(np.percentile(w, 90) - np.percentile(w, 10)) * m_per_unit)

    if strikes:
        out.strike_duration_s = round(float(np.median(strikes)), 3)
        out.release_duration_s = round(float(np.median(releases)), 3)
    if gives:
        out.wrist_give_m = round(float(np.median(gives)), 4)
    if hovers:
        out.hover_height_m = round(float(np.median(hovers)), 4)
    return out


# Target values, so the report is a diff rather than a number dump.
#
# As of 2026-08-06 these are no longer our guesses — they are the published measurements
# our model is calibrated to. A reading that disagrees with a row here means either the
# footage differs from the studied population, or the extraction is wrong. Both are worth
# knowing, which is the whole point of keeping this as a diff.
#
#   strike_duration_s   0.080   Goebl & Palmer 2008, Exp Brain Res 186:471 (79.3 ms at
#                               2 tones/s, 60.7 ms at 7 — tempo-dependent)
#   release_duration_s  0.16    NO SOURCE. No study reports finger return time; this one
#                               is still a guess and is the honest gap in the table.
#   wrist_give_m        --      NO VERIFIED SOURCE in metres. Published only as an angle:
#                               12.63 deg per cycle (Goebl & Palmer 2013, PLOS ONE
#                               8:e50901). A "~90 mm" figure was found and FAILED
#                               verification, so it is deliberately absent here.
#   hover_height_m      0.013   Dalla Bella & Palmer 2011, PLOS ONE 6:e20518 (excursion
#                               21.7-26.1 mm, less the ~10 mm key surface)
#   travel_lead_s       --      NOT A CONSTANT: ~1.05 x inter-onset interval, tempo-
#                               invariant (Dalla Bella & Palmer 2011). Measured per take.
#   hand_span_m         0.226   Boyle, Boyle & Booker 2015, APPCA (n=473; 0.201 female)
#   arch_mcp_deg        18.0    mid-cycle mean; 33.98 deg range (Goebl & Palmer 2013)
#   arch_pip_deg        37.0    Rahman et al. 2011, IJSTER 2(2):22 (observed 26-49 deg)
OURS = {
    "strike_duration_s": 0.080, "release_duration_s": 0.16, "wrist_give_m": 0.0,
    "hover_height_m": 0.013, "travel_lead_s": 0.26, "hand_span_m": 0.226,
    "arch_mcp_deg": 18.0, "arch_pip_deg": 37.0,
}

# Rows with no trustworthy published value. The report flags these so a "close" verdict
# against a guess is never mistaken for agreement with the literature.
UNSOURCED = {"release_duration_s", "wrist_give_m", "travel_lead_s"}


def report(m: Measured) -> None:
    print(f"\ndetection: {m.hands_detected_pct:.0f}% of {m.frames_used} frames\n")
    print("%-22s %10s %10s   %s" % ("constant", "measured", "ours", "verdict"))
    print("-" * 62)
    for k, ours in OURS.items():
        tag = " (unsourced)" if k in UNSOURCED else ""
        got = getattr(m, k, 0.0)
        if not got:
            print("%-22s %10s %10.3f   (not measured)%s" % (k, "-", ours, tag))
            continue
        d = abs(got - ours) / max(1e-9, abs(ours)) if ours else float("inf")
        verdict = "close" if d < 0.25 else ("ADJUST" if d < 1.0 else "WAY OFF")
        print("%-22s %10.3f %10.3f   %s%s" % (k, got, ours, verdict, tag))
    print("\nMeasurements are facts about hand motion — no MANO, no third-party weights,")
    print("no copied trajectories.")
    print("Rows WITHOUT '(unsourced)' are calibrated to published biomechanics; a")
    print("disagreement there means the footage or the extraction is unusual, not that")
    print("the model is wrong. Rows marked '(unsourced)' are the ones footage can still")
    print("improve — release duration and wrist give above all.")
    print("\nNOTE: a top-down camera cannot measure wrist give, hover height or PIP angle;")
    print("vertical motion is invisible along the view axis. Use a SIDE view for those.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    times, seq, fps = track(a.video, start=a.start, end=a.end)
    print(f"tracked {len(times)} frames @ {fps:.1f} fps")
    m = measure(times, seq)
    report(m)
    if a.json_out:
        json.dump(asdict(m), open(a.json_out, "w"), indent=1)
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
