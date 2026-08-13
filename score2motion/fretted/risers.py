"""Find the RISERS: single-note pitch ramps where the hand slides up the neck.

slides.py finds glides that BRIDGE two planned notes. A riser is a different
gesture: ONE finger holds, the others lift, and the whole hand travels toward
the instrument body so f0 climbs continuously -- >= 3 semitones over >= 300 ms
-- typically right before a drop or a section change, not necessarily landing
on a clean note. The plan either misses it or chops it into stair-steps; only
the stem shows it.

Discriminators (each measured and reported, not assumed):
  vs note-pair slide : a slide starts on planned note a and lands on planned
                       note b, and is short. A candidate overlapping a known
                       slide window is reclassified to that slide.
  vs vibrato / drift : vibrato oscillates about a center. A riser accumulates:
                       max pullback <= 0.7 st AND travels >= 2.5 st/s.
  vs fretted run     : a run climbs in ~1-st stair steps; a riser climbs in
                       tiny per-frame increments. step_frac caps the share of
                       the climb delivered by >0.6 st jumps.

Needs the 'audio' extra (librosa + scipy):  pip install score2motion[audio]

    python3 -m score2motion.fretted.risers placed.json stem.wav --t0 0 \
        --out risers.json [--slides slides.json]
"""
import argparse
import json

MIN_ST, MIN_DUR = 3.0, 0.30       # the gesture: >= 3 semitones over >= 300 ms
RATE_MIN = 2.5                    # st/s -- slower is drift, not a hand
DD = 0.7                          # max pullback inside one riser (kills vibrato)
JUMP = 1.0                        # frame continuity, same law as slides.py
STEP_ST, STEP_FRAC_MAX = 0.6, 0.4  # stair-step share cap (kills fretted runs)
MONO_MIN = 0.6
GAP = 6                           # bridge unvoiced blinks <= 30 ms, <= 1 st


def main(argv=None):
    ap = argparse.ArgumentParser(description="detect single-note risers")
    ap.add_argument("plan")
    ap.add_argument("stem")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slides", help="known note-pair slides json (cross-check)")
    a = ap.parse_args(argv)

    import librosa
    import numpy as np
    from scipy.signal import medfilt

    KNOWN = json.load(open(a.slides))["slides"] if a.slides else []
    notes = json.load(open(a.plan))["notes"]
    if not notes:
        json.dump({"risers": []}, open(a.out, "w"), indent=1)
        print("0 riser(s) found (empty plan)")
        return
    SR = 16000
    end = max(n["t_off"] for n in notes) + 2.0  # a riser may overrun the last note
    y, sr = librosa.load(a.stem, sr=SR, offset=a.t0, duration=end)
    HOP = 80                                    # 5ms
    f0, _, _ = librosa.pyin(y, fmin=30, fmax=400, sr=sr, hop_length=HOP,
                            frame_length=2048)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0]
    L = min(len(f0), len(rms))
    f0, rms = f0[:L], rms[:L]
    t_of = lambda i: i * HOP / sr
    midi = 69 + 12 * np.log2(np.where(np.isnan(f0), 1e-9, f0) / 440.0)
    midi[np.isnan(f0)] = np.nan
    print(f"f0 track: {L} frames over {end:.1f}s, "
          f"voiced {np.mean(~np.isnan(f0)) * 100:.0f}%")

    track = midi.copy()           # blink-filled so one ramp reads as one ramp
    i = 0
    while i < L:
        if np.isnan(track[i]):
            j = i
            while j < L and np.isnan(track[j]):
                j += 1
            if 0 < i and j < L and j - i <= GAP \
                    and abs(track[j] - track[i - 1]) <= JUMP:
                track[i:j] = np.linspace(track[i - 1], track[j],
                                         j - i + 2)[1:-1]
            i = j
        else:
            i += 1

    def swings_up(m, dd):
        """Maximal rising swings tolerating pullback <= dd -> (lo, hi)."""
        out, lo, hi = [], 0, 0
        for k in range(1, len(m)):
            if m[k] >= m[hi]:
                hi = k
            elif m[hi] - m[k] > dd or m[k] < m[lo]:
                if hi > lo:
                    out.append((lo, hi))
                lo = hi = k
        if hi > lo:
            out.append((lo, hi))
        return out

    cands = []                    # (frame_lo, frame_hi, pitch_lo, pitch_hi)
    g = 0
    while g < L:
        if np.isnan(track[g]):
            g += 1
            continue
        h = g
        while h < L and not np.isnan(track[h]):
            h += 1
        if h - g >= 40:           # >= 200 ms voiced at all
            run = track[g:h]
            mf = medfilt(run, 7) if h - g >= 7 else run
            cuts = [0] + [k + 1 for k in
                          np.where(np.abs(np.diff(mf)) > JUMP)[0]] + [len(mf)]
            for c0, c1 in zip(cuts, cuts[1:]):
                if c1 - c0 < 20:
                    continue
                for lo, hi in swings_up(mf[c0:c1], DD):
                    cands.append((g + c0 + lo, g + c0 + hi,
                                  float(mf[c0 + lo]), float(mf[c0 + hi])))
        g = h

    def note_at(t):
        k = None
        for ix, n in enumerate(notes):
            if n["t_on"] <= t + 1e-9:
                k = ix
            else:
                break
        return k

    risers, nearmiss, reclassed = [], [], []
    for a_i, b_i, p_lo, p_hi in cands:
        net, dur = p_hi - p_lo, t_of(b_i - a_i)
        if net < 2.0 or dur < 0.20:
            continue
        t_lo, t_hi = t_of(a_i), t_of(b_i)
        known = next((s for s in KNOWN if t_lo < s["to_t"] + 0.05
                      and t_hi > s["from_t"] - 0.05), None)
        seg = track[a_i:b_i + 1]
        d = np.diff(seg)
        sig = d[np.abs(d) > 0.02]
        mono = float(np.mean(sig > 0)) if len(sig) else 0.0
        step_frac = float(d[d > STEP_ST].sum()) / net
        mfseg = medfilt(seg, 7) if len(seg) >= 7 else seg
        pull = float(np.max(np.maximum.accumulate(mfseg) - mfseg))
        bridge = None             # endpoints match one planned a->b step
        for ka, (na, nb) in enumerate(zip(notes, notes[1:])):
            if (abs(nb["t_on"] - t_hi) <= 0.10
                    and nb["t_on"] - na["t_off"] <= 0.15
                    and abs(p_hi - nb["pitch"]) <= 0.75
                    and abs(p_lo - na["pitch"]) <= 1.0):
                bridge = (ka, ka + 1)
                break
        rate = net / max(dur, 1e-9)
        fails = []
        if net < MIN_ST:
            fails.append(f"net {net:.1f} st < {MIN_ST:.0f}")
        if dur < MIN_DUR:
            fails.append(f"dur {dur * 1000:.0f} ms < {MIN_DUR * 1000:.0f}")
        if rate < RATE_MIN:
            fails.append(f"rate {rate:.1f} st/s < {RATE_MIN} (drift)")
        if mono < MONO_MIN:
            fails.append(f"mono {mono:.0%} < {MONO_MIN:.0%}")
        if step_frac > STEP_FRAC_MAX:
            fails.append(f"stair-stepped {step_frac:.0%} (fretted run)")
        if fails:
            if bridge:
                fails.append(f"= planned slide {bridge[0]}->{bridge[1]}")
            nearmiss.append((t_lo, p_lo, net, dur, ", ".join(fails)))
            continue
        if known is not None:
            reclassed.append((t_lo, t_hi, net, dur, known))
            continue              # it IS one of the known note-pair slides

        # what follows the ramp top: cut? register jump? drop-out?
        cut_ms = 0
        for k in range(b_i + 1, min(L, b_i + 41)):
            if np.isnan(midi[k]):
                k2 = k
                while k2 < L and np.isnan(midi[k2]):
                    k2 += 1
                cut_ms = (k2 - k) * HOP / sr * 1000
                break
        w0 = b_i + 4
        w1 = min(L, w0 + int(0.5 * sr / HOP))
        after_v = float(np.mean(~np.isnan(midi[w0:w1]))) if w1 > w0 else 0.0
        r_during = float(np.mean(rms[max(0, b_i - int(0.3 * sr / HOP)):b_i + 1]))
        r_after = float(np.mean(rms[w0:w1])) if w1 > w0 else 0.0
        nxt = next((n for n in notes if n["t_on"] > t_hi - 0.02), None)
        parts = []
        if cut_ms >= 50:
            parts.append(f"f0 cuts for {cut_ms:.0f} ms")
        if r_after < 0.2 * r_during or after_v < 0.25:
            parts.append(f"drops out (RMS falls to "
                         f"{r_after / max(r_during, 1e-9):.0%})")
        if nxt is None:
            parts.append("end of piece")
        else:
            dp_next = nxt["pitch"] - p_hi
            word = ("drop: resolves DOWN" if dp_next <= -5
                    else "leaps UP" if dp_next >= 5 else "lands: continues")
            parts.append(f"{word} {dpэ:+.1f} st to p{nxt['pitch']}")
        anch = note_at(t_lo)
        a_n = notes[anch] if anch is not None else None
        risers.append({
            "t": round(t_lo, 3), "t_end": round(t_hi, 3),
            "from_pitch": round(p_lo, 1), "to_pitch": round(p_hi, 1),
            "semitones": round(net, 2), "duration_s": round(dur, 3),
            "rate_st_per_s": round(rate, 1), "cut_after_ms": round(cut_ms),
            "mono": round(mono, 2), "pullback": round(pull, 2),
            "step_frac": round(step_frac, 2),
            "anchor_idx": anch,
            "anchor": (f"p{a_n['pitch']} s{a_n.get('string')} "
                       f"f{a_n.get('fret')}" if a_n else "none"),
            "bridge": bridge, "followed_by": "; ".join(parts)})

    json.dump({"risers": risers}, open(a.out, "w"), indent=1)
    print(f"{len(risers)} riser(s) found:")
    for r in risers:
        print(f"   t {r['t']:7.2f}s  {r['from_pitch']:.1f} -> "
              f"{r['to_pitch']:.1f} midi  +{r['semitones']:.1f} st over "
              f"{r['duration_s'] * 1000:.0f} ms -> {r['followed_by']}")
    if nearmiss:
        print(f"{len(nearmiss)} near-miss up-glide(s) rejected:")
        for t, p, net, dur, why in sorted(nearmiss):
            print(f"   t {t:7.2f}s  from {p:.1f}  +{net:.1f} st / "
                  f"{dur * 1000:.0f} ms: {why}")


if __name__ == "__main__":
    main()
