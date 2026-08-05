"""The two-stage driver: prompt + MIDI in, playing-skeleton BVH out.

    stage A  base take        "a person standing playing a keyboard" (unconstrained)
    stage B  author the set   place the virtual keyboard from the base take, evaluate the
                              MIDI hand timelines into per-frame wrist positions
    stage C  constrained take regenerate with EndEffectorConstraintSet on both hands
    stage D  fingers          inject press/travel motion into the finger channels
    stage E  export           somaskel77 BVH, ready for any retarget pipeline
"""
from __future__ import annotations

import numpy as np
import torch

from .constraints_mod import build_hand_constraints
from .fingers_mod import inject_fingers
from .keyboard import PlacedKeyboard
from .timeline import HandTimeline

PROMPT_DEFAULT = ("a person standing in place and playing an electronic keyboard with "
                  "both hands, upper body moving gently with the music")


def generate_player(
    plan,
    *,
    seconds: float,
    prompt: str = PROMPT_DEFAULT,
    model_name: str = "soma",
    device: str = "cuda",
    seed: int = 7,
    diffusion_steps: int = 30,
    out_bvh: str = "played.bvh",
    up_axis: int = 1,
) -> dict:
    from kimodo.model.load_model import load_model
    from kimodo.exports.bvh import save_motion_bvh

    torch.manual_seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA not available in this torch build — falling back to CPU "
              "(slower but works; a 4s take is a few minutes)")
        device = "cpu"
    model = load_model(modelname=model_name, device=device)
    motion_rep = model.motion_rep
    skel30 = motion_rep.skeleton
    fps = motion_rep.fps
    n_frames = int(round(seconds * fps))
    print(f"[A] base take: '{prompt}' {n_frames}f @ {fps}fps")
    base = model([prompt], [n_frames], diffusion_steps, multi_prompt=True,
                 cfg_weight=[2.0, 2.0], num_samples=1)

    joints = base["posed_joints"]
    j_np = (joints[0] if joints.dim() == 4 else joints).detach().cpu().numpy()
    names = list(skel30.bone_order_names)
    kb = PlacedKeyboard.place_from_take(j_np, names, plan, up_axis=up_axis)
    up = np.zeros(3); up[up_axis] = 1.0
    tl_L = HandTimeline(kb, plan, "Left", up)
    tl_R = HandTimeline(kb, plan, "Right", up)
    targets = {
        "LeftHand": tl_L.sample(n_frames, fps) if tl_L else None,
        "RightHand": tl_R.sample(n_frames, fps) if tl_R else None,
    }
    print("[B] keyboard placed; hand timelines sampled")

    cset = build_hand_constraints(skel30, base, targets, device=device)
    print("[C] constrained take...")
    final = model([prompt], [n_frames], diffusion_steps, multi_prompt=True,
                  cfg_weight=[2.0, 2.0], num_samples=1, constraint_lst=[cset])

    torch.save({k: v for k, v in final.items() if torch.is_tensor(v)},
               out_bvh + ".raw.pt")            # cache: iterate on D/E without re-diffusing
    skel77 = skel30.somaskel77
    n77 = len(skel77.bone_order_names)
    local77 = final["local_rot_mats"]
    while local77.dim() > 4:                    # the model batches: [1, T, J, 3, 3]
        local77 = local77[0]
    if local77.shape[1] != n77:                 # genuinely 30-joint: upgrade
        out77 = skel30.output_to_SOMASkeleton77(
            {"local_rot_mats": local77, "root_positions": final["root_positions"]})
        local77 = out77["local_rot_mats"]
        if local77.dim() == 4 and local77.shape[0] == 1:
            local77 = local77[0]
    print("[D] injecting fingers...")
    local77 = inject_fingers(local77, skel77, plan, fps=fps)

    root_pos = final["root_positions"]
    if torch.is_tensor(root_pos) and root_pos.dim() == 3:
        root_pos = root_pos[0]
    save_motion_bvh(out_bvh, local77, root_pos, skeleton=skel77, fps=fps)
    print(f"[E] saved {out_bvh}")

    # [V] built-in verification: how far are the constrained wrists from their timelines?
    rot77, pos77, *_ = skel77.fk(local77, root_pos)
    ixw = {n: i for i, n in enumerate(skel77.bone_order_names)}
    import numpy as _np
    for jname, tl in targets.items():
        if tl is None:
            continue
        got = pos77[:, ixw[jname]].detach().cpu().numpy()
        want = _np.asarray(tl)[: got.shape[0]]
        err = _np.linalg.norm(got - want, axis=1)
        print(f"[V] {jname}: wrist-to-timeline mean {err.mean()*100:.1f} cm, "
              f"p95 {_np.percentile(err, 95)*100:.1f} cm")
    return {"bvh": out_bvh, "keyboard": kb, "fps": fps, "frames": n_frames}
