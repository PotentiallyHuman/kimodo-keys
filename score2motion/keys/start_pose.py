"""Author the opening pose, then let Kimodo generate outward from it.

August's insight (2026-08-05): rather than generate a generic take and rotate/reposition it
afterwards, put the skeleton in the correct keyboard-playing stance and make THAT the
take's keyframe. Kimodo supports this natively — `FullBodyConstraintSet` fixes full-body
global positions/rotations on chosen keyframes, and the constraint is visible to the model
at every denoising step (it is a time-series handed over before generation, not a prompt).

The stance follows the same law the Blender side uses: upper arm 45 deg down-and-forward,
forearm horizontal, hands at keyboard height in front of the body, palms down.

Note on fingers: Kimodo predicts a 30-joint skeleton whose hands are a wrist plus two
direction markers — the 50 finger joints exist only in the export skeleton and are added
from a fixed relaxed pose. Finger motion therefore cannot be conditioned here; it is
injected after generation (see fingers_mod).
"""
from __future__ import annotations

import numpy as np
import torch


def playing_stance(skeleton, base_out, *, up_axis: int = 1, board_drop: float = 0.22,
                   board_reach: float = 0.38):
    """Edit the base take's FIRST frame into a keyboard-playing stance.

    Returns (positions, rotations) for frame 0, shaped for FullBodyConstraintSet.
    Only the arms are re-authored; the rest of the pose is the model's own, so the
    stance stays inside the distribution the model knows.
    """
    joints = base_out["posed_joints"]
    if joints.dim() == 4:
        joints = joints[0]
    rots = base_out["global_rot_mats"]
    if rots.dim() == 5:
        rots = rots[0]
    n30 = len(skeleton.bone_order_names)
    if joints.shape[1] != n30:
        upgraded = getattr(skeleton, "somaskel77", None)
        if upgraded is not None and joints.shape[1] == len(upgraded.bone_order_names):
            idx = torch.tensor(skeleton.get_skel_slice(upgraded), device=joints.device)
            joints, rots = joints[:, idx], rots[:, idx]

    ix = {n: i for i, n in enumerate(skeleton.bone_order_names)}
    p = joints[0].clone()

    # body frame from the shoulders: forward = the way the chest faces
    up = np.zeros(3); up[up_axis] = 1.0
    up_t = torch.tensor(up, dtype=p.dtype, device=p.device)
    l_sh, r_sh = p[ix["LeftShoulder"]], p[ix["RightShoulder"]]
    across = (l_sh - r_sh)
    across[up_axis] = 0.0
    across = across / max(1e-6, float(torch.linalg.norm(across)))
    fwd = torch.linalg.cross(up_t, across)
    fwd = fwd / max(1e-6, float(torch.linalg.norm(fwd)))

    for side, sgn in (("Left", 1.0), ("Right", -1.0)):
        sh = p[ix[f"{side}Shoulder"]]
        arm, fore, hand = ix[f"{side}Arm"], ix[f"{side}ForeArm"], ix[f"{side}Hand"]
        l_up = float(torch.linalg.norm(joints[0][fore] - joints[0][arm]))
        l_fa = float(torch.linalg.norm(joints[0][hand] - joints[0][fore]))
        # 45 deg down-and-forward, then horizontal to the hand — the stance law
        elbow = sh + (fwd - up_t) / (2 ** 0.5) * l_up
        wrist = elbow + fwd * l_fa
        # hands sit a comfortable width apart, at board height in front of the body
        wrist = wrist + across * (sgn * 0.14)
        wrist[up_axis] = sh[up_axis] - board_drop
        elbow[up_axis] = sh[up_axis] - board_drop * 0.55
        p[arm], p[fore], p[hand] = sh, elbow, wrist
        # keep the hand-direction markers ahead of the wrist so the hand reads palm-down
        for marker in (f"{side}HandThumbEnd", f"{side}HandMiddleEnd"):
            if marker in ix:
                p[ix[marker]] = wrist + fwd * 0.06
    return p, rots[0]


def build_start_pose_constraint(skeleton, base_out, *, device, n_keyframes: int = 1,
                                up_axis: int = 1):
    """FullBodyConstraintSet pinning the authored stance on the opening frame(s)."""
    from kimodo.constraints import FullBodyConstraintSet

    pos0, rot0 = playing_stance(skeleton, base_out, up_axis=up_axis)
    frames = torch.arange(n_keyframes, device=pos0.device)
    pos = pos0.unsqueeze(0).repeat(n_keyframes, 1, 1)
    rot = rot0.unsqueeze(0).repeat(n_keyframes, 1, 1, 1)
    return FullBodyConstraintSet(
        skeleton, frames, pos, rot, None, to_crop=False,
    ).to(device)
