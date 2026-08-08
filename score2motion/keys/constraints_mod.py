"""Turn hand timelines into Kimodo generation constraints.

Uses Kimodo's own `EndEffectorConstraintSet` — "fixing selected end-effector positions on
given frames". Strategy is two-stage self-conditioning:

  1. a BASE take is generated unconstrained ("a person standing playing a keyboard")
  2. its LeftHand/RightHand tracks are EDITED to the MIDI hand timelines, and the edited
     take is fed back as an end-effector constraint set
  3. the constrained generation re-imagines the whole body so that reaching those keys at
     those times moves like a person

The diffusion owns body coherence; the MIDI owns where the hands are. Fingers are dead in
the generative model and are injected afterwards (see fingers_mod.py).
"""
from __future__ import annotations

import numpy as np
import torch


def build_hand_constraints(
    skeleton,
    base_out: dict,
    targets: dict,          # {"LeftHand": [T,3] | None, "RightHand": [T,3] | None}
    *,
    device,
):
    """EndEffectorConstraintSet pinning the hands to the MIDI timelines on every frame."""
    from kimodo.constraints import EndEffectorConstraintSet

    joints = base_out["posed_joints"]
    if joints.dim() == 4:                     # [B,T,J,3] -> first sample
        joints = joints[0]
    rots = base_out["global_rot_mats"]
    if rots.dim() == 5:
        rots = rots[0]
    joints = joints.clone()

    # the model APPLIES constraints in its 30-joint prediction space (the crash that
    # taught us: index 3642 vs size 121*30). If the base output is already upgraded to the
    # 77-joint export skeleton, slice it back DOWN using kimodo's own index mapping.
    n30 = len(skeleton.bone_order_names)
    if joints.shape[1] != n30:
        upgraded = getattr(skeleton, "somaskel77", None)
        if upgraded is not None and joints.shape[1] == len(upgraded.bone_order_names):
            idx = torch.tensor(skeleton.get_skel_slice(upgraded), device=joints.device)
            joints = joints[:, idx]
            rots = rots[:, idx]
        else:
            raise ValueError(f"unexpected joint count {joints.shape[1]} (want {n30})")

    ix = {n: i for i, n in enumerate(skeleton.bone_order_names)}
    names = []
    T = joints.shape[0]
    for jname, tl in targets.items():
        if tl is None:
            continue
        arr = torch.as_tensor(np.asarray(tl), dtype=joints.dtype, device=joints.device)
        n = min(T, arr.shape[0])
        joints[:n, ix[jname]] = arr[:n]
        names.append(jname)
    if not names:
        raise ValueError("no hand targets given")

    frame_indices = torch.arange(T, device=joints.device)
    return EndEffectorConstraintSet(
        skeleton,
        frame_indices,
        joints,
        rots,
        None,
        joint_names=names,
        to_crop=False,
    ).to(device)
