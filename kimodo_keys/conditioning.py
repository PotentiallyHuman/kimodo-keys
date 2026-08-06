"""Hard hand conditioning — v0.2's core.

Kimodo's denoiser already receives constraints as (observed_motion, motion_mask): known
motion-rep channels plus a boolean mask, attended to as soft guidance. This module makes
those observations BINDING, the way a start frame binds a video model: at every denoising
step, the model's clean-motion estimate is corrected so the observed channels equal their
targets exactly, and the next step generates the rest of the body around them.

Not a post-fix: the clamp runs inside the sampling loop, before each DDIM update, so the
whole body is generated conditioned on hands that are already where the MIDI says.
"""
from __future__ import annotations

import torch


class _DenoiserProxy(torch.nn.Module):
    """Calls through to the real denoiser, capturing (motion_mask, observed_motion)."""

    def __init__(self, inner, state):
        super().__init__()
        self._inner = inner
        self._state = state

    def forward(self, *args, **kwargs):
        # call signature: (cfg_weight, motion, pad_mask, text_feat, text_pad_mask,
        #                  t, first_heading_angle, motion_mask, observed_motion, ...)
        if len(args) >= 9:
            self._state["mask"] = args[7]
            self._state["obs"] = args[8]
        else:
            self._state["mask"] = kwargs.get("motion_mask")
            self._state["obs"] = kwargs.get("observed_motion")
        return self._inner(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._inner, name)


class _SamplerProxy(torch.nn.Module):
    """Clamps the observed channels of pred_xstart before every DDIM update."""

    def __init__(self, inner, state):
        super().__init__()
        self._inner = inner
        self._state = state

    def forward(self, use_timesteps, x_t, pred_xstart, t):
        mask, obs = self._state.get("mask"), self._state.get("obs")
        if mask is not None and obs is not None and mask.any():
            m = mask.bool()                       # arrives as Long from the batching path
            if not hasattr(self, "_dbg_done"):
                delta = (obs.to(pred_xstart.dtype) - pred_xstart)[m if m.shape == pred_xstart.shape else m[..., :pred_xstart.shape[-2], :]]
                print(f"[hard-dbg] clamping {int(m.sum())} channels/frame-batch, "
                      f"mean |correction| {float(delta.abs().mean()):.4f} (normalized units)")
                self._dbg_done = True
            if m.shape != pred_xstart.shape:      # tolerate length padding differences
                T = min(m.shape[-2], pred_xstart.shape[-2])
                pred_xstart[..., :T, :] = torch.where(
                    m[..., :T, :], obs[..., :T, :], pred_xstart[..., :T, :])
            else:
                pred_xstart = torch.where(m, obs.to(pred_xstart.dtype), pred_xstart)
        return self._inner(use_timesteps, x_t, pred_xstart, t)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._inner, name)


def enable_hard_hand_conditioning(model) -> None:
    """Wrap the model in place. Safe when no constraints are active (mask empty = no-op)."""
    state: dict = {}
    if isinstance(model.denoiser, _DenoiserProxy):
        return                                     # already enabled
    model.denoiser = _DenoiserProxy(model.denoiser, state)
    model.sampler = _SamplerProxy(model.sampler, state)
    print("[hard] hand conditioning ENABLED: observed channels clamp every denoise step")
