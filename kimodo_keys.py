"""Backwards-compatible shim: kimodo-keys is now score2motion.keys."""
import warnings

from score2motion.keys import *  # noqa: F401,F403

warnings.warn(
    "kimodo_keys has moved to score2motion.keys; this shim will be removed in 0.4",
    DeprecationWarning,
    stacklevel=2,
)
