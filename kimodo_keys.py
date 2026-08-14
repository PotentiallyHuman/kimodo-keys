"""Backwards-compatible shim: kimodo-keys is now score2motion.keys.

The rename left this as a FLAT module that re-exported names. That covered
``from kimodo_keys import is_white`` but silently broke every
``from kimodo_keys.keyboard import ...`` -- the submodule path most of the old
code actually used -- with "kimodo_keys is not a package". A compatibility shim
that only covers half the old surface is worse than none, because the failure
lands on the user rather than here.

So the submodules are registered under the old names too. CPython's importer
re-checks sys.modules after importing a parent (the "crazy side-effects" branch
in _find_and_load_unlocked), which is what lets a non-package forward its
children this way.
"""
import importlib
import pkgutil
import sys
import warnings

from score2motion import keys as _keys
from score2motion.keys import *  # noqa: F401,F403

# forward every submodule under its old name, so the old import paths resolve
for _m in pkgutil.iter_modules(_keys.__path__):
    try:
        sys.modules[f"{__name__}.{_m.name}"] = importlib.import_module(
            f"score2motion.keys.{_m.name}")
    except ImportError:
        # a submodule needing Blender or a heavy optional dep is not a reason to
        # break the whole shim -- the rest still resolves
        pass
__path__ = _keys.__path__          # so `import kimodo_keys.x` also finds new ones

warnings.warn(
    "kimodo_keys has moved to score2motion.keys; this shim will be removed in 0.4",
    DeprecationWarning,
    stacklevel=2,
)
