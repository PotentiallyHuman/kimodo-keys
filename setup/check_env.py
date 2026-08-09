"""Is this machine ready to run the pipeline? Say what is missing and how to get it.

Deliberately does NOT install anything by default. Silently downloading a few
hundred megabytes of Blender is the fastest way to break a first run: the platform
matrix is wide, PATH and permissions differ, managed machines forbid it, and every
Blender release moves the download. So this checks, reports, and prints the exact
one-line command for the platform it finds itself on.

`--auto-install` is available for people who want it done for them, and it is
opt-in on purpose.

    python3 setup/check_env.py
    python3 setup/check_env.py --auto-install
"""
import importlib
import os
import platform
import shutil
import subprocess
import sys

AUTO = "--auto-install" in sys.argv
OK, MISSING = [], []


def cmd_for(pkg):
    """The install line for the platform we are actually on."""
    sysname = platform.system()
    if sysname == "Darwin":
        return f"brew install {pkg}"
    if sysname == "Windows":
        return f"winget install {pkg}"
    for mgr, line in (("apt-get", f"sudo apt install {pkg}"),
                      ("dnf", f"sudo dnf install {pkg}"),
                      ("pacman", f"sudo pacman -S {pkg}"),
                      ("zypper", f"sudo zypper install {pkg}")):
        if shutil.which(mgr):
            return line
    return f"install {pkg} with your package manager"


def binary(name, why, version_flag="--version", pkg=None):
    path = shutil.which(name)
    if path:
        try:
            v = subprocess.run([path, version_flag], capture_output=True, text=True,
                               timeout=20).stdout.strip().split("\n")[0]
        except Exception:
            v = "(version unknown)"
        OK.append(f"{name:10s} {v}")
    else:
        MISSING.append((name, why, cmd_for(pkg or name)))


def module(name, why, pip_name=None):
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "")
        OK.append(f"{name:10s} {v}".rstrip())
    except ImportError:
        MISSING.append((name, why, f"pip install {pip_name or name}"))


print(f"python     {sys.version.split()[0]}  ({platform.system()} {platform.machine()})")
if sys.version_info < (3, 9):
    MISSING.append(("python3.9+", "the engine uses modern typing syntax",
                    "install a newer Python"))

binary("blender", "renders the performance, and is where the rig lives")
# ffmpeg takes a single dash; --version prints nothing and the check silently
# reported it as present with no version at all
binary("ffmpeg", "turns the rendered frames into a video file", version_flag="-version")
module("mido", "reads MIDI parts", "mido")
module("numpy", "geometry and audio maths")

# The model weights are a separate licence and a separate download. They are not
# redistributed with this repository -- fetch_kimodo.py pulls them from HuggingFace
# and shows NVIDIA's terms before it does.
models = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "models")
if os.path.isdir(models) and os.listdir(models):
    OK.append(f"{'weights':10s} present in models/")
else:
    MISSING.append(("motion weights",
                    "generates the body motion; separate licence, not shipped here",
                    "python3 setup/fetch_kimodo.py"))

print("\nready:")
for line in OK:
    print(f"  {line}")

if not MISSING:
    print("\neverything the pipeline needs is here.")
    sys.exit(0)

print("\nmissing:")
for name, why, how in MISSING:
    print(f"  {name}")
    print(f"      {why}")
    print(f"      {how}")

if not AUTO:
    print("\nRun those, or re-run this with --auto-install to attempt them for you.")
    print("Nothing was installed. Blender in particular is a large download and is\n"
          "better installed the way the rest of your system installs things.")
    sys.exit(1)

print("\n--auto-install given; attempting the commands above.")
failed = 0
for name, _why, how in MISSING:
    if how.startswith(("install ", "python3 ")):
        print(f"  skipping {name}: run `{how}` yourself")
        continue
    print(f"  $ {how}")
    if subprocess.run(how, shell=True).returncode != 0:
        print(f"  FAILED: {name} -- install it by hand and re-run this check")
        failed += 1
sys.exit(1 if failed else 0)
