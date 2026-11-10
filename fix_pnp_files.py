"""
fix_pnp_files.py: put the Pick & Place files into the M0NARCH repo as real files.

Problem: H1_robot-Pick_Place/ (and the unused top-level mujoco_menagerie/) are saved in git as
submodule links (mode 160000). A plain `git clone` gets empty folders, so Pick & Place cannot load.

What this does (run once in ~/M0NARCH, venv active; safe to run again):
  1. Moves the nested .git links/folders inside H1_robot-Pick_Place/ to ~/M0NARCH_backup_<time>/
     (moved, not deleted), so git can track the files themselves.
  2. Removes the two submodule links from git's index (your files stay on disk).
  3. .gitignore: keeps only what the simulator needs. Not committed: checkpoints/ and logs/
     (training output, already on the ROS-integration branch) and every menagerie robot except
     unitree_h1 and robotiq_2f85 (the full menagerie is ~2 GB).
  4. Stages H1_robot-Pick_Place/ and .gitignore (nothing else).
  5. Checks the staged files alone: exports them to a temp folder, as a fresh clone would get
     them, and loads the Pick & Place model from there with server.py's own loader.
"""
import datetime
import os
import shutil
import subprocess
import sys
import tempfile

PNP = "H1_robot-Pick_Place"
MODELS = ("unitree_h1", "robotiq_2f85")
NEEDED = [f"{PNP}/assets/h1_pick_place_scene.xml"] + [f"{PNP}/mujoco_menagerie/{m}" for m in MODELS]
IGNORE = [
    "# Pick & Place: only what the simulator needs (fix_pnp_files.py)",
    f"{PNP}/checkpoints/",
    f"{PNP}/logs/",
    f"{PNP}/MJDATA.TXT",
    f"{PNP}/mujoco_menagerie/*",
    f"!{PNP}/mujoco_menagerie/unitree_h1/",
    f"!{PNP}/mujoco_menagerie/robotiq_2f85/",
    f"!{PNP}/mujoco_menagerie/LICENSE",
    "__pycache__/",
    "MUJOCO_LOG.TXT",
]


def git(*args, check=True):
    r = subprocess.run(("git",) + args, text=True, capture_output=True)
    if check and r.returncode != 0:
        sys.exit(f"FAILED: git {' '.join(args)}\n{r.stderr}")
    return r.stdout


def is_link(path):
    return any(line.startswith("160000") for line in git("ls-files", "-s", "--", path, check=False).splitlines())


if not (os.path.isfile("server.py") and os.path.isdir(".git")):
    sys.exit("Run this inside ~/M0NARCH (the folder with server.py and .git).")
missing = [p for p in NEEDED if not os.path.exists(p)]
if missing:
    sys.exit("These Pick & Place files are missing on this computer, nothing changed:\n  " + "\n  ".join(missing))

# 1. nested .git links/folders -> backup
backup = os.path.expanduser(f"~/M0NARCH_backup_{datetime.datetime.now():%Y%m%d_%H%M%S}")
moved = []
for root, dirs, files in os.walk(PNP):
    for name in list(dirs) + files:
        if name == ".git":
            src = os.path.join(root, name)
            dst = os.path.join(backup, src)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            moved.append(src)
    dirs[:] = [d for d in dirs if d != ".git"]
print(f"1. nested git links moved to {backup}: {len(moved)}" if moved else "1. no nested git links")

# 2. submodule links out of the index
for path in (PNP, "mujoco_menagerie"):
    if is_link(path):
        git("rm", "-q", "--cached", path)
        print(f"2. {path}: submodule link removed from git (files stay on disk)")
if os.path.isfile(".gitmodules"):
    for path in (PNP, "mujoco_menagerie"):
        git("config", "-f", ".gitmodules", "--remove-section", f"submodule.{path}", check=False)
    if not open(".gitmodules").read().strip():
        git("rm", "-q", "-f", ".gitmodules", check=False)
    else:
        git("add", ".gitmodules")

# 3. .gitignore
have = open(".gitignore").read().splitlines() if os.path.exists(".gitignore") else []
add = [x for x in IGNORE if x not in have]
if add:
    with open(".gitignore", "a") as f:
        f.write(("\n" if have and have[-1] else "") + "\n".join(add) + "\n")
print(f"3. .gitignore: {len(add)} lines added" if add else "3. .gitignore already complete")

# 4. stage
git("add", ".gitignore", PNP)
if is_link(PNP):
    sys.exit(f"FAILED: {PNP} is still a link. Send me this message.")
staged = [line for line in git("ls-files", "--", PNP).splitlines()]
size = sum(os.path.getsize(p) for p in staged if os.path.isfile(p))
print(f"4. staged {len(staged)} files in {PNP}/ ({size / 1e6:.1f} MB)")

# 5. check the staged files alone (what a fresh clone will get)
tmp = tempfile.mkdtemp(prefix="m0narch_check_")
git("checkout-index", "-a", f"--prefix={tmp}/")
repo = os.getcwd()
chk = subprocess.run([sys.executable, "-c", f'''
import math, os, sys
sys.path.insert(0, "{repo}")
import mujoco, numpy as np
from pnp_sorting import prepare_sorting_scene
src = open("{repo}/server.py").read()
i, j = src.index("def load_pick_place_model"), src.index("model_pnp, data_pnp = load_pick_place_model()")
ns = dict(os=os, mujoco=mujoco, np=np, math=math, prepare_sorting_scene=prepare_sorting_scene)
exec(src[i:j], ns)
os.chdir("{tmp}")                       # load ONLY the staged files, as a fresh clone would
m, d = ns["load_pick_place_model"]()
print(f"5. fresh-clone check OK: Pick & Place model loads from the staged files ({{m.nbody}} bodies)")
'''], text=True, capture_output=True)
shutil.rmtree(tmp, ignore_errors=True)
print(chk.stdout.strip() or chk.stderr.strip()[-600:])
if chk.returncode != 0:
    sys.exit("Check FAILED: do not commit yet; send me the message above.")
print("\nDone. Commit together with the Pick & Place code (see the git steps).")
