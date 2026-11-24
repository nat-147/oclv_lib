"""Turn a 2-way continuous-shifting-sprites dataset (train / test) into a 3-way one (train / valid / test), like CLEVR.

The existing test split is cut in two without regenerating anything (sprites are generated from the global RNG, so a
regeneration would not reproduce the same data):
  test/{images,vs}_0 .. N_valid-1        -> valid/{images,vs}_0 .. N_valid-1
  test/{images,vs}_N_valid .. N_test-1   -> test/{images,vs}_0 .. N_test-N_valid-1
train/ is untouched. config.txt is rewritten in the 3-way format of ContinuousShiftingSpritesloops (original kept as
config.txt.bak_2way) and the move is logged in PROVENANCE.txt.

Usage: python scripts/split_sprites_dataset.py <dataset_dir> [--valid 500]
"""
import argparse
import datetime
import os
import re
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--valid", type=int, default=500)
args = ap.parse_args()
D = args.path.rstrip("/")
cfg_path = os.path.join(D, "config.txt")

cfg = open(cfg_path).read()
assert "num validation" not in cfg, f"{D} is already split"
n_train = int(re.search(r"num training: (\d+)", cfg).group(1))
n_test = int(re.search(r"num testing: (\d+)", cfg).group(1))
n_valid, n_new_test = args.valid, n_test - args.valid
assert 0 < n_valid < n_test

test_dir, valid_dir = os.path.join(D, "test"), os.path.join(D, "valid")
for kind in ("images", "vs"):
    have = sum(1 for f in os.listdir(test_dir) if re.fullmatch(rf"{kind}_\d+\.pt", f))
    assert have == n_test, f"{test_dir}: {have} {kind} files, expected {n_test}"
    for i in range(n_test):
        assert os.path.isfile(os.path.join(test_dir, f"{kind}_{i}.pt")), (kind, i)
assert not os.path.exists(valid_dir) or not os.listdir(valid_dir), f"{valid_dir} is not empty"
os.makedirs(valid_dir, exist_ok=True)

for kind in ("images", "vs"):
    for i in range(n_valid):                       # first part -> valid (same index)
        os.rename(os.path.join(test_dir, f"{kind}_{i}.pt"), os.path.join(valid_dir, f"{kind}_{i}.pt"))
    for i in range(n_valid, n_test):               # rest -> test, renumbered from 0 (targets were freed above)
        os.rename(os.path.join(test_dir, f"{kind}_{i}.pt"), os.path.join(test_dir, f"{kind}_{i - n_valid}.pt"))

shutil.copy2(cfg_path, cfg_path + ".bak_2way")
lines = cfg.rstrip("\n").split("\n")
out = []
for line in lines:
    if line.startswith("num testing:"):
        out.append(f"num validation: {n_valid}")
        out.append(f"num testing: {n_new_test}")
    else:
        out.append(line)
with open(cfg_path, "w") as f:
    f.write("\n".join(out) + "\n\n")   # same layout as create_config(): repr ends with \n, print adds one more
with open(os.path.join(D, "PROVENANCE.txt"), "a") as f:
    f.write(f"[{datetime.date.today()}] split test ({n_test}) into valid 0..{n_valid - 1} (old test 0..{n_valid - 1}) and "
            f"test 0..{n_new_test - 1} (old test {n_valid}..{n_test - 1}); train ({n_train}) unchanged. "
            "Original config.txt kept as config.txt.bak_2way (scripts/split_sprites_dataset.py)\n")
print(f"{D}: train {n_train} / valid {n_valid} / test {n_new_test}")
