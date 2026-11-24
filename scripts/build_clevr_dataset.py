"""Generate the CLEVR dataset of config/dataset/clevr.yaml (train / valid / test) with the current generator,
exactly as the training script would build it (same dataset args, same seeds), but without starting a run.

Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/build_clevr_dataset.py [hydra overrides, e.g. dataset.num_gen_procs=16]
"""
import os
import sys
from pathlib import Path

import hydra
from hydra.experimental import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from data.datasets import hydra_to_dataset_args  # noqa: E402

with initialize_config_dir(config_dir=str(ROOT / "config")):
    cfg = compose("train_object_discovery", overrides=["model=sa-velocity_clevr", "dataset=clevr"] + sys.argv[1:])

cls = hydra.utils.get_class(cfg.dataset._target_)
args = hydra_to_dataset_args(cfg)
print(f"path={args['path']} train={args['num_training']} valid={args.get('num_validation', 0)} "
      f"test={args['num_testing']}", flush=True)
splits = [("train", dict(train=True))]
if args.get("num_validation", 0) > 0:
    splits.append(("valid", dict(train=False, split="valid")))
splits.append(("test", dict(train=False, split="test")))
for name, kw in splits:
    ds = cls(**{**args, **kw})
    print(f"{name}: {len(ds)} sequences ready", flush=True)
print("done", flush=True)
