"""Prepare a trial run dir that resumes from the shared 10k checkpoint.

Usage: python scripts/make_trial_config.py <shared_dir> <trial_dir> <override...>
Exit 0: trial dir prepared (resume from shared checkpoint).
Exit 3: warmup starts before the shared step, so the trial must train from scratch.
"""
import shutil
import sys
import uuid
from pathlib import Path

from omegaconf import OmegaConf, open_dict

SHARED_STEP = 10_000
FILES = ["train_checkpoint.pt", "model.pt", "best_model.pt", "train_state.yaml"]

shared, trial = Path(sys.argv[1]), Path(sys.argv[2])
overrides = sys.argv[3:]

cfg = OmegaConf.load(shared / "train_config.yaml")
cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

if cfg.model.lambda_warmup_steps_interval[0] < SHARED_STEP:
    sys.exit(3)

with open_dict(cfg):
    cfg.curr_dir = trial.name
    cfg.uuid = str(uuid.uuid4())  # separate wandb run per trial
    cfg.trainer.steps = 50_000

trial.mkdir(parents=True, exist_ok=False)
for f in FILES:
    shutil.copy2(shared / f, trial / f)
OmegaConf.save(cfg, trial / "train_config.yaml")
print(f"prepared {trial} from {shared}")
