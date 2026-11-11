"""Prepare a trial run dir that resumes from the latest usable shared checkpoint.

Shared checkpoints live in outputs/runs/<PREFIX>shared_<N>k. One at step S can be reused by any
trial whose lambda warmup starts at or after S, because the velocity loss weight is 0 until then.

Usage: python scripts/make_trial_config.py <trial_dir> <override...>
Prints "START=<warmup start>" and "SHARED=<step or 0>".
SHARED=0 means no usable checkpoint: the trial trains from scratch (trial dir is still created).
"""
import shutil
import sys
import uuid
from pathlib import Path

from omegaconf import OmegaConf, open_dict

RUNS = Path("outputs/runs")
PREFIX = "sav-mean_z-occ-temp-clevr6-hparamtune-"
MODEL_YAML = Path("config/model/sa-velocity_clevr.yaml")
FILES = ["train_checkpoint.pt", "model.pt", "best_model.pt", "train_state.yaml"]

trial = Path(sys.argv[1])
overrides = OmegaConf.from_dotlist(sys.argv[2:])

# Tunable keys come from the current model yaml, then CLI overrides, never from the shared run.
model_yaml = OmegaConf.load(MODEL_YAML).model
tunable = OmegaConf.create({"model": {
    "lambda_warmup_steps_interval": model_yaml.lambda_warmup_steps_interval,
    "vel_loss_weights": model_yaml.vel_loss_weights,
}})
start = OmegaConf.merge(tunable, overrides).model.lambda_warmup_steps_interval[0]

shared_step = 0
for step in (30_000, 20_000, 10_000):
    d = RUNS / f"{PREFIX}shared_{step // 1000}k"
    if step <= start and all((d / f).exists() for f in FILES + ["train_config.yaml"]):
        shared_step, shared = step, d
        break

trial.mkdir(parents=True, exist_ok=False)
if shared_step:
    cfg = OmegaConf.merge(OmegaConf.load(shared / "train_config.yaml"), tunable, overrides)
    with open_dict(cfg):
        cfg.curr_dir = trial.name
        cfg.uuid = str(uuid.uuid4())  # separate wandb run per trial
        cfg.trainer.steps = 50_000
    for f in FILES:
        shutil.copy2(shared / f, trial / f)
    OmegaConf.save(cfg, trial / "train_config.yaml")

print(f"START={start}")
print(f"SHARED={shared_step}")
