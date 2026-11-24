"""Prepare a trial run dir that resumes from the latest usable shared checkpoint.

Shared checkpoints live in outputs/runs/<PREFIX>shared_<N>k. One at step S can be reused by any
trial whose lambda warmup starts at or after S, because the velocity loss weight is 0 until then.

Usage: [CAMPAIGN_PREFIX=... MODEL_YAML=...] python scripts/make_trial_config.py <trial_dir> <override...>
(defaults: the sa-velocity CLEVR6 campaign; the MONet campaign sets
 CAMPAIGN_PREFIX=mon-mean_z-clevr6-hparamtune- MODEL_YAML=config/model/monet-velocity_clevr.yaml)
Shared checkpoints are found by globbing <PREFIX>shared_<N>k; a trial with use_lambda_warmup=false starts the
vel loss at step 0 and therefore always trains from scratch.
Prints "START=<warmup start>" and "SHARED=<step or 0>".
SHARED=0 means no usable checkpoint: the trial trains from scratch (trial dir is still created).
"""
import os
import re
import shutil
import sys
import uuid
from pathlib import Path

from omegaconf import OmegaConf, open_dict

RUNS = Path("outputs/runs")
PREFIX = os.environ.get("CAMPAIGN_PREFIX", "sav-mean_z-clevr6-hparamtune-")
MODEL_YAML = Path(os.environ.get("MODEL_YAML", "config/model/sa-velocity_clevr.yaml"))
FILES = ["train_checkpoint.pt", "model.pt", "best_model.pt", "train_state.yaml"]

trial = Path(sys.argv[1])
overrides = OmegaConf.from_dotlist(sys.argv[2:])

# Tunable keys come from the current model yaml, then CLI overrides, never from the shared run.
model_yaml = OmegaConf.load(MODEL_YAML).model
tunable = OmegaConf.create({"model": {
    "use_lambda_warmup": model_yaml.get("use_lambda_warmup", True),
    "lambda_warmup_steps_interval": model_yaml.lambda_warmup_steps_interval,
    "vel_loss_weights": model_yaml.vel_loss_weights,
}})
merged = OmegaConf.merge(tunable, overrides).model
start = merged.lambda_warmup_steps_interval[0] if merged.use_lambda_warmup else 0

shared_step = 0
steps = sorted((int(m.group(1)) * 1000, d) for d in RUNS.glob(f"{PREFIX}shared_*k")
               if (m := re.fullmatch(re.escape(PREFIX) + r"shared_(\d+)k", d.name)))
for step, d in reversed(steps):
    if step <= start and all((d / f).exists() for f in FILES + ["train_config.yaml"]):
        shared_step, shared = step, d
        break

trial.mkdir(parents=True, exist_ok=False)
if shared_step:
    cfg = OmegaConf.merge(OmegaConf.load(shared / "train_config.yaml"), tunable, overrides)
    with open_dict(cfg):
        cfg.curr_dir = trial.name
        cfg.uuid = str(uuid.uuid4())  # separate wandb run per trial
        if OmegaConf.select(overrides, "trainer.steps") is None:
            cfg.trainer.steps = 50_000  # shared checkpoints were saved with their own (shorter) budget
    for f in FILES:
        shutil.copy2(shared / f, trial / f)
    OmegaConf.save(cfg, trial / "train_config.yaml")

print(f"START={start}")
print(f"SHARED={shared_step}")
