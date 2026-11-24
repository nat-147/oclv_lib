"""Re-evaluate a run's best_model.pt with hungarian_matching=True and write <run>/train_state_hm.yaml.

Reproduces the in-training validation of train_object_discovery_accelerate.py on one process:
same valid split, batch_size from the run config, no shuffle, `--steps` batches (trainer.eval_steps, default 100),
same MetricsEvaluator / ForwardPass (no `step` → full vel-loss weight in valid_loss, like the trainer).
Checkpoints trained without hungarian matching lack the matcher's `perms` buffer; it is a fixed table of slot
permutations, so the freshly built one is kept (only that key may be missing).

Usage: CUDA_VISIBLE_DEVICES=0 python scripts/eval_hm.py <run_dir> [--steps 100] [--ckpt best_model.pt]
"""
import argparse
import sys
from pathlib import Path

import hydra
import torch
import yaml
from accelerate import Accelerator
from omegaconf import OmegaConf, open_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.datasets import hydra_to_dataset_args  # noqa: E402
from evaluation.metrics.metrics_evaluator_vs_acc import MetricsEvaluator  # noqa: E402
from models.utils import ForwardPass  # noqa: E402
from velocity_extraction.utils import get_dataset_dataloader  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run_dir", type=Path)
ap.add_argument("--steps", type=int, default=100)
ap.add_argument("--ckpt", default="best_model.pt")
ap.add_argument("--out", default="train_state_hm.yaml")
ap.add_argument("--model-step", type=int, default=None,
                help="training step for the vel-loss lambda warmup in valid_loss (default: full weight, like the trainer without eval_lambda_warmup)")
ap.add_argument("--no-hm", action="store_true", help="sanity check: keep the run's own hungarian_matching")
args = ap.parse_args()

cfg = OmegaConf.load(args.run_dir / "train_config.yaml")
with open_dict(cfg):
    if not args.no_hm:
        cfg.model.hungarian_matching = True
torch.manual_seed(cfg.seed)

acc = Accelerator()
model = hydra.utils.instantiate(cfg.model)
sd = torch.load(args.run_dir / args.ckpt, "cpu")
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not unexpected and all(k.endswith("hungarian_matcher.perms") for k in missing), (missing, unexpected)

dataconfig = dict(
    dataset_class=hydra.utils.get_class(cfg.dataset._target_),
    dataset_args=hydra_to_dataset_args(cfg),
    dataloader_args=dict(batch_size=cfg.batch_size, pin_memory=True, num_workers=cfg.num_workers),
)
if dataconfig["dataset_args"].get("num_validation", 0) > 0:
    dataconfig["eval_split"] = "valid"
_, _, _, valid_loader = get_dataset_dataloader(dataconfig)
model, valid_loader = acc.prepare(model, valid_loader)
model.eval()

loss_terms = ["loss", "x_recon_loss"]
if model.use_vel_extractor:
    loss_terms += ["loop_loss", "z_pred_loss", "shortcut_loss", "isometry_loss"]
ev = MetricsEvaluator(dataloader=valid_loader, device=acc.device, loss_terms=loss_terms, skip_background=True,
                      _forward_pass=ForwardPass(model, acc.device), accelerator=acc)
losses, metrics = ev.eval(model, steps=args.steps, model_step=args.model_step)

state = yaml.safe_load(open(args.run_dir / "train_state.yaml"))
src = state["best_model"] if args.ckpt == "best_model.pt" else {"step": state["step"]}
out = {
    "best_model": {
        "image_ari": float(metrics["image_ari"]), "video_ari": float(metrics["video_ari"]),
        "image_mbo": float(metrics["image_mbo"]), "video_mbo": float(metrics["video_mbo"]),
        "mse": float(metrics["mse"]), "valid_loss": float(losses["loss"]), "step": src["step"],
    },
    "eval": {"hungarian_matching": bool(cfg.model.hungarian_matching), "ckpt": args.ckpt, "steps": args.steps, "batch_size": cfg.batch_size,
             "num_processes": acc.num_processes, "model_step": args.model_step, "split": dataconfig.get("eval_split", "test")},
}
with open(args.run_dir / args.out, "w") as f:
    yaml.safe_dump(out, f)
print(yaml.safe_dump(out))
