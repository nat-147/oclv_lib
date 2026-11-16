#!/bin/bash
# Run one hparam trial on a single GPU (no DDP), resuming from the latest usable shared checkpoint.
# Shared checkpoints were trained without hungarian matching, so the matcher's fixed `perms` buffer
# (all slot permutations, see models/utils.py HungarianMatcher) is added to the copied model weights.
# Usage: scripts/autoresearch_trial_1gpu.sh <trial_no> <gpu> <hydra overrides...>
set -u
cd /workspace/oclv_lib
N=$1; GPU=$2; shift 2
RUN=sav-mean_z-occ-temp-clevr6-hparamtune-trial_$N
D=outputs/runs/$RUN

if [ -e "$D" ]; then echo "$D already exists" >&2; exit 1; fi

out=$(python scripts/make_trial_config.py "$D" "$@") || { echo "failed to prepare $D" >&2; exit 1; }
echo "$out" | sed 's/^/trial '"$N"': /'

python - "$D" <<'EOF' || { echo "failed to add matcher perms" >&2; exit 1; }
import itertools, sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
d = Path(sys.argv[1])
cfg = OmegaConf.load(d / "train_config.yaml")
if cfg.model.get("hungarian_matching") and cfg.model.use_vel_extractor:
    k = OmegaConf.to_container(cfg, resolve=True)["model"]["num_slots"]
    perms = torch.tensor(list(itertools.permutations(range(k))), dtype=torch.long)
    for f in ("model.pt", "best_model.pt"):
        if (d / f).exists():
            sd = torch.load(d / f, "cpu")
            sd.setdefault("hungarian_matcher.perms", perms)
            torch.save(sd, d / f)
    print(f"added hungarian_matcher.perms ({len(perms)}x{k})")
EOF

CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --main_process_port $((29600 + GPU)) \
  train_object_discovery_accelerate.py model=sa-velocity_clevr dataset=clevr \
  curr_dir=$RUN "$@" > "$D/run.log" 2>&1
rc=$?
echo "exit=$rc"
