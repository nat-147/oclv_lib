#!/bin/bash
# Run one hparam trial of a campaign on a single GPU, resuming from the latest usable shared checkpoint
# (<PREFIX>shared_<N>k, see scripts/make_trial_config.py).
#
# Env:
#   MODEL            hydra model config (default monet-velocity_clevr)
#   CAMPAIGN_PREFIX  run-dir prefix      (default mon-mean_z-occ-clevr6-hparamtune-)
#   SNAPSHOT_EVERY   if set (e.g. 10000), copy this run's checkpoint to <PREFIX>shared_<N>k every SNAPSHOT_EVERY
#                    steps (only meaningful for a base run whose vel-loss weight stays 0, so the state is
#                    trial-independent), skipping snapshots that already exist
# Usage: scripts/campaign_trial_1gpu.sh <trial_no> <gpu> <hydra overrides...>
set -u
cd /workspace/oclv_lib
N=$1; GPU=$2; shift 2
MODEL=${MODEL:-monet-velocity_clevr}
export CAMPAIGN_PREFIX=${CAMPAIGN_PREFIX:-mon-mean_z-occ-clevr6-hparamtune-}
export MODEL_YAML=config/model/$MODEL.yaml
RUN=${CAMPAIGN_PREFIX}trial_$N
D=outputs/runs/$RUN

if [ -e "$D" ]; then echo "$D already exists" >&2; exit 1; fi

out=$(python scripts/make_trial_config.py "$D" "$@") || { echo "failed to prepare $D" >&2; exit 1; }
echo "$out" | sed 's/^/trial '"$N"': /'
SHARED=$(echo "$out" | sed -n 's/^SHARED=//p')

if [ "$SHARED" != "0" ]; then
python - "$D" "$MODEL" "$@" <<'EOF' || { echo "failed to prepare trial config/weights" >&2; exit 1; }
import itertools, os, sys
from pathlib import Path
import torch
from hydra.experimental import compose, initialize_config_dir
from omegaconf import OmegaConf
d, model = Path(sys.argv[1]), sys.argv[2]
cfg = OmegaConf.load(d / "train_config.yaml")
# keep the dataset section equal to the current dataset config (+ CLI dataset overrides) so the trainer can resume
with initialize_config_dir(config_dir=os.path.abspath("config")):
    cur = compose("train_object_discovery", overrides=[f"model={model}", "dataset=clevr"]
                  + [a for a in sys.argv[3:] if a.startswith("dataset.")])
if OmegaConf.to_container(cfg.dataset) != OmegaConf.to_container(cur.dataset):
    cfg.dataset = cur.dataset
    OmegaConf.save(cfg, d / "train_config.yaml")
    print("dataset config synced to current clevr.yaml")
if cfg.model.get("hungarian_matching") and cfg.model.use_vel_extractor:
    k = OmegaConf.to_container(cfg, resolve=True)["model"]["num_slots"]
    k = k - 1 if "monet" in model else k   # MONet matches foreground slots only
    perms = torch.tensor(list(itertools.permutations(range(k))), dtype=torch.long)
    for f in ("model.pt", "best_model.pt"):
        sd = torch.load(d / f, "cpu")
        sd.setdefault("hungarian_matcher.perms", perms)
        torch.save(sd, d / f)
    print(f"added hungarian_matcher.perms ({len(perms)}x{k})")
EOF
fi

snapshot_loop() {  # copy the run's checkpoint to <PREFIX>shared_<S/1000>k each time it reaches a multiple of S
  local every=$1 done_steps=""
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    if [ -f "$D/train_state.yaml" ]; then
      local s; s=$(sed -n 's/^step: //p' "$D/train_state.yaml")
      if [ -n "$s" ] && [ "$s" -gt 0 ] && [ $(( s % every )) -eq 0 ] && [[ " $done_steps " != *" $s "* ]]; then
        local dst=outputs/runs/${CAMPAIGN_PREFIX}shared_$(( s / 1000 ))k
        if [ ! -e "$dst" ]; then
          mkdir -p "$dst.tmp"
          cp "$D"/{train_checkpoint.pt,model.pt,best_model.pt,train_state.yaml,train_config.yaml} "$dst.tmp/"
          if grep -q "^step: $s$" "$dst.tmp/train_state.yaml"; then mv "$dst.tmp" "$dst"; echo "saved $dst"
          else rm -rf "$dst.tmp"; echo "snapshot $s raced, retrying"; continue; fi
        fi
        done_steps="$done_steps $s"
      fi
    fi
    sleep 5
  done
}

CUDA_VISIBLE_DEVICES=$GPU accelerate launch --num_processes=1 --main_process_port $((29600 + GPU)) \
  train_object_discovery_accelerate.py model=$MODEL dataset=clevr curr_dir=$RUN "$@" > "$D/run.log" 2>&1 &
TRAIN_PID=$!
if [ -n "${SNAPSHOT_EVERY:-}" ]; then snapshot_loop "$SNAPSHOT_EVERY" & fi
wait $TRAIN_PID
rc=$?
wait
echo "exit=$rc"
