#!/bin/bash
# Run one hparam trial with 4-GPU DDP, resuming from the shared 10k checkpoint when possible.
# Usage: scripts/autoresearch_trial.sh <trial_no> <hydra overrides...>
set -u
cd /workspace/oclv_lib
N=$1; shift
RUN=sav-mean_z-occ-temp-clevr6-hparamtune-trial_$N
SHARED=outputs/runs/sav-mean_z-occ-temp-clevr6-hparamtune-shared_10k
D=outputs/runs/$RUN

if [ -e "$D" ]; then echo "$D already exists" >&2; exit 1; fi

python scripts/make_trial_config.py "$SHARED" "$D" "$@"
rc=$?
if [ $rc -eq 3 ]; then
  echo "warmup starts before 10k: training from scratch"
  mkdir -p "$D"
elif [ $rc -ne 0 ]; then
  echo "failed to prepare $D" >&2; exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 --multi_gpu \
  train_object_discovery_accelerate.py model=sa-velocity_clevr dataset=clevr \
  curr_dir=$RUN "$@" > "$D/run.log" 2>&1
echo "exit=$?"
