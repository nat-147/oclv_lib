#!/bin/bash
# Run one hparam trial with 4-GPU DDP, resuming from the latest usable shared checkpoint.
# While it trains, it snapshots shared 20k/30k checkpoints if this trial's warmup starts late enough
# and they don't exist yet (vel loss weight is still 0 at that point, so the state is trial-independent).
# Usage: scripts/autoresearch_trial.sh <trial_no> <hydra overrides...>
set -u
cd /workspace/oclv_lib
N=$1; shift
PREFIX=sav-mean_z-clevr6-hparamtune-
RUN=${PREFIX}trial_$N
D=outputs/runs/$RUN

if [ -e "$D" ]; then echo "$D already exists" >&2; exit 1; fi

out=$(python scripts/make_trial_config.py "$D" "$@") || { echo "failed to prepare $D" >&2; exit 1; }
START=$(echo "$out" | sed -n 's/^START=//p')
SHARED=$(echo "$out" | sed -n 's/^SHARED=//p')
echo "trial $N: warmup start=$START, resume from shared step=$SHARED"

snapshot() {  # copy the trial's checkpoint once it reaches step $1
  local S=$1 dst=outputs/runs/${PREFIX}shared_$(( $1 / 1000 ))k
  while kill -0 "$TRAIN_PID" 2>/dev/null; do
    if [ -f "$D/train_state.yaml" ] && grep -q "^step: $S$" "$D/train_state.yaml"; then
      mkdir -p "$dst.tmp"
      cp "$D"/{train_checkpoint.pt,model.pt,best_model.pt,train_state.yaml,train_config.yaml} "$dst.tmp/"
      if grep -q "^step: $S$" "$dst.tmp/train_state.yaml"; then mv "$dst.tmp" "$dst"; echo "saved $dst"
      else rm -rf "$dst.tmp"; echo "snapshot $S raced, skipped"; fi
      return
    fi
    sleep 5
  done
}

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 --multi_gpu \
  train_object_discovery_accelerate.py model=sa-velocity_clevr dataset=clevr \
  curr_dir=$RUN "$@" > "$D/run.log" 2>&1 &
TRAIN_PID=$!

for S in 20000 30000; do
  if [ "$START" -ge "$S" ] && [ "$SHARED" -lt "$S" ] && [ ! -d "outputs/runs/${PREFIX}shared_$(( S / 1000 ))k" ]; then
    snapshot $S &
  fi
done

wait $TRAIN_PID
rc=$?
wait
echo "exit=$rc"
