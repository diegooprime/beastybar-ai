#!/usr/bin/env bash
# Train the PBS value network (bbar/value/train.py) on a RunPod RTX 4090 pod. From the laptop:
#
#   scripts/pod.sh bg $id vt-v1 scripts/value_train.sh v1 data/value/v1    # stdout in runs/vt-v1.log
#   scripts/pod.sh ssh $id tail -n 3 runs/vt-v1.log                        # val metrics every 2000 steps; at
#                                                                          # the end best.pt's test-split error
#   scripts/pod.sh pull $id runs/value/v1; scripts/pod.sh pull $id checkpoints/value/v1
#
# NAME names runs/value/NAME (metrics.jsonl) and checkpoints/value/NAME ({latest,best}.pt); DATA are shard
# directories, several as one quoted argument (pod.sh bg passes it on unchanged):
#   scripts/pod.sh bg $id vt-v3 scripts/value_train.sh v3 "data/value/v1 data/value/v2" --arch opt
# The labels live on the GPU (~9 KB each with PBS v1 features, ~11.5 KB with v2's option features: 1M labels =
# 11.5 GB). Extra arguments override the flags below (python -m bbar.value.train --help), e.g. a bigger network:
# --width 1024 --blocks 6, or the option-feature architecture: --arch opt --hand-width 256 --tok-width 128.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: scripts/value_train.sh NAME DATA [train flags...]}
DATA=${2:?usage: scripts/value_train.sh NAME DATA [train flags...]}
shift 2
source ~/.cargo/env 2>/dev/null || true
export PYTHONUNBUFFERED=1
command -v maturin >/dev/null || pip install -q maturin numpy
PYTHON=python scripts/build_core.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
exec python -m bbar.value.train --name "$NAME" --data $DATA --device cuda \
  --width 512 --blocks 3 --expansion 1 --steps 100000 --batch 2048 --lr 1e-3 --min-lr 1e-5 --warmup 1000 \
  --weight-decay 1e-4 --huber 0.1 --all-weight 0.1 --val-permille 50 --test-permille 50 --eval-every 2000 \
  --log-every 200 \
  "$@"
