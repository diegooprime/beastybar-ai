#!/usr/bin/env bash
# The value-network training cascade (bbar/value/cascade.py) on a RunPod GPU pod: the network on the exact labels
# (plies 10-13, bbar/value/gen.py), then labels at plies 8-9 from depth-limited solves valued by it at ply 10, retrain
# on all, then 6-7 (leaves at 8-9), ... down to 0-1. From the laptop:
#
#   id=$(scripts/pod.sh up cascade)                                         # RTX 4090, 16 vCPUs
#   scripts/pod.sh setup $id
#   scripts/pod.sh push $id checkpoints/bp-v1/step_30518.pt                 # positions' policy mix + Bayes ranges
#   scripts/pod.sh push $id data/value/v1                                   # the exact labels (plies 10-13)
#   scripts/pod.sh push $id checkpoints/value/v1/best.pt                    # optional: V0 already trained on them
#   scripts/pod.sh bg $id cascade-c1 scripts/cascade.sh c1 data/value/v1 --net checkpoints/value/v1/best.pt
#   scripts/pod.sh ssh $id grep cascade runs/cascade-c1.log                 # levels, labels/s per worker
#   scripts/pod.sh pull $id checkpoints/value                               # c1-1 .. c1-5 (best.pt each)
#   scripts/pod.sh pull $id runs/value                                      # their metrics (per-ply errors)
#   scripts/pod.sh pull $id data/value                                      # the cascade shards (c1-p8-9, ...)
#
# Without --net the first step trains V0 (c1-0) on DATA. Each level writes data/value/NAME-pA-B and trains
# checkpoints/value/NAME-k on everything so far (exact + all levels); --resume (on by default here) skips trainings
# that ran to their end (an interrupted one is trained again: its best.pt is not marked complete) and levels whose
# summary names the current network, then redoes everything after the first redone step, so a rerun after an
# interruption picks up where it stopped. Extra arguments override the flags below (python -m bbar.value.cascade run
# --help); label counts per level: --levels 8,9:200000 6,7:200000 ...
#
# Cascade labels are right on their ranges only (bbar/value/cascade.py): bbar.value.train leaves them out of its
# all-feasible-hands term (--cascade-all-weight 0, its default).
#
# Throughput: each of --workers processes runs --jobs solves in lockstep with one batched network call per
# half-iteration on the GPU and the solver on --threads threads. Measure before the long run (labels/s per worker
# in the summary; same flags, cpu vs cuda):
#   scripts/pod.sh ssh $id python -m bbar.value.cascade labels --net checkpoints/value/v1/best.pt \
#       --out data/value/tput-cuda --plies 8,9 --labels 1024 --workers 1 --jobs 128 --threads 8 --device cuda \
#       --blueprint checkpoints/bp-v1/step_30518.pt
#   (and --device cpu --out data/value/tput-cpu). Laptop (M4, 512x3 network, 64 jobs, 4 threads): 3.5 labels/s per
#   worker on CPU, 3.9 on MPS, 80% of it in the network.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: scripts/cascade.sh NAME DATA [cascade run flags...]}
DATA=${2:?usage: scripts/cascade.sh NAME DATA [cascade run flags...]}
shift 2
source ~/.cargo/env 2>/dev/null || true
export PYTHONUNBUFFERED=1
command -v maturin >/dev/null || pip install -q maturin numpy
PYTHON=python scripts/build_core.sh
python -m bbar.golden --engine core
device=cpu
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then device=cuda; fi

# shellcheck disable=SC2086
exec python -m bbar.value.cascade run --name "$NAME" --data $DATA --device "$device" \
  --levels 8,9:200000 6,7:200000 4,5:200000 2,3:200000 0,1:200000 --d 2 --max-leaf-ply 10 --iters 200 \
  --workers 4 --jobs 128 --threads 4 --resume \
  --label-args "--mix random:0.2,pimc:0.4,bp:0.4 --blueprint checkpoints/bp-v1/step_30518.pt --bayes 0.3 \
--ranges-per-state 4 --shard-size 4000" \
  --train-args "--width 512 --blocks 3 --expansion 1 --steps 100000 --batch 2048 --lr 1e-3 --min-lr 1e-5 \
--warmup 1000 --eval-every 2000 --log-every 200" \
  "$@"
