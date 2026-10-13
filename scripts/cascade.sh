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
# half-iteration (worker k on GPU k mod the GPU count) and the solver on --threads threads. Measure before the long
# run, with the run's flags (the summary has labels/s per worker, the network's share and ms per half-iteration):
#   scripts/pod.sh ssh $id python -m bbar.value.cascade labels --net checkpoints/value/v3m/best.pt \
#       --out data/value/tput-cuda --plies 8,9 --labels 1024 --iters 200 --workers 4 --jobs 128 --threads 4 \
#       --device cuda --blueprint checkpoints/bp-v1/step_30518.pt
#   then again with --stub-net --out data/value/tput-stub (a do-nothing network: the value function's cost without
#   the forward passes, no shards) and with --device cpu --out data/value/tput-cpu.
# GPU-bound estimate for the flags below (--iters 200: 400 half-iterations per label, each a forward pass over every
# value leaf at 2 x parameters FLOP, fp32; ~68 value leaves per solve at plies 8-9, leaves at 10, 68-71 in two
# samples; other levels have other counts), at 22-25 TFLOPS achieved (A40 fp32 peak 37.4):
#   1024x6 MLP (v3m, 15.37M parameters: labels level 1 when the run gets --net checkpoints/value/v3m/best.pt):
#   0.84 TFLOP per label, 26-30 labels/s per A40, 13-15 per worker with 2 workers per GPU (4 on 2xA40);
#   512x3 MLP (--train-args below, 2.96M parameters: the networks that label levels 2-5, and level 1 without --net):
#   0.16 TFLOP per label, 137-155 labels/s per A40: the GPU is then not the bound.
# Not in the estimate: the value function's per-call cost besides the forward passes (NetValues.session: copies,
# scatter, gather) and the solver. On an M4 with a do-nothing network on MPS (4 workers x 5 threads, 128 jobs) they
# take 19-23 ms per half-iteration per worker, 15-18 of it the session: at most 14-17 labels/s per worker at --iters
# 200 whatever the network. Measure them on the pod (--stub-net) before counting on the GPU-bound figures. Laptop
# with a 512x3 network (64 jobs, 4 threads, 1 worker): 20 ms per half-iteration on MPS, 40 on CPU (1 torch thread),
# i.e. 7.9 and 4.0 labels/s per worker at --iters 200.
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
