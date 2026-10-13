#!/usr/bin/env bash
# Experiment E2 (bbar/value/e2.py): exact exploitability, in full ply-9 windows, of depth-limited solving with
# value-network leaves at ply 10 and 11 (composite: trunk + safe re-solves below every leaf), against exact solving,
# exact leaf values (oracle: what a perfect network gives) and uniform play. On a RunPod pod (16 vCPUs: 4 positions
# at a time x 4 solver threads). From the laptop:
#
#   scripts/pod.sh push $id checkpoints/bp-v1/step_8000.pt                 # the positions' policy mix
#   scripts/pod.sh push $id checkpoints/value/v1/best.pt                   # or train on the pod (value_train.sh)
#   scripts/pod.sh bg $id e2-v1 scripts/e2.sh v1 checkpoints/value/v1/best.pt
#   scripts/pod.sh ssh $id tail -n 20 runs/e2-v1.log                       # summary table at the end
#   scripts/pod.sh ssh $id grep '"runs/e2/v1.jsonl"' runs/results.jsonl >> runs/results.jsonl   # its summary line
#   scripts/pod.sh pull $id runs/e2                                        # runs/e2/v1.jsonl: positions + summary
#
# The summary goes to the pod's evidence trail runs/results.jsonl, which (like bbar.arena) refuses uncommitted code:
# sync a committed tree, or pass --allow-dirty (the line then records the code hash).
#
# Cost per ply-9 position, M4 laptop, 4 threads (pods are slower per thread), measured on one position
# (runs/e2/smoke-S9.jsonl and its line in runs/results.jsonl; build c99a27b6a8738f09, code 5e6564ad9074b907):
# ~2 minutes in all (109 s): exact solve 10 s; the network's depth-limited solves themselves 0.2 s (leaves at 10)
# and 0.8 s (at 11); the rest is evaluation (leaf windows, both players' re-solves below every leaf, leaf-error
# check, scoring: 62 s and 35 s). The oracle (every half-iteration solves every leaf's window) is extrapolated:
# 4 iterations with leaves at 10 took 50 s (runs/e2/smoke-S9-oracle4.jsonl: a cold query ~23 s, as smoke-S9's
# leaf-error check, then ~4 s per warm-started one), so 200 iterations ~30 minutes at 10 and less at 11 (its cold
# query is half as long), ~45 minutes per position for both
# (unmeasured beyond that: the first oracle positions in the log give the real figure), hence --oracle-positions 8
# (~2 hours in all with 4 workers). For ply 8 (--ply 8: windows ~10x larger, ~1.5 GB each) use fewer workers and
# fewer oracle positions. Extra arguments override the flags below.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: scripts/e2.sh NAME NET [e2 flags...]}
NET=${2:?usage: scripts/e2.sh NAME NET [e2 flags...]}
shift 2
source ~/.cargo/env 2>/dev/null || true
export PYTHONUNBUFFERED=1
command -v maturin >/dev/null || pip install -q maturin numpy
PYTHON=python scripts/build_core.sh
python -m bbar.golden --engine core

exec python -m bbar.value.e2 --net "$NET" --positions 64 --ply 9 --leaf-plies 10,11 --iters 200 \
  --oracle-positions 8 --oracle-target 1e-6 --exact-target 1e-6 --gadget-target 1e-6 \
  --workers 4 --threads 4 --seed 0 --out "runs/e2/$NAME.jsonl" \
  "$@"
