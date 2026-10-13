#!/usr/bin/env bash
# A bbar.match match on a RunPod pod (GPU if present), e.g. the exact-tail player against its base network:
#
#   id=$(scripts/pod.sh up tail)
#   scripts/pod.sh setup $id
#   scripts/pod.sh push $id checkpoints/bp-v1/step_8000.pt
#   scripts/pod.sh bg $id tail-vs-bp scripts/match.sh \
#       tail:nn:checkpoints/bp-v1/step_8000.pt,F=10,mode=gadget,workers=12,cache=256 nn:checkpoints/bp-v1/step_8000.pt
#   scripts/pod.sh ssh $id tail -n 2 runs/tail-vs-bp.log
#   scripts/pod.sh ssh $id tail -n 1 runs/results.jsonl >> runs/results.jsonl     # the evidence line
#   scripts/pod.sh down $id
#
# Defaults: 2000 deal pairs, 128 deals at a time, random seed (logged). A tail agent plays with two instances (one per
# seat assignment of the duplicate pair), each keeping one solved window per game in play: 128 windows each with
# --batch 128 (about 12 MB per F=10 window, ~3 GB in all). Its cache= must be at least --batch, or windows are solved
# again while in use (same moves, more compute: bbar.match warns, and agent_stats counts the re-solves).
# The evidence trail runs/results.jsonl takes committed code only (sync a committed tree, or --allow-dirty).
set -euo pipefail
cd "$(dirname "$0")/.."
A=${1:?usage: scripts/match.sh A B [bbar.match flags...]}
B=${2:?usage: scripts/match.sh A B [bbar.match flags...]}
shift 2
source ~/.cargo/env 2>/dev/null || true
export PYTHONUNBUFFERED=1
command -v maturin >/dev/null || pip install -q maturin numpy
PYTHON=python scripts/build_core.sh
python -m bbar.golden --engine core
device=cpu
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then device=cuda; fi
exec python -m bbar.match "$A" "$B" --deals 2000 --batch 128 --device "$device" "$@"
