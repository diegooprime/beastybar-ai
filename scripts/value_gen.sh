#!/usr/bin/env bash
# Generate exact value-network labels (bbar/value/gen.py) on a RunPod pod (16 vCPUs: one solver process per vCPU).
# From the laptop:
#
#   id=$(scripts/pod.sh up value)                                          # create the pod (a CPU job: any GPU pod)
#   scripts/pod.sh setup $id                                               # sync the repo, build, check golden
#   scripts/pod.sh push $id checkpoints/bp-v1/step_8000.pt                 # the blueprint of the policy mix
#   scripts/pod.sh bg $id gen-v1 scripts/value_gen.sh v1                   # 2 hours; stdout in runs/gen-v1.log
#   scripts/pod.sh ssh $id tail -n 8 runs/gen-v1.log                       # labels/s per thread by ply at the end
#   scripts/pod.sh pull $id data/value/v1                                  # shards + summary-s0.json
#
# Shards land in data/value/NAME (pod.sh sync never touches data/). Extra arguments override the flags below
# (python -m bbar.value.gen --help); a second run into the same NAME needs another --seed (--force replaces the
# run of its seed: that seed's shards are deleted first). Laptop rates (M4, per thread): ply 10 ~1.2 labels/s,
# 11 ~15, 12 ~150, 13 ~1200 — the ply weights below put most of the time into ply 10 and 11, the leaf plies of E2
# from ply 9.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: scripts/value_gen.sh NAME [gen flags...]}
shift
source ~/.cargo/env 2>/dev/null || true
export PYTHONUNBUFFERED=1
command -v maturin >/dev/null || pip install -q maturin numpy
PYTHON=python scripts/build_core.sh
python -m bbar.golden --engine core

exec python -m bbar.value.gen --out "data/value/$NAME" --workers "$(nproc)" --minutes 120 --labels 100000000 \
  --plies 10,11,12,13 --ply-weights 0.4,0.3,0.2,0.1 --ranges-per-state 4 --target 1e-5 \
  --mix random:0.2,pimc:0.4,bp:0.4 --blueprint checkpoints/bp-v1/step_8000.pt --shard-size 2000 --seed 0 \
  "$@"
