#!/usr/bin/env bash
# E1 (docs/PLAN.md): exact exploitability delta_T of an agent in late-game windows, by root ply T (bbar/delta.py),
# on a RunPod pod (GPU if present, else CPU; 16 vCPU is plenty). From the laptop:
#
#   id=$(scripts/pod.sh up e1)                                             # create the pod
#   scripts/pod.sh setup $id                                               # sync the repo, build, check golden
#   scripts/pod.sh push $id checkpoints/bp-v1/step_8000.pt                 # networks are not synced
#   scripts/pod.sh bg $id e1-bp scripts/e1.sh nn:checkpoints/bp-v1/step_8000.pt
#   scripts/pod.sh bg $id e1-tail scripts/e1.sh tail:nn:checkpoints/bp-v1/step_8000.pt,F=10,mode=gadget,cache=256
#   scripts/pod.sh bg $id e1-pimc scripts/e1.sh pimc:w=8,d=4 --T 12,11,10 --N 100   # built-in specs are slower
#   scripts/pod.sh ssh $id tail -n 8 runs/e1-bp.log                        # summary table at the end
#   scripts/pod.sh pull $id runs/delta                                     # JSON lines with provenance
#   scripts/pod.sh down $id                                                # stop paying
#
# Defaults: T in {12, 11, 10, 9}, N = 200 self-play positions per T, seed 1001 (the same deals for every agent),
# all cores. Extra arguments override them (python -m bbar.delta --help). Tail agents self-play 32 games at a time
# (--batch) and keep those games' solved windows (one per game and seat, plus the sub-windows of the position being
# evaluated) until each game is evaluated at every T: cache=256 covers that (about 12 MB per F=10 window). For tail
# agents, compare modes by the "safe max" column (per-hand gain of an opponent best response over the base network),
# not by delta_T alone: plain DCFR has delta_T ~ 0 by construction.
set -euo pipefail
cd "$(dirname "$0")/.."
AGENT=${1:?usage: scripts/e1.sh AGENT [bbar.delta flags...]}
shift
source ~/.cargo/env 2>/dev/null || true
export PYTHONUNBUFFERED=1
command -v maturin >/dev/null || pip install -q maturin numpy
PYTHON=python scripts/build_core.sh
python -m bbar.golden --engine core
device=cpu
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then device=cuda; fi
exec python -m bbar.delta "$AGENT" --T 12,11,10,9 --N 200 --seed 1001 --threads 0 --device "$device" "$@"
