#!/usr/bin/env bash
# Train the blueprint policy pi (MMD-PPO self-play, bbar/rl/mmd.py) on a RunPod RTX 4090 pod. From the laptop:
#
#   id=$(scripts/pod.sh up rl-blueprint)                                   # create the pod
#   scripts/pod.sh setup $id                                               # sync the repo, build, check golden
#   scripts/pod.sh bg $id bp-v1 scripts/train_blueprint.sh bp-v1           # train; stdout in runs/bp-v1.log
#   scripts/pod.sh ssh $id tail -n 3 runs/bp-v1.log                        # progress (metrics: runs/bp-v1/)
#   scripts/pod.sh pull $id runs/bp-v1; scripts/pod.sh pull $id checkpoints/bp-v1
#   scripts/pod.sh down $id                                                # stop paying
#
# Extra arguments override the flags below (python -m bbar.rl.mmd --help lists them all); after an interruption,
# continue with: scripts/pod.sh bg $id bp-v1 scripts/train_blueprint.sh bp-v1 --resume
# (--resume exits if checkpoints/bp-v1/latest.pt is missing, e.g. on a new pod: sync skips checkpoints/. A fresh run
# refuses an existing runs/bp-v1 or checkpoints/bp-v1; --force renames them to bp-v1.replaced-<time> first.)
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?usage: scripts/train_blueprint.sh NAME [mmd flags...]}
shift
source ~/.cargo/env 2>/dev/null || true
export PYTHONUNBUFFERED=1

# Build the engine from the synced sources and put the extension next to them (./bbar shadows a wheel installed in
# site-packages, and rsync keeps an old extension), then gate on the golden games.
command -v maturin >/dev/null || pip install -q maturin numpy
maturin build --release -q -o target/wheels
python - <<'PY'
import glob, os, zipfile
whl = max(glob.glob("target/wheels/bbar-*.whl"), key=os.path.getmtime)
with zipfile.ZipFile(whl) as z:
    for name in z.namelist():
        if name.startswith("bbar/_core") and name.endswith(".so"):
            open(name, "wb").write(z.read(name))
PY
python -m bbar.golden --engine core
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -m bbar.rl.bench --device cuda --updates 3 2>&1 | tail -1

exec python -m bbar.rl.mmd --name "$NAME" --device cuda \
  --n-envs 4096 --rollout-steps 16 --env-threads 12 \
  --width 512 --blocks 4 --critic-width 512 --critic-blocks 4 \
  --lr 3e-4 --critic-lr 3e-4 --epochs 2 --minibatch 8192 --clip 0.2 --gae-lambda 0.95 \
  --alpha0 0.1 --alpha-min 0.005 --alpha-schedule exp \
  --aux-coef 0.1 --total-decisions 2e9 \
  --eval-every 200 --eval-deals 5000 --ckpt-every 1000 --log-every 20 \
  "$@"
