#!/usr/bin/env bash
# Drive RunPod pods from the laptop. All computation runs on pods; stop them when idle.
#
#   scripts/pod.sh up NAME [GPU_ID]      create a pod (tries community, then secure), wait for ssh, print its id
#   scripts/pod.sh setup ID              sync the repo, install rust, build the engine, check golden
#   scripts/pod.sh sync ID               rsync the repo to /workspace/beastybar-ai (no .git, .venv, target, runs)
#   scripts/pod.sh ssh ID [CMD...]       run a command in the repo dir (interactive shell without CMD)
#   scripts/pod.sh bg ID NAME CMD...     run CMD detached; log at runs/NAME.log on the pod
#   scripts/pod.sh pull ID PATH          rsync PATH (relative to repo) back from the pod
#   scripts/pod.sh down ID               delete the pod
set -euo pipefail

KEY="$HOME/.runpod/ssh/RunPod-Key-Go"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE=/workspace/beastybar-ai
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR
          -o ServerAliveInterval=30 -o ConnectTimeout=15)

target() {  # prints "user@host port"
  runpodctl ssh info "$1" 2>/dev/null | python3 -c '
import json, re, sys
d = json.load(sys.stdin)
if "error" in d: sys.exit(1)
ip, port = d.get("ip"), d.get("port")
if not ip:
    m = re.search(r"(\S+@[\d.]+)\s+-p\s+(\d+)", d.get("command", ""))
    if not m: sys.exit(1)
    print(m.group(1), m.group(2)); sys.exit()
print("root@" + ip, port)'
}

rsh() {
  local id=$1; shift
  read -r host port < <(target "$id")
  ssh "${SSH_OPTS[@]}" -p "$port" "$host" "$@"
}

wait_ssh() {
  for _ in $(seq 1 90); do
    if target "$1" >/dev/null && rsh "$1" true 2>/dev/null; then return 0; fi
    sleep 5
  done
  echo "pod $1: ssh not ready after 7.5 min" >&2; return 1
}

cmd=${1:-}; shift || true
case "$cmd" in
  up)
    name=$1; gpu=${2:-"NVIDIA GeForce RTX 4090"}
    for cloud in COMMUNITY SECURE; do
      out=$(runpodctl pod create --name "$name" --template-id runpod-torch-v280 --gpu-id "$gpu" \
            --cloud-type "$cloud" --container-disk-in-gb 40 2>&1 || true)
      id=$(echo "$out" | python3 -c 'import json,sys
try: print(json.load(sys.stdin)["id"])
except Exception: pass' 2>/dev/null || true)
      if [ -n "$id" ]; then
        echo "created $id ($gpu, $cloud, $(echo "$out" | grep -o '"costPerHr": [0-9.]*'))" >&2
        wait_ssh "$id"; echo "$id"; exit 0
      fi
    done
    echo "no capacity for $gpu" >&2; exit 1 ;;
  sync)
    id=$1
    read -r host port < <(target "$id")
    rsh "$id" "mkdir -p $REMOTE"
    rsync -az --delete -e "ssh ${SSH_OPTS[*]} -p $port" \
      --exclude .git --exclude .venv --exclude target --exclude runs --exclude '__pycache__' \
      --exclude '*.so' --exclude checkpoints \
      "$REPO/" "$host:$REMOTE/" ;;
  setup)
    id=$1
    "$0" sync "$id"
    rsh "$id" "bash -lc 'set -e
      command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync >/dev/null)
      command -v cargo >/dev/null || [ -x ~/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
      source ~/.cargo/env
      cd $REMOTE
      pip install -q maturin numpy pytest 2>&1 | tail -1
      maturin build --release -q -o target/wheels 2>&1 | tail -2
      pip install -q --force-reinstall --no-deps target/wheels/*.whl
      python -m bbar.golden --engine core'" ;;
  ssh)
    id=$1; shift
    if [ $# -eq 0 ]; then
      read -r host port < <(target "$id"); exec ssh "${SSH_OPTS[@]}" -p "$port" -t "$host" "cd $REMOTE; bash -l"
    fi
    rsh "$id" "cd $REMOTE && source ~/.cargo/env 2>/dev/null; $*" ;;
  bg)
    id=$1; name=$2; shift 2
    rsh "$id" "cd $REMOTE && mkdir -p runs && source ~/.cargo/env 2>/dev/null; nohup bash -c '$*' > runs/$name.log 2>&1 &"
    echo "started $name on $id; log: runs/$name.log" ;;
  pull)
    id=$1; path=$2
    read -r host port < <(target "$id")
    mkdir -p "$REPO/$(dirname "$path")"
    rsync -az -e "ssh ${SSH_OPTS[*]} -p $port" "$host:$REMOTE/$path" "$REPO/$(dirname "$path")/" ;;
  down)
    runpodctl pod delete "$1" ;;
  *)
    sed -n '2,12p' "$0"; exit 1 ;;
esac
