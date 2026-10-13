#!/usr/bin/env bash
# Drive RunPod pods from the laptop. All computation runs on pods; stop them when idle.
#
#   scripts/pod.sh up NAME [GPU_ID]      create a pod (CLOUDS="COMMUNITY SECURE" order, GPUS=1), wait for ssh, print its id
#   scripts/pod.sh setup ID              sync the repo, install rust, build the engine, check golden
#   scripts/pod.sh sync ID               rsync the repo to /workspace/beastybar-ai (no .git, .venv, target, runs, data),
#                                        write its provenance to REVISION there, rebuild the engine if needed
#   scripts/pod.sh ssh ID [CMD...]       run a command in the repo dir (interactive shell without CMD)
#   scripts/pod.sh bg ID NAME CMD...     run CMD detached; log at runs/NAME.log on the pod
#                                        CMD (ssh, bg): several arguments reach the command as they are, quoted once
#                                        as for a local command (scripts/value_train.sh v3 "data/value/a data/value/b");
#                                        a single argument is a shell command line, run as written ('a && b | c')
#   scripts/pod.sh pull ID PATH          rsync PATH (relative to repo) back from the pod
#   scripts/pod.sh backup ID             pull data/, checkpoints/ and runs/ into backups/<pod id>/ (no deletes): pods
#                                        can vanish (e.g. when the account balance runs out), so back up often
#   scripts/pod.sh autobackup ID [MIN]   backup every MIN minutes (default 10) until the pod is gone
#   scripts/pod.sh guard [USD]           every 5 min: if the account balance < USD (default 2.5), back up and delete
#                                        every pod (RunPod terminates pods, losing their disks, once it is negative)
#   scripts/pod.sh push ID PATH          rsync PATH (relative to repo) to the pod (e.g. a checkpoint: sync skips them)
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

cmdline() {  # CMD... as one line for the pod's shell: several arguments each quoted (they arrive unchanged), one as is
  if [ $# -eq 1 ]; then printf '%s' "$1"; else printf '%q ' "$@"; fi
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
    for cloud in ${CLOUDS:-COMMUNITY SECURE}; do
      out=$(runpodctl pod create --name "$name" --template-id runpod-torch-v280 --gpu-id "$gpu" --gpu-count "${GPUS:-1}" \
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
      --exclude '*.so' --exclude checkpoints --exclude data \
      "$REPO/" "$host:$REMOTE/"
    # The pod has no .git: record the commit and code hash (bbar/provenance.py) for the results log.
    py="$REPO/.venv/bin/python"; [ -x "$py" ] || py=python3
    (cd "$REPO" && "$py" -m bbar.provenance) | rsh "$id" "cat > $REMOTE/REVISION"
    # Never leave an extension built from older sources next to new ones (bbar.arena would refuse to log anyway).
    rsh "$id" "bash -lc 'source ~/.cargo/env 2>/dev/null; cd $REMOTE
      if command -v maturin >/dev/null; then PYTHON=python scripts/build_core.sh; else echo \"no toolchain yet: run setup\"; fi'" ;;
  setup)
    id=$1
    "$0" sync "$id"
    rsh "$id" "bash -lc 'set -e
      command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync >/dev/null)
      command -v cargo >/dev/null || [ -x ~/.cargo/bin/cargo ] || curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
      source ~/.cargo/env
      cd $REMOTE
      pip install -q --break-system-packages maturin numpy pytest 2>&1 | tail -1
      PYTHON=python scripts/build_core.sh
      python -m bbar.golden --engine core'" ;;
  ssh)
    id=$1; shift
    if [ $# -eq 0 ]; then
      read -r host port < <(target "$id"); exec ssh "${SSH_OPTS[@]}" -p "$port" -t "$host" "cd $REMOTE; bash -l"
    fi
    rsh "$id" "cd $REMOTE && source ~/.cargo/env 2>/dev/null; $(cmdline "$@")" ;;
  bg)
    id=$1; name=$2; shift 2
    line=$(cmdline "$@")
    rsh "$id" "cd $REMOTE && mkdir -p runs && source ~/.cargo/env 2>/dev/null; nohup bash -c $(printf '%q' "$line") \
      > runs/$name.log 2>&1 &"
    echo "started $name on $id; log: runs/$name.log" ;;
  pull)
    id=$1; path=$2
    read -r host port < <(target "$id")
    mkdir -p "$REPO/$(dirname "$path")"
    rsync -az -e "ssh ${SSH_OPTS[*]} -p $port" "$host:$REMOTE/$path" "$REPO/$(dirname "$path")/" ;;
  push)
    id=$1; path=$2
    read -r host port < <(target "$id")
    rsh "$id" "mkdir -p $REMOTE/$(dirname "$path")"
    rsync -az -e "ssh ${SSH_OPTS[*]} -p $port" "$REPO/$path" "$host:$REMOTE/$(dirname "$path")/" ;;
  backup)
    id=$1
    read -r host port < <(target "$id")
    mkdir -p "$REPO/backups/$id"
    for d in data checkpoints runs; do
      rsync -az -e "ssh ${SSH_OPTS[*]} -p $port" --exclude '*.tmp.npz' "$host:$REMOTE/$d" "$REPO/backups/$id/" 2>/dev/null || true
    done
    du -sh "$REPO/backups/$id" ;;
  autobackup)
    id=$1; every=${2:-10}
    while target "$id" >/dev/null; do "$0" backup "$id" >/dev/null 2>&1 || true; sleep $((60 * every)); done
    echo "pod $id is gone; last backup in backups/$id" ;;
  guard)
    floor=${1:-2.5}
    while true; do
      bal=$(runpodctl user 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["clientBalance"])' 2>/dev/null || echo 999)
      if python3 -c "import sys; sys.exit(0 if float('$bal') < float('$floor') else 1)"; then
        echo "$(date -u +%FT%TZ) balance $bal < $floor: backing up and deleting all pods"
        for id in $(runpodctl pod list 2>/dev/null | python3 -c 'import json,sys; print(" ".join(p["id"] for p in json.load(sys.stdin)))'); do
          "$0" backup "$id" || true; runpodctl pod delete "$id" || true
        done
        exit 0
      fi
      sleep 300
    done ;;
  down)
    runpodctl pod delete "$1" ;;
  *)
    awk 'NR > 1 && /^set /{exit} NR > 1' "$0"; exit 1 ;;
esac
