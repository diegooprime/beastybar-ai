#!/usr/bin/env bash
# Gather label directories from several pods onto a hub pod (through the laptop; shards are small).
#
#   scripts/gather.sh HUB DIR [DIR...] -- POD [POD...]
#
# Each DIR (relative to the repo, e.g. data/value/x9) is pulled from every POD into the laptop's DIR (no deletes,
# unfinished *.tmp.npz skipped), then pushed to the HUB's DIR. Shard names carry their run's seed, so pods never
# overwrite each other.
set -euo pipefail
cd "$(dirname "$0")/.."
hub=${1:?usage: scripts/gather.sh HUB DIR [DIR...] -- POD [POD...]}; shift
dirs=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do dirs+=("$1"); shift; done
shift || true
pods=("$@")
KEY="$HOME/.runpod/ssh/RunPod-Key-Go"
rs() {  # rs POD ARGS...: rsync with that pod's ssh target
  local pod=$1; shift
  read -r host port < <(scripts/pod.sh target "$pod")
  rsync -az --exclude '*.tmp.npz' -e "ssh -i $KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p $port" "${@/#POD:/$host:/workspace/beastybar-ai/}"
}
for d in "${dirs[@]}"; do
  mkdir -p "$d"
  for pod in "${pods[@]}"; do
    [ "$pod" = "$hub" ] && continue
    rs "$pod" "POD:$d/" "$d/" < /dev/null || echo "  (nothing at $d on $pod)"
  done
  scripts/pod.sh ssh "$hub" "mkdir -p $d" < /dev/null
  rs "$hub" "$d/" "POD:$d/" < /dev/null
  echo "$d: $(ls "$d" | grep -c 'npz$') shards on the laptop, pushed to $hub"
done
