#!/usr/bin/env bash
# Builds the engine extension in place, bbar/_core.abi3.so (release), unless it is already built from the current
# core/ sources. `python -m bbar...` run from the repo then uses it (an installed copy elsewhere would be shadowed by
# the repo's bbar/). For pods (scripts/pod.sh setup and sync); on the laptop `maturin develop --release` does the same.
#
#   scripts/build_core.sh        # PYTHON (default python3) and MATURIN (default maturin) override the tools
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
if "$PY" -m bbar.provenance --check-build 2>/dev/null; then exit 0; fi
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
"${MATURIN:-maturin}" build --release -q -o "$out"
"$PY" - "$out" <<'EOF'
import glob, os, sys, zipfile
(wheel,) = glob.glob(os.path.join(sys.argv[1], "*.whl"))
with zipfile.ZipFile(wheel) as z:
    (name,) = [n for n in z.namelist() if n.startswith("bbar/_core.") and not n.endswith(".py")]
    with open(name + ".tmp", "wb") as f:
        f.write(z.read(name))
os.chmod(name + ".tmp", 0o755)
os.replace(name + ".tmp", name)  # a new file: processes still running the old one are unaffected
EOF
"$PY" -m bbar.provenance --check-build
