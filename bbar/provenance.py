"""Which code produced a result. bbar.arena stores these with every line of runs/results.jsonl.

- commit: short hash of HEAD, plus "-dirty" when the code differs from it. Without git (pods get the repo without
  .git) it comes from the REVISION file that `scripts/pod.sh sync` writes.
- code: content hash of the code (bbar/, core/ and pyproject.toml; build outputs, caches and hidden files excluded).
  Two different trees never share it, and a checkout holding the same code gives the same hash, so a result from a
  dirty tree can be matched to the commit that later holds its code.
- build: the fingerprint of the Rust sources the engine extension was compiled from (core/build.rs, exported as
  bbar._core.build_id()). It must equal core_source_id() of the current files; otherwise the extension is stale and
  its results would be credited to code that plays differently.

    python -m bbar.provenance                 # {"commit": ..., "code": ...}: what scripts/pod.sh writes to REVISION
    python -m bbar.provenance --check-build   # exit 1 (and say why) if bbar._core is missing or stale

Standard library only (scripts/pod.sh runs it with any python3).
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ("bbar", "core", "pyproject.toml")
_SKIP_DIRS = {"target", "__pycache__"}
_SKIP_SUFFIXES = {".so", ".pyd", ".dylib", ".dll", ".pyc"}


def _digest(root, paths):
    """SHA-256 over `path NUL length NUL content` of each file (paths relative to root, '/' separators, sorted),
    first 16 hex digits. core/build.rs computes core_source_id the same way."""
    h = hashlib.sha256()
    for rel in sorted(paths):
        data = (Path(root) / rel).read_bytes()
        h.update(rel.encode() + b"\0" + str(len(data)).encode() + b"\0" + data)
    return h.hexdigest()[:16]


def core_source_id(root=ROOT):
    """Fingerprint of the Rust sources in root/core, as core/build.rs embeds it in the extension it builds."""
    core = Path(root) / "core"
    paths = [f for f in ("Cargo.toml", "Cargo.lock", "build.rs") if (core / f).is_file()]
    for dirpath, _, files in os.walk(core / "src"):
        rel = Path(dirpath).relative_to(core).as_posix()
        paths += [f"{rel}/{f}" for f in files
                  if f.endswith(".rs") and (Path(dirpath) / f).is_file() and not (Path(dirpath) / f).is_symlink()]
    return _digest(core, paths)


def code_files(root=ROOT):
    """Paths (relative to root) of the code files that `code_id` hashes."""
    root = Path(root)
    out = []
    for top in CODE:
        p = root / top
        if p.is_file():
            out.append(top)
            continue
        for dirpath, dirs, files in os.walk(p):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith(".")]
            for f in files:
                path = Path(dirpath) / f
                if not f.startswith(".") and path.suffix not in _SKIP_SUFFIXES and path.is_file():
                    out.append(path.relative_to(root).as_posix())
    return out


def code_id(root=ROOT):
    """Content hash of the code in root (see the module docstring)."""
    return _digest(root, code_files(root))


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def revision(root=ROOT):
    """{"commit", "code"} of the code in root."""
    root = Path(root)
    code = code_id(root)
    try:
        head = _git(root, "rev-parse", "--short", "HEAD")
        dirty = _git(root, "status", "--porcelain", "--", *CODE)
        return {"commit": head + ("-dirty" if dirty else ""), "code": code}
    except (OSError, subprocess.CalledProcessError):
        pass
    try:  # a pod: the revision synced from the laptop, if the code is still what was synced
        synced = json.loads((root / "REVISION").read_text())
        commit = synced["commit"]
        if synced["code"] != code and not commit.endswith("-dirty"):
            commit += "-dirty"
        return {"commit": commit, "code": code}
    except (OSError, ValueError, KeyError, TypeError):
        return {"commit": "unknown", "code": code}


def is_clean(commit):
    """Whether `commit` names a commit holding exactly the code (no "-dirty", not "unknown")."""
    return re.fullmatch(r"[0-9a-f]{7,40}", commit) is not None


def build_id():
    """bbar._core's build fingerprint, or None if the extension is missing or predates fingerprints."""
    try:
        from bbar import _core
    except ImportError:
        return None
    return getattr(_core, "build_id", lambda: None)()


class StaleBuild(RuntimeError):
    pass


def check_build(root=ROOT):
    """The extension's build id; raises StaleBuild unless it was compiled from the current root/core sources."""
    build, source = build_id(), core_source_id(root)
    if build != source:
        raise StaleBuild(
            f"bbar._core was built from other sources (build {build}, core/ is {source}): rebuild it with "
            "`maturin develop --release` (laptop) or scripts/build_core.sh (pods)")
    return build


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if args == ["--check-build"]:
        try:
            print(f"bbar._core is current (build {check_build()})")
        except StaleBuild as e:
            print(e, file=sys.stderr)
            return 1
        return 0
    if args:
        print(__doc__, file=sys.stderr)
        return 2
    print(json.dumps(revision()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
