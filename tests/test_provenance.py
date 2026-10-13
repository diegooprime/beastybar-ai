"""Provenance of results (bbar/provenance.py): the log must name the code, and the build, that produced a result."""
import json
import shutil
import subprocess

import pytest

from bbar import provenance


def git(root, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root, check=True,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A small git repo shaped like this one, with one commit."""
    root = tmp_path / "repo"
    for rel, text in {
        "bbar/__init__.py": "",
        "bbar/arena.py": "x = 1\n",
        "core/Cargo.toml": "[package]\n",
        "core/src/eval.rs": "pub const POS_WEIGHT: [f32; 2] = [0.6, 0.45];\n",
        "pyproject.toml": "[project]\n",
        "README.md": "readme\n",
        ".gitignore": "target/\n__pycache__/\n*.so\nruns/*\n",
    }.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


def test_dirty_trees_are_told_apart_by_their_code_hash(repo):
    clean = provenance.revision(repo)
    assert provenance.is_clean(clean["commit"])
    eval_rs = repo / "core/src/eval.rs"
    eval_rs.write_text(eval_rs.read_text().replace("[0.6, 0.45]", "[-0.6, -0.45]"))  # the review's POS_WEIGHT edit
    one = provenance.revision(repo)
    (repo / "bbar/new.py").write_text("y = 2\n")                                       # an untracked file
    two = provenance.revision(repo)
    assert one["commit"] == two["commit"] == clean["commit"] + "-dirty"
    assert not provenance.is_clean(one["commit"])
    assert len({clean["code"], one["code"], two["code"]}) == 3
    # Build outputs, caches, runs/ and docs do not change it.
    for junk in ["core/target/release/x", "bbar/__pycache__/a.pyc", "bbar/_core.abi3.so", "runs/r.jsonl",
                 "README.md", "core/.DS_Store"]:
        (repo / junk).parent.mkdir(parents=True, exist_ok=True)
        (repo / junk).write_text("junk")
    assert provenance.revision(repo) == two
    # Committing exactly that code gives a clean commit with the same code hash: a result logged from the dirty tree
    # can be matched to the commit that holds its code.
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "change")
    committed = provenance.revision(repo)
    assert provenance.is_clean(committed["commit"]) and committed["commit"] != clean["commit"]
    assert committed["code"] == two["code"]


def test_pods_use_the_synced_revision(repo, tmp_path):
    # scripts/pod.sh sync copies the tree without .git and writes `python -m bbar.provenance` output to REVISION.
    synced = provenance.revision(repo)
    pod = tmp_path / "pod"
    shutil.copytree(repo, pod, ignore=shutil.ignore_patterns(".git"))
    assert provenance.revision(pod)["commit"] == "unknown"
    (pod / "REVISION").write_text(json.dumps(synced))
    assert provenance.revision(pod) == synced
    (pod / "bbar/arena.py").write_text("x = 2\n")  # edited on the pod after the sync
    edited = provenance.revision(pod)
    assert edited["commit"] == synced["commit"] + "-dirty" and edited["code"] != synced["code"]


def test_extension_build_id_is_the_fingerprint_of_its_sources():
    # core/build.rs (Rust SHA-256) and provenance.core_source_id (hashlib) agree on the current sources.
    core = pytest.importorskip("bbar._core")
    assert core.build_id() == provenance.core_source_id() == provenance.check_build()
    assert core.play_match("random", "random", 1, seed=0)["build"] == core.build_id()


def test_a_stale_extension_is_detected(tmp_path):
    # The review's scenario: POS_WEIGHT negated in core/src/eval.rs (and committed) without rebuilding the extension.
    pytest.importorskip("bbar._core")
    root = tmp_path / "copy"
    for rel in ["core/Cargo.toml", "core/Cargo.lock", "core/build.rs"]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(provenance.ROOT / rel, root / rel)
    shutil.copytree(provenance.ROOT / "core/src", root / "core/src")
    assert provenance.check_build(root) == provenance.core_source_id(root)
    eval_rs = root / "core/src/eval.rs"
    eval_rs.write_text(eval_rs.read_text().replace("[0.6, 0.45, 0.25, 0.1]", "[-0.6, -0.45, -0.25, -0.1]"))
    with pytest.raises(provenance.StaleBuild):
        provenance.check_build(root)
    # Only Rust sources count: notes next to them do not make the build stale.
    eval_rs.write_text((provenance.ROOT / "core/src/eval.rs").read_text())
    (root / "core/src/NOTES.txt").write_text("notes")
    assert provenance.check_build(root)


def test_cli_prints_the_revision_and_checks_the_build(capsys):
    assert provenance.main([]) == 0
    assert json.loads(capsys.readouterr().out) == provenance.revision()
    pytest.importorskip("bbar._core")
    assert provenance.main(["--check-build"]) == 0
