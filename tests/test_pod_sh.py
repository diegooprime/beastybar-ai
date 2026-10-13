"""scripts/pod.sh hands the pod the command it was given, word for word (`ssh` and `bg`).

runpodctl and ssh are stand-ins: ssh runs the remote command line with a local bash, as sshd would with the pod's
shell, in a scratch directory standing for the pod's repo. The check that matters: a quoted argument with spaces (a
list of DATA directories for scripts/value_train.sh) reaches the command as one argument, so the trainer's parser
reads every directory, and a single argument is still a shell command line (pipes, &&).
"""
import argparse
import json
import os
import shutil
import stat
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
POD_SH = REPO / "scripts" / "pod.sh"

RUNPODCTL = """#!/usr/bin/env bash
echo '{"ip": "127.0.0.1", "port": 22}'
"""
# ssh's own options, then user@host, then the command words, which ssh joins with spaces for the remote shell.
SSH = """#!/usr/bin/env bash
while [ $# -gt 0 ]; do
  case $1 in
    -i|-o|-p) shift 2 ;;
    -*) shift ;;
    *@*) shift; break ;;
    *) shift ;;
  esac
done
exec bash -c "cd() { builtin cd \\"\\$FAKE_REMOTE\\"; }; $*"
"""
# python on the pod: record argv (value_train.sh execs python -m bbar.value.train ...).
PYTHON = """#!/usr/bin/env bash
"$REAL_PYTHON" -c 'import json, sys; json.dump(sys.argv[1:], open("py_argv.json", "w"))' "$@"
"""
ARGV = "import json, sys; json.dump(sys.argv[1:], open('argv.json', 'w'))"
WORDS = ["data/value/v1 data/value/v2", "it's", "$HOME", "*", "", "a\"b", "back\\slash", "--x=1 2"]


def _bashes():
    found = {}
    for b in (shutil.which("bash"), "/bin/bash"):
        if b and os.path.exists(b):
            v = subprocess.run([b, "-c", "echo $BASH_VERSION"], capture_output=True, text=True).stdout.strip()
            found.setdefault(v, b)
    return sorted(found.values())


def _exe(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def pod(tmp_path):
    stubs, remote, home = tmp_path / "bin", tmp_path / "remote", tmp_path / "home"
    for d in (stubs, remote, home):
        d.mkdir()
    _exe(stubs / "runpodctl", RUNPODCTL)
    _exe(stubs / "ssh", SSH)
    _exe(stubs / "python", PYTHON)
    _exe(stubs / "maturin", "#!/usr/bin/env bash\n")
    env = dict(os.environ, PATH=f"{stubs}:{os.environ['PATH']}", HOME=str(home), FAKE_REMOTE=str(remote),
               REAL_PYTHON=shutil.which("python3") or "python3")

    def run(bash, *args):
        return subprocess.run([bash, str(POD_SH), *args], env=env, capture_output=True, text=True, timeout=60)
    return run, remote


def _wait(done, seconds=20):
    end = time.time() + seconds
    while not done():
        if time.time() > end:
            raise AssertionError("the detached command did not finish")
        time.sleep(0.05)


def _wait_for(path):
    """The JSON a detached command writes to `path` (written in one go by json.dump: wait for it to parse)."""
    def ready():
        try:
            json.loads(path.read_text())
            return True
        except (OSError, ValueError):
            return False
    _wait(ready)
    return json.loads(path.read_text())


@pytest.mark.parametrize("bash", _bashes())
def test_ssh_and_bg_pass_every_word_unchanged(pod, bash):
    run, remote = pod
    out = run(bash, "ssh", "POD", "python3", "-c", ARGV, *WORDS)
    assert out.returncode == 0, out.stderr
    assert json.loads((remote / "argv.json").read_text()) == WORDS
    (remote / "argv.json").unlink()
    out = run(bash, "bg", "POD", "job", "python3", "-c", ARGV, *WORDS)
    assert out.returncode == 0 and "started job" in out.stdout, out.stderr
    assert _wait_for(remote / "argv.json") == WORDS


@pytest.mark.parametrize("bash", _bashes())
def test_one_argument_is_a_command_line(pod, bash):
    run, remote = pod
    out = run(bash, "ssh", "POD", "echo a && echo 'b c' | tr b B")
    assert out.returncode == 0 and out.stdout.split("\n")[:2] == ["a", "B c"], out
    out = run(bash, "bg", "POD", "line", "echo x > one.txt && echo y >> one.txt")
    assert out.returncode == 0, out.stderr
    one = remote / "one.txt"
    _wait(lambda: one.exists() and one.read_text().split() == ["x", "y"])
    assert (remote / "one.txt").read_text().split() == ["x", "y"]


@pytest.mark.parametrize("bash", _bashes())
def test_value_train_gets_every_data_directory(pod, bash, monkeypatch):
    """The command for a training run on several label directories (scripts/value_train.sh's header) reaches the
    trainer's parser with all of them, and the flags after them."""
    run, remote = pod
    (remote / "scripts").mkdir()
    shutil.copy(REPO / "scripts" / "value_train.sh", remote / "scripts" / "value_train.sh")
    _exe(remote / "scripts" / "build_core.sh", "#!/usr/bin/env bash\n")
    out = run(bash, "bg", "POD", "vt-v3opt", "scripts/value_train.sh", "v3opt",
              "data/value/v1 data/value/v2 data/value/v2q", "--arch", "opt", "--tok-width", "128")
    assert out.returncode == 0, out.stderr
    argv = _wait_for(remote / "py_argv.json")
    assert argv[:2] == ["-m", "bbar.value.train"]
    # The trainer's own parser reads them (and stops there: no training).
    pytest.importorskip("torch")
    from bbar.value import train
    seen, parse = [], argparse.ArgumentParser.parse_args

    class Parsed(Exception):
        pass

    def spy(self, args=None, namespace=None):
        seen.append(parse(self, args, namespace))
        raise Parsed
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", spy)
    with pytest.raises(Parsed):
        train.main(argv[2:])
    a = seen[0]
    assert a.name == "v3opt" and a.data == ["data/value/v1", "data/value/v2", "data/value/v2q"]
    assert a.arch == "opt" and a.tok_width == 128 and a.device == "cuda"
    assert "unrecognized" not in (remote / "runs" / "vt-v3opt.log").read_text()
