"""Arena: reproducible duplicate-deal matches, agent specs, interruption, and the results log."""
import json
import signal
import subprocess
import sys
import time

import pytest

core = pytest.importorskip("bbar._core")

from bbar import arena, provenance  # noqa: E402


def test_same_seed_same_results_for_any_thread_count():
    for a, b in [("random", "greedy"), ("pimc:w=4,d=3", "greedy")]:
        one = core.play_match(a, b, 24, seed=5, threads=1)
        many = core.play_match(a, b, 24, seed=5, threads=4)
        for key in ("deal", "a_seat", "start", "a_points", "b_points", "moves"):
            assert one[key] == many[key], key
        assert one["moves"] != core.play_match(a, b, 24, seed=6, threads=4)["moves"]


def test_games_are_duplicate_pairs_and_replay():
    seed = 1
    r = core.play_match("random", "greedy", 6, seed=seed, threads=2)
    assert r["a"] == "random" and r["b"] == "greedy"
    assert r["deal"] == [k // 2 for k in range(12)]
    assert r["a_seat"] == [0, 1] * 6
    assert r["start"] == [(k // 2) % 2 for k in range(12)]
    for k, moves in enumerate(r["moves"]):
        game = core.arena_deal(seed, r["deal"][k])
        assert game.start == r["start"][k]
        for m in moves:
            game.play_action(m)
        assert game.is_over()
        a_seat = r["a_seat"][k]
        assert (game.score()[a_seat], game.score()[1 - a_seat]) == (r["a_points"][k], r["b_points"][k])


def test_greedy_beats_random():
    result, _ = arena.run("greedy", "random", 200, seed=3, threads=0)
    assert result.deals == 200
    assert result.score_lo > 0.5, str(result)


def test_specs_are_validated_and_canonical():
    assert core.parse_spec("pimc:d=4, w=16") == "pimc:w=16,d=4"
    assert core.parse_spec("pimc") == "pimc:w=32,d=6"
    for bad in ["minimax", "pimc:x=1", "random:w=2", "pimc:w=0", "pimc:d=0", "pimc:w=1,d=0", "cheat:d=0"]:
        with pytest.raises(ValueError):
            core.parse_spec(bad)
    with pytest.raises(ValueError):
        core.play_match("greedy", "pimc:q=3", 1, seed=0)
    with pytest.raises(TypeError):
        core.play_match("greedy", "random", 1)  # no default seed


def test_cli_logs_one_line_per_match(tmp_path, capsys):
    log = tmp_path / "results.jsonl"
    arena.main(["--ladder", "random", "greedy", "pimc:w=2,d=2", "--deals", "8", "--seed", "4",
                "--threads", "2", "--log", str(log), "--quiet"])
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(x["a"], x["b"]) for x in lines] == [
        ("random", "greedy"), ("random", "pimc:w=2,d=2"), ("greedy", "pimc:w=2,d=2")]
    for x in lines:
        assert x["deals"] == 8 and x["seed"] == 4 and x["threads"] == 2
        assert x["wins"] + x["draws"] + x["losses"] == 16
        for key in ("score", "score_lo", "score_hi", "margin", "margin_ci95", "seconds", "commit", "code", "build",
                    "hostname", "command", "time"):
            assert key in x, key
        assert x["score_lo"] < x["score_hi"] and x["score_lo"] <= x["score"] <= x["score_hi"]
        assert x["command"].startswith("python -m bbar.arena --ladder random greedy")
        assert x["commit"] == provenance.revision()["commit"]
        assert x["code"] == provenance.code_id()
        assert x["build"] == core.build_id() == provenance.core_source_id()
    out = capsys.readouterr().out
    assert "mean" in out and "random vs greedy" in out

    arena.main(["greedy", "random", "--deals", "3", "--log", str(log), "--quiet"])
    assert len(log.read_text().splitlines()) == 4


def test_cli_rejects_bad_specs_before_playing(tmp_path):
    log = tmp_path / "results.jsonl"
    with pytest.raises(SystemExit):
        arena.main(["--ladder", "random", "greedy", "pimc:w=2,x=1", "--deals", "2", "--log", str(log)])
    assert not log.exists()


def test_results_log_is_not_gitignored():
    import subprocess
    r = subprocess.run(["git", "check-ignore", "-q", "runs/results.jsonl"], cwd=arena.ROOT)
    assert r.returncode == 1  # 1 = not ignored


def test_default_seed_is_random_64_bit_and_logged(tmp_path):
    # Explicit seeds are guessable: a cheating agent that knows the seed can recompute every deal
    # (core/tests/info_leaks.rs). Without --seed the arena draws 64 random bits and logs them.
    log = tmp_path / "results.jsonl"
    for _ in range(2):
        arena.main(["random", "greedy", "--deals", "2", "--log", str(log), "--quiet"])
    seeds = [json.loads(line)["seed"] for line in log.read_text().splitlines()]
    assert seeds[0] != seeds[1] and all(2 ** 32 <= s < 2 ** 64 for s in seeds)  # (fails with chance ~2^-31)
    # The logged seed reproduces the match.
    x = json.loads(log.read_text().splitlines()[0])
    result, _ = arena.run("random", "greedy", 2, seed=x["seed"], threads=1)
    assert (result.wins, result.draws, result.losses) == (x["wins"], x["draws"], x["losses"])
    with pytest.raises(SystemExit):
        arena.main(["random", "greedy", "--deals", "2", "--seed", str(2 ** 64), "--log", str(log)])


def test_evidence_trail_refuses_uncommitted_code(tmp_path, monkeypatch):
    trail = tmp_path / "results.jsonl"
    monkeypatch.setattr(arena, "LOG", trail)
    monkeypatch.setattr(arena, "ROOT", tmp_path)
    monkeypatch.setattr(provenance, "revision", lambda root=None: {"commit": "abc1234-dirty", "code": "0" * 16})
    with pytest.raises(SystemExit):
        arena.main(["random", "greedy", "--deals", "2", "--seed", "1", "--quiet"])
    assert not trail.exists()
    # --allow-dirty logs it, with the code hash that identifies the tree.
    arena.main(["random", "greedy", "--deals", "2", "--seed", "1", "--quiet", "--allow-dirty"])
    x = json.loads(trail.read_text())
    assert (x["commit"], x["code"]) == ("abc1234-dirty", "0" * 16)
    # Committed code needs no flag.
    monkeypatch.setattr(provenance, "revision", lambda root=None: {"commit": "abc1234", "code": "1" * 16})
    arena.main(["random", "greedy", "--deals", "2", "--seed", "1", "--quiet"])
    assert len(trail.read_text().splitlines()) == 2


def test_stale_engine_build_is_refused(tmp_path, monkeypatch):
    # The extension that would play was compiled from other sources than core/ holds now: refuse before playing.
    log = tmp_path / "results.jsonl"
    monkeypatch.setattr(provenance, "core_source_id", lambda root=None: "0123456789abcdef")
    with pytest.raises(SystemExit):
        arena.main(["random", "greedy", "--deals", "2", "--seed", "1", "--log", str(log), "--quiet"])
    assert not log.exists()


def test_ctrl_c_interrupts_a_match_stuck_in_one_long_search():
    # pimc to depth 24 from the opening needs hours per move. SIGINT must still end the match within seconds.
    code = "from bbar._core import play_match; play_match('pimc:w=1,d=24', 'random', 1000, seed=0, threads=1)"
    p = subprocess.Popen([sys.executable, "-c", code], cwd=arena.ROOT, stderr=subprocess.PIPE, text=True)
    try:
        time.sleep(1.5)
        assert p.poll() is None, p.stderr.read()
        p.send_signal(signal.SIGINT)
        t = time.monotonic()
        p.wait(timeout=10)
        assert time.monotonic() - t < 5
        assert "KeyboardInterrupt" in p.stderr.read()
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
