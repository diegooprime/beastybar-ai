"""bbar.match: any agent against any other (built-in Rust agents through VecEnv.act, networks from checkpoints), with
the arena's protocol: Rust-vs-Rust matches replay bbar.arena move for move; determinism; seats; the evidence trail."""
import hashlib
import json
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

core = pytest.importorskip("bbar._core")
torch = pytest.importorskip("torch")

from bbar import arena, match, provenance  # noqa: E402
from bbar._core import OBS_DIM, OBS_VERSION, VecEnv, encode_moves  # noqa: E402
from bbar.rl.net import PolicyNet  # noqa: E402

COLUMNS = ("deal", "a_seat", "start", "a_points", "b_points", "moves")


def tiny_checkpoint(path, width=16, seed=0):
    torch.manual_seed(seed)
    net = PolicyNet(width, 1)
    with torch.no_grad():   # sharper than the near-uniform initialisation, so argmax and sampling differ
        net.policy.weight.mul_(100)
    torch.save({"obs_version": OBS_VERSION, "obs_dim": OBS_DIM, "policy_hparams": net.hparams,
                "policy": net.state_dict()}, path)
    return path


@pytest.mark.parametrize("a,b", [("random", "greedy"), ("pimc:w=3,d=2", "cheat:d=3"),
                                 ("pimc-hand:w=2,d=2", "pimc:w=2,d=3")])
def test_rust_agents_replay_the_arena_move_for_move(a, b):
    ref = core.play_match(a, b, 25, seed=17, threads=2)
    for batch, threads in [(25, 1), (6, 3)]:
        result, raw, _ = match.run(a, b, 25, seed=17, batch=batch, threads=threads)
        for key in COLUMNS:
            assert raw[key] == ref[key], (key, batch, threads)
        want, _ = arena.run(a, b, 25, seed=17)
        assert result == want


def test_vecenv_act_bindings():
    env = VecEnv(6, seed=0, threads=2)
    env.reset_arena(3, [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1])
    for j in range(6):
        g = core.arena_deal(3, j // 2)
        assert env.movers()[0][j] == g.to_move and env.movers()[1][j] == 0
    acts = env.act("pimc:w=2,d=2", np.array([5, 0, 2]))
    assert acts.dtype == np.int64 and len(acts) == 3
    obs, mask, _, _ = env.observe()
    assert all(mask[j, a] for j, a in zip([5, 0, 2], acts))
    u = env.uniforms([1, 4])
    assert u.dtype == np.float64 and ((0 <= u) & (u < 1)).all()
    for bad in ([0, 0], [6], [-1]):
        with pytest.raises(ValueError):
            env.act("random", bad)
    with pytest.raises(ValueError):
        env.act("pimc:x=1", [0])
    with pytest.raises(ValueError):
        env.reset_arena(3, [0] * 6, [2] * 6)


def test_inverse_cdf_sampling():
    p = np.array([[0.0, 0.2, 0.0, 0.5, 0.3, 0.0]])
    u = (np.arange(10000) + 0.5) / 10000
    a = match.sample_inverse_cdf(np.repeat(p, len(u), 0), u)
    assert np.allclose(np.bincount(a, minlength=6) / len(u), p[0], atol=1e-3)
    # Never an action of probability zero, even at the edges.
    a = match.sample_inverse_cdf(np.repeat(p, 3, 0), np.array([0.0, 0.2, 1 - 2 ** -53]))
    assert a.tolist() == [1, 3, 4]


def test_network_matches_are_reproducible_for_any_batch_and_thread_count(tmp_path):
    ck = tiny_checkpoint(tmp_path / "tiny.pt")
    for b in ("pimc:w=2,d=2", f"nn:{ck},argmax"):
        runs = [match.run(f"nn:{ck}", b, 12, seed=4, batch=batch, threads=threads)[1]
                for batch, threads in [(12, 1), (5, 3)]]
        for key in COLUMNS:
            assert runs[0][key] == runs[1][key], (b, key)
        other = match.run(f"nn:{ck}", b, 12, seed=5)[1]
        assert other["moves"] != runs[0]["moves"]


def test_seats_swap_and_the_network_plays_from_its_own_seat(tmp_path):
    # A = the network's argmax, B = random. Every move made from A's seat (as recorded) must be the network's argmax
    # on the standard observation of that position, recomputed from the deal; B's moves mostly are not.
    ck = tiny_checkpoint(tmp_path / "tiny.pt")
    net = PolicyNet(16, 1)
    net.load_state_dict(torch.load(ck, weights_only=True)["policy"])
    seed = 8
    _, raw, _ = match.run(f"nn:{ck},argmax", "random", 15, seed=seed)
    assert raw["deal"] == [k // 2 for k in range(30)] and raw["a_seat"] == [0, 1] * 15
    assert raw["start"] == [(k // 2) % 2 for k in range(30)]
    deviations = 0
    for k, moves in enumerate(raw["moves"]):
        g = core.arena_deal(seed, raw["deal"][k])
        hands, decks, start = [g.hand(0), g.hand(1)], [g.deck(0), g.deck(1)], g.start
        a_seat = raw["a_seat"][k]
        for ply, m in enumerate(moves):
            obs, mask, _ = encode_moves(start, hands, decks, moves[:ply])
            with torch.no_grad():
                logits, _ = net(torch.from_numpy(obs)[None])
            best = int(torch.where(torch.from_numpy(mask).bool(), logits[0], -1e9).argmax())
            if g.to_move == a_seat:
                assert m == best, f"game {k} ply {ply}: A's seat did not play the network's move"
            else:
                deviations += m != best
            g.play_action(m)
        assert g.is_over()
        assert (g.score()[a_seat], g.score()[1 - a_seat]) == (raw["a_points"][k], raw["b_points"][k])
    assert deviations > 30


def test_cli_logs_checkpoint_hashes_and_provenance(tmp_path, capsys):
    ck = tiny_checkpoint(tmp_path / "tiny.pt")
    log = tmp_path / "results.jsonl"
    match.main([f"nn:{ck}", "greedy", "--deals", "5", "--seed", "2", "--device", "cpu", "--log", str(log),
                "--quiet"])
    match.main(["random", "greedy", "--deals", "3", "--seed", "2", "--log", str(log), "--quiet"])
    first, second = [json.loads(line) for line in log.read_text().splitlines()]
    sha = hashlib.sha256(ck.read_bytes()).hexdigest()[:16]
    assert first["checkpoints"] == {str(ck): sha} and first["device"] == "cpu" and first["runner"] == "bbar.match"
    assert first["a"] == f"nn:{ck}" and first["b"] == "greedy" and first["seed"] == 2 and first["deals"] == 5
    assert first["wins"] + first["draws"] + first["losses"] == 10
    assert first["score_lo"] <= first["score"] <= first["score_hi"]
    assert first["commit"] == provenance.revision()["commit"] and first["code"] == provenance.code_id()
    assert first["build"] == core.build_id() == provenance.core_source_id()
    assert first["command"].startswith("python -m bbar.match nn:")
    assert second["checkpoints"] == {} and second["device"] is None
    # The same games as the arena's.
    want, _ = arena.run("random", "greedy", 3, seed=2)
    assert (second["wins"], second["draws"], second["losses"]) == (want.wins, want.draws, want.losses)
    assert "nn:" in capsys.readouterr().out


AUTO = match.resolve_device("auto")


@pytest.mark.skipif(AUTO.type == "cpu", reason="no accelerator: auto is the cpu")
def test_networks_sample_on_the_auto_device(tmp_path):
    # The default device (auto: mps on a Mac, which has no float64) samples, and plays the cpu's games.
    ck = tiny_checkpoint(tmp_path / "tiny.pt")
    env = VecEnv(8, seed=0, threads=2)
    env.reset_arena(5, np.repeat(np.arange(4), 2), np.tile([0, 1], 4))
    obs, mask, _, _ = env.observe()
    idx = np.arange(8)
    for argmax in (False, True):
        acts = match.NetPlayer(ck, AUTO, argmax=argmax).act(env, idx, obs, mask)
        assert acts.dtype == np.int64 and mask[idx, acts].all()
    want = match.run(f"nn:{ck}", "greedy", 10, seed=3, device="cpu")[1]
    got = match.run(f"nn:{ck}", "greedy", 10, seed=3, device=AUTO)[1]
    assert got["moves"] == want["moves"]


def test_cli_plays_sampled_networks_on_the_default_device(tmp_path):
    ck = tiny_checkpoint(tmp_path / "tiny.pt")
    log = tmp_path / "results.jsonl"
    match.main([f"nn:{ck}", "greedy", "--deals", "5", "--seed", "1", "--log", str(log), "--quiet"])  # --device auto
    assert json.loads(log.read_text())["device"] == str(AUTO)


def test_networks_play_with_full_fp32_matmuls_whatever_the_process_set(tmp_path, monkeypatch):
    # bbar.rl.exploit turns TF32 on to train on CUDA; its logged matches must replay as `python -m bbar.match` (TF32
    # off, PyTorch's default) does on the same device.
    ck = tiny_checkpoint(tmp_path / "tiny.pt")
    matmul = torch.backends.cuda.matmul
    seen, act = [], match.NetPlayer.act

    def spy(self, *args):
        seen.append(matmul.allow_tf32)
        return act(self, *args)

    monkeypatch.setattr(match.NetPlayer, "act", spy)
    old = matmul.allow_tf32
    try:
        matmul.allow_tf32 = True
        match.run(f"nn:{ck}", "greedy", 3, seed=1)
        assert seen and not any(seen)
        assert matmul.allow_tf32, "the caller's setting is restored after the match"

        class Broken(match.RustPlayer):
            def act(self, *args):
                raise RuntimeError("broken agent")

        with pytest.raises(RuntimeError, match="broken agent"):
            match.play(match.make_player(f"nn:{ck}"), Broken("greedy"), 2, 1)
        assert matmul.allow_tf32, "restored after a failed match too"
    finally:
        matmul.allow_tf32 = old


def test_cli_refuses_bad_agents_and_uncommitted_code_before_playing(tmp_path, monkeypatch):
    log = tmp_path / "results.jsonl"
    for bad in (["nn:" + str(tmp_path / "missing.pt"), "random"], ["random", "pimc:x=2"], ["nn:", "random"]):
        with pytest.raises(SystemExit):
            match.main([*bad, "--deals", "2", "--log", str(log)])
    assert not log.exists()
    trail = tmp_path / "trail.jsonl"
    monkeypatch.setattr(match, "LOG", trail)
    monkeypatch.setattr(arena, "ROOT", tmp_path)
    monkeypatch.setattr(provenance, "revision", lambda root=None: {"commit": "abc1234-dirty", "code": "0" * 16})
    with pytest.raises(SystemExit):
        match.main(["random", "greedy", "--deals", "2", "--seed", "1", "--quiet", "--log", str(trail)])
    assert not trail.exists()
    match.main(["random", "greedy", "--deals", "2", "--seed", "1", "--quiet", "--log", str(trail), "--allow-dirty"])
    assert json.loads(trail.read_text())["commit"] == "abc1234-dirty"


def test_ctrl_c_interrupts_an_agent_stuck_in_one_long_search():
    # pimc to depth 24 from the opening needs hours per move; SIGINT must end VecEnv.act within seconds.
    code = "from bbar._core import VecEnv; VecEnv(4, seed=0, threads=2).act('pimc:w=1,d=24', [0, 1, 2, 3])"
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


def test_an_interrupted_match_keeps_its_finished_batches():
    class StopsLate(match.RustPlayer):
        calls = 0

        def act(self, *args):
            StopsLate.calls += 1
            if StopsLate.calls > 30:   # well into the second batch of 2 deals
                raise KeyboardInterrupt
            return super().act(*args)

    raw = match.play(match.make_player("greedy"), StopsLate("random"), 6, seed=5, batch=2)
    assert raw["stopped_early"]
    assert raw["deal"] == [0, 0, 1, 1]
    assert all(len(raw[k]) == 4 for k in ("a_seat", "start", "a_points", "b_points", "moves"))
    full = match.play(match.make_player("greedy"), match.make_player("random"), 2, seed=5, batch=2)
    assert raw["a_points"] == full["a_points"] and not full["stopped_early"]
    with pytest.raises(KeyboardInterrupt):   # nothing finished: nothing to keep
        StopsLate.calls = 25
        match.play(match.make_player("greedy"), StopsLate("random"), 6, seed=5, batch=2)
