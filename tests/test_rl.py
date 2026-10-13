"""The training environment (bbar._core.VecEnv) against the engine, and one tiny MMD-PPO training run."""
import dataclasses
import math

import numpy as np
import pytest

core = pytest.importorskip("bbar._core")
from bbar._core import NUM_ACTIONS, OBS_DIM, PRIV_DIM, VecEnv, encode_moves  # noqa: E402

ANIMALS = ["skunk", "parrot", "kangaroo", "monkey", "chameleon", "seal",
           "zebra", "giraffe", "snake", "crocodile", "hippo", "lion"]


def test_vecenv_matches_engine():
    seeds, starts = [3, 4, 5, 6, 7], [0, 1, 0, 1, 1]
    env = VecEnv(5, seed=0, threads=2)
    env.reset_deals(np.array(seeds, dtype=np.uint64), np.array(starts))
    games = [core.Game.random(s, st) for s, st in zip(seeds, starts)]
    rng = np.random.default_rng(0)
    finished = np.zeros(5, dtype=bool)
    plies = 0
    while not finished.all():
        obs, mask, mover, ply = env.observe()
        assert obs.shape == (5, OBS_DIM) and obs.dtype == np.float32 and mask.shape == (5, NUM_ACTIONS)
        assert env.privileged().shape == (5, PRIV_DIM)
        actions = np.full(5, -1)
        for i, g in enumerate(games):
            if finished[i]:
                continue
            assert np.flatnonzero(mask[i]).tolist() == g.legal_actions()
            assert mover[i] == g.to_move and ply[i] == plies
            assert mover[i] == g.start ^ (ply[i] % 2), "the starting player moves at even plies"
            actions[i] = rng.choice(np.flatnonzero(mask[i]))
            g.play_action(int(actions[i]))
        done, result, margin = env.step(actions)
        plies += 1
        for i, g in enumerate(games):
            if finished[i]:
                assert not done[i]
                continue
            assert done[i] == g.is_over()
            if done[i]:
                a, b = g.score()
                assert margin[i] == a - b
                assert result[i] == (1.0 if a > b else 0.5 if a == b else 0.0)
                assert tuple(env.final_scores()[i]) == (a, b)
                finished[i] = True
    with pytest.raises(ValueError):
        env.step(np.full(5, 99))


def test_vecenv_reproducible():
    def run(threads):
        env = VecEnv(300, seed=11, threads=threads)
        rng = np.random.default_rng(1)
        out = []
        for _ in range(30):
            obs, mask, _, _ = env.observe()
            a = np.array([rng.choice(np.flatnonzero(m)) for m in mask])
            out.append((obs.sum(1), env.step(a)[1]))
        return out
    for (o1, r1), (o2, r2) in zip(run(1), run(4)):
        assert np.array_equal(o1, o2) and np.array_equal(r1, r2)


def test_observation_uses_only_the_movers_information():
    p0_hand, p0_deck = ANIMALS[:4], ANIMALS[4:]
    # Player 0 to move at ply 0: player 1's hand and both deck orders are hidden from them.
    a = encode_moves(0, [p0_hand, ANIMALS[8:]], [p0_deck, ANIMALS[:8]], [])
    b = encode_moves(0, [p0_hand, ANIMALS[:4]], [p0_deck[::-1], ANIMALS[4:]], [])
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[2], b[2]), "the privileged features do see the hidden information"
    # Player 1 to move after player 0 played the lion (action 35): player 0's new hand is hidden from player 1.
    p1 = [ANIMALS[2:6], ANIMALS[6:] + ANIMALS[:2]]
    a = encode_moves(0, [["lion", "skunk", "parrot", "seal"], p1[0]],
                     [[x for x in ANIMALS if x not in ("lion", "skunk", "parrot", "seal")], p1[1]], [35])
    b = encode_moves(0, [["lion", "zebra", "hippo", "monkey"], p1[0]],
                     [[x for x in ANIMALS if x not in ("lion", "zebra", "hippo", "monkey")][::-1], p1[1]], [35])
    assert np.array_equal(a[0], b[0])


torch = pytest.importorskip("torch")


def test_gae():
    from bbar.rl.mmd import gae
    G, P = 1, 24
    values = torch.zeros(G, P)
    values[0, :5] = torch.tensor([0.5, 0.4, 0.6, 0.3, 0.7])
    valid = torch.zeros(G, P, dtype=torch.bool)
    valid[0, :5] = True
    outcome = torch.zeros(G, P)
    outcome[0, :5] = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0])   # player at even plies won
    # lambda = 1: Monte Carlo, A = outcome - V.
    adv = gae(values, outcome, valid, 1.0)
    assert torch.allclose(adv[0, :5], outcome[0, :5] - values[0, :5])
    assert (adv[0, 5:] == 0).all()
    # lambda = 0: one-step TD along the player's own decisions (ply k -> k + 2), the final score after the last.
    adv = gae(values, outcome, valid, 0.0)
    assert torch.allclose(adv[0, :5], torch.tensor([0.6 - 0.5, 0.3 - 0.4, 0.7 - 0.6, 0.0 - 0.3, 1.0 - 0.7]))


def tiny_config(tmp_path, **kw):
    from bbar.rl.mmd import Config
    cfg = Config(name="tiny", device="cpu", n_envs=32, rollout_steps=24, width=32, blocks=1, critic_width=32,
                 critic_blocks=1, minibatch=256, warmup_updates=1, eval_deals=8,
                 runs_dir=str(tmp_path / "runs"), ckpt_dir=str(tmp_path / "ckpt"), magnet_reset=2)
    return dataclasses.replace(cfg, **kw)


def test_tiny_training_step_and_checkpoint(tmp_path):
    from bbar.rl.agent import NetAgent, load_checkpoint
    from bbar.rl.mmd import Trainer
    cfg = tiny_config(tmp_path)
    tr = Trainer(cfg)
    recs = [tr.train_step() for _ in range(3)]
    trained = [r for r in recs if "loss" in r]
    assert trained, "24 steps finish games in some of the 32 envs"
    for r in trained:
        for k in ("loss", "pg_loss", "kl_magnet", "entropy", "critic_ce", "aux_bce", "critic_ece"):
            assert math.isfinite(r[k]), (k, r)
        assert r["games"] > 0 and r["transitions"] >= r["games"] * 10
    assert tr.magnet is not None, "magnet reset every 2 updates"
    m = tr.evaluate("random", "random")
    assert m.deals == 8 and m.wins + m.draws + m.losses == 16

    tr.save(step=True)
    path = tmp_path / "ckpt" / "tiny" / "latest.pt"
    assert (tmp_path / "ckpt" / "tiny" / f"step_{tr.update}.pt").exists()
    env = VecEnv(16, seed=5)
    obs, mask, _, _ = env.observe()
    agent = NetAgent(path)
    want = tr.policy_logp(tr.policy, torch.from_numpy(obs), torch.from_numpy(mask).bool())
    assert torch.allclose(agent.logp(obs, mask), want)
    acts = agent.act(obs, mask)
    assert all(mask[i, a] for i, a in enumerate(acts))

    resumed = Trainer(cfg, load_checkpoint(path))
    assert (resumed.update, resumed.decisions) == (tr.update, tr.decisions)
    for p, q in zip(resumed.policy.parameters(), tr.policy.parameters()):
        assert torch.equal(p, q)
    assert resumed.train_step()["update"] == tr.update
