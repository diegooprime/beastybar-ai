"""Adversarial review of the blueprint pipeline: information leakage into the policy and feature-encoding bugs.

Independent of the Rust encoder's own tests: observations are re-derived in Python from the layout spec in
core/src/features.rs using only the Game API, and hidden information is perturbed by building whole alternative deals
that reproduce the same public history (not only by redrawing the state at one ply).
"""
import dataclasses

import numpy as np
import pytest

core = pytest.importorskip("bbar._core")
from bbar._core import NUM_ACTIONS, OBS_DIM, PRIV_DIM, Game, VecEnv, encode_moves, obs_layout  # noqa: E402

ANIMALS = ["skunk", "parrot", "kangaroo", "monkey", "chameleon", "seal",
           "zebra", "giraffe", "snake", "crocodile", "hippo", "lion"]
AID = {a: i for i, a in enumerate(ANIMALS)}
L = obs_layout()


# ------------------------------------------------------------------------------------------------ spec encoder

def spec_queue(out, off, queue, me):
    """queue: [(animal, owner)] front first; 4 slots x (animal one-hot 12, mine, theirs)."""
    for i, (a, o) in enumerate(queue):
        out[off + 14 * i + AID[a]] = 1
        out[off + 14 * i + 12 + (o != me)] = 1


def spec_obs(g, hist):
    """Observation of g's player to move from the documented layout. hist: [(player, action, queue_before)]."""
    out = np.zeros(OBS_DIM, dtype=np.float32)
    me, op = g.to_move, 1 - g.to_move
    for a in g.hand(me):
        out[0 + AID[a]] = 1
    for a in g.deck(me):
        out[12 + AID[a]] = 1
    for a in g.hand(op) + g.deck(op):
        out[24 + AID[a]] = 1
    if not g.deck(op):
        for a in g.hand(op):
            out[36 + AID[a]] = 1
    for a, o in g.bar():
        out[(48 if o == me else 60) + AID[a]] = 1
    for a, o in g.thats_it():
        out[(72 if o == me else 84) + AID[a]] = 1
    spec_queue(out, 96, g.queue(), me)
    ply = len(hist)
    out[152 + ply] = 1
    s = g.score()
    left = sum(len(g.hand(p)) + len(g.deck(p)) for p in (0, 1)) + len(g.queue())
    out[177:185] = [s[me] / 20, s[op] / 20, (s[me] - s[op]) / 20, len(g.deck(me)) / 8, len(g.deck(op)) / 8,
                    float(g.start == me), len(g.queue()) / 4, left / 24]
    for j, (p, a, q) in enumerate(reversed(hist)):
        base = 185 + 94 * j
        out[base + (0 if p == me else 1)] = 1
        out[base + 2 + a] = 1
        spec_queue(out, base + 38, q, me)
    return out


def random_game_prefix(rng, seed, start, stop_ply=None):
    """A random game from Game.random(seed, start); returns (game at stop, actions, hist) with hist entries
    (player, action, queue before) and the game not over."""
    g = Game.random(seed, start)
    actions, hist = [], []
    stop = stop_ply if stop_ply is not None else rng.integers(0, 24)
    while not g.is_over() and len(actions) < stop:
        a = int(rng.choice(g.legal_actions()))
        hist.append((g.to_move, a, g.queue()))
        g.play_action(a)
        actions.append(a)
    if g.is_over():   # back off one move so a player is to move
        return random_game_prefix(rng, seed, start, len(actions) - 1)
    return g, actions, hist


def test_layout_matches_spec_offsets():
    assert L["my_hand"] == (0, 12) and L["opp_hand_public"] == (36, 12) and L["queue"] == (96, 56)
    assert L["ply"] == (152, 25) and L["scalars"] == (177, 8) and L["history"] == (185, 24 * 94)
    assert L["history_queue"] == (38, 1) and OBS_DIM == 185 + 24 * 94 and PRIV_DIM == 36


def test_encode_moves_matches_spec_encoder():
    """Every position of 400 random games, encoded by Rust (encode_moves) and by the spec encoder above."""
    rng = np.random.default_rng(1)
    checked = 0
    for seed in range(400):
        start = seed % 2
        g0 = Game.random(seed, start)
        hands = [g0.hand(0), g0.hand(1)]
        decks = [g0.deck(0), g0.deck(1)]
        g = g0.copy()
        actions, hist = [], []
        while not g.is_over():
            obs, mask, priv = encode_moves(start, hands, decks, actions)
            want = spec_obs(g, hist)
            bad = np.flatnonzero(obs != want)
            assert len(bad) == 0, (seed, len(actions), [(int(i), obs[i], want[i]) for i in bad[:10]])
            assert np.flatnonzero(mask).tolist() == g.legal_actions()
            # Privileged: opponent hand, my next draw, opponent's next draw.
            me, op = g.to_move, 1 - g.to_move
            pw = np.zeros(PRIV_DIM, dtype=np.float32)
            for a in g.hand(op):
                pw[AID[a]] = 1
            if g.deck(me):
                pw[12 + AID[g.deck(me)[0]]] = 1
            if g.deck(op):
                pw[24 + AID[g.deck(op)[0]]] = 1
            assert np.array_equal(priv, pw)
            a = int(rng.choice(g.legal_actions()))
            hist.append((g.to_move, a, g.queue()))
            g.play_action(a)
            actions.append(a)
            checked += 1
    assert checked > 8000


def test_vecenv_observe_matches_spec_encoder_through_auto_reset():
    """The VecEnv path (history kept by VecEnv.step, auto-reset) against the spec encoder, over two games per env."""
    n = 64
    rng = np.random.default_rng(2)
    env = VecEnv(n, seed=0, threads=3)
    seeds = np.arange(100, 100 + n, dtype=np.uint64)
    starts = np.arange(n) % 2
    env.reset_deals(seeds, starts)
    games = [Game.random(int(s), int(st)) for s, st in zip(seeds, starts)]
    hists = [[] for _ in range(n)]
    finished = np.zeros(n, dtype=bool)
    steps = 0
    while not finished.all():
        obs, mask, mover, ply = env.observe()
        priv = env.privileged()
        actions = np.full(n, -1)
        for i in range(n):
            if finished[i]:
                continue
            want = spec_obs(games[i], hists[i])
            assert np.array_equal(obs[i], want), (i, np.flatnonzero(obs[i] != want)[:10])
            assert ply[i] == len(hists[i]) and mover[i] == games[i].to_move
            assert priv[i, :12].sum() == len(games[i].hand(1 - games[i].to_move))
            a = int(rng.choice(np.flatnonzero(mask[i])))
            hists[i].append((games[i].to_move, a, games[i].queue()))
            games[i].play_action(a)
            actions[i] = a
        done, result, margin = env.step(actions)
        steps += 1
        for i in range(n):
            if not finished[i] and done[i]:
                assert games[i].is_over()
                finished[i] = True
    # After auto-reset the fresh game must have an empty history and ply 0.
    obs, mask, mover, ply = env.observe()
    assert (ply == 0).all()
    assert (obs[:, 185:] == 0).all(), "history of a fresh game after auto-reset must be empty"
    assert (obs[:, 152] == 1).all()


# ----------------------------------------------------------------------------- hidden-information invariance

def alternative_deal(rng, g0, actions, k_plies):
    """A different deal that reproduces the same public history for `actions[:k_plies]`: for each player, the
    animals still unplayed after k plies are randomly re-assigned among the positions (initial hand slots and
    deck positions) those animals occupied. Played animals keep their positions, so every played card is in hand
    when it was played and the public history is identical. The mover at ply k keeps their current hand; only
    the order of their undrawn deck is shuffled. Returns (hands, decks)."""
    g = g0.copy()
    for a in actions[:k_plies]:
        g.play_action(a)
    mover = g.to_move
    hands, decks = [], []
    for p in (0, 1):
        init = g0.hand(p) + g0.deck(p)                 # positions: 4 hand slots, then the deck top first
        unplayed = set(g.hand(p)) | set(g.deck(p))
        if p == mover:
            # keep the current hand fixed: shuffle only positions of animals still in the deck
            pos = [i for i, a in enumerate(init) if a in set(g.deck(p))]
        else:
            pos = [i for i, a in enumerate(init) if a in unplayed]
        vals = [init[i] for i in pos]
        rng.shuffle(vals)
        alt = list(init)
        for i, v in zip(pos, vals):
            alt[i] = v
        hands.append(alt[:4])
        decks.append(alt[4:])
    return hands, decks


def test_observation_invariant_to_alternative_deals_with_same_public_history():
    rng = np.random.default_rng(3)
    changed_opp_hand = changed_any = 0
    for seed in range(600):
        start = seed % 2
        g0 = Game.random(seed, start)
        g, actions, _ = random_game_prefix(rng, seed, start)
        k = len(actions)
        base_obs, base_mask, base_priv = encode_moves(start, [g0.hand(0), g0.hand(1)], [g0.deck(0), g0.deck(1)],
                                                      actions)
        for _ in range(3):
            hands, decks = alternative_deal(rng, g0, actions, k)
            obs, mask, priv = encode_moves(start, hands, decks, actions)   # raises if the history became illegal
            assert np.array_equal(obs, base_obs), (seed, k, np.flatnonzero(obs != base_obs)[:10])
            assert np.array_equal(mask, base_mask)
            changed_any += not np.array_equal(priv, base_priv)
            changed_opp_hand += not np.array_equal(priv[:12], base_priv[:12])
    assert changed_opp_hand > 300 and changed_any > 600, "the test must perturb hidden information"


def test_policy_output_invariant_to_hidden_information():
    """End to end through a (randomly initialised) PolicyNet and NetAgent: identical logits and aux output."""
    torch = pytest.importorskip("torch")
    from bbar.rl.agent import NetAgent
    from bbar.rl.net import PolicyNet
    torch.manual_seed(0)
    agent = NetAgent(PolicyNet(width=64, blocks=2))
    rng = np.random.default_rng(4)
    base, alt, masks = [], [], []
    for seed in range(200):
        start = seed % 2
        g0 = Game.random(seed, start)
        _, actions, _ = random_game_prefix(rng, seed, start)
        o, m, _ = encode_moves(start, [g0.hand(0), g0.hand(1)], [g0.deck(0), g0.deck(1)], actions)
        hands, decks = alternative_deal(rng, g0, actions, len(actions))
        o2, _, _ = encode_moves(start, hands, decks, actions)
        base.append(o), alt.append(o2), masks.append(m)
    base, alt, masks = map(np.stack, (base, alt, masks))
    assert torch.equal(agent.logp(base, masks), agent.logp(alt, masks))
    assert np.array_equal(agent.opponent_hand(base), agent.opponent_hand(alt))


# ------------------------------------------------------------------------------------- trainer data alignment

def tiny_config(tmp_path, **kw):
    from bbar.rl.mmd import Config
    cfg = Config(name="review", device="cpu", n_envs=48, rollout_steps=10, width=32, blocks=1, critic_width=32,
                 critic_blocks=1, minibatch=512, warmup_updates=1, eval_deals=8, epochs=1,
                 runs_dir=str(tmp_path / "runs"), ckpt_dir=str(tmp_path / "ckpt"))
    return dataclasses.replace(cfg, **kw)


class RecordingEnv:
    """Wraps the trainer's VecEnv: logs every step's actions and results, and each env's full game."""

    def __init__(self, env):
        self.env = env
        self.steps = []          # (actions, done, result)
        self.cur = [[] for _ in range(len(env))]   # per env: (state obs, action) of the current game
        self.games = []          # (env, actions, p0 score, hist from VecEnv.history)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, actions):
        hist_before = [self.env.history(i) for i in range(len(self.env))]
        done, result, margin = self.env.step(actions)
        self.steps.append((np.array(actions), done.copy(), result.copy()))
        for i in np.flatnonzero(done):
            self.games.append((int(i), hist_before[i] + [(None, int(actions[i]))], float(result[i])))
        return done, result, margin


def test_trainer_batches_align_obs_actions_outcomes_and_privileged(tmp_path):
    """Every transition the trainer learns from: the stored obs is the position the stored action was played in,
    plies run 0..L-1 within a game, the outcome is the mover's real result, and priv belongs to the same position.
    Covers carried-over games (rollout_steps < game length) and the staggered start."""
    torch = pytest.importorskip("torch")
    from bbar.rl.mmd import Trainer
    tr = Trainer(tiny_config(tmp_path))
    rec = RecordingEnv(tr.env)
    tr.env = rec
    batches = []
    orig_make = tr.make_batch

    def make_batch(ended):
        b = orig_make(ended)
        batches.append((tuple(x.copy() for x in ended), b))
        return b

    tr.make_batch = make_batch
    for _ in range(8):
        tr.train_step()
    assert batches and sum(len(e[0]) for e, _ in batches) > 40
    by_env_games = {}
    for env_i, hist, score in rec.games:
        by_env_games.setdefault(env_i, []).append((hist, score))
    used = {}
    for (env_idx, first, length, p0, margin), b in batches:
        pos = 0
        obs = b.obs.float().numpy()
        priv = b.priv.float().numpy()
        act = b.action.numpy()
        outc = b.outcome.numpy()
        mask = b.mask.numpy()
        for e, f, ln, sc in zip(env_idx, first, length, p0):
            # the j-th completed game of env e (games complete in order)
            j = used.get(e, 0)
            used[e] = j + 1
            hist, score = by_env_games[int(e)][j]
            assert len(hist) == ln and score == sc
            for k in range(ln):
                o, p, a = obs[pos + k], priv[pos + k], act[pos + k]
                assert o[152 + k] == 1, "ply one-hot must equal the ply of the stored transition"
                assert a == hist[k][1], "stored action must be the action played at that ply"
                assert mask[pos + k, a], "stored action must be legal in the stored mask"
                if k > 0:
                    # most recent history slot = the previous move, by the opponent
                    assert o[185 + 2 + hist[k - 1][1]] == 1 and o[185 + 1] == 1
                mover = hist[k][0] if hist[k][0] is not None else 1 - hist[k - 1][0]
                assert outc[pos + k] == (sc if mover == 0 else 1 - sc)
                # privileged belongs to the same position as the obs
                opp_unplayed, my_deck = o[24:36], o[12:24]
                assert np.all(p[:12] <= opp_unplayed)
                assert p[:12].sum() == min(4, opp_unplayed.sum())
                if p[12:24].any():
                    assert np.all(p[12:24] <= my_deck)
                if p[24:36].any():
                    assert np.all(p[24:36] <= opp_unplayed - p[:12])
            pos += ln
        assert pos == b.n


# ------------------------------------------------------------------------- review findings: CLI, agent, encoder

def cli(tmp_path, name, *extra):
    """bbar.rl.mmd argv for a tiny CPU run of one update under tmp_path."""
    return ["--name", name, "--device", "cpu", "--n-envs", "32", "--rollout-steps", "24", "--env-threads", "2",
            "--width", "16", "--blocks", "1", "--critic-width", "16", "--critic-blocks", "1", "--minibatch", "256",
            "--total-decisions", "768", "--eval-deals", "4", "--ckpt-every", "1",
            "--runs-dir", str(tmp_path / "runs"), "--ckpt-dir", str(tmp_path / "ckpt"), *extra]


def records(path):
    import json
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_resume_without_a_checkpoint_fails_and_touches_nothing(tmp_path):
    pytest.importorskip("torch")
    from bbar.rl.mmd import main
    (tmp_path / "runs" / "typo").mkdir(parents=True)
    (tmp_path / "runs" / "typo" / "metrics.jsonl").write_text('{"type": "start"}\n')
    with pytest.raises(SystemExit, match="not found"):
        main(cli(tmp_path, "typo", "--resume"))
    assert (tmp_path / "runs" / "typo" / "metrics.jsonl").read_text() == '{"type": "start"}\n'
    assert not (tmp_path / "ckpt").exists()
    with pytest.raises(SystemExit, match="exclude"):
        main(cli(tmp_path, "typo", "--resume", "--force"))


def test_fresh_run_never_mixes_with_an_old_one(tmp_path):
    """Without --force an existing run is refused (even with no latest.pt); with --force it is moved aside whole, so
    the new run's metrics.jsonl and checkpoints hold nothing of the old run."""
    pytest.importorskip("torch")
    from bbar.rl.mmd import main
    run, ck = tmp_path / "runs" / "r", tmp_path / "ckpt" / "r"
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text('{"type": "start", "old": true}\n{"type": "train", "update": 0}\n')
    with pytest.raises(SystemExit, match="--force") as e:
        main(cli(tmp_path, "r"))
    assert "--resume" not in str(e.value), "no checkpoint: resuming is not an option"
    ck.mkdir(parents=True)
    for f in ("latest.pt", "step_4.pt"):
        (ck / f).write_bytes(b"old")
    with pytest.raises(SystemExit, match="--resume"):
        main(cli(tmp_path, "r"))
    assert (ck / "step_4.pt").read_bytes() == b"old"

    main(cli(tmp_path, "r", "--force"))
    recs = records(run / "metrics.jsonl")
    assert recs[0]["type"] == "start" and "old" not in recs[0]
    assert [r["type"] for r in recs].count("start") == 1
    assert [r["update"] for r in recs if r["type"] == "train"] == [0]
    assert sorted(p.name for p in ck.iterdir()) == ["latest.pt", "step_1.pt"]
    (old_run,) = (tmp_path / "runs").glob("r.replaced-*")
    (old_ck,) = (tmp_path / "ckpt").glob("r.replaced-*")
    assert records(old_run / "metrics.jsonl")[0]["old"] is True
    assert sorted(p.name for p in old_ck.iterdir()) == ["latest.pt", "step_4.pt"]

    # A second --force keeps both earlier runs.
    main(cli(tmp_path, "r", "--force"))
    assert len(list((tmp_path / "ckpt").glob("r.replaced-*"))) == 2
    # --resume continues in place, appending.
    main(cli(tmp_path, "r", "--resume", "--total-decisions", "1536"))
    recs = records(run / "metrics.jsonl")
    assert [r["type"] for r in recs].count("start") == 2
    assert [r["update"] for r in recs if r["type"] == "train"] == [0, 1]


def test_cli_flag_types_follow_the_config_annotations():
    pytest.importorskip("torch")
    import typing
    from bbar.rl.mmd import Config, build_parser
    args = build_parser().parse_args(["--alpha-decay-decisions", "2e9", "--total-decisions", "4e9"])
    assert args.alpha_decay_decisions == 2e9 and isinstance(args.alpha_decay_decisions, float)
    hints = typing.get_type_hints(Config)
    floats = [f.name for f in dataclasses.fields(Config) if hints[f.name] is float]
    assert "alpha_decay_decisions" in floats
    for name in floats:
        flag = "--" + name.replace("_", "-")
        assert getattr(build_parser().parse_args([flag, "1e3"]), name) == 1000.0
        assert getattr(build_parser(explicit_only=True).parse_args([flag, "1e3"]), name) == 1000.0
    defaults = vars(build_parser().parse_args([]))
    for f in dataclasses.fields(Config):
        assert isinstance(defaults[f.name], hints[f.name]), (f.name, defaults[f.name])


def test_opponent_hand_is_consistent_with_the_observation():
    """Played animals get 0, a public hand is returned exactly, a hidden hand's probabilities sum to 4 and keep the
    head's order. Holds for any weights, so a random network is enough."""
    torch = pytest.importorskip("torch")
    from bbar.rl.agent import NetAgent
    from bbar.rl.net import OPP_HAND_PUBLIC, OPP_UNPLAYED, PolicyNet
    torch.manual_seed(1)
    agent = NetAgent(PolicyNet(width=32, blocks=1))
    with torch.no_grad():
        agent.net.opp_hand.bias.add_(2.0)          # raw marginals well above a consistent range
    env = VecEnv(256, seed=3, threads=2)
    rng = np.random.default_rng(0)
    hidden_rows = public_rows = 0
    raw_sums = []
    for _ in range(40):
        obs, mask, _, _ = env.observe()
        priv = env.privileged()
        p, raw = agent.opponent_hand(obs), agent.opponent_hand(obs, exact_size=False)
        unplayed, public = obs[:, OPP_UNPLAYED] > 0, obs[:, OPP_HAND_PUBLIC]
        assert p.dtype == np.float32 and ((0 <= p) & (p <= 1)).all()
        assert (p[~unplayed] == 0).all() and (raw[~unplayed] == 0).all()
        pub = public.any(1)
        assert np.array_equal(p[pub], public[pub]) and np.array_equal(raw[pub], public[pub])
        assert np.array_equal(p[pub], priv[pub, :12]), "a public hand is the true hand"
        hid = ~pub & (unplayed.sum(1) > 0)
        assert np.allclose(p[hid].sum(1), 4, atol=1e-4) and np.array_equal(priv[hid, :12].sum(1), np.full(hid.sum(), 4))
        d = (p[hid][:, :, None] - p[hid][:, None, :]) * (raw[hid][:, :, None] - raw[hid][:, None, :])
        assert (d >= 0).all(), "the size correction keeps the head's order"
        hidden_rows += hid.sum()
        raw_sums.append(raw[hid].sum(1))
        public_rows += pub.sum()
        env.step(np.array([rng.choice(np.flatnonzero(m)) for m in mask]))
    assert hidden_rows > 1000 and public_rows > 1000
    assert np.concatenate(raw_sums).mean() > 5, "the test must start from inconsistent raw marginals"


def test_encode_moves_rejects_impossible_deals():
    a = ANIMALS
    ok_hands, ok_decks = [a[:4], a[8:]], [a[4:], a[:8]]
    encode_moves(0, ok_hands, ok_decks, [])
    bad = {
        "two lions": ([["lion", "skunk", "parrot", "seal"], a[:4]], [["lion"] + a[2:9], a[4:]]),   # hand + deck
        "two skunks": ([["skunk", "skunk", "parrot", "seal"], a[:4]],
                       [[x for x in a if x not in ("skunk", "parrot", "seal")][:8], a[4:]]),
        "in hand and 8": ([a[:3], a[8:]], [a[3:11], a[:8]]),        # 3 in hand
        "and 7": ([a[:4], a[8:]], [a[4:11], a[:8]]),                  # deck of 7
        "and 9": ([a[:4], a[8:]], [a[4:] + ["skunk"], a[:8]]),        # deck of 9
        "unknown animal": ([a[:3] + ["dragon"], a[8:]], [a[4:], a[:8]]),
    }
    for msg, (hands, decks) in bad.items():
        with pytest.raises(ValueError, match=msg):
            encode_moves(0, hands, decks, [35])
    with pytest.raises(ValueError, match="two lions"):
        Game(0, *bad["two lions"])
