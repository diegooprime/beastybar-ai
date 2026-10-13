"""Policy tables on windows, Python agents in matches, the exact-tail agent and the delta_T tool (E1).

- WindowTree has Window's layout; NN tables are valid distributions over each hand's legal actions and equal the
  network on the real positions; built-in agent tables are valid and do not depend on the thread count.
- Python agents see only the mover's information (VecEnv.views), get the seat RNG's one draw per decision, play
  reproducibly whatever the batch, and get a fresh instance per stream of games: none sees both games of a duplicate
  pair (a stateful agent that tries to read its opponent's hand from the twin game does no better than chance).
- The tail agent plays legal, reproducible full games; it plays its base network move for move before F; its play does
  not depend on its cache (a fresh agent per call plays the same), which holds one window per game in play and counts
  re-solves; its name tells apart every option that changes play; its gadget re-solve is safe; bbar.delta logs
  consistent lines, solves each window once, and reports per-hand safety margins against the base network.
"""
import json

import numpy as np
import pytest

core = pytest.importorskip("bbar._core")
torch = pytest.importorskip("torch")

from bbar import belief, delta, match, policy, tail, window_check  # noqa: E402
from bbar._core import (OBS_DIM, OBS_VERSION, VecEnv, Window, WindowTree, arena_deal, encode_moves,  # noqa: E402
                        hand_rank)
from bbar.actions import decode, encode  # noqa: E402
from bbar.golden import load  # noqa: E402
from bbar.rl.net import PolicyNet  # noqa: E402


def tiny_checkpoint(path, width=16, seed=0, sharp=30.0):
    torch.manual_seed(seed)
    net = PolicyNet(width, 1)
    with torch.no_grad():
        net.policy.weight.mul_(sharp)
    torch.save({"obs_version": OBS_VERSION, "obs_dim": OBS_DIM, "policy_hparams": net.hparams,
                "policy": net.state_dict()}, path)
    return path


@pytest.fixture(scope="module")
def ck(tmp_path_factory):
    return tiny_checkpoint(tmp_path_factory.mktemp("ck") / "tiny.pt")


@pytest.fixture(scope="module")
def soft_ck(tmp_path_factory):
    """A network close to uniform: its samples depend on every draw."""
    return tiny_checkpoint(tmp_path_factory.mktemp("ck") / "soft.pt", seed=1, sharp=1.0)


def golden_prefix(g, ply):
    """(public start Game, actions) of the first `ply` moves of golden game g."""
    deal = belief.start_game(g["start"])
    actions = [encode(*s["move"]) for s in g["steps"][:ply]]
    return deal, actions


@pytest.fixture(scope="module")
def games():
    return load()


def test_window_tree_has_the_window_layout(games):
    for gi, ply in [(0, 12), (3, 11), (5, 10)]:
        deal, actions = golden_prefix(games[gi], ply)
        tree = WindowTree(deal, actions)
        w = Window(window_check.public_game(games[gi], ply))
        assert tree.num_entries == w.num_entries and tree.num_nodes == w.sizes()["internal"]
        assert tree.ply == ply
        for n, (hist, mover, offset, hands, acts) in enumerate(w.nodes()):
            node = tree.node(n)
            assert (node["path"], node["mover"], node["entry"], node["actions"]) == (hist, mover, offset, acts)
            assert tree.hands(n) == [hand_rank(list(h)) for h in hands]
            assert tree.find(hist) == n
        assert sum(tree.num_rows(p) for p in (0, 1)) == sum(len(x[3]) for x in w.nodes())


def test_nn_table_rows_are_distributions_and_equal_the_network_on_real_positions(ck):
    model = policy.NetModel(str(ck), "cpu")
    rng = np.random.default_rng(1)
    checked = 0
    for d in range(6):
        g0 = core.arena_deal(21, d)
        hands0, decks0, start = [g0.hand(0), g0.hand(1)], [g0.deck(0), g0.deck(1)], g0.start
        g = g0.copy()
        actions = []
        for _ in range(11):
            legal = g.legal_actions()
            actions.append(int(legal[rng.integers(len(legal))]))
            g.play_action(actions[-1])
        deal = belief.start_game(start)
        tree = WindowTree(deal, actions)
        w = Window(belief.replay(deal, actions))
        table = model.window_table(tree, 0) + model.window_table(tree, 1)
        for hist, mover, offset, hands, acts in w.nodes():
            for i, hand in enumerate(hands):
                row = table[offset + i * len(acts): offset + (i + 1) * len(acts)]
                legal = np.array([decode(a)[0] in hand for a in acts])
                assert abs(row.sum() - 1) < 1e-9 and np.all(row[~legal] == 0) and np.all(row >= 0)
        # Real continuations inside the window: the table row of the true hand is the network on the real position.
        for _ in range(10):
            r = g.copy()
            path = []
            for _ in range(int(rng.integers(4))):
                legal = r.legal_actions()
                path.append(int(legal[rng.integers(len(legal))]))
                r.play_action(path[-1])
            n = tree.find(path)
            if n is None:
                continue
            obs, legal, _ = encode_moves(start, hands0, decks0, actions + path)
            want = model.probs(obs[None], legal[None])[0]
            got = w.policy(path, table)[hand_rank(r.hand(r.to_move))]
            assert np.allclose(got, want, rtol=1e-5, atol=1e-7)
            checked += 1
    assert checked > 20


def test_nn_table_on_a_ply10_window_takes_seconds(ck):
    import time
    deal, actions = golden_prefix(load()[5], 10)
    tree = WindowTree(deal, actions)
    model = policy.NetModel(str(ck), "cpu")
    t = time.time()
    model.window_table(tree, 0)
    model.window_table(tree, 1)
    assert time.time() - t < 30


def test_rust_agent_tables(games):
    deal, actions = golden_prefix(games[2], 12)
    tree = WindowTree(deal, actions)
    w = Window(belief.replay(deal, actions))
    a = tree.agent_table("pimc:w=2,d=2", 0, samples=2, seed=5, threads=1) + tree.agent_table(
        "pimc:w=2,d=2", 1, samples=2, seed=5, threads=1)
    b = tree.agent_table("pimc:w=2,d=2", 0, samples=2, seed=5, threads=3) + tree.agent_table(
        "pimc:w=2,d=2", 1, samples=2, seed=5, threads=3)
    assert np.array_equal(a, b)
    for hist, mover, offset, hands, acts in w.nodes():
        for i, hand in enumerate(hands):
            row = a[offset + i * len(acts): offset + (i + 1) * len(acts)]
            assert abs(row.sum() - 1) < 1e-12 and set(np.unique(row)) <= {0, 0.5, 1}
            assert all(decode(x)[0] in hand for x, p in zip(acts, row) if p > 0)
    assert 0 <= w.exploitability(a, a)["exploitability"] <= 0.5
    with pytest.raises(ValueError):
        tree.agent_table("cheat:d=2", 0)


def test_vecenv_views_hold_only_the_movers_information():
    env = VecEnv(8, seed=0, threads=2)
    env.reset_arena(9, np.repeat(np.arange(4), 2), np.tile([0, 1], 4))
    twin = VecEnv(8, seed=0, threads=2)   # asks for a network's draws where env asks for views
    twin.reset_arena(9, np.repeat(np.arange(4), 2), np.tile([0, 1], 4))
    rng = np.random.default_rng(0)
    differs = 0
    for _ in range(12):
        idx = np.array([7, 2, 4, 1])
        games, hists, seeds, uniforms = env.views(idx)
        assert seeds.dtype == np.uint64 and len(set(seeds.tolist())) == 4
        assert uniforms.dtype == np.float64 and np.array_equal(uniforms, twin.uniforms(idx))
        for j, g, h in zip(idx, games, hists):
            assert h == [a for _, a in env.history(j)]
            true = core.arena_deal(9, j // 2)
            for a in h:
                true.play_action(a)
            p = true.to_move
            assert g.to_move == p and g.hand(p) == true.hand(p) and g.queue() == true.queue()
            assert sorted(g.deck(p)) == sorted(true.deck(p)) and g.score() == true.score()
            assert sorted(g.hand(1 - p) + g.deck(1 - p)) == sorted(true.hand(1 - p) + true.deck(1 - p))
            differs += g.hand(1 - p) != true.hand(1 - p)
        _, mask, _, _ = env.observe()
        acts = np.array([rng.choice(np.flatnonzero(mask[j])) for j in range(8)])
        env.step(acts)
        twin.step(acts)
    assert differs > 5
    with pytest.raises(ValueError):
        env.views([1, 1])


class Recorder:
    """A Python agent that records what it sees and plays a random legal move with its private RNG."""
    name = "recorder"
    checkpoints = {}
    needs_obs = True

    def __init__(self):
        self.seen = []

    def act(self, batch):
        out = []
        for g, h, s, o, l in zip(batch.games, batch.histories, batch.seeds, batch.obs, batch.legal):
            self.seen.append((g, list(h)))
            legal = g.legal_actions()
            assert sorted(legal) == list(np.flatnonzero(l))
            out.append(legal[np.random.default_rng(int(s)).integers(len(legal))])
        return out


def test_python_agents_play_matches_reproducibly_with_their_own_information():
    runs = []
    for batch, threads in [(6, 1), (4, 3)]:
        player = match.PyPlayer(Recorder)
        _, raw, _ = match.run(player, "greedy", 6, seed=13, batch=batch, threads=threads)
        runs.append(raw)
        assert sorted(player.instances) == [0, 1]   # A in seat 0, A in seat 1
        for stream, rec in player.instances.items():
            assert rec.seen and all(g.to_move == stream for g, _ in rec.seen)   # one seat per instance
            for g, h in rec.seen:
                assert g.start ^ (len(h) % 2) == g.to_move
    assert runs[0]["moves"] == runs[1]["moves"]

    class Cheater(Recorder):
        def act(self, batch):
            return [99] * len(batch.games)

    with pytest.raises(ValueError):
        match.run(match.PyPlayer(Cheater), "random", 2, seed=1)
    with pytest.raises(TypeError):
        match.PyPlayer(Recorder())   # an instance would play every stream: a factory is required


class Peeker:
    """A stateful agent that tries to read its opponent's hand from the other game of the duplicate pair: at ply 0
    (the games where it starts, one per deal, in deal order) it remembers its hand; at ply 1 (the twin games of the
    same deals, same order, if it plays them too) it predicts the starter's 3 remaining cards as that hand minus the
    animal just played."""
    name = "peeker"
    checkpoints = {}
    needs_obs = False

    def __init__(self):
        self.first, self.preds = None, []

    def act(self, batch):
        plies = {len(h) for h in batch.histories}
        if plies == {0}:
            self.first = [set(g.hand(g.to_move)) for g in batch.games]
        elif plies == {1} and self.first is not None:
            self.preds += [own0 - {decode(h[0])[0]} for own0, h in zip(self.first, batch.histories)]
        return [g.legal_actions()[np.random.default_rng(int(s)).integers(len(g.legal_actions()))]
                for g, s in zip(batch.games, batch.seeds)]


def peek_hits(seed, raw, preds, games):
    """How many predictions (in the order of `games`) name 3 cards all in the opponent's true hand at ply 1."""
    hits = 0
    for pred, e in zip(preds, games):
        g = arena_deal(seed, raw["deal"][e])
        g.play_action(raw["moves"][e][0])
        hits += pred <= set(g.hand(1 - raw["a_seat"][e]))
    return hits


def test_a_stateful_python_agent_cannot_read_the_twin_game():
    seed, deals = 123, 50
    second = lambda e: raw["a_seat"][e] != raw["start"][e]   # noqa: E731   A moves second in game e

    # The hole: one instance seeing every game (as before streams) reads all 50 opponent hands.
    class OneInstance(match.PyPlayer):
        def act(self, env, idx, obs, mask, streams=None):
            return super().act(env, idx, obs, mask)

    shared = OneInstance(Peeker)
    _, raw, _ = match.run(shared, "random", deals, seed=seed, batch=deals)
    preds = shared.instance(0).preds
    assert len(preds) == deals and peek_hits(seed, raw, preds, [e for e in range(2 * deals) if second(e)]) == deals

    # PyPlayer: one instance per seat assignment, so the "twin" it remembers is another deal: chance level
    # (all 3 named cards in a random 4-card hand of the 11 left: ~0.024 per guess).
    player = match.PyPlayer(Peeker)
    _, raw, _ = match.run(player, "random", deals, seed=seed, batch=deals)
    hits = n = 0
    for stream, agent in player.instances.items():
        games = [e for e in range(2 * deals) if e % 2 == stream and second(e)]
        hits += peek_hits(seed, raw, agent.preds, games)
        n += len(agent.preds)
    assert n >= deals - 2 and hits <= 6

    # The same in self-play (bbar.delta): one instance per seat, each moving for its seat only.
    rec = match.PyPlayer(Recorder)
    delta.self_play(rec, 6, 8, seed=3, batch=6)
    assert sorted(rec.instances) == [0, 1]
    assert all(g.to_move == seat for seat, r in rec.instances.items() for g, _ in r.seen)


def test_tail_spec_parsing(ck):
    path, opts = tail.parse_spec(f"nn:{ck},F=11,mode=plain,target=1e-5")
    assert path == str(ck) and opts["F"] == 11 and opts["mode"] == "plain" and opts["target"] == 1e-5
    for bad in ["random", f"nn:{ck},F=x", f"nn:{ck},mode=greedy", f"nn:{ck},F=1,F=2", f"nn:{ck},foo=1"]:
        with pytest.raises(ValueError):
            tail.parse_spec(bad)
    assert match.parse_agent(f"tail:nn:{ck}")[0] == "py"


@pytest.mark.parametrize("mode", ["gadget", "plain"])
def test_tail_agent_plays_legal_reproducible_full_games(ck, mode):
    spec = f"tail:nn:{ck},F=11,mode={mode}"
    runs = []
    for batch, threads, workers in [(4, 1, 1), (3, 2, 3)]:
        result, raw, players = match.run(f"{spec},workers={workers}", f"nn:{ck}", 4, seed=7, batch=batch,
                                         threads=threads, device="cpu")
        assert players[0].name == spec + ",target=0.0001,iters=10000"
        runs.append(raw)
        assert all(len(m) >= 16 for m in raw["moves"])
        s = players[0].summary()
        assert s["instances"] == 2 and s["solves"] == s["windows"] == 8 and s["resolves"] == 0   # one per game
        assert s["decisions"]["window"] > 0 and s["decisions"]["minimax"] > 0
        assert s["decisions"]["base"] > 0 and s["gap"]["max"] <= 1e-4 + 1e-12
        assert all(x["ply"] in (11, 12) for a in players[0].instances.values() for x in a.log)
        line = match.record(result, raw, "cmd", {"commit": "x", "code": "y"}, players, "cpu")
        assert line["agent_stats"][players[0].name] == s
    assert runs[0]["moves"] == runs[1]["moves"]
    other = match.run(spec, f"nn:{ck}", 4, seed=8, device="cpu")[1]
    assert other["moves"] != runs[0]["moves"]


def test_tail_plays_its_base_network_move_for_move_before_F(soft_ck):
    # Before F the tail agent samples its network with the seat's own draws, as NetPlayer does: same games.
    F = 11
    _, nn, _ = match.run(f"nn:{soft_ck}", f"nn:{soft_ck}", 4, seed=21, batch=4, device="cpu")
    _, tl, _ = match.run(f"tail:nn:{soft_ck},F={F},mode=plain", f"nn:{soft_ck}", 4, seed=21, batch=4, device="cpu")
    assert [m[:F] for m in tl["moves"]] == [m[:F] for m in nn["moves"]]
    assert tl["moves"] != nn["moves"]
    # bbar.delta's self-play positions are paired too: the same games up to ply F.
    games_nn = delta.self_play(match.NetPlayer(str(soft_ck)), 6, F, seed=4)
    games_tl = delta.self_play(match.make_player(f"tail:nn:{soft_ck},F={F}"), 6, F, seed=4)
    assert games_tl == games_nn


def test_tail_cache_keeps_one_window_per_game_and_never_changes_play(ck):
    spec, deals = f"tail:nn:{ck},F=11,mode=gadget", 6
    _, raw, players = match.run(spec, f"nn:{ck}", deals, seed=31, batch=deals, device="cpu")
    s = players[0].summary()
    assert s["solves"] == s["windows"] == 2 * deals and s["resolves"] == 0   # each game's window solved once
    assert all(not a._cache for a in players[0].instances.values())   # dropped after their last use
    # A cache smaller than the games in play: a warning, windows evicted while in use and re-solved (counted apart
    # from the distinct windows), the same moves.
    with pytest.warns(UserWarning, match="cache=2"):
        _, raw2, players2 = match.run(spec + ",cache=2", f"nn:{ck}", deals, seed=31, batch=deals, device="cpu")
    s2 = players2[0].summary()
    assert raw2["moves"] == raw["moves"] and s2["windows"] == 2 * deals and s2["resolves"] > 0
    assert s2["solves"] == s2["windows"] + s2["resolves"]
    for key in ("internal", "entries", "iterations", "impossible_opponent_decisions"):
        assert s2[key] == s[key]   # statistics over distinct windows
    # Nothing carried between calls matters: a fresh agent at every call plays the same games.
    model = players[0].instance(0).model

    class Amnesiac:
        name, checkpoints, needs_obs = "amnesiac", {}, True

        def act(self, batch):
            return tail.TailAgent(model, F=11, mode="gadget", cache=0).act(batch)

    _, raw3, _ = match.run(match.PyPlayer(Amnesiac), f"nn:{ck}", deals, seed=31, batch=4, device="cpu")
    assert raw3["moves"] == raw["moves"]


def test_tail_name_tells_apart_every_option_that_changes_play(ck, games):
    body = f"nn:{ck}"
    name = tail.TailAgent.from_spec(body).name
    assert name == f"tail:{body},F=10,mode=gadget,target=0.0001,iters=10000"
    for speed_only in [",threads=2", ",workers=3", ",cache=7", ",eps=1e-9", ",check_every=10", ",algo=dcfr"]:
        assert tail.TailAgent.from_spec(body + speed_only).name == name
    names = {name}
    for opt in [",eps=0.01", ",eps=0.2", ",check_every=5", ",algo=cfr+", ",target=1e-5", ",target=1.0000001e-4",
                ",iters=99", ",F=12", ",mode=plain"]:
        agent = tail.TailAgent.from_spec(body + opt)
        assert agent.name not in names
        names.add(agent.name)
        # The name is a spec of the same agent.
        path, opts = tail.parse_spec(agent.name[len("tail:"):])
        again = tail.TailAgent(path, **opts)
        assert again.name == agent.name
        assert (again.F, again.mode, again.target, again.iters, again.algo, again.eps, again.check_every) == (
            agent.F, agent.mode, agent.target, agent.iters, agent.algo, agent.eps, agent.check_every)
    # eps does change play: it floors the likelihoods of the ranges the window is solved for.
    deal, actions = golden_prefix(games[4], 11)
    seat = (deal.start + 11) % 2
    a, b = (tail.TailAgent(str(ck), F=11, mode="plain", eps=e) for e in (belief.EPS, 0.2))
    pa, pb = (x.solution(deal.start, tuple(actions), seat).window.root_policy() for x in (a, b))
    assert np.abs(pa - pb).max() > 0.01


def test_tail_minimax_plays_optimal_moves():
    rng = np.random.default_rng(0)
    for i in range(30):
        G = core.arena_deal(3, i)
        for _ in range(15 + i % 3):
            G.play_action(G.legal_actions()[0])
        if G.is_over():
            continue
        assert tail.perfect_info(G)
        vals = tail.minimax_values(G)
        a = tail.minimax_choice(G, rng)
        assert vals[a] == max(vals.values())


def test_gadget_resolve_is_safe(ck, games):
    # The re-solved strategy concedes the opponent no more, per opponent hand, than the base table did.
    agent = tail.TailAgent(str(ck), F=11, mode="gadget", target=1e-6)
    deal, actions = golden_prefix(games[4], 11)
    seat = (deal.start + 11) % 2
    sol = agent.solution(deal.start, tuple(actions), seat)
    w = sol.window
    base = agent.model.window_table(sol.tree, seat)
    cfv_base = w.br_values(1 - seat, base)
    cfv_new = w.br_values(1 - seat, sol.profile)
    hands = w.hands(1 - seat)
    assert np.all(cfv_new[hands] <= cfv_base[hands] + 1e-4)
    assert sol.info["gap"] <= 1e-6 and sol.info["internal"] > 0


def test_delta_tool(ck, tmp_path):
    out = tmp_path / "d.jsonl"
    delta.main([f"nn:{ck}", "--T", "12,11", "--N", "3", "--seed", "4", "--out", str(out), "--quiet"])
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    header, rows, sums = lines[0], [x for x in lines if x["type"] == "position"], [x for x in lines
                                                                                    if x["type"] == "summary"]
    assert header["type"] == "run" and header["seed"] == 4 and str(ck) in header["checkpoints"]
    assert header["build"] == core.build_id() and header["ranges"] == "bayes"
    assert len(rows) == 6 and [s["T"] for s in sums] == [12, 11]
    model = policy.NetModel(str(ck), "cpu")
    for r in rows:
        assert 0 <= r["delta"] <= 0.5 and abs((r["loss"][0] + r["loss"][1]) / 2 - r["delta"]) < 1e-12
        assert min(r["loss"]) >= -2 * r["eq_gap"] - 1e-9
        assert all(p > 0 for p in r["true_hand_prob"])
        # Recompute from scratch.
        deal = belief.start_game(r["start"])
        acts = r["actions"]
        rs = [belief.model_range(model, deal, acts, p) for p in (0, 1)]
        w = Window(belief.replay(deal, acts), rs[0].range, rs[1].range)
        tree = WindowTree(deal, acts)
        t = model.window_table(tree, 0) + model.window_table(tree, 1)
        e = w.exploitability(t, t)
        assert abs(e["exploitability"] - r["delta"]) < 1e-9
    # Rust specs and tail agents run too (sampled ranges for specs).
    out2 = tmp_path / "d2.jsonl"
    delta.main(["pimc:w=2,d=2", "--T", "12", "--N", "2", "--seed", "4", "--samples", "2", "--out", str(out2),
                "--quiet"])
    assert json.loads(out2.read_text().splitlines()[0])["ranges"].startswith("bayes-sampled")
    out3 = tmp_path / "d3.jsonl"
    delta.main([f"tail:nn:{ck},F=11,mode=plain", "--T", "12", "--N", "2", "--seed", "4", "--out", str(out3),
                "--quiet"])
    assert len(out3.read_text().splitlines()) == 4


def test_delta_on_tail_agents_solves_each_window_once_and_reports_safety(ck, tmp_path, monkeypatch):
    solved = []
    real = tail.TailAgent._solve

    def spy(self, *key):
        solved.append(key)
        return real(self, *key)

    monkeypatch.setattr(tail.TailAgent, "_solve", spy)
    rows = {}
    for mode in ("gadget", "plain"):
        solved.clear()
        out = tmp_path / f"{mode}.jsonl"
        delta.main([f"tail:nn:{ck},F=11,mode={mode}", "--T", "12,11,10", "--N", "3", "--batch", "2", "--seed", "6",
                    "--out", str(out), "--quiet"])
        # Self-play windows are reused by the evaluation, every T of a game before the next game: no re-solve.
        assert solved and len(solved) == len(set(solved))
        lines = [json.loads(x) for x in out.read_text().splitlines()]
        rows[mode] = [x for x in lines if x["type"] == "position"]
        assert len(rows[mode]) == 9 and sorted({r["T"] for r in rows[mode]}) == [10, 11, 12]
        sums = [x for x in lines if x["type"] == "summary"]
        assert [x["T"] for x in sums] == [12, 11, 10] and all("safety" in x for x in sums)
        for r in rows[mode]:
            assert len(r["safety"]) == 2 and len(r["safety_weighted"]) == 2
    # Gadget re-solving is safe per opponent hand (up to its gap); plain DCFR has no such guarantee.
    assert max(max(r["safety"]) for r in rows["gadget"]) <= 2e-3
    # A margin recomputed from scratch: on the tail's own window for the seat moving at ply 11.
    r = next(r for r in rows["gadget"] if r["T"] == 12)
    deal = belief.start_game(r["start"])
    agent = tail.TailAgent(str(ck), F=11, mode="gadget")
    for p in (0, 1):
        w, tree, table = agent.solved_root(deal, r["actions"], p)
        q = 1 - p
        gain = w.br_values(q, table) - w.br_values(q, agent.model.window_table(tree, p))
        assert abs(gain[w.hands(q)].max() - r["safety"][p]) < 1e-9
    # Networks are their own base: no safety field.
    out = tmp_path / "nn.jsonl"
    delta.main([f"nn:{ck}", "--T", "12", "--N", "2", "--seed", "6", "--out", str(out), "--quiet"])
    assert all("safety" not in json.loads(x) for x in out.read_text().splitlines())
