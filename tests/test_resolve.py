"""The continual re-solving player (bbar/resolve.py) and the solver pieces it stands on.

- Depth-limited re-solving with the gadget (Window.resolve_external): with exact leaf values, the opponent's
  best-response values against the re-solved trunk stay below the carried ones (up to the gadget's gap), in the
  depth-limited game and in the full window once the leaves are re-solved too; node_values at the root are br_values.
- A depth-limited WindowTree has the depth-limited Window's layout (base-network tables on depth-limited windows).
- NetValues with fixed-size chunks: a row's values do not depend on what else is in the batch.
- The agent: legal, reproducible full games in both modes whatever the batch, threads and workers; the base network
  move for move before P0; play independent of its cache (a fresh agent per call plays the same); with P0 >= F it is
  the tail agent; its name tells apart every option that changes play.
- With exact leaf values (V=exact), every carried opponent value (from a value leaf or an internal node of the
  previous solve, and at the exact window) is nearly achievable: the re-solve's gadget violations are small, which
  checks their units and perspective.
- The safe gadget's prior keeps every opponent hand's weight up (default mix: half Bayes, half uniform): with the
  Bayes prior alone, hands it nearly rules out go unconstrained after finitely many iterations (excess 3.7e-3 against
  3e-5 in the case below; up to 0.17 in blueprint self-play), and every re-solve of the agent (depth-limited and
  exact) uses the mixed prior.
- bbar.delta on resolve agents: the table at ply T is the agent's own play (its depth-limited root strategies and exact
  windows), recomputed from scratch by a fresh agent; plain mode with exact leaves has delta_T ~ 0.
"""
import functools
import json

import numpy as np
import pytest

core = pytest.importorskip("bbar._core")
torch = pytest.importorskip("torch")

from bbar import belief, delta, match, resolve, window_check  # noqa: E402
from bbar._core import OBS_DIM, OBS_VERSION, Window, WindowTree  # noqa: E402
from bbar.actions import encode  # noqa: E402
from bbar.golden import load  # noqa: E402
from bbar.policy import NetModel  # noqa: E402
from bbar.rl.net import PolicyNet  # noqa: E402
from bbar.value import dl as D  # noqa: E402
from bbar.value import pbs  # noqa: E402
from bbar.value.net import NetValues, ValueNet  # noqa: E402


def tiny_policy(path, width=16, seed=0, sharp=30.0):
    torch.manual_seed(seed)
    net = PolicyNet(width, 1)
    with torch.no_grad():
        net.policy.weight.mul_(sharp)
    torch.save({"obs_version": OBS_VERSION, "obs_dim": OBS_DIM, "policy_hparams": net.hparams,
                "policy": net.state_dict()}, path)
    return path


def tiny_value(path, seed=0):
    torch.manual_seed(seed)
    v = ValueNet(width=16, blocks=1)
    torch.save(dict(model=v.state_dict(), hparams=v.hparams, pbs_version=pbs.PBS_VERSION), path)
    return path


@pytest.fixture(scope="module")
def nets(tmp_path_factory):
    d = tmp_path_factory.mktemp("nets")
    return str(tiny_policy(d / "pi.pt")), str(tiny_policy(d / "soft.pt", seed=1, sharp=1.0)), str(
        tiny_value(d / "v.pt"))


@pytest.fixture(scope="module")
def games():
    return load()


def spec(nets, **kw):
    pi, _, v = nets
    opts = dict(P0=11, F=13, iters=30)
    opts.update(kw)
    return f"resolve:V={v},base=nn:{pi}," + ",".join(f"{k}={x}" for k, x in opts.items())


def ranges_for(G, seed):
    w0 = Window(G)
    rng = np.random.default_rng(seed)
    out = []
    for p in range(2):
        r = np.zeros(495)
        h = w0.hands(p)
        r[h] = rng.dirichlet(np.ones(len(h)) * 0.5)
        out.append(r)
    return out


@pytest.mark.parametrize("gi,ply,leaf_ply", [(5, 11, 13), (2, 12, 13)])
def test_depth_limited_gadget_keeps_the_carried_values(games, gi, ply, leaf_ply):
    G = window_check.public_game(games[gi], ply)
    r = ranges_for(G, gi)
    full = Window(G, *r)
    full.solve(target=1e-8)
    star = full.profile_array()
    p = G.to_move      # we re-solve for the player to move at the root (as the agent does)
    o = 1 - p
    cfvs = full.br_values(o, star)
    dl = Window(G, *r, leaf_ply=leaf_ply)
    ws = D.leaf_windows(dl)
    dl.resolve_external(p, cfvs, D.LeafOracle(dl, target=1e-8, windows=ws), iters=100)
    avg = dl.leaf_values_avg()
    hands = dl.hands(o)
    after = dl.br_values(o, None, leaf_values=avg)
    g = dl.gadget_stats(avg)
    assert g["exploitability"] < 1e-4 and g["max_violation"] < 2e-4
    assert np.max(after[hands] - cfvs[hands]) == pytest.approx(g["max_violation"], abs=1e-12)
    best, follow, mass = dl.node_values(o, [], None, avg)
    assert mass == pytest.approx(1.0) and np.array_equal(best, after)
    # In the full window: our trunk and our safe re-solves below the leaves; the opponent's best response there.
    prof, info = D.composite(full, dl, ws, gadget_target=1e-8)
    assert info["entries"] == full.num_entries
    assert np.max(full.br_values(o, prof)[hands] - cfvs[hands]) < 2e-4


def test_gadget_prior_keeps_every_opponent_hand_in_play(games, nets):
    """A re-solve whose carried values are exactly achievable, with a Bayes-like prior that nearly rules out a third
    of the opponent's hands: alone it leaves those hands unconstrained after 100 iterations, the agent's mix (and
    uniform) do not."""
    G = window_check.public_game(games[3], 11)
    rng = np.random.default_rng(3)
    w0 = Window(G)
    r = [np.zeros(495), np.zeros(495)]
    for p in (0, 1):
        r[p][w0.hands(p)] = rng.dirichlet(np.full(len(w0.hands(p)), 0.5))
    full = Window(G, *r)
    full.solve(target=1e-8)
    p = G.to_move
    o = 1 - p
    cfvs = full.br_values(o, full.profile_array())
    hands = full.hands(o)
    low = rng.choice(hands, len(hands) // 3, replace=False)
    bayes = r[o].copy()
    bayes[low] = 1e-9
    bayes /= bayes.sum()
    excess = {}
    for prior in ("bayes", "mix", "uniform"):
        agent = resolve.ResolveAgent(NetModel(nets[0]), "exact", prior=prior)
        dl = Window(G, *r, leaf_ply=13)
        pr = agent._gadget_prior(bayes, dl, o)
        assert pr.sum() == pytest.approx(1) and np.all(pr[np.setdiff1d(np.arange(495), hands)] == 0)
        if prior != "bayes":
            assert pr[hands].min() >= (0.5 if prior == "mix" else 1) / len(hands) - 1e-15
        dl.resolve_external(p, cfvs, D.LeafOracle(dl, target=1e-8), iters=100, prior=pr)
        g = dl.gadget_stats(dl.leaf_values_avg())
        excess[prior] = g["max_violation"]        # over all opponent hands, unweighted
        assert g["exploitability"] < 2e-5
    assert excess["bayes"] > 1e-3 and excess["mix"] < 2e-4 and excess["uniform"] < 2e-4, excess


def test_every_safe_resolve_uses_the_mixed_prior(nets, monkeypatch):
    seen = []
    real = resolve.ResolveAgent._gadget_prior

    def spy(self, bayes, w, o):
        pr = real(self, bayes, w, o)
        seen.append((w.num_value_leaves > 0, pr[w.hands(o)].min() * len(w.hands(o))))
        return pr
    monkeypatch.setattr(resolve.ResolveAgent, "_gadget_prior", spy)
    match.run(spec(nets, mode="safe"), f"nn:{nets[0]}", 2, seed=7, batch=2, device="cpu")
    assert {dl for dl, _ in seen} == {True, False}      # depth-limited re-solves and exact windows
    assert min(x for _, x in seen) >= 0.5 - 1e-12


def test_cut_window_tree_has_the_cut_window_layout(games):
    for gi, ply, leaf_ply in [(0, 11, 13), (3, 10, 12), (5, 12, 13)]:
        deal = belief.start_game(games[gi]["start"])
        actions = [encode(*s["move"]) for s in games[gi]["steps"][:ply]]
        tree = WindowTree(deal, actions, leaf_ply=leaf_ply)
        w = Window(belief.replay(deal, actions), leaf_ply=leaf_ply)
        assert tree.num_entries == w.num_entries and tree.num_nodes == w.sizes()["internal"]
        for n, (hist, mover, offset, hands, acts) in enumerate(w.nodes()):
            node = tree.node(n)
            assert (node["path"], node["mover"], node["entry"], node["actions"]) == (hist, mover, offset, acts)
    with pytest.raises(ValueError):
        WindowTree(deal, actions, leaf_ply=len(actions))


def test_value_chunks_are_batch_independent(nets):
    from bbar.value.net import load_value_net
    net, _ = load_value_net(nets[2])
    rng = np.random.default_rng(0)
    n = 150
    pub = rng.random((n, pbs.PUB_DIM)).astype(np.float32)
    mov = rng.integers(0, 2, n)
    bel = rng.dirichlet(np.ones(495), size=(n, 2))
    nv = NetValues(net, chunk=64)
    full = nv(pub, mov, bel)
    for a, b in [(0, 1), (7, 8), (3, 70), (64, 150)]:
        assert np.array_equal(nv(pub[a:b], mov[a:b], bel[a:b]), full[a:b])
    assert np.allclose(NetValues(net)(pub, mov, bel), full, atol=1e-6)


def test_spec_parsing_and_names(nets):
    pi, _, v = nets
    body = f"V={v},base=nn:{pi}"
    opts = resolve.parse_spec(body + ",P0=7,F=11,d=3,mode=plain,iters=50")
    assert (opts["P0"], opts["F"], opts["d"], opts["mode"], opts["iters"], opts["base"]) == (7, 11, 3, "plain", 50, pi)
    for bad in [f"base=nn:{pi}", f"V={v}", f"V={v},base={pi}", body + ",mode=gadget", body + ",d=1",
                body + ",F=15", body + ",P0=x", body + ",P0=1,P0=2", body + ",foo=1", body + ",prior=belief"]:
        with pytest.raises(ValueError):
            resolve.parse_spec(bad)
    assert resolve.parse_spec(body + ",d=1,mode=plain")["d"] == 1
    assert match.parse_agent("resolve:" + body)[0] == "py"
    name = resolve.ResolveAgent.from_spec(body).name
    assert name == (f"resolve:V={v},base=nn:{pi},P0=8,F=10,d=2,iters=200,mode=safe,target=0.0001,"
                    f"exact_iters=10000")
    for speed_only in [",threads=2", ",workers=3", ",cache=7", ",algo=dcfr", ",check_every=10", ",prior=mix"]:
        assert resolve.ResolveAgent.from_spec(body + speed_only).name == name
    names = {name}
    for opt in [",P0=9", ",F=11", ",d=3", ",iters=10", ",mode=plain", ",target=1e-5", ",exact_iters=5",
                ",algo=cfr+", ",eps=0.01", ",check_every=3", ",prior=uniform", ",prior=bayes"]:
        agent = resolve.ResolveAgent.from_spec(body + opt)
        assert agent.name not in names
        names.add(agent.name)
        again = resolve.ResolveAgent.from_spec(agent.name[len("resolve:"):])
        assert again.name == agent.name
    assert set(resolve.ResolveAgent.from_spec(body).checkpoints) == {pi, v}


@pytest.mark.parametrize("mode", ["safe", "plain"])
def test_resolve_agent_plays_legal_reproducible_full_games(nets, mode):
    s = spec(nets, mode=mode)
    runs = []
    for batch, threads, workers in [(4, 1, 1), (3, 2, 2)]:
        result, raw, players = match.run(f"{s},workers={workers}", f"nn:{nets[0]}", 4, seed=7, batch=batch,
                                         threads=threads, device="cpu")
        runs.append(raw)
        assert all(len(m) >= 16 for m in raw["moves"])
        st = players[0].summary()
        assert st["instances"] == 2 and st["decisions"]["dl"] > 0 and st["decisions"]["window"] > 0
        assert st["decisions"]["base"] > 0 and st["decisions"]["minimax"] > 0
        assert st["dl"]["value_leaves"]["max"] > 0 and st["dl"]["iterations"]["max"] == 30
        assert st["exact"]["n"] == 8 and set(st["seconds_by_ply"]["exact"]) == {13}   # one exact window per game
        assert set(st["seconds_by_ply"]["dl"]) == {11, 12}
        log = [x for a in players[0].instances.values() for x in a.log]
        assert all(x["ply"] == 13 for x in log if x["kind"] == "exact")
        assert all(x["leaf_ply"] == 13 for x in log if x["kind"] in ("dl", "opp"))
        if mode == "safe":
            assert {x["cfvs"] for x in log if x["kind"] == "dl"} == {"base"}
            assert {x["cfvs"] for x in log if x["kind"] == "exact"} == {"leaf"}
        else:
            assert any(x["kind"] == "opp" for x in log)   # the opponent moved first in the phase
        line = match.record(result, raw, "cmd", {"commit": "x", "code": "y"}, players, "cpu")
        assert line["agent_stats"][players[0].name] == st
    assert runs[0]["moves"] == runs[1]["moves"]


def test_resolve_plays_its_base_network_move_for_move_before_P0(nets):
    _, soft, v = nets
    s = f"resolve:V={v},base=nn:{soft},P0=11,F=13,iters=10"
    _, nn, _ = match.run(f"nn:{soft}", f"nn:{soft}", 4, seed=21, batch=4, device="cpu")
    _, rs, _ = match.run(s, f"nn:{soft}", 4, seed=21, batch=4, device="cpu")
    assert [m[:11] for m in rs["moves"]] == [m[:11] for m in nn["moves"]]
    assert rs["moves"] != nn["moves"]


@pytest.mark.parametrize("mode", ["safe", "plain"])
def test_resolve_play_does_not_depend_on_its_cache(nets, mode):
    s = spec(nets, mode=mode, d=3, P0=10)
    _, raw, players = match.run(s, f"nn:{nets[0]}", 3, seed=31, batch=3, device="cpu")
    agent = players[0].instance(0)
    assert not agent._exact and not agent._dl      # dropped once done
    make = functools.partial(resolve.ResolveAgent, agent.model, agent.values, P0=10, F=13, d=3, iters=30, mode=mode)

    class Amnesiac:
        name, checkpoints, needs_obs = "amnesiac", {}, True

        def act(self, batch):
            return make(cache=0).act(batch)

    _, raw2, _ = match.run(match.PyPlayer(Amnesiac), f"nn:{nets[0]}", 3, seed=31, batch=2, device="cpu")
    assert raw2["moves"] == raw["moves"]


@pytest.mark.parametrize("rmode,tmode", [("safe", "gadget"), ("plain", "plain")])
def test_without_a_depth_limited_phase_it_is_the_tail_agent(nets, rmode, tmode):
    pi, _, v = nets
    a = match.run(f"resolve:V={v},base=nn:{pi},P0=12,F=11,mode={rmode}", f"nn:{pi}", 3, seed=9, device="cpu")[1]
    b = match.run(f"tail:nn:{pi},F=11,mode={tmode}", f"nn:{pi}", 3, seed=9, device="cpu")[1]
    assert a["moves"] == b["moves"]


def exact_agent(pi, **kw):
    return functools.partial(resolve.ResolveAgent, NetModel(pi), "exact", **kw)


def test_carried_values_are_achievable_with_exact_leaves(nets):
    """With exact leaf values the carried opponent values are (nearly) what the opponent can get: small violations at
    every re-solve that carries them (value leaf, internal node, exact window)."""
    pi = nets[0]
    for kw, sources in [(dict(P0=11, F=14, d=2), {"dl": {"base", "leaf"}, "exact": {"leaf"}}),
                        (dict(P0=11, F=14, d=3), {"dl": {"base", "node"}, "exact": {"leaf"}})]:
        player = match.PyPlayer(exact_agent(pi, iters=40, mode="safe", target=1e-6, **kw))
        match.run(player, f"nn:{pi}", 1, seed=3, batch=1, device="cpu")
        log = [x for a in player.instances.values() for x in a.log]
        for kind, want in sources.items():
            assert {x["cfvs"] for x in log if x["kind"] == kind} == want
        for x in log:
            if x.get("cfvs") in ("leaf", "node"):
                assert x["max_violation"] < 2e-3, x
            assert x["gap"] < 1e-3, x


def test_delta_on_resolve_agents_is_its_own_play(nets, tmp_path):
    pi, _, v = nets
    s = spec(nets, mode="safe", P0=12)
    out = tmp_path / "d.jsonl"
    delta.main([s, "--T", "12,11", "--N", "2", "--seed", "4", "--out", str(out), "--quiet"])
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    rows = [x for x in lines if x["type"] == "position"]
    assert len(rows) == 4 and all("safety" in r for r in rows)     # T = 11 < P0: base-network entries too
    # Recomputed from scratch with fresh agents (one per seat): ranges from its likelihoods, its table.
    fresh = resolve.ResolveAgent.factory(s[len("resolve:"):])
    for r in rows:
        deal = belief.start_game(r["start"])
        acts = r["actions"]
        agents = [fresh(), fresh()]
        rs = [belief.bayes_range(deal, acts, p, agents[p].likelihoods(deal, acts, p)) for p in (0, 1)]
        w = Window(belief.replay(deal, acts), rs[0].range, rs[1].range)
        tree = WindowTree(deal, acts)
        tables = [agents[p].window_table(tree, p, deal, acts) for p in (0, 1)]
        t = tables[0] + tables[1]
        assert abs(w.exploitability(t, t)["exploitability"] - r["delta"]) < 1e-9
        # The table at the root mover's node is the agent's depth-limited root strategy there.
        if len(acts) < 12:
            continue
        p = deal.start ^ (len(acts) % 2)
        pol = agents[p]._own(r["start"], tuple(acts)).policy
        node = tree.node(0)
        hands, a = tree.hands(0), node["actions"]
        assert np.allclose(t[:len(hands) * len(a)], pol[np.ix_(hands, a)].ravel())


def test_plain_resolving_with_exact_leaves_is_nearly_unexploitable_in_its_beliefs(nets, tmp_path):
    pi = nets[0]
    out = tmp_path / "d.jsonl"
    delta.main([f"resolve:V=exact,base=nn:{pi},P0=12,F=14,iters=100,mode=plain,target=1e-5", "--T", "13,12",
                "--N", "1", "--seed", "5", "--out", str(out), "--quiet"])
    rows = [json.loads(x) for x in out.read_text().splitlines() if '"position"' in x]
    assert len(rows) == 2 and max(r["delta"] for r in rows) < 2e-3
