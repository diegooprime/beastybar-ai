"""Depth-limited solving with external leaf values and the PBS value-network pipeline (bbar/value/).

- (a) Oracle check: with leaf values from exact solves of each leaf's own window (the beliefs of the current iterate
  as its root ranges; per-hand best-response values against the other's solution), the depth-limited solve's average
  trunk, completed below every leaf by both players' safe re-solves (gadget with the averaged opponent values and the
  average belief), has small exact exploitability in the full window. Observed (DCFR, 100 iterations, leaf solves to
  1e-7, gadgets to 1e-8): 1.4e-6 (golden game 5, ply 11, leaves at 12) and 1.4e-5 (game 3, ply 11, leaves at 13);
  200 iterations: 1.9e-7 and 5.8e-6. Where it comes from:
  - the trunk, mostly (CFR-D convergence: the leaf values react to the trunk strategy through the beliefs, so the
    average converges more slowly than ordinary CFR, about 1/sqrt(T) on hard positions, e.g. 2.2e-4 -> 1.8e-4 ->
    1.1e-4 from 100 to 200 to 400 iterations on game 1 ply 11 with leaves at 12);
  - the safe re-solves below the leaves: they must respect the averaged opponent values, which are only upper bounds
    of the opponent's best-response values (a best response to a mixture gains at most the mixture of best
    responses), so the gadgets' targets are loose. Against the same trunk with the full window's equilibrium below
    the leaves, the composite is 0.96x as exploitable on game 5, 1.18x on game 3 and, on the slow game 1, 1.44x /
    1.65x / 1.43x at 100 / 200 / 400 iterations (1.5e-4, 1.1e-4, 7.6e-5 for trunk + equilibrium): on slow positions
    roughly a third of the composite's exploitability; it shrinks with the trunk's (excess 6.9e-5 -> 3.3e-5 from 100
    to 400 iterations) but stays about the same share;
  - not the numerical errors of the leaf solves and gadgets: game 1's composite is 2.25e-4, 2.23e-4, 2.23e-4 with
    oracle leaf solves to 1e-5, 1e-7, 1e-9, and the gadgets reach ~1e-8.
  Tolerances: ~3x the composite observed with these settings (gadgets to 1e-7). Neither they nor the safety bound
  pin how the opponent values are averaged (uniform weights, or values at the final average beliefs, also pass on
  one position or the other): the callback's arguments do, recomputed exactly (also test_leaf_values_avg_weights).
- (b) A leaf ply where every leaf is a frontier or terminal node: the depth-limited solve is the ordinary solve, bit
  for bit.
- Consistency: leaf values taken from the full window reproduce its value exactly; embed round-trips profiles.
- The label generator, the PBS encoding and the network's leaf mapping agree with the solver's conventions.
- Pipeline bookkeeping: a generator rerun with --force replaces its seed's run whole and logs its own arguments; the
  trainer reports best.pt on a test split disjoint from the selection split; E2's cost columns count deployable work
  only, and its summary goes to the evidence trail (refused from uncommitted code unless --allow-dirty).
"""
import json
import shlex

import numpy as np
import pytest

core = pytest.importorskip("bbar._core")
torch = pytest.importorskip("torch")
from bbar import window_check  # noqa: E402
from bbar.golden import load  # noqa: E402
from bbar.value import dl as D  # noqa: E402
from bbar.value import gen, pbs  # noqa: E402
from bbar.value.net import NetValues, ValueNet  # noqa: E402
from bbar.value.sample import StateSampler, public_game  # noqa: E402

Window = core.Window


@pytest.fixture(scope="module")
def games():
    return load()


def ranges_for(G, seed, zeros=0.3):
    w0 = Window(G)
    rng = np.random.default_rng(seed)
    out = []
    for p in range(2):
        r = np.zeros(495)
        h = w0.hands(p)
        r[h] = rng.dirichlet(np.ones(len(h)) * 0.5) * (rng.random(len(h)) > zeros)
        if r.sum() == 0:
            r[h[0]] = 1
        out.append(r)
    return out


def test_no_value_leaves_is_the_ordinary_solve(games):
    for gi, ply, leaf_ply in ((3, 12, 15), (4, 13, 15), (6, 12, 16)):
        G = window_check.public_game(games[gi], ply)
        r = ranges_for(G, gi)
        w = Window(G, *r)
        w.solve(target=0, max_iters=40, check_every=40)
        d = Window(G, *r, leaf_ply=leaf_ply)
        assert d.num_value_leaves == 0 and d.sizes()["value_leaves"] == 0 and d.leaf_ply == leaf_ply
        calls = []
        out = d.solve_external(lambda p, t, b, m: calls.append((p, t, b.shape, m.shape)) or np.zeros((0, 2, 495)),
                               iters=40)
        assert out["iterations"] == 40 and len(calls) == 80 and calls[:2] == [(0, 1, (0, 2, 495), (0, 2)),
                                                                              (1, 1, (0, 2, 495), (0, 2))]
        assert np.array_equal(w.profile_array(), d.profile_array())
        assert w.exploitability() == d.exploitability()


def test_leaf_values_from_the_full_window_reproduce_its_value(games):
    G = window_check.public_game(games[2], 10)
    r = ranges_for(G, 2)
    full = Window(G, *r, threads=2)
    full.solve(target=1e-6)
    for prof in (full.profile_array(), full.profile_array("uniform")):
        dl = Window(G, *r, leaf_ply=12)
        assert dl.ply == 10 and dl.num_value_leaves > 10
        trunk, n = dl.embed(dl.profile_array("uniform"), full, prof)
        assert n == dl.num_entries
        bel, mass = dl.leaf_query(trunk)
        assert bel.shape == (dl.num_value_leaves, 2, 495) and mass.shape == (dl.num_value_leaves, 2)
        assert np.allclose(bel.sum(-1), 1) and np.all(bel * (1 - dl.leaf_feasible()) == 0)
        vals = np.zeros_like(bel)
        parts = []
        for j, (hist, mover) in enumerate(dl.value_leaves()):
            assert dl.leaf_game(j).to_move == mover
            w = Window(dl.leaf_game(j), bel[j, 0], bel[j, 1])
            sub, m = w.embed(w.profile_array("uniform"), full, prof, [], hist)
            assert m == w.num_entries
            vals[j] = [w.cfvs(0, sub), w.cfvs(1, sub)]
            parts.append((w, sub, hist))
        e = dl.exploitability(trunk, trunk, leaf_values=vals)
        assert abs(e["value"] - full.exploitability(prof, prof)["value"]) < 1e-12
        back, k = full.embed_all(full.embed(full.profile_array("uniform"), dl, trunk)[0], parts)
        # (profiles passed as arrays are re-normalised per row: equal up to rounding)
        assert k + n == full.num_entries and np.abs(back - prof).max() < 1e-14


def averaged_leaf_values(record, weight, shape):
    """leaf_values_avg recomputed from the callback's arguments, record = [(player, t, mass, values)]: per leaf j and
    owner o, sum_t weight(t) m_i^t[j] v^t[j, o] / sum_t weight(t) m_i^t[j] over the half-iterations of i = 1 - o;
    the last half-iteration's values where that sum of weights is 0. Also returns the sums of weights [n, 2]."""
    num, den = np.zeros(shape), np.zeros(shape[:2])
    for i, t, m, v in record:
        o = 1 - i
        x = weight(t) * m[:, i]
        num[:, o] += x[:, None] * v[:, o]
        den[:, o] += x
    pos = den > 0
    out = record[-1][3].copy()
    out[pos] = num[pos] / den[pos][:, None]
    return out, den


# Composite exploitability observed with these settings: 1.571e-6 (game 5) and 1.375e-5 (game 3); tolerance ~3x.
@pytest.mark.parametrize("gi,ply,leaf_ply,tol", [(5, 11, 12, 5e-6), (3, 11, 13, 4e-5)])
def test_oracle_depth_limited_solve_is_nearly_unexploitable(games, gi, ply, leaf_ply, tol):
    G = window_check.public_game(games[gi], ply)
    r = ranges_for(G, gi)
    full = Window(G, *r)
    dl = Window(G, *r, leaf_ply=leaf_ply)
    ws = D.leaf_windows(dl)
    oracle = D.LeafOracle(dl, target=1e-7, windows=ws)
    record = []

    def fn(player, t, b, m):
        v = oracle(player, t, b, m)
        record.append((player, t, m.copy(), v.copy()))
        return v
    res = dl.solve_external(fn, iters=100)
    assert res["iterations"] == 100 and oracle.stats["calls"] == 200
    # The opponent values for re-solving are the oracle's values weighted by t^2 (DCFR) x the re-solving player's
    # reach mass (not uniformly, not the values at the final average beliefs: both also pass the bounds below).
    want, _ = averaged_leaf_values(record, lambda t: t * t, (dl.num_value_leaves, 2, pbs.SLOTS))
    assert np.abs(dl.leaf_values_avg() - want * dl.leaf_feasible()).max() <= 1e-12
    prof, info = D.composite(full, dl, ws, gadget_target=1e-7)
    assert info["entries"] == full.num_entries and info["gadget_exploitability"] <= 1e-7
    e = full.exploitability(prof, prof)
    assert e["exploitability"] <= tol, e
    assert full.exploitability("uniform", "uniform")["exploitability"] > 0.1
    # The averaged opponent values bound the opponent's best response below each leaf (what makes re-solving safe):
    # the full-window best response of each player against the composite is at most the depth-limited one.
    opp = dl.leaf_values_avg()
    trunk_br = [dl.exploitability(None, None, leaf_values=opp)["br"][p] for p in (0, 1)]
    for p in (0, 1):
        assert e["br"][p] <= trunk_br[p] + 1e-6


@pytest.mark.parametrize("algo,weight", [("dcfr", lambda t: t * t), ("pcfr+", lambda t: t * t),
                                         ("cfr+", lambda t: t)])
def test_leaf_values_avg_weights(games, algo, weight):
    """leaf_values_avg = the values the callback returned, averaged with the weights of the average strategy (the
    algorithm's iteration weight x the other player's reach mass), with arbitrary values that change every call;
    a leaf the other player never reaches keeps the last half-iteration's values."""
    G = window_check.public_game(games[5], 11)
    mv = G.to_move
    r = ranges_for(G, 5, zeros=0.0)
    # The mover holds no hand with animal X: the leaves below the mover's playing X are never reached by the mover.
    hands = np.array(Window(G).hands(mv))
    has = pbs.SLOT_ONEHOT[hands] > 0
    x = next(a for a in range(pbs.NUM_ANIMALS) if 0 < has[:, a].sum() < len(hands))
    r[mv][hands[has[:, x]]] = 0
    dl = Window(G, *r, leaf_ply=13)
    rng = np.random.default_rng(0)
    record = []

    def fn(player, t, b, m):
        v = rng.random(b.shape)
        record.append((player, t, m.copy(), v))
        return v
    dl.solve_external(fn, iters=30, algo=algo)
    assert [(p, t) for p, t, _, _ in record] == [(i % 2, i // 2 + 1) for i in range(60)]
    want, den = averaged_leaf_values(record, weight, (dl.num_value_leaves, 2, pbs.SLOTS))
    assert (den[:, 1 - mv] == 0).any() and (den > 0).sum() > dl.num_value_leaves   # both cases are exercised
    got = dl.leaf_values_avg()
    feas = dl.leaf_feasible()
    assert np.all(got[feas == 0] == 0)
    assert np.abs(got - want * feas).max() <= 1e-12
    # Any other weighting differs (the check above has teeth).
    other, _ = averaged_leaf_values(record, lambda t: 1.0, (dl.num_value_leaves, 2, pbs.SLOTS))
    assert np.abs(got - other * feas).max() > 1e-3


def test_depth_limited_api_errors(games):
    G = window_check.public_game(games[0], 11)
    dl = Window(G, leaf_ply=12)
    with pytest.raises(ValueError):
        dl.solve()
    with pytest.raises(ValueError):
        dl.resolve(0, np.zeros(495))
    with pytest.raises(ValueError):
        dl.exploitability("uniform", "uniform")      # leaf values needed
    with pytest.raises(ValueError):
        dl.ext_begin()                               # no ext_start
    dl.ext_start()
    dl.ext_begin()
    with pytest.raises(ValueError):
        dl.ext_begin()
    with pytest.raises(ValueError):
        dl.ext_end(np.zeros((1, 2, 495)))
    dl.ext_end(np.full((dl.num_value_leaves, 2, 495), 0.5))
    with pytest.raises(ValueError):
        dl.ext_end(np.full((dl.num_value_leaves, 2, 495), 0.5))
    with pytest.raises(ValueError):
        Window(G, leaf_ply=11)
    with pytest.raises(ValueError):
        Window(G).leaf_values_avg()


def test_pbs_encoding_matches_the_solver(games):
    rng = np.random.default_rng(0)
    for gi in range(6):
        for ply in (10, 12, 13):
            G = window_check.public_game(games[gi], ply)
            key = core.public_key(G)
            mover, uu, kply = pbs.key_info([key])
            assert mover[0] == G.to_move and kply[0] == ply
            w = Window(G)
            m = pbs.masks_of_keys([key])[0]
            assert list(np.flatnonzero(m[0])) == w.hands(G.to_move)
            assert list(np.flatnonzero(m[1])) == w.hands(1 - G.to_move)
            f = pbs.public_features([key], [G.start])[0]
            assert f.shape == (pbs.PUB_DIM,) and f[108 + ply] == 1 and f[133] == (G.start == G.to_move)
            s = G.score()
            d = s[G.to_move] - s[1 - G.to_move]
            assert abs(f[82] * 20 - d) < 1e-6 and f[83 + max(-12, min(12, d)) + 12] == 1
            queue = [(a, o) for a, o in G.queue()]
            for i, (a, o) in enumerate(queue):
                slot = f[i * 14:(i + 1) * 14]
                assert slot[pbs.ANIMAL_NAMES.index(a)] == 1 and slot[12 + (o != G.to_move)] == 1
            assert f[len(queue) * 14:56].sum() == 0
            for kind in pbs.RANGE_KINDS:
                r, k = pbs.sample_range(uu[0, 0], rng, kind)
                assert k == kind and abs(r.sum() - 1) < 1e-12 and np.all(r[~m[0]] == 0) and np.all(r >= 0)


def test_labels_are_the_oracle_values(games):
    """A generator label equals the oracle's values at those ranges (mover-relative vs absolute)."""
    G = window_check.public_game(games[7], 11)
    dl = Window(G, leaf_ply=12)
    rng = np.random.default_rng(3)
    for j in range(min(3, dl.num_value_leaves)):
        g = dl.leaf_game(j)
        (lab,), _ = gen.label_state(g, 1, rng, 1e-7, 20_000)
        mv = g.to_move
        bel = dl.leaf_query("uniform")[0].copy()
        bel[j] = lab["ranges"][::1 if mv == 0 else -1]
        v = D.LeafOracle(dl, target=1e-7).values(bel, warm=False)
        assert np.abs(pbs.mover_relative(v[j], mv) - lab["values"]).max() < 1e-12
        assert lab["exploitability"] <= 1e-7
        assert abs((lab["ranges"][0] * lab["values"][0]).sum() + (lab["ranges"][1] * lab["values"][1]).sum() - 1) \
            <= 2 * lab["exploitability"] + 1e-12


class _Fake(torch.nn.Module):
    """Mover's values = mover's belief + 0.1, opp's values = 0.25 (to check the mover-relative mapping)."""

    def forward(self, pub, r):
        v = torch.empty_like(r)
        v[:, 0] = r[:, 0] + 0.1
        v[:, 1] = 0.25
        return v


def test_net_leaf_values_are_mover_relative(games):
    for ply in (10, 11):    # leaves at ply + 1: movers 1 - (root mover)
        G = window_check.public_game(games[1], ply)
        dl = Window(G, leaf_ply=ply + 1)
        bel, _ = dl.leaf_query("uniform")
        out = D.NetLeaf(dl, NetValues(_Fake()))(0, 1, bel, None)
        for j, (_, m) in enumerate(dl.value_leaves()):
            assert np.allclose(out[j, m], np.clip(bel[j, m] + 0.1, 0, 1)) and np.allclose(out[j, 1 - m], 0.25)
    # A real (untrained) network: shapes, zero-sum outputs on feasible hands.
    net = ValueNet(width=32, blocks=1)
    pub = torch.zeros(3, pbs.PUB_DIM)
    r = torch.softmax(torch.randn(3, 2, 495), -1)
    v = net(pub, r)
    assert v.shape == (3, 2, 495)
    assert torch.allclose((r * v).sum((1, 2)), torch.ones(3), atol=1e-5)


def test_sampler_and_generator_shards(tmp_path):
    s = StateSampler(5, mix={"random": 1.0, "pimc": 1.0}, envs=4)
    out = s.batch((10, 12))
    assert sorted({x.ply for x in out}) == [10, 12] and len(out) == 8
    for x in out:
        assert core.public_key(public_game(x.start, x.actions)) == core.public_key(x.game)
        assert len(x.actions) == x.ply
    summary = gen.main(["--out", str(tmp_path), "--labels", "12", "--workers", "1", "--plies", "13",
                        "--ply-weights", "1", "--blueprint", "", "--mix", "random:1", "--ranges-per-state", "3",
                        "--shard-size", "5"])
    assert summary["labels"] >= 12 and summary["by_ply"][0]["labels_per_sec_per_thread"] > 0
    from bbar.value.train import load_labels
    d, paths = load_labels([str(tmp_path)])
    assert len(paths) == 2 and len(d["key"]) == summary["labels"] == 12
    assert np.all(d["ply"] == 13) and d["pub"].shape == (len(d["key"]), pbs.PUB_DIM)
    assert np.all(d["values"][~d["mask"]] == 0) and np.allclose(d["ranges"].sum(-1), 1, atol=1e-5)


GEN_ARGS = ["--workers", "1", "--plies", "13", "--ply-weights", "1", "--blueprint", "", "--mix", "random:1"]


def test_generator_force_replaces_the_run(tmp_path):
    from bbar.value.train import load_labels
    args = ["--out", str(tmp_path), *GEN_ARGS, "--ranges-per-state", "3", "--shard-size", "3"]
    s1 = gen.main([*args, "--labels", "12"])
    assert len(list(tmp_path.glob("s0-w*.npz"))) == 4
    # The command is the generator's arguments, not the caller's command line.
    assert s1["command"] == shlex.join(["python", "-m", "bbar.value.gen", *args, "--labels", "12"])
    with pytest.raises(SystemExit):
        gen.main([*args, "--labels", "3"])                  # same seed: refused without --force
    s2 = gen.main([*args, "--labels", "3", "--force"])
    d, paths = load_labels([str(tmp_path)])
    assert s2["labels"] == len(d["key"]) == 3 and len(paths) == 1   # no stale shard of the first run left
    with np.load(paths[0]) as z:
        assert json.loads(z["meta"].item())["command"] == s2["command"] and s2["command"].endswith("--force")
    assert json.loads((tmp_path / "summary-s0.json").read_text())["labels"] == 3
    s3 = gen.main([*args, "--labels", "3", "--seed", "1"])     # another seed adds to the directory
    assert len(load_labels([str(tmp_path)])[0]["key"]) == 6 and s3["labels"] == 3


def test_trainer_reports_best_on_an_unselected_test_split(tmp_path, monkeypatch):
    from bbar.value import train
    data = tmp_path / "data"
    gen.main(["--out", str(data), *GEN_ARGS, "--labels", "240", "--ranges-per-state", "2", "--plies", "12,13",
              "--ply-weights", "1,1"])
    d, _ = train.load_labels([str(data)])
    keys = d["key"]
    val, test = train.split(keys, 200, 250)
    assert np.array_equal(val, train.split(keys, 200)[0])      # adding a test split leaves the validation split
    assert val.any() and test.any() and (~val & ~test).any() and not (val & test).any()
    for sel in (val, test):                                    # whole public states on each side
        assert not set(keys[sel].tolist()) & set(keys[~sel].tolist())
    monkeypatch.setattr(train, "ROOT", tmp_path)
    train.main(["--name", "t", "--data", str(data), "--steps", "6", "--batch", "16", "--width", "16", "--blocks",
                "1", "--warmup", "1", "--eval-every", "2", "--log-every", "2", "--device", "cpu",
                "--val-permille", "200", "--test-permille", "250"])
    lines = [json.loads(x) for x in (tmp_path / "runs/value/t/metrics.jsonl").read_text().splitlines()]
    cfg = lines[0]
    assert cfg["n_val"] == val.sum() and cfg["n_test"] == test.sum() and cfg["n_train"] == (~val & ~test).sum()
    vals = [x for x in lines if x["kind"] == "val"]
    (t,) = [x for x in lines if x["kind"] == "test"]
    best = min(vals, key=lambda x: x["all"]["rw_mae"])
    assert t["step"] == best["step"] and t["all"]["n"] == test.sum()
    from bbar.value.net import load_value_net
    net, ck = load_value_net(tmp_path / "checkpoints/value/t/best.pt")
    assert ck["step"] == best["step"] and ck["test_metrics"] == {k: v for k, v in t.items() if k not in ("kind", "step")}
    err = train.evaluate(net, d, np.flatnonzero(test), "cpu")
    assert abs(err["rw_abs"].mean() - t["all"]["rw_mae"]) < 1e-6


def test_e2_costs_and_evidence_trail(tmp_path, monkeypatch):
    from bbar import arena, match, provenance
    from bbar.value import e2
    net = tmp_path / "net.pt"
    torch.manual_seed(0)
    v = ValueNet(width=16, blocks=1)
    torch.save(dict(model=v.state_dict(), hparams=v.hparams, pbs_version=pbs.PBS_VERSION), net)
    out, log = tmp_path / "e2.jsonl", tmp_path / "results.jsonl"
    args = ["--net", str(net), "--positions", "2", "--ply", "12", "--leaf-plies", "13", "--iters", "10",
            "--oracle-positions", "1", "--mix", "random:1", "--blueprint", "", "--workers", "1", "--threads", "1",
            "--seed", "3", "--out", str(out)]
    e2.main([*args, "--log", str(log)])
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    header, rows, summary = lines[0], [x for x in lines if x["kind"] == "position"], lines[-1]
    assert len(rows) == 2 and summary["kind"] == "summary"
    for row in rows:
        r = row["L13"]
        # Deployable cost: depth-limited window + solve (+ the leaf windows the oracle solves); the rest is overhead.
        n = r["net"]
        assert n["seconds"] == pytest.approx(n["dl_build_seconds"] + n["solve_seconds"], abs=1e-9)
        assert 0 <= n["trunk_eq_exploitability"] < row["uniform"]
        if "oracle" in r:
            o = r["oracle"]
            assert o["leaf_build_seconds"] == n["leaf_build_seconds"] > 0   # built once, recorded in both rows
            assert o["seconds"] == pytest.approx(o["dl_build_seconds"] + o["solve_seconds"] + o["leaf_build_seconds"],
                                                 abs=1e-9)
    assert sum("oracle" in r["L13"] for r in rows) == 1
    m = summary["metrics"]
    assert m["L13_net_seconds"]["mean"] == pytest.approx(np.mean([r["L13"]["net"]["seconds"] for r in rows]))
    # The evidence-trail line: the summary with its intervals and the run's provenance.
    (t,) = [json.loads(x) for x in log.read_text().splitlines()]
    assert t["experiment"] == "E2" and t["positions"] == 2 and t["seed"] == 3 and t["leaf_plies"] == [13]
    assert t["command"] == shlex.join(["python", "-m", "bbar.value.e2", *args, "--log", str(log)])
    assert (t["commit"], t["code"]) == (header["revision"]["commit"], header["revision"]["code"])
    assert t["build"] == header["build"] == core.build_id() and t["net_sha256"] == header["net_sha256"] is not None
    x = t["metrics"]["L13_net_exploitability"]
    assert x["n"] == 2 and len(x["mean_ci95"]) == 2 and len(x["median_ci95"]) == 2
    assert x["mean"] == pytest.approx(m["L13_net_exploitability"]["mean"], rel=1e-3)
    assert t["metrics"]["L13_oracle_exploitability"]["n"] == 1
    # The default trail only takes committed code (as bbar.arena's), unless --allow-dirty.
    trail = tmp_path / "trail.jsonl"
    monkeypatch.setattr(arena, "LOG", trail)
    monkeypatch.setattr(match, "LOG", trail)
    monkeypatch.setattr(provenance, "revision", lambda root=None: {"commit": "0123abc-dirty", "code": "x"})
    with pytest.raises(SystemExit):
        e2.main(args)
    assert not trail.exists()
