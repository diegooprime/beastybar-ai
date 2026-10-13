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
- Batched solving (dl.solve_many through bbar._core.WindowBatch; float32 beliefs and values of the feasible hands
  with NetValues.session): every window follows exactly the trajectory of the per-window reference loop (ext_begin /
  ext_end, float64 arrays concatenated around the value function), with a network (any chunking) and with a float64
  value function (slot layout), plain and gadget windows, windows without value leaves, on any number of threads;
  WindowBatch's layouts are ext_begin's arrays rearranged, NetValues.session gives __call__'s values, and bad input
  is refused before any window changes; its arrays are real numpy arrays' memory, never an address an object claims.
- Consistency: leaf values taken from the full window reproduce its value exactly; embed round-trips profiles.
- The label generator, the PBS encoding and the network's leaf mapping agree with the solver's conventions.
- Pipeline bookkeeping: a generator rerun with --force replaces its seed's run whole and logs its own arguments; the
  trainer reports best.pt on a test split disjoint from the selection split; E2's cost columns count deployable work
  only, and its summary goes to the evidence trail (refused from uncommitted code unless --allow-dirty).
"""
import json
from pathlib import Path
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


def dl_jobs(games, seed=0):
    """Fresh depth-limited jobs of several sizes: plain windows, a gadget re-solve and a window without value
    leaves (ply 13, leaves at 15: all frontier)."""
    rng = np.random.default_rng(seed)
    jobs = []
    for gi, ply, leaf_ply in ((1, 10, 12), (2, 11, 12), (4, 12, 14), (3, 13, 15), (6, 11, 13)):
        G = window_check.public_game(games[gi], ply)
        jobs.append(D.DLJob(Window(G, *ranges_for(G, gi), leaf_ply=leaf_ply)))
    G = window_check.public_game(games[5], 11)
    w = Window(G, *ranges_for(G, 5), leaf_ply=13)
    cfvs = np.zeros(495)
    cfvs[w.hands(1)] = rng.uniform(0.3, 0.7, len(w.hands(1)))
    prior = np.zeros(495)
    prior[w.hands(1)] = rng.uniform(0.5, 1.5, len(w.hands(1)))
    jobs.append(D.DLJob(w, gadget=(0, cfvs, prior)))
    assert jobs[3].leaves == 0 and all(j.leaves > 0 for j in jobs if j is not jobs[3])
    return jobs


def solve_state(job):
    w = job.w
    avg = w.leaf_values_avg()
    out = [w.iterations, w.profile_array(), avg, *w.leaf_query()]
    if job.gadget is not None:
        out += list(w.gadget_stats(avg).values())
    return out


def synthetic_values(pub, movers, beliefs):
    """A float64 value function of everything it is given (absolute players)."""
    v = 0.3 + 20 * beliefs[:, ::-1] + 0.05 * movers[:, None, None] + 0.01 * pub[:, :1, None]
    return np.clip(v, 0, 1)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + (
    ["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("kind", ["net", "net-chunk", "synthetic", "exact"] + [f"net-{d}" for d in DEVICES[1:]])
def test_batched_solves_follow_the_reference_trajectories(games, kind):
    torch.manual_seed(0)
    dev = kind[4:] if kind[4:] in DEVICES else "cpu"
    net = ValueNet(width=32, blocks=1).eval().to(dev)
    iters = 12
    values = {"net": NetValues(net), "net-chunk": NetValues(net, chunk=16), "synthetic": synthetic_values,
              "exact": D.ExactValues(target=1e-5, warm=False)}.get(kind) or NetValues(net, dev, chunk=64)
    ref = dl_jobs(games)
    if kind == "exact":     # (slow: the smallest windows)
        ref = [ref[1], ref[3], ref[4]]
        iters = 4
    st_ref = D.solve_many(ref, values, iters, fast=False)
    want = [solve_state(j) for j in ref]
    for threads in (1, 3):
        jobs = dl_jobs(games)
        if kind == "exact":
            jobs = [jobs[1], jobs[3], jobs[4]]
        st = D.solve_many(jobs, values, iters, threads=threads)
        assert st["calls"] == st_ref["calls"] == 2 * iters and st["leaves"] == st_ref["leaves"]
        for job, w0 in zip(jobs, want):
            got = solve_state(job)
            assert len(got) == len(w0)
            for a, b in zip(got, w0):
                assert np.array_equal(a, b), (kind, threads)


def test_window_batch_layouts_and_session(games):
    a, b, c = dl_jobs(games), dl_jobs(games), dl_jobs(games)
    for js in (a, b, c):
        D._ext_start(js, "dcfr")
    want = [j.w.ext_begin()[2] for j in a]               # per window: float64 (leaves, 2, 495), absolute players
    full = np.concatenate(want)
    compact = core.WindowBatch([j.w for j in b], threads=2)
    slots_ = core.WindowBatch([j.w for j in c], layout="slots", relative=False)
    n = compact.num_leaves
    assert n == slots_.num_leaves == len(full) and compact.rows()[-1] == n and slots_.width == 495
    movers = compact.movers()
    assert np.array_equal(movers, np.concatenate([j.movers for j in a]))
    buf = np.full((n, 2, compact.width), np.nan, dtype=np.float32)
    compact.begin(buf)
    out64 = np.full((n, 2, 495), np.nan)
    slots_.begin(out64)
    assert np.array_equal(out64, full)                  # slots layout: ext_begin's arrays, concatenated
    idx, counts = compact.slots()
    assert compact.width == counts.max() and idx.shape == buf.shape
    for row in range(n):
        for r in range(2):
            p = r ^ int(movers[row])                    # mover first
            k = counts[row, r]
            feas = np.flatnonzero(full[row, p] > 0) if (full[row, p] > 0).sum() == k else None
            assert list(idx[row, r, :k]) == sorted(idx[row, r, :k])
            assert np.array_equal(buf[row, r, :k], full[row, p, idx[row, r, :k]].astype(np.float32))
            assert np.all(buf[row, r, k:] == 0) and np.all(full[row, p, idx[row, r, k:]] == 0)
            if feas is not None:
                assert np.array_equal(idx[row, r, :k], feas)
    # The session's values are __call__'s, at the same slots (mover first), bit for bit.
    torch.manual_seed(1)
    nv = NetValues(ValueNet(width=32, blocks=1).eval(), chunk=8)
    pub = np.concatenate([j.pub for j in a])
    got = nv.session(pub, idx)(buf)
    ref = nv(pub, movers, full)
    for r in range(2):
        p = r ^ movers
        assert np.array_equal(got[:, r], np.take_along_axis(ref[np.arange(n), p], idx[:, r], 1).astype(np.float32))


def test_window_batch_errors(games):
    jobs = dl_jobs(games)
    ws = [j.w for j in jobs]
    with pytest.raises(ValueError):
        core.WindowBatch([ws[0], ws[1], ws[0]])         # a window twice
    with pytest.raises(ValueError):
        core.WindowBatch(ws, layout="dense")
    batch = core.WindowBatch(ws)
    n, wd = batch.num_leaves, batch.width
    buf = np.zeros((n, 2, wd), dtype=np.float32)
    with pytest.raises(ValueError):
        batch.begin(buf)                                # no ext_start
    D._ext_start(jobs, "dcfr")
    for bad in (np.zeros((n, 2, wd), dtype=np.int64), np.zeros((n, 2, wd + 1), dtype=np.float32),
                np.zeros((n, 2, 2 * wd), dtype=np.float32)[:, :, ::2]):
        with pytest.raises(ValueError):
            batch.begin(bad)
    ro = np.zeros((n, 2, wd), dtype=np.float32)
    ro.flags.writeable = False
    with pytest.raises(ValueError):
        batch.begin(ro)
    with pytest.raises(ValueError):
        batch.end(buf)                                  # end without begin
    batch.begin(buf)
    with pytest.raises(ValueError):
        batch.begin(buf)                                # begin twice
    with pytest.raises(ValueError):
        ws[0].ext_begin()                               # (the window's own calls see the pending half-iteration)
    with pytest.raises(ValueError):
        batch.end(buf, buf)                             # values and out overlap
    _, counts = batch.slots()
    vals = np.full((n, 2, wd), 0.5, dtype=np.float32)
    row = int(np.flatnonzero(counts[:, 1] < wd)[0]) if (counts[:, 1] < wd).any() else None
    if row is not None:
        vals[row, 1, -1] = np.nan                       # padding is never read
    last = n - 1
    vals[last, 0, 0] = np.inf
    with pytest.raises(ValueError, match="values"):
        batch.end(vals)
    assert [w.iterations for w in ws] == [1] * len(ws)  # nothing changed: the half-iteration can still end
    vals[last, 0, 0] = 0.5
    out = np.zeros_like(buf)
    batch.end(vals, out)
    batch.end(vals.astype(np.float64))                  # float64 values work too
    assert batch.stats()["ends"] == 2 and batch.stats()["begins"] == 2
    assert [w.iterations for w in ws] == [1] * len(ws)
    st = jobs[0].w.ext_begin()                          # the windows carry on alone
    jobs[0].w.ext_end(np.full(st[2].shape, 0.5))
    assert jobs[0].w.iterations == 2


def fake_arrays(decoy):
    """(objects that describe `decoy`'s memory without being a numpy array: one with an __array_interface__, one that
    also claims __class__ numpy.ndarray so that isinstance passes, a memoryview, a list; LyingArray, a real ndarray
    subclass whose __array_interface__ describes `decoy`)."""
    class Claims:
        __array_interface__ = decoy.__array_interface__

    class Spoof(Claims):
        __class__ = np.ndarray

    class LyingArray(np.ndarray):
        @property
        def __array_interface__(self):
            return decoy.__array_interface__
    assert isinstance(Spoof(), np.ndarray)
    return [Claims(), Spoof(), memoryview(decoy), decoy.tolist()], LyingArray


def test_window_batch_takes_only_numpy_arrays(games):
    """begin / end take the memory of real numpy arrays only, as numpy reports it (its own __array_interface__
    descriptor): an object with a made-up __array_interface__ once made begin write to that address with the GIL
    released (SIGSEGV at 0x10000). Here the fakes describe a real decoy array, so a regression fails the test instead
    of crashing it: anything that is not an ndarray is refused before a window changes, and an ndarray subclass
    lying about its interface is written and read at its own memory."""
    ref, jobs = dl_jobs(games), dl_jobs(games)
    for js in (ref, jobs):
        D._ext_start(js, "dcfr")
    rb, batch = core.WindowBatch([j.w for j in ref]), core.WindowBatch([j.w for j in jobs])
    shape = (batch.num_leaves, 2, batch.width)
    decoy = np.full(shape, np.nan, dtype=np.float32)
    fakes, LyingArray = fake_arrays(decoy)
    for bad in fakes:
        with pytest.raises(ValueError, match="not a numpy array"):
            batch.begin(bad)
    assert np.isnan(decoy).all()
    want = np.zeros(shape, dtype=np.float32)
    rb.begin(want)
    lie = np.full(shape, np.nan, dtype=np.float32).view(LyingArray)
    batch.begin(lie)
    assert np.isnan(decoy).all() and np.array_equal(np.asarray(lie), want)
    vals = np.full(shape, 0.5, dtype=np.float32)
    its = [j.w.iterations for j in jobs]
    for bad in fakes:
        with pytest.raises(ValueError, match="not a numpy array"):
            batch.end(bad)
        with pytest.raises(ValueError, match="not a numpy array"):
            batch.end(vals, bad)
    assert [j.w.iterations for j in jobs] == its and np.isnan(decoy).all()
    rb.end(vals, want)
    batch.end(vals.view(LyingArray), lie)               # values read from vals (the decoy is all NaN), out to lie
    assert np.isnan(decoy).all() and np.array_equal(np.asarray(lie), want)


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
            for kind in pbs.SYNTHETIC_KINDS:
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


def test_bayes_ranges_are_the_model_posterior(tmp_path):
    """--bayes labels start with both players' exact Bayes ranges under the model ("bayes"), or those mixed with a
    synthetic range ("bayes-mix"); either way only hands the player can hold carry mass."""
    from bbar.belief import model_range, start_game
    from bbar.policy import NetModel
    ck = "checkpoints/bp-v1/step_8000.pt"
    if not Path(ck).is_file():
        pytest.skip("no blueprint checkpoint")
    model = NetModel(ck, "cpu")
    s = StateSampler(7, mix={"random": 1.0}, envs=3)
    rng = np.random.default_rng(0)
    for x in s.batch((12,)):
        pair = gen.bayes_pair(model, x, rng)
        _, _, u, _, _ = pbs.decode_keys([core.public_key(x.game)])
        for p, (r, kind) in enumerate(pair):
            assert kind in ("bayes", "bayes-mix") and abs(r.sum() - 1) < 1e-12
            outside = np.ones(len(r), bool)
            outside[pbs.feasible_slots(int(u[0, p]))] = False
            assert np.all(r[outside] == 0)
            if kind == "bayes":
                assert np.allclose(r, model_range(model, start_game(x.start), x.actions, p).range)
    summary = gen.main(["--out", str(tmp_path), "--labels", "6", "--ranges-per-state", "1", "--bayes", "1",
                        "--bayes-model", ck, *GEN_ARGS])
    from bbar.value.train import load_labels
    d, _ = load_labels([str(tmp_path)])
    kinds = {pbs.RANGE_KINDS[k] for k in d["kinds"].ravel()}
    assert summary["labels"] >= 6 and kinds & {"bayes", "bayes-mix"}




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
    # A PBS version-1 network (E2 reads both versions; tests/test_options.py): the untrained network whose depth-limited
    # trunk is less exploitable than uniform play on these positions (a random network need not be).
    v = ValueNet(width=16, blocks=1, pub_dim=pbs.PUB_DIMS[1])
    torch.save(dict(model=v.state_dict(), hparams=v.hparams, pbs_version=1), net)
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


def test_hand_net_is_zero_sum_and_round_trips(tmp_path):
    import torch
    from bbar.value.net import load_value_net, make_value_net
    torch.manual_seed(0)
    net = make_value_net("hand", width=32, blocks=1, hand_width=16)
    pub = torch.rand(4, pbs.PUB_DIM)
    r = torch.rand(4, 2, pbs.SLOTS)
    r = r / r.sum(-1, keepdim=True)
    v = net(pub, r)
    assert v.shape == (4, 2, pbs.SLOTS) and torch.allclose((r * v).sum((1, 2)), torch.ones(4), atol=1e-5)
    path = tmp_path / "hand.pt"
    torch.save(dict(model=net.state_dict(), hparams=net.hparams, pbs_version=pbs.PBS_VERSION), path)
    back, _ = load_value_net(path)
    assert torch.allclose(back(pub, r), v.detach())


def test_harvest_labels_the_queried_beliefs(tmp_path):
    """bbar.value.harvest: labels at the leaf ply, kind dl-query, beliefs normalised, solved to the target."""
    import torch
    from bbar.value import harvest
    from bbar.value.net import make_value_net
    torch.manual_seed(0)
    net = make_value_net("hand", width=16, blocks=1, hand_width=8)
    path = tmp_path / "v.pt"
    torch.save(dict(model=net.state_dict(), hparams=net.hparams, pbs_version=pbs.PBS_VERSION), path)
    summary = harvest.main(["--net", str(path), "--out", str(tmp_path / "q"), "--labels", "4", "--workers", "1",
                            "--root-plies", "12", "--iters", "20", "--leaves-per-state", "4", "--bayes", "0",
                            "--mix", "random:1", "--blueprint", ""])
    from bbar.value.train import load_labels
    d, _ = load_labels([str(tmp_path / "q")])
    assert summary["labels"] >= 4 and np.all(d["ply"] == 13)
    assert np.all(d["kinds"] == pbs.RANGE_KINDS.index("dl-query")) and np.all(d["exploitability"] <= 1e-5)
    assert np.allclose(d["ranges"].sum(-1), 1, atol=1e-4) and np.all(d["values"][~d["mask"]] == 0)
