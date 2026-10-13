"""Depth-limited solving with external leaf values, and the composite profile it implies in the full window.

    dl = Window(game, r0, r1, leaf_ply=L)                    # value leaves at ply L
    dl.solve_external(NetLeaf(dl, NetValues(net)), iters=100)   # or LeafOracle(dl): exact leaf solves
    prof, info = composite(full, dl, leaf_windows(dl))       # full = Window(game, r0, r1): the whole window
    full.exploitability(prof, prof)                           # exact exploitability of depth-limited play

Leaf value functions are called once per half-iteration with the current iterate's beliefs at every value leaf
(CFR-D; core/src/window.rs) and return both players' per-hand values (win probability of the hand's owner against
the other player's belief):
  - NetLeaf: the value network (bbar/value/net.py) on the leaves' PBS encodings.
  - LeafOracle: the exact values: each leaf's own window (frontier leaves at ply 15) solved with the query's beliefs
    as root ranges (DCFR to `target`, warm-started from the previous query's regrets), per-hand best-response values
    against the other player's solution — exactly how bbar/value/gen.py labels, so a perfect network would return
    these.

Many solves at once (solve_many): DLJob wraps a depth-limited window (and, for a re-solve, the gadget: our range, the
opponent's carried cfvs and its prior); solve_many runs the jobs' half-iterations in lockstep and values every value
leaf of every job with ONE call of the value function per half-iteration (GPU batching for bbar/resolve.py and
bbar/value/cascade.py). The solver work of all jobs is one call into Rust per half-iteration (bbar._core.WindowBatch,
core/src/python_batch.rs: `threads` threads, GIL released, leaf arrays read and written in place); with a network
(NetValues.session) only float32 beliefs and values of the feasible hands cross to the device, the leaves' public
features once per solve_many. Same trajectories, bit for bit, as the per-window loop it replaced (fast=False;
tests/test_value.py). Cost per half-iteration, 128 cascade solves at plies 8-9 (9132 value leaves), 5 threads, M4,
NetValues with a do-nothing network on MPS (everything but the forward passes), before -> after: 30 -> 7 ms (5.5 ms
of the 7 the session call: copies to and from the device, scatter, gather; the solver 1.3 ms); 4 such workers at
once: 91-129 -> 19-23 ms each (15-18 ms the session call). So 4.4-5.6x, and on MPS the session's per-call cost,
not the solver, is the floor of a half-iteration (bbar.value.cascade labels --stub-net measures it on any device).

The composite profile in the full window: plies [S, L) from the depth-limited solve's average strategy; below each
value leaf, for each player p, a safe re-solve of the leaf's subtree (core/src/window.rs gadget): p's range = p's
belief at the leaf under the average strategy, the opponent's cfvs = `dl.leaf_values_avg()` (the opponent's values
against p's average strategy there, which the gadget lets the opponent take instead of entering). Each player's
subtree strategy comes from their own re-solve. Every entry of the full window is covered.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from bbar._core import Window, WindowBatch, public_key
from bbar.value import pbs


def _map(fn, items, threads):
    if threads > 1 and len(items) > 1:
        with ThreadPoolExecutor(threads) as ex:   # the solver releases the GIL
            return list(ex.map(fn, items))
    return [fn(x) for x in items]


def leaf_windows(dl, threads=1):
    """One window per value leaf of `dl` (the leaf's public position, frontier leaves; ranges set later), built
    `threads` at a time (each window solves single-threaded: leaves are solved in parallel instead)."""
    games = [dl.leaf_game(j) for j in range(dl.num_value_leaves)]
    return _map(lambda g: Window(g, threads=1), games, threads)


class LeafOracle:
    """Exact leaf values (see the module docstring). `stats`: solves and DCFR iterations spent."""

    def __init__(self, dl, target=1e-6, max_iters=20_000, threads=1, windows=None):
        self.windows = windows if windows is not None else leaf_windows(dl, threads)
        self.target, self.max_iters, self.threads = target, max_iters, threads
        self.stats = {"solves": 0, "iterations": 0, "calls": 0}

    def values(self, beliefs, warm=True):
        # Warm starts only from this oracle's own previous query: reused windows may hold another solve's state.
        warm = warm and self.stats["calls"] > 0

        def one(j):
            w = self.windows[j]
            w.set_ranges(beliefs[j, 0], beliefs[j, 1])
            r = w.solve(target=self.target, max_iters=self.max_iters, warm=warm)
            return np.stack([w.br_values(0), w.br_values(1)]), r["iterations"]
        res = _map(one, range(len(self.windows)), self.threads)
        self.stats["solves"] += len(res)
        self.stats["iterations"] += sum(r[1] for r in res)
        self.stats["calls"] += 1
        return np.stack([r[0] for r in res]) if res else np.zeros((0, 2, pbs.SLOTS))

    def __call__(self, player, t, beliefs, mass):
        return self.values(beliefs)


def leaf_features(dl):
    """(public features float32 [n, PUB_DIM], movers int64 [n]) of the value leaves of `dl` (what NetValues takes)."""
    games = [dl.leaf_game(j) for j in range(dl.num_value_leaves)]
    if not games:
        return np.zeros((0, pbs.PUB_DIM), dtype=np.float32), np.zeros(0, dtype=np.int64)
    keys = np.array([public_key(g) for g in games], dtype=np.uint64)
    return pbs.public_features(keys, [g.start for g in games]), np.array([g.to_move for g in games], dtype=np.int64)


class DLJob:
    """One depth-limited solve for solve_many: the window `w` (built with leaf_ply; its ranges are the solve's) and,
    for a re-solve, `gadget` = (player, opp_cfvs, prior or None) as Window.ext_start takes them."""

    def __init__(self, w, gadget=None):
        self.w, self.gadget = w, gadget
        self.pub, self.movers = leaf_features(w)
        self._games = None

    @property
    def leaves(self):
        return len(self.movers)

    @property
    def games(self):
        """The value leaves' public positions (for value functions that need them: ExactValues)."""
        if self._games is None:
            self._games = [self.w.leaf_game(j) for j in range(self.w.num_value_leaves)]
        return self._games


class ExactValues:
    """The exact value function, a drop-in for a value network in solve_many and the agents (for tests and as a
    reference): each leaf's own window (frontier leaves at ply 15; built once per public position and kept) solved by
    DCFR to `target` with the query's beliefs as root ranges (warm-started from that window's previous solve), per-hand
    best-response values against the other player's solution: what bbar/value/gen.py labels, what LeafOracle gives.
    Only for late leaves (the windows must be small). Takes the leaves' games (needs_games). warm=False solves every
    query from scratch: the values are then a function of the query alone (an agent using them replays exactly)."""
    needs_games = True

    def __init__(self, target=1e-6, max_iters=20_000, threads=1, warm=True):
        self.target, self.max_iters, self.threads, self.warm = target, max_iters, threads, warm
        self.windows = {}
        self.stats = {"solves": 0, "iterations": 0, "calls": 0}

    def __call__(self, pub, movers, beliefs, games):
        self.stats["calls"] += 1
        keys = [public_key(g) for g in games]
        for k, g in zip(keys, games):
            if k not in self.windows:
                self.windows[k] = [Window(g, threads=1), False]

        def one(j):
            w, warm = self.windows[keys[j]]
            w.set_ranges(beliefs[j, 0], beliefs[j, 1])
            r = w.solve(target=self.target, max_iters=self.max_iters, warm=warm)
            return np.stack([w.br_values(0), w.br_values(1)]), r["iterations"]
        out = []
        for j in range(len(games)):     # sequential per window object: one window may serve several leaves
            v, it = one(j)
            self.windows[keys[j]][1] = self.warm
            out.append(v)
            self.stats["iterations"] += it
        self.stats["solves"] += len(out)
        return np.stack(out) if out else np.zeros((0, 2, pbs.SLOTS))


def solve_many(jobs, values, iters, algo="dcfr", threads=1, fast=True):
    """Runs `iters` CFR-D iterations of every job (DLJob) in lockstep: each half-iteration, ext_begin on every window,
    ONE call of the value function over the value leaves of all jobs, ext_end on every window. Each window's result is
    what its own solve_external / resolve_external with the same value function would give, up to the value
    function's float rounding on a different batch. Returns seconds, callback_seconds (value function), leaves (value
    leaves in all), calls, and the solver's seconds inside begin / end (begin_seconds, end_seconds; GIL released).

    The windows run through one bbar._core.WindowBatch on `threads` threads (GIL released; leaf arrays written and
    read in place). A value function with session() (NetValues, resolve.ValueModel) gets the fast path: float32
    beliefs and values of the feasible hands only, mover first, its public features on the device once
    (NetValues.session); any other is called as values(pub, movers, beliefs[, games]) with float64 arrays over the 495
    slots, absolute players, a fresh array per call. Both give exactly the trajectories of the per-window reference
    loop (fast=False: ext_begin / ext_end of each window, np.concatenate / np.split around the value function;
    tests/test_value.py)."""
    if not fast:
        return _solve_many_reference(jobs, values, iters, algo, threads)
    t0 = time.perf_counter()
    jobs = list(jobs)
    _ext_start(jobs, algo)
    session = getattr(values, "session", None)
    batch = WindowBatch([j.w for j in jobs], layout="compact" if session else "slots", relative=bool(session),
                        threads=threads)
    n = batch.num_leaves
    cb, calls = 0.0, 0
    if session is not None:
        fn = session(np.concatenate([j.pub for j in jobs]), batch.slots()[0]) if n else None
        buf = np.empty((n, 2, batch.width), dtype=np.float32)
        batch.begin(buf)
        for h in range(2 * iters):
            if n:
                t1 = time.perf_counter()
                vals = fn(buf)
                cb += time.perf_counter() - t1
                calls += 1
            else:
                vals = buf
            batch.end(vals, buf if h + 1 < 2 * iters else None)
    else:
        pub = np.concatenate([j.pub for j in jobs]) if jobs else np.zeros((0, pbs.PUB_DIM), dtype=np.float32)
        movers = np.concatenate([j.movers for j in jobs]) if jobs else np.zeros(0, dtype=np.int64)
        extra = ([g for j in jobs for g in j.games],) if getattr(values, "needs_games", False) else ()
        for _ in range(2 * iters):
            beliefs = np.empty((n, 2, batch.width))
            batch.begin(beliefs)
            if n:
                t1 = time.perf_counter()
                vals = np.ascontiguousarray(values(pub, movers, beliefs, *extra), dtype=np.float64)
                cb += time.perf_counter() - t1
                calls += 1
            else:
                vals = beliefs
            batch.end(vals)
    batch.release()     # the solver's between-half-iteration caches (the windows may be kept: bbar/resolve.py)
    st = batch.stats()
    return dict(seconds=time.perf_counter() - t0, callback_seconds=cb, leaves=n, calls=calls,
                begin_seconds=st["begin_seconds"], end_seconds=st["end_seconds"])


def _ext_start(jobs, algo):
    for j in jobs:
        if j.gadget is None:
            j.w.ext_start(algo)
        else:
            player, cfvs, prior = j.gadget
            j.w.ext_start(algo, player, cfvs, prior)


def _solve_many_reference(jobs, values, iters, algo="dcfr", threads=1):
    """solve_many through each window's own ext_begin / ext_end (float64 arrays over the 495 slots, concatenated for
    the value function and split back): the reference the batched path is tested against."""
    t0 = time.perf_counter()
    jobs = list(jobs)
    _ext_start(jobs, algo)
    sizes = [j.leaves for j in jobs]
    cuts = np.cumsum(sizes)[:-1]
    n = int(sum(sizes))
    pub = np.concatenate([j.pub for j in jobs]) if jobs else np.zeros((0, pbs.PUB_DIM), dtype=np.float32)
    movers = np.concatenate([j.movers for j in jobs]) if jobs else np.zeros(0, dtype=np.int64)
    extra = ([g for j in jobs for g in j.games],) if getattr(values, "needs_games", False) else ()
    pool = ThreadPoolExecutor(threads) if threads > 1 and len(jobs) > 1 else None
    run = (lambda f, xs: list(pool.map(f, xs))) if pool else (lambda f, xs: [f(x) for x in xs])
    cb, calls = 0.0, 0
    try:
        for _ in range(2 * iters):
            qs = run(lambda j: j.w.ext_begin(), jobs)
            if n:
                beliefs = np.concatenate([q[2] for q in qs])
                t1 = time.perf_counter()
                vals = values(pub, movers, beliefs, *extra)
                cb += time.perf_counter() - t1
                calls += 1
                parts = np.split(vals, cuts)
            else:
                parts = [np.zeros((0, 2, pbs.SLOTS))] * len(jobs)
            run(lambda jv: jv[0].w.ext_end(jv[1]), list(zip(jobs, parts)))
    finally:
        if pool:
            pool.shutdown()
    return dict(seconds=time.perf_counter() - t0, callback_seconds=cb, leaves=n, calls=calls)


class NetLeaf:
    """Leaf values from a value network (bbar.value.net.NetValues) at the leaves' PBS."""

    def __init__(self, dl, net_values):
        self.pub, self.movers = leaf_features(dl)
        self.nv = net_values
        self.games = [dl.leaf_game(j) for j in range(dl.num_value_leaves)] if getattr(
            net_values, "needs_games", False) else None
        self.calls = 0

    def values(self, beliefs):
        self.calls += 1
        if not len(self.movers):
            return np.zeros((0, 2, pbs.SLOTS))
        if self.games is not None:
            return self.nv(self.pub, self.movers, beliefs, self.games)
        return self.nv(self.pub, self.movers, beliefs)

    def __call__(self, player, t, beliefs, mass):
        return self.values(beliefs)


def composite(full, dl, windows, gadget_target=1e-6, gadget_iters=50_000, prior="uniform", threads=1, opp=None):
    """(full-window profile, info) of depth-limited play: the trunk of `dl`'s average strategy and, below each of
    its value leaves, both players' safe re-solves (module docstring). `windows`: leaf_windows(dl) (re-ranged here).
    prior: the gadget's opponent prior, "uniform" or "belief" (half the opponent's belief, half uniform).
    info: entries covered (must equal full.num_entries), worst gadget exploitability and violation."""
    bel, mass = dl.leaf_query()
    opp = dl.leaf_values_avg() if opp is None else opp
    leaves = dl.value_leaves()

    def one(j):
        w = windows[j]
        w.set_ranges(bel[j, 0], bel[j, 1])
        subs, worst = [], [0.0, -np.inf]
        for p in (0, 1):
            o = 1 - p
            pr = None
            if prior == "belief":
                feas = np.zeros(pbs.SLOTS)
                feas[w.hands(o)] = 1 / len(w.hands(o))
                pr = 0.5 * bel[j, o] + 0.5 * feas
            r = w.resolve(p, opp[j, o], target=gadget_target, max_iters=gadget_iters, prior=pr)
            worst = [max(worst[0], r["exploitability"]), max(worst[1], r["max_violation"])]
            subs.append(w.profile_array())
        return w.compose(subs[0], subs[1]), worst
    res = _map(one, range(len(leaves)), threads)
    prof, n = full.embed(full.profile_array("uniform"), dl)
    parts = [(windows[j], res[j][0], leaves[j][0]) for j in range(len(leaves))]
    prof, m = full.embed_all(prof, parts)
    info = dict(entries=n + m, full_entries=full.num_entries,
                gadget_exploitability=max((r[1][0] for r in res), default=0.0),
                gadget_max_violation=max((r[1][1] for r in res), default=0.0))
    return prof, info


def leaf_error(dl, values, exact):
    """Errors of leaf values against exact ones at dl's average-strategy beliefs: per value leaf j and player p,
    sum_h b_p(h) |values - exact|, averaged over the leaves with the probability of reaching each (product of the
    players' reach masses) and over both players: (on-path range-weighted MAE, max over leaves)."""
    bel, mass = dl.leaf_query()
    if not len(mass):
        return 0.0, 0.0
    e = (bel * np.abs(values - exact)).sum(-1)       # [n, 2]
    w = mass[:, 0] * mass[:, 1]
    w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
    return float((w * e.mean(1)).sum()), float(e.max())
