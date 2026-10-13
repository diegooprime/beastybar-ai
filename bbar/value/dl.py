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

The composite profile in the full window: plies [S, L) from the depth-limited solve's average strategy; below each
value leaf, for each player p, a safe re-solve of the leaf's subtree (core/src/window.rs gadget): p's range = p's
belief at the leaf under the average strategy, the opponent's cfvs = `dl.leaf_values_avg()` (the opponent's values
against p's average strategy there, which the gadget lets the opponent take instead of entering). Each player's
subtree strategy comes from their own re-solve. Every entry of the full window is covered.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from bbar._core import Window, public_key
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


class NetLeaf:
    """Leaf values from a value network (bbar.value.net.NetValues) at the leaves' PBS."""

    def __init__(self, dl, net_values):
        games = [dl.leaf_game(j) for j in range(dl.num_value_leaves)]
        keys = np.array([public_key(g) for g in games], dtype=np.uint64)
        self.pub = pbs.public_features(keys, [g.start for g in games]) if games else np.zeros((0, pbs.PUB_DIM))
        self.movers = np.array([g.to_move for g in games], dtype=np.int64)
        self.nv = net_values
        self.calls = 0

    def values(self, beliefs):
        self.calls += 1
        if not len(self.movers):
            return np.zeros((0, 2, pbs.SLOTS))
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
