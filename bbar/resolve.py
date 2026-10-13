"""Continual re-solving player (milestone M4 of docs/PLAN.md): a base network early, depth-limited re-solves with a
public-belief-state value network in the middle game, one exact window late, exact minimax at the end.

Spec (bbar.match, bbar.delta):
    resolve:V=PATH,base=nn:PATH[,P0=8][,F=10][,d=2][,iters=200][,mode=safe|plain][,target=1e-4]
            [,exact_iters=10000][,prior=mix|uniform|bayes][,algo=dcfr][,eps=1e-9][,check_every=10][,threads=1]
            [,workers=1][,cache=256]
V: a value network (bbar/value/net.py, trained by bbar/value/train.py), or V=exact: exact leaf values (each value leaf's
own window solved to 1e-6 at every half-iteration: bbar/value/dl.ExactValues; a reference for late leaves only, slow);
base: a policy network. The agent's name is its
spec with every option that changes its play (V, base, P0, F, d, iters, mode, target, exact_iters, and prior, algo,
eps, check_every when not the default); threads, workers and cache change only speed and memory. Ply k = k moves
played.

Decisions of the agent (seat s; o = 1 - s), by ply t:
  - perfect information (ply 15 for the second mover, every ply >= 16): exact minimax, ties broken with the private
    seed (bbar/tail.py);
  - t < min(P0, F): sample the base network with the seat's draw, exactly as bbar.match.NetPlayer does (move for move);
  - P0 <= t < F: a depth-limited window (core/src/window.rs) from the current public state, value leaves at ply
    min(t + d, F) (terminal and frontier leaves exact as always) valued by V (bbar/value/dl.py: CFR-D, the iterate's
    leaf beliefs), `iters` iterations; the move is sampled (private seed; never argmax) from the AVERAGE strategy at the
    root for the actual hand;
  - t >= F: one exact window per game, rooted at ply R = F (or, with no depth-limited phase (P0 >= F), at the agent's
    first decision >= F, exactly as tail:), solved to `target` (at most exact_iters iterations) and followed, sampled
    with the seat's draw, until perfect information.

What the agent knows at a public node (all a function of the public history: its own play enters only through its
strategy for EVERY hand, so the agent is a behaviour strategy, and fresh instances replay it exactly):
  - our range: exact Bayes over our own decisions with the strategy actually used for every hand (the base network
    before P0, each depth-limited solve's root average strategy, the exact window's strategy), plus the draws
    (bbar/belief.py; floored at eps like the tail agent's);
  - the opponent, by mode:
      plain (ReBeL-style common-knowledge beliefs): its Bayes range: base network before P0; in [P0, F) the opponent's
        policy at that node in our previous solve (internal there when d >= 2); if the node is outside our previous
        tree (it moved first in the phase, or d = 1), a solve at its decision point (a plain depth-limited window rooted
        there with the ranges then) gives its policy for all hands. Every window is solved plainly on these ranges.
      safe (DeepStack continual re-solving): the opponent's counterfactual values at the current public node carried
        from our previous solve: averaged leaf values (Window.leaf_values_avg: the opponent's values against our average
        strategy at that leaf, in units where our reach there sums to 1) when the node is a value leaf of it, else the
        opponent's best-response cfvs at that internal node against our average strategy with the averaged leaf values
        below (Window.node_values, divided by our reach mass there). The window is re-solved with the gadget
        (Window.resolve_external / resolve): the opponent may take those values instead of entering, so at the
        gadget's equilibrium no opponent hand gains more than they say (up to V's errors). After finitely many
        iterations a hand's excess shrinks with its gadget prior (the convergence target and the terminate/follow
        updates are prior-weighted), so the prior keeps every hand's weight up: `prior` = mix (default): half the
        opponent's Bayes range (under the previous solve's strategy; uniform over its feasible hands before any
        solve), half uniform over its feasible hands, as dl.composite's prior="belief" (every hand >= 1/(2n)); uniform:
        uniform over its feasible hands (CFR-D / DeepStack's gadget); bayes: the Bayes range alone (hands the previous
        solve thought would not play the observed action get a prior near eps and go nearly unconstrained). With
        exact leaves (base bp-v1 step 30518, P0=11, F=14, 100 iterations, 324 re-solves on and off the played path)
        the largest excess is 0.014 with mix, 0.010 uniform, 0.17 bayes (README). Logged per re-solve: gap (the gadget
        game's exploitability, prior-weighted), max_violation (the largest excess over all opponent hands, unweighted)
        and mean_violation (the prior-weighted mean excess). Our first depth-limited decision has no previous solve:
        its opponent values are the base network's, the opponent's best-response cfvs
        against the base network on the same depth-limited window (leaf values: V at the beliefs the base network
        induces, both players' ranges Bayes under the base network). d >= 2 (with d = 1 our next decision would lie
        outside every previous tree).
    The exact window at F: plain, a DCFR solve on the tracked ranges; safe, the gadget with the opponent values carried
    to the ply-F node (a value leaf of our last depth-limited solve; rooting at F rather than at our first decision >= F
    keeps them for the seat that moves at F + 1) and the prior pushed along (mixed as above). Without a depth-limited
    phase (P0 >= F) it is the tail agent's window (gadget against the base network's cfvs, prior = Bayes under the base
    network, whatever `prior` says).

Every solve is logged (ResolveAgent.log): kind (dl: our decision, opp: plain mode's solve at an opponent node, exact),
ply, window size (internal nodes, value leaves, entries), iterations, seconds (build + solve; batched depth-limited
solves share the batch's wall time equally), value-network seconds, the opponent values' source and the re-solve's
gap / max_violation / mean_violation (see above; plain solves: gap only). Depth-limited
solves of every game in play at the same point are run in lockstep with one batched value-network call per
half-iteration (bbar/value/dl.solve_many): results equal one-at-a-time solving up to the network's float rounding on a
different batch.

Cache: depth-limited states per game until its exact window is built; exact windows (about 12 MB each at F=10) LRU-bound
by `cache` and dropped after the agent's last decision in them, as the tail agent does (bbar.match: cache >= --batch).
"""
import functools
import os
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from bbar import belief
from bbar._core import Window, WindowTree, belief_rows, hand_rank
from bbar.policy import NN, NetModel, sha256_short
from bbar.tail import (CHECK_EVERY, DEFAULT_CACHE, PERFECT_INFO_PLY, Solution, first_root, last_window_decision,
                       minimax_choice, num, perfect_info, sample)
from bbar.value import dl as D

MODES = ("safe", "plain")
PRIORS = ("mix", "uniform", "bayes")    # the safe gadget's opponent prior (module docstring)
DEFAULTS = dict(P0=8, F=10, d=2, iters=200, mode="safe", target=1e-4, exact_iters=10_000, prior="mix", algo="dcfr",
                eps=belief.EPS, check_every=CHECK_EVERY, threads=1, workers=1, cache=DEFAULT_CACHE)
CASTS = dict(V=str, base=str, P0=int, F=int, d=int, iters=int, mode=str, target=float, exact_iters=int, prior=str,
             algo=str, eps=float, check_every=int, threads=int, workers=int, cache=int)
SPEED_ONLY = ("threads", "workers", "cache")


def parse_spec(body):
    """'V=PATH,base=nn:PATH,P0=8,...' (after 'resolve:') -> options dict (base without its 'nn:' prefix)."""
    opts = dict(DEFAULTS)
    seen = set()
    for part in body.split(","):
        if "=" not in part:
            raise ValueError(f"resolve:{body!r}: expected key=value, got {part!r}")
        k, v = (x.strip() for x in part.split("=", 1))
        if k not in CASTS:
            raise ValueError(f"resolve:{body!r}: unknown key {k!r} (allowed: {', '.join(CASTS)})")
        if k in seen:
            raise ValueError(f"resolve:{body!r}: repeated key {k!r}")
        seen.add(k)
        try:
            opts[k] = CASTS[k](v)
        except ValueError:
            raise ValueError(f"resolve:{body!r}: bad value for {k}: {v!r}") from None
    if "V" not in seen or "base" not in seen:
        raise ValueError(f"resolve:{body!r}: V=PATH (value network) and base=nn:PATH (policy network) are required")
    if not opts["base"].startswith(NN) or not opts["base"][len(NN):]:
        raise ValueError(f"resolve:{body!r}: the base must be a network, base=nn:PATH")
    opts["base"] = opts["base"][len(NN):]
    if opts["mode"] not in MODES:
        raise ValueError(f"resolve:{body!r}: mode must be safe or plain")
    if opts["prior"] not in PRIORS:
        raise ValueError(f"resolve:{body!r}: prior must be one of {', '.join(PRIORS)}")
    if not 1 <= opts["F"] <= 14 or opts["P0"] < 0:
        raise ValueError(f"resolve:{body!r}: need 1 <= F <= 14 and P0 >= 0")
    if opts["d"] < 1 or opts["iters"] < 1 or opts["exact_iters"] < 1:
        raise ValueError(f"resolve:{body!r}: d, iters and exact_iters must be at least 1")
    if opts["mode"] == "safe" and opts["d"] < 2:
        raise ValueError(f"resolve:{body!r}: mode=safe needs d >= 2 (the next own decision must lie in the previous "
                         "tree to carry the opponent's values to it)")
    if opts["cache"] < 0 or opts["check_every"] < 1:
        raise ValueError(f"resolve:{body!r}: cache must be at least 0, check_every at least 1")
    return opts


def agent_name(V, base, P0, F, d, iters, mode, target, exact_iters, algo="dcfr", eps=belief.EPS,
               check_every=CHECK_EVERY, prior="mix"):
    """The spec of everything that changes the agent's play (threads, workers and cache do not)."""
    name = (f"resolve:V={V},base={base},P0={P0},F={F},d={d},iters={iters},mode={mode},target={num(target)},"
            f"exact_iters={exact_iters}")
    if prior != "mix":
        name += f",prior={prior}"
    if algo != "dcfr":
        name += f",algo={algo}"
    if eps != belief.EPS:
        name += f",eps={num(eps)}"
    if check_every != CHECK_EVERY:
        name += f",check_every={check_every}"
    return name


def mover(start, t):
    """The player to move at ply t of a game started by `start` (the starter moves at even plies)."""
    return (int(start) + int(t)) % 2


class ValueModel:
    """A PBS value network checkpoint as a leaf value function (bbar.value.net.NetValues), with its path and hash."""

    def __init__(self, path, device="cpu", chunk=None):
        from bbar.value.net import NetValues, load_value_net
        self.path = str(path)
        self.sha = sha256_short(Path(path).read_bytes())
        net, ck = load_value_net(path, device)
        # Fixed-size forward passes: a leaf's value never depends on which other solves share the batch.
        chunk = chunk or (64 if str(device) == "cpu" else 1024)
        self.fn = NetValues(net, device, chunk=chunk)
        self.plies = sorted({int(p) for p in (ck.get("metrics") or {}) if p.isdigit()})

    def __call__(self, pub, movers, beliefs):
        return self.fn(pub, movers, beliefs)

    def session(self, pub, slots):
        """NetValues.session: the fast path of bbar/value/dl.solve_many (same values)."""
        return self.fn.session(pub, slots)


def exact_values():
    """V=exact: the exact value function (cold solves: the agent's play does not depend on what it solved before)."""
    v = D.ExactValues(target=1e-6, warm=False)
    v.path = "exact"
    return v


class DLState:
    """A solved depth-limited window at a public node: `seat` moves at its root (us, or the opponent for plain mode's
    solves at its decision points); `policy` its root average strategy (float64 [slots, 36])."""

    def __init__(self, key, seat, window, info):
        self.key, self.seat, self.window, self.info = key, seat, window, info
        self.policy = window.root_policy()
        self._avg = None
        self._leaves = None

    @property
    def leaf_avg(self):
        if self._avg is None:
            self._avg = self.window.leaf_values_avg()
        return self._avg

    def leaf_index(self, path):
        if self._leaves is None:
            self._leaves = {tuple(h): j for j, (h, _) in enumerate(self.window.value_leaves())}
        return self._leaves.get(tuple(path))


class ResolveAgent:
    needs_obs = True

    def __init__(self, base, V, P0=8, F=10, d=2, iters=200, mode="safe", target=1e-4, exact_iters=10_000,
                 prior="mix", algo="dcfr", eps=belief.EPS, check_every=CHECK_EVERY, threads=1, workers=1,
                 cache=DEFAULT_CACHE, device="cpu", name=None):
        self.model = base if isinstance(base, NetModel) else NetModel(base, device)
        if isinstance(V, (str, os.PathLike)):
            V = exact_values() if str(V) == "exact" else ValueModel(V, device)
        self.values = V
        if mode not in MODES:
            raise ValueError(f"mode must be safe or plain, not {mode!r}")
        if mode == "safe" and d < 2:
            raise ValueError("mode=safe needs d >= 2")
        if prior not in PRIORS:
            raise ValueError(f"prior must be one of {', '.join(PRIORS)}, not {prior!r}")
        self.P0, self.F, self.d, self.iters, self.mode, self.prior = P0, F, d, iters, mode, prior
        self.target, self.exact_iters, self.algo, self.eps, self.check_every = target, exact_iters, algo, eps, \
            check_every
        self.threads, self.workers, self.cache_size = threads, workers, cache
        self.checkpoints = dict(self.model.checkpoints)
        vpath = getattr(self.values, "path", "<value>")
        if hasattr(self.values, "sha"):
            self.checkpoints[vpath] = self.values.sha
        self.name = name or agent_name(vpath, self.model.name, P0, F, d, iters, mode, target, exact_iters, algo, eps,
                                       check_every, prior)
        self._dl = OrderedDict()           # (kind, start, history) -> DLState
        self._exact = OrderedDict()        # (start, root history, seat) -> Solution
        self._evicted = set()
        self._base = OrderedDict()         # (start, history, player) -> base-network likelihood rows
        self.drop_finished = True          # drop a game's states once done with them (bbar.delta: off)
        self.log = []
        self.counts = {"base": 0, "dl": 0, "window": 0, "minimax": 0, "fallback": 0}
        self.resolves = 0
        self._warned = False

    @classmethod
    def factory(cls, body, device="cpu"):
        """A zero-argument callable making fresh agents for the spec body 'V=PATH,base=nn:PATH,...' that share the
        loaded networks (bbar.match.PyPlayer)."""
        opts = parse_spec(body)
        model = NetModel(opts.pop("base"), device)
        values = exact_values() if opts["V"] == "exact" else ValueModel(opts["V"], device)
        opts.pop("V")
        return functools.partial(cls, model, values, device=device, **opts)

    @classmethod
    def from_spec(cls, body, device="cpu"):
        return cls.factory(body, device)()

    def prepare(self, live):
        """bbar.match: up to `live` of this agent's games are in play at once, each holding one exact window."""
        if live > self.cache_size:
            warnings.warn(f"{self.name}: {live} games in play at once but cache={self.cache_size}: exact windows will "
                          f"be evicted while in use and solved again (same moves, more compute)", stacklevel=2)
            self._warned = True

    # ---- the public state machine -------------------------------------------------------------------------------

    def dl_phase(self, t):
        return self.P0 <= t < self.F

    def exact_root(self, start, seat):
        """Ply of the exact window `seat` plays from: F after a depth-limited phase, else (P0 >= F) its first decision
        at ply >= F, as the tail agent."""
        return self.F if self.P0 < self.F else first_root(start, seat, self.F)

    def _base_lik(self, start, hist, p):
        key = (int(start), tuple(hist), p)
        lik = self._base.get(key)
        if lik is None:
            lik = self.model.likelihoods(belief.start_game(start), list(hist), p)
            self._base[key] = lik
            while len(self._base) > 4096:
                self._base.popitem(last=False)
        return lik

    def _source(self, start, hist, t, p, seat, kind):
        """What player p's decision at ply t of `hist` contributes to a range computed by `seat`'s agent: 'base' (the
        base network's likelihoods), a policy float64 [slots, 36] over p's hands, or None (uninformative).
        kind: 'plain' (the opponent's range), 'prior' (the safe gadget's prior), 'base' (Bayes under the base network);
        our own decisions always contribute the strategy we used."""
        H = tuple(hist[:t])
        if p == seat:
            if t < min(self.P0, self.F):
                return "base"
            if t < self.F:
                return self._own(start, H).policy
            R = self.exact_root(start, seat)
            sol = self._get_exact(start, tuple(hist[:R]), seat)
            return sol.window.policy(list(hist[R:t]))
        if kind == "base" or t < self.P0 or t >= self.F:
            return None if kind == "prior" and self.P0 < self.F else "base"
        prev = self._own(start, H[:-1]) if t >= 1 and self.dl_phase(t - 1) else None
        if prev is not None and self.d >= 2:
            return prev.window.policy([H[-1]])
        if kind == "plain":
            return self._get_dl(("opp", int(start), H)).policy
        return None

    def _range(self, start, hist, p, seat, kind):
        """belief.Range of player p after `hist` as `seat`'s agent tracks it (see _source)."""
        deal = belief.start_game(start)
        ply, ranks, _ = belief_rows(deal, list(hist), p)
        lik = np.ones(len(ply))
        for t in np.unique(ply):
            t = int(t)
            sel = ply == t
            src = self._source(start, hist, t, p, seat, kind)
            if isinstance(src, str):
                lik[sel] = self._base_lik(start, hist, p)[sel]
            elif src is not None:
                lik[sel] = src[ranks[sel], int(hist[t])]
        return belief.bayes_range(deal, hist, p, lik, eps=self.eps)

    def _deps(self, start, hist, seat):
        """Depth-limited solves `seat`'s state at the node `hist` depends on: ours before it, and plain mode's solves
        at the opponent's decision points outside our previous trees."""
        out = []
        for t in range(self.P0, min(len(hist), self.F)):
            if mover(start, t) == seat:
                out.append(("own", int(start), tuple(hist[:t])))
            elif self.mode == "plain" and not (self.d >= 2 and self.dl_phase(t - 1)):
                out.append(("opp", int(start), tuple(hist[:t])))
        return out

    def _own(self, start, hist):
        return self._get_dl(("own", int(start), tuple(hist)))

    def _get_dl(self, req):
        st = self._dl.get(req)
        if st is None:
            self._ensure([req])
            st = self._dl[req]
        else:
            self._dl.move_to_end(req)
        return st

    def _ensure(self, reqs):
        """Solves the depth-limited windows `reqs` (and what they depend on) that are not cached, all of one level in
        lockstep (bbar/value/dl.solve_many)."""
        reqs = [r for r in dict.fromkeys(reqs) if r not in self._dl]
        if not reqs:
            return
        deps = []
        for kind, start, hist in reqs:
            seat = mover(start, len(hist)) if kind == "own" else 1 - mover(start, len(hist))
            deps += self._deps(start, hist, seat)
        self._ensure(deps)
        reqs = [r for r in reqs if r not in self._dl]
        if not reqs:
            return
        prepared = [self._prepare(r) for r in reqs]
        stats = D.solve_many([job for job, _ in prepared], self.values, self.iters, self.algo, self.workers)
        for r, (job, info) in zip(reqs, prepared):
            sz = job.w.sizes()
            share = stats["seconds"] / len(reqs)
            info.update(internal=sz["internal"], entries=sz["entries"], value_leaves=job.leaves,
                        iterations=job.w.iterations, batch=len(reqs), solve_s=share,
                        net_s=stats["callback_seconds"] / len(reqs))
            avg = job.w.leaf_values_avg()
            if job.gadget is not None:    # the gadget game's gap and violations, with the averaged leaf values
                g = job.w.gadget_stats(avg)
                info.update(gap=g["exploitability"], max_violation=g["max_violation"],
                            mean_violation=g["mean_violation"])
            else:                         # the depth-limited game's exploitability, with the averaged leaf values
                info["gap"] = job.w.exploitability(None, None, leaf_values=avg)["exploitability"]
            info["seconds"] = info.pop("prep_s") + share
            st = DLState(r, mover(r[1], len(r[2])), job.w, info)
            self._dl[r] = st
            self.log.append(info)
        self._trim_dl()

    def _trim_dl(self):
        while len(self._dl) > max(64, 32 * self.cache_size):
            self._dl.popitem(last=False)

    def _gadget_prior(self, bayes, w, o):
        """The safe gadget's prior over the opponent o's root hands of window `w` (slot layout; module docstring):
        `bayes` (its Bayes range), uniform over its feasible hands, or (mix) half of each."""
        if self.prior == "bayes":
            return bayes
        uniform = np.zeros(len(bayes))
        hands = w.hands(o)
        uniform[hands] = 1.0 / len(hands)
        if self.prior == "uniform":
            return uniform
        return 0.5 * bayes / bayes.sum() + 0.5 * uniform

    def _carried(self, prev, path, o):
        """(the opponent o's values at the node `path` below prev's root, carried from prev's solve, in units where our
        reach there sums to 1; 'leaf' or 'node')."""
        j = prev.leaf_index(path)
        if j is not None:
            return prev.leaf_avg[j, o].copy(), "leaf"
        best, _, mass = prev.window.node_values(o, list(path), None, prev.leaf_avg)
        return best / mass, "node"

    def _base_cfvs(self, w, deal, hist, seat, leaf_ply):
        """The opponent's best-response cfvs against the base network on the depth-limited window `w` (its ranges:
        ours and the opponent's Bayes range under the base network), leaf values V at the beliefs the base network
        induces."""
        tree = WindowTree(deal, list(hist), leaf_ply=leaf_ply)
        table = self.model.window_table(tree, 0) + self.model.window_table(tree, 1)
        bel, _ = w.leaf_query(table)
        vals = D.NetLeaf(w, self.values).values(bel)
        return w.br_values(1 - seat, table, leaf_values=vals)

    def _prepare(self, req):
        """(DLJob, info) of a depth-limited solve: own decision or (plain) the opponent's decision point."""
        t0 = time.perf_counter()
        kind, start, hist = req
        t = len(hist)
        root_mover = mover(start, t)
        seat = root_mover if kind == "own" else 1 - root_mover
        o = 1 - seat
        deal = belief.start_game(start)
        game = belief.replay(deal, hist)
        L = min(t + self.d, self.F)
        ours = self._range(start, hist, seat, seat, "ours").range
        r = [None, None]
        r[seat] = ours
        info = dict(kind="dl" if kind == "own" else "opp", ply=t, seat=root_mover, mode=self.mode, leaf_ply=L)
        if self.mode == "plain" or kind == "opp":
            opp = self._range(start, hist, o, seat, "plain")
            r[o] = opp.range
            info["impossible"] = opp.impossible
            job = D.DLJob(Window(game, r[0], r[1], leaf_ply=L, threads=self.threads))
        else:
            prior = self._range(start, hist, o, seat, "prior").range
            prev = self._own(start, hist[:t - 2]) if t >= 2 and self.dl_phase(t - 2) else None
            if prev is not None:
                cfvs, info["cfvs"] = self._carried(prev, hist[t - 2:], o)
                r[o] = prior
                w = Window(game, r[0], r[1], leaf_ply=L, threads=self.threads)
            else:
                r[o] = self._range(start, hist, o, seat, "base").range
                w = Window(game, r[0], r[1], leaf_ply=L, threads=self.threads)
                cfvs, info["cfvs"] = self._base_cfvs(w, deal, hist, seat, L), "base"
            job = D.DLJob(w, gadget=(seat, cfvs, self._gadget_prior(prior, w, o)))
        info["prep_s"] = time.perf_counter() - t0
        return job, info

    # ---- exact windows --------------------------------------------------------------------------------------------

    def _exact_deps(self, start, root, seat):
        return self._deps(start, root, seat)

    def _get_exact(self, start, root, seat):
        key = (int(start), tuple(root), int(seat))
        sol = self._exact.get(key)
        if sol is not None:
            self._exact.move_to_end(key)
            return sol
        self._ensure(self._exact_deps(*key))
        sol = self._store_exact(key, self._solve_exact(*key))
        self._trim_exact()
        return sol

    def _store_exact(self, key, sol):
        if key in self._evicted:
            self._evicted.discard(key)
            sol.info["resolve"] = True
            self.resolves += 1
            if not self._warned:
                self._warned = True
                warnings.warn(f"{self.name}: re-solving an exact window the cache (cache={self.cache_size}) evicted "
                              f"while still in use: same moves, more compute", stacklevel=3)
        self._exact[key] = sol
        self.log.append(sol.info)
        if self.drop_finished:   # the game's depth-limited states are no longer needed
            self._release_dl(key[0], key[1])
        return sol

    def _prefetch_exact(self, keys):
        """Solves the exact windows `keys` that are not cached: their depth-limited dependencies in one lockstep batch,
        then the windows on `workers` threads (the solver releases the GIL)."""
        todo = [key for key in dict.fromkeys(keys) if key not in self._exact]
        self._ensure([d for key in todo for d in self._exact_deps(*key)])
        if self.workers > 1 and len(todo) > 1:
            with ThreadPoolExecutor(self.workers) as ex:
                solved = list(ex.map(lambda key: self._solve_exact(*key), todo))
            for key, sol in zip(todo, solved):
                self._store_exact(key, sol)

    def _trim_exact(self):
        while len(self._exact) > self.cache_size:
            key, _ = self._exact.popitem(last=False)
            self._evicted.add(key)

    def _solve_exact(self, start, root, seat):
        """The exact window rooted at public history `root` for `seat` (its depth-limited dependencies cached)."""
        t0 = time.time()
        deal = belief.start_game(start)
        o = 1 - seat
        R = len(root)
        ours = self._range(start, root, seat, seat, "ours")
        prev, tp = None, None
        for tp in (R - 1, R - 2):
            if tp >= 0 and self.dl_phase(tp) and mover(start, tp) == seat:
                prev = self._own(start, root[:tp])
                break
        if self.mode == "plain":
            opp = self._range(start, root, o, seat, "plain")
        elif prev is None:     # no depth-limited decision before: the tail agent's prior (Bayes under the base network)
            opp = self._range(start, root, o, seat, "base")
        else:
            opp = self._range(start, root, o, seat, "prior")
        r = [None, None]
        r[seat], r[o] = ours.range, opp.range
        t1 = time.time()
        game = belief.replay(deal, root)
        w = Window(game, r[0], r[1], threads=self.threads)
        sz = w.sizes()
        t2 = time.time()
        info = dict(kind="exact", ply=R, seat=seat, mode=self.mode, internal=sz["internal"], entries=sz["entries"],
                    value_leaves=0, hands=list(sz["hands"]), impossible=[ours.impossible, opp.impossible],
                    belief_s=t1 - t0, build_s=t2 - t1, table_s=0.0, resolve=False)
        sol = Solution(tuple(root), seat, w, deal, info)
        if sz["internal"] == 0:
            info.update(solve_s=0.0, iterations=0, gap=0.0)
        elif self.mode == "plain":
            res = w.solve(target=self.target, max_iters=self.exact_iters, check_every=self.check_every,
                          algo=self.algo)
            info.update(solve_s=res["seconds"], iterations=res["iterations"], gap=res["exploitability"])
        else:
            if prev is not None:
                cfvs, info["cfvs"] = self._carried(prev, root[tp:], o)
            else:
                cfvs, info["cfvs"] = w.br_values(o, self.model.window_table(sol.tree, seat)), "base"
            t3 = time.time()
            # Without a depth-limited phase this is the tail agent's window: its prior, Bayes under the base network.
            prior = opp.range if self.P0 >= self.F else self._gadget_prior(opp.range, w, o)
            res = w.resolve(seat, cfvs, target=self.target, max_iters=self.exact_iters,
                            check_every=self.check_every, algo=self.algo, prior=prior)
            info.update(table_s=t3 - t2, solve_s=res["seconds"], iterations=res["iterations"],
                        gap=res["exploitability"], max_violation=res["max_violation"],
                        mean_violation=res["mean_violation"])
        info["seconds"] = time.time() - t0
        return sol

    def _release_dl(self, start, prefix):
        k = len(prefix)
        for key in [key for key in self._dl if key[1] == start and key[2] == tuple(prefix[:len(key[2])])
                    and len(key[2]) <= k]:
            del self._dl[key]

    def release(self, start, prefix):
        """Forgets every state of games below the public history `prefix` from start seat `start` (bbar.delta, once a
        position is evaluated): not counted as evictions."""
        prefix = tuple(int(a) for a in prefix)
        k = len(prefix)
        for key in [key for key in self._dl if key[1] == start and key[2][:k] == prefix]:
            del self._dl[key]
        for key in [key for key in self._exact if key[0] == start and key[1][:k] == prefix]:
            del self._exact[key]
        self._evicted = {key for key in self._evicted if not (key[0] == start and key[1][:k] == prefix)}

    # ---- acting -------------------------------------------------------------------------------------------------

    def act(self, batch):
        """int64 [k]: one action per decision of `batch` (bbar.match.Decisions)."""
        k = len(batch.games)
        out = np.full(k, -1, dtype=np.int64)
        u = np.asarray(batch.uniforms, dtype=np.float64)
        base, dl, exact = [], [], []
        for j, (g, hist) in enumerate(zip(batch.games, batch.histories)):
            t = len(hist)
            if perfect_info(g):
                out[j] = minimax_choice(g, np.random.default_rng(int(batch.seeds[j])))
                self.counts["minimax"] += 1
            elif t < min(self.P0, self.F):
                base.append(j)
            elif t < self.F:
                dl.append(j)
            else:
                exact.append(j)
        if base:
            out[base] = self.model.sample(batch.obs[base], batch.legal[base], u[base])
            self.counts["base"] += len(base)
        if dl:
            reqs = [("own", int(batch.games[j].start), tuple(int(a) for a in batch.histories[j])) for j in dl]
            self._ensure(reqs)
            for j, req in zip(dl, reqs):
                g = batch.games[j]
                probs = self._get_dl(req).policy[hand_rank(g.hand(g.to_move))]
                out[j] = sample(probs, np.random.default_rng(int(batch.seeds[j])).random())
                self.counts["dl"] += 1
        if exact:
            keys = []
            for j in exact:
                g, hist = batch.games[j], batch.histories[j]
                R = self.exact_root(g.start, g.to_move)
                keys.append((int(g.start), tuple(int(a) for a in hist[:R]), int(g.to_move)))
            self._prefetch_exact(keys)
            finished = []
            for j, key in zip(exact, keys):
                g, hist = batch.games[j], batch.histories[j]
                sol = self._get_exact(*key)
                if sol.info["internal"] == 0:
                    out[j] = self.model.sample(batch.obs[[j]], batch.legal[[j]], u[[j]])[0]
                    self.counts["fallback"] += 1
                else:
                    probs = sol.policy(hist[len(key[1]):], hand_rank(g.hand(g.to_move)))
                    out[j] = sample(probs, u[j])
                    self.counts["window"] += 1
                if self.drop_finished and last_window_decision(len(hist)):
                    finished.append(key)
            for key in finished:
                self._exact.pop(key, None)
            self._trim_exact()
        return out

    @classmethod
    def summarize(cls, agents):
        """Statistics of one or more agents (bbar.match.PyPlayer instances): decisions by kind; solves by kind (dl:
        our depth-limited decisions, opp: plain mode's solves at opponent decision points, exact) with mean / max of
        window sizes, iterations, seconds (batched solves: their share of the batch), value-network seconds, gaps
        (prior-weighted) and gadget violations (max_violation: over all opponent hands, unweighted; mean_violation:
        prior-weighted); and seconds by kind and ply (a dl solve is one decision; an exact window's solve is
        charged to the first decision played from it, the later ones are free)."""
        counts = {k: sum(a.counts[k] for a in agents) for k in agents[0].counts}
        log = [x for a in agents for x in a.log]
        out = {"decisions": counts, "instances": len(agents), "solves": len(log),
               "resolves": sum(a.resolves for a in agents)}
        for kind in ("dl", "opp", "exact"):
            xs = [x for x in log if x["kind"] == kind and not x.get("resolve")]
            if not xs:
                continue
            s = {"n": len(xs)}
            for key in ("internal", "entries", "value_leaves", "iterations", "seconds", "solve_s", "net_s", "gap",
                        "max_violation", "mean_violation", "batch"):
                v = [x[key] for x in xs if key in x]
                if v:
                    s[key] = {"mean": float(np.mean(v)), "max": float(np.max(v))}
            out[kind] = s
        by_ply = {}
        for x in log:
            by_ply.setdefault(x["kind"], {}).setdefault(int(x["ply"]), []).append(x["seconds"])
        out["seconds_by_ply"] = {kind: {p: {"n": len(v), "mean": float(np.mean(v)), "max": float(np.max(v))}
                                        for p, v in sorted(d.items())} for kind, d in by_ply.items()}
        return out

    def summary(self):
        return self.summarize([self])

    # ---- the agent's behaviour as tables (bbar.delta) -------------------------------------------------------------

    def window_table(self, tree, seat, deal, actions):
        """`seat`'s part of this agent's strategy table on the window `tree` rooted after `actions` (from the start
        position `deal`): base network before P0, the root strategy of its depth-limited solve at every node of
        [P0, F), its exact windows (rooted at ply R) below."""
        T = len(actions)
        start = deal.start
        actions = tuple(int(a) for a in actions)
        R = self.exact_root(start, seat)
        mine = tree.entry_movers() == seat
        if R <= T:
            sol = self._get_exact(start, actions[:R], seat)
            n = sol.tree.find(list(actions[R:]))
            if n is None:
                raise ValueError(f"window at ply {T} is not inside the exact window at ply {R}")
            node = sol.tree.node(n)
            prof = sol.profile[node["entry"]:node["entry_end"]]
            assert len(prof) == tree.num_entries
            return np.where(mine, prof, 0.0)
        # The base network's entries only where it plays (plies < P0): at T >= P0 a ply-8 window has millions of rows.
        table = self.model.window_table(tree, seat) if T < self.P0 else np.zeros(tree.num_entries)
        nodes = [n for depth in range(max(self.P0 - T, 0), self.F - T) for n in tree.nodes_at(depth, mover=seat)]
        paths = {n: tuple(tree.node(n)["path"]) for n in nodes}
        roots = [actions + tuple(tree.node(n)["path"]) for n in tree.nodes_at(R - T)]
        deps = [("own", int(start), actions + paths[n]) for n in nodes]
        deps += [d for root in roots for d in self._exact_deps(start, root, seat)]
        self._ensure(deps)
        for n in nodes:
            node = tree.node(n)
            pol = self._own(start, actions + paths[n]).policy
            hands, acts = tree.hands(n), node["actions"]
            a = node["entry"]
            table[a:a + len(hands) * len(acts)] = pol[np.ix_(hands, acts)].ravel()
        keys = [(int(start), root, int(seat)) for root in roots]
        for i in range(0, len(keys), max(1, self.cache_size)):   # at most `cache` windows held at once
            chunk = keys[i:i + max(1, self.cache_size)]
            self._prefetch_exact(chunk)
            for key in chunk:
                n = tree.find(list(key[1][T:]))
                node = tree.node(n)
                sol = self._get_exact(*key)
                a, b = node["entry"], node["entry_end"]
                assert b - a == sol.window.num_entries
                table[a:b] = np.where(mine[a:b], sol.profile, table[a:b])
        return table

    def solved_root(self, deal, actions, seat):
        """(Window, WindowTree, seat's solved table) of the exact window `seat` plays from after `actions` if its root
        is at or before ply len(actions), else None (bbar.delta safety margins)."""
        R = self.exact_root(deal.start, seat)
        if R > len(actions):
            return None
        sol = self._get_exact(deal.start, tuple(int(a) for a in actions[:R]), seat)
        return sol.window, sol.tree, np.where(sol.tree.entry_movers() == seat, sol.profile, 0.0)

    def likelihoods(self, game, actions, player):
        """Per-row likelihoods (rows of bbar._core.belief_rows) of `player`'s actions under this agent (playing that
        seat): the base network before P0, its depth-limited solves' root strategies in [P0, F), its exact window from
        there."""
        actions = tuple(int(a) for a in actions)
        lik = self.model.likelihoods(game, actions, player)
        ply, ranks, _ = belief_rows(game, list(actions), player)
        late = np.flatnonzero(ply >= min(self.P0, self.F))
        if len(late):
            if ply[late].max() >= PERFECT_INFO_PLY:
                raise ValueError("likelihoods at perfect-information plies are not implemented")
            self._ensure([("own", int(game.start), actions[:int(t)]) for t in np.unique(ply[late])
                          if self.dl_phase(int(t))])
            pols = {}
            for i in late:
                t = int(ply[i])
                if t not in pols:
                    pols[t] = self._source(game.start, actions, t, player, player, "ours")
                lik[i] = pols[t][ranks[i], actions[t]]
        return lik
