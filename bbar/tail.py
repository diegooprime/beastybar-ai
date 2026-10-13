"""Exact-tail player: a base policy network early, an exactly solved window late, exact minimax at the end.

Spec (bbar.match, bbar.delta): tail:nn:PATH[,F=10][,mode=gadget|plain][,target=1e-4][,iters=10000][,algo=dcfr]
                                    [,eps=1e-9][,check_every=10][,threads=1][,workers=1][,cache=256]
The agent's name is its spec with every option that changes its play (F, mode, target, iters, and algo, eps and
check_every when not the default); threads, workers and cache change only speed and memory.

Decisions, by ply (ply k = k moves played):
  - perfect information (the mover knows every card: ply 15 for the second mover, every ply >= 16): exact minimax
    (bbar._core.pi_value on every child), uniformly random among the optimal moves;
  - ply < F: sample the base network's policy with the seat's draw for the decision (Decisions.uniforms), exactly as
    bbar.match.NetPlayer does: before F the agent replays its base network move for move;
  - the first own decision at ply >= F (ply F or F + 1): build the window below the current public state (down to the
    perfect-information frontier, bbar._core.Window) with
      our range      = Bayes over our own past decisions, assuming we played the base policy (we did);
      opponent range = Bayes over the opponent's past decisions, assuming it played the base policy
                       (bbar/belief.py; floored at eps, so off-model opponents are survivable),
    and solve it:
      mode=plain   DCFR on the window with both ranges, to exploitability `target` (or `iters` iterations);
      mode=gadget  safe re-solving (CFR-D / DeepStack gadget): fill our base-policy table on the window, compute the
                   opponent's best-response counterfactual values against it (our exact range), and re-solve with
                   the gadget (opponent prior = its Bayes range): the result is no more exploitable by the opponent's
                   best response (per opponent hand) than the base policy in this window, up to the gadget gap;
    then sample the solved strategy for our hand;
  - later decisions inside that window: the solved strategy at the current node for our current hand (sampled with
    the seat's draw: never argmax outside perfect-information positions).

The agent sees only its own information: the view game (own hand true, opponent hand and deck orders randomized),
the public history, the seat's draw and a private seed per decision (bbar.match.PyPlayer; minimax ties use the
private seed). The solved window depends only on the public history up to its root, so solutions are cached by that
history, and a re-solve gives the same result: the agent's play is a deterministic function of (public history, own
hand, draw, private seed), whatever its cache holds (tests/test_tail.py replays matches with a fresh agent per call).

Cache. In play, a window serves one game from its root (ply F or F + 1) to the agent's last decision before the
perfect-information frontier (ply 13 or 14), after which it is dropped; the LRU bound `cache` caps memory (about
12 MB per F=10 window). A match keeps one window per game in play per instance (bbar.match: one game per deal of
--batch), so `cache` must be at least --batch, or windows are evicted while still in use and solved again: same moves,
more compute (prepare() warns; summary() counts these re-solves). bbar.delta reuses its self-play windows for the
E1 evaluation and releases each game's windows when done (release()).

Every solve is logged (TailAgent.log): window size, belief/table/solve times, iterations, final gap and whether it
re-solved an evicted window.
"""
import functools
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from bbar import belief
from bbar._core import Window, WindowTree, hand_rank, pi_value
from bbar.policy import NN, NetModel

PERFECT_INFO_PLY = 15   # the second mover knows everything from here; everybody from 16
DEFAULT_CACHE = 256     # solved windows kept (bbar.match keeps one per game in play: --batch <= this)
CHECK_EVERY = 10


def parse_spec(body):
    """'nn:PATH,F=10,mode=gadget,...' (after 'tail:') -> (base path, options dict)."""
    parts = body.split(",")
    base = parts[0]
    if not base.startswith(NN) or not base[len(NN):]:
        raise ValueError(f"tail:{body!r}: the base must be a network, tail:nn:PATH[,key=value...]")
    opts = dict(F=10, mode="gadget", target=1e-4, iters=10_000, algo="dcfr", eps=belief.EPS, check_every=CHECK_EVERY,
                threads=1, workers=1, cache=DEFAULT_CACHE)
    casts = dict(F=int, mode=str, target=float, iters=int, algo=str, eps=float, check_every=int, threads=int,
                 workers=int, cache=int)
    seen = set()
    for part in parts[1:]:
        if "=" not in part:
            raise ValueError(f"tail:{body!r}: expected key=value, got {part!r}")
        k, v = (x.strip() for x in part.split("=", 1))
        if k not in casts:
            raise ValueError(f"tail:{body!r}: unknown key {k!r} (allowed: {', '.join(casts)})")
        if k in seen:
            raise ValueError(f"tail:{body!r}: repeated key {k!r}")
        seen.add(k)
        opts[k] = casts[k](v)
    if opts["mode"] not in ("gadget", "plain"):
        raise ValueError(f"tail:{body!r}: mode must be gadget or plain")
    if not 0 <= opts["F"] <= 24:
        raise ValueError(f"tail:{body!r}: F must be in 0..24")
    if opts["cache"] < 0 or opts["check_every"] < 1:
        raise ValueError(f"tail:{body!r}: cache must be at least 0 (0: no window kept between calls), check_every "
                         f"at least 1")
    return base[len(NN):], opts


def num(x):
    """A float as short as '{:g}' when that is exact, else its exact repr (names must tell values apart)."""
    short = f"{x:g}"
    return short if float(short) == x else repr(float(x))


def agent_name(base_name, F, mode, target, iters, algo="dcfr", eps=belief.EPS, check_every=CHECK_EVERY):
    """The spec of everything that changes the agent's play (threads, workers and cache do not)."""
    name = f"tail:{base_name},F={F},mode={mode},target={num(target)},iters={iters}"
    if algo != "dcfr":
        name += f",algo={algo}"
    if eps != belief.EPS:
        name += f",eps={num(eps)}"
    if check_every != CHECK_EVERY:
        name += f",check_every={check_every}"
    return name


def cache_key(start, root, seat):
    return int(start), tuple(int(a) for a in root), int(seat)


def first_root(start, seat, F):
    """The first ply >= F at which `seat` moves (the starting player moves at even plies)."""
    return F if (start + F) % 2 == seat else F + 1


def perfect_info(game):
    """Whether the player to move knows every card: own deck at most one card (known by elimination), opponent's
    deck empty (their hand is then public)."""
    p = game.to_move
    return len(game.deck(p)) <= 1 and len(game.deck(1 - p)) == 0


def minimax_values(game):
    """Exact value for the mover of every legal action at a perfect-information position: {action: value}."""
    p = game.to_move
    out = {}
    for a in game.legal_actions():
        c = game.copy()
        c.play_action(a)
        if c.is_over():
            s = c.score()
            v0 = 1.0 if s[0] > s[1] else (0.5 if s[0] == s[1] else 0.0)
        else:
            v0 = pi_value(c)
        out[a] = v0 if p == 0 else 1.0 - v0
    return out


def minimax_choice(game, rng):
    vals = minimax_values(game)
    best = max(vals.values())
    ties = [a for a, v in vals.items() if v == best]
    return ties[int(rng.integers(len(ties)))]


def sample(probs, u):
    """Inverse-CDF sample of one action from probabilities over 36 actions (bbar.match.sample_inverse_cdf)."""
    from bbar.match import sample_inverse_cdf
    return int(sample_inverse_cdf(np.asarray(probs, dtype=np.float64)[None], np.array([u]))[0])


def last_window_decision(ply):
    """Whether the mover's decision at `ply` is its last before the perfect-information frontier (its next one, at
    ply + 2, is perfect information): the window it was played from is no longer needed."""
    return ply + 2 >= PERFECT_INFO_PLY


@dataclass
class Solution:
    """A solved window rooted at public history `root` (for `seat`'s decisions)."""
    root: tuple
    seat: int
    window: Window
    deal: object          # public start Game
    info: dict
    _tree: object = None
    _profile: object = None

    @property
    def tree(self):
        if self._tree is None:
            self._tree = WindowTree(self.deal, list(self.root))
        return self._tree

    @property
    def profile(self):
        """The solved (average) profile, both players' entries, Window layout."""
        if self._profile is None:
            self._profile = self.window.profile_array()
        return self._profile

    def policy(self, path, hand):
        """float64 [36]: the solved strategy at the node `path` below the root, for hand slot `hand`."""
        return self.window.policy(list(path))[hand]


class TailAgent:
    needs_obs = True

    def __init__(self, base, F=10, mode="gadget", target=1e-4, iters=10_000, algo="dcfr", eps=belief.EPS,
                 check_every=CHECK_EVERY, threads=1, workers=1, cache=DEFAULT_CACHE, device="cpu", name=None):
        self.model = base if isinstance(base, NetModel) else NetModel(base, device)
        self.F, self.mode, self.target, self.iters, self.algo = F, mode, target, iters, algo
        self.eps, self.check_every = eps, check_every
        self.threads, self.workers, self.cache_size = threads, workers, cache
        self.checkpoints = dict(self.model.checkpoints)
        self.name = name or agent_name(self.model.name, F, mode, target, iters, algo, eps, check_every)
        self._cache = OrderedDict()
        self._evicted = set()              # keys the LRU bound dropped (solving one again is a re-solve)
        self.drop_finished = True          # drop a window after our last decision in it (bbar.delta: off)
        self.log = []                      # one dict per solve
        self.counts = {"base": 0, "window": 0, "minimax": 0, "fallback": 0}
        self.resolves = 0
        self._warned = False

    @classmethod
    def factory(cls, body, device="cpu"):
        """A zero-argument callable making fresh agents for the spec body 'nn:PATH,...' that share one loaded network
        (bbar.match.PyPlayer)."""
        path, opts = parse_spec(body)
        return functools.partial(cls, NetModel(path, device), device=device, **opts)

    @classmethod
    def from_spec(cls, body, device="cpu"):
        return cls.factory(body, device)()

    def prepare(self, live):
        """bbar.match: up to `live` of this agent's games are in play at once, each holding one solved window."""
        if live > self.cache_size:
            warnings.warn(f"{self.name}: {live} games in play at once but cache={self.cache_size}: their windows will "
                          f"be evicted while in use and solved again (same moves, more compute); play at most "
                          f"{self.cache_size} deals at a time (--batch) or raise cache= (about 12 MB per F=10 window)",
                          stacklevel=2)
            self._warned = True

    # ---- solving ------------------------------------------------------------------------------------------------

    def ranges(self, deal, actions):
        """Both players' Bayes ranges after `actions`, assuming both play the base policy."""
        return [belief.model_range(self.model, deal, actions, p, eps=self.eps) for p in (0, 1)]

    def solution(self, start, root, seat):
        """The solved window at public history `root` (a tuple of actions) for `seat` (cached)."""
        sol = self._get(cache_key(start, root, seat))
        self._trim()
        return sol

    def _get(self, key):
        sol = self._cache.get(key)
        if sol is not None:
            self._cache.move_to_end(key)
            return sol
        return self._store(key, self._solve(*key))

    def _store(self, key, sol):
        """Caches a fresh solution and logs it (in the calling thread, in a deterministic order)."""
        if key in self._evicted:
            self._evicted.discard(key)
            sol.info["resolve"] = True
            self.resolves += 1
            if not self._warned:
                self._warned = True
                warnings.warn(f"{self.name}: re-solving a window the cache (cache={self.cache_size}) evicted while "
                              f"still in use: same moves, more compute; raise cache= or play fewer games at once",
                              stacklevel=3)
        self._cache[key] = sol
        self.log.append(sol.info)
        return sol

    def _trim(self):
        while len(self._cache) > self.cache_size:
            key, _ = self._cache.popitem(last=False)
            self._evicted.add(key)

    def release(self, start, prefix):
        """Forgets every cached window rooted below the public history `prefix` from start seat `start` (bbar.delta,
        once a position is evaluated): not counted as evictions."""
        prefix = tuple(int(a) for a in prefix)
        k = len(prefix)
        for key in [key for key in self._cache if key[0] == start and key[1][:k] == prefix]:
            del self._cache[key]
        self._evicted = {key for key in self._evicted if not (key[0] == start and key[1][:k] == prefix)}

    def _solve(self, start, root, seat):
        """Solves the window (no shared state: runs in worker threads)."""
        t0 = time.time()
        deal = belief.start_game(start)
        rs = self.ranges(deal, root)
        t1 = time.time()
        game = belief.replay(deal, root)
        w = Window(game, rs[0].range, rs[1].range, threads=self.threads)
        sz = w.sizes()
        t2 = time.time()
        info = dict(ply=len(root), seat=seat, mode=self.mode, internal=sz["internal"], entries=sz["entries"],
                    hands=list(sz["hands"]), impossible=[rs[seat].impossible, rs[1 - seat].impossible],
                    belief_s=t1 - t0, build_s=t2 - t1, table_s=0.0, resolve=False)
        sol = Solution(root, seat, w, deal, info)
        if sz["internal"] == 0:
            info.update(solve_s=0.0, iterations=0, gap=0.0)
        elif self.mode == "plain":
            r = w.solve(target=self.target, max_iters=self.iters, check_every=self.check_every, algo=self.algo)
            info.update(solve_s=r["seconds"], iterations=r["iterations"], gap=r["exploitability"])
        else:
            table = self.model.window_table(sol.tree, seat)
            t3 = time.time()
            cfvs = w.br_values(1 - seat, table)
            r = w.resolve(seat, cfvs, target=self.target, max_iters=self.iters, check_every=self.check_every,
                          algo=self.algo, prior=rs[1 - seat].range)
            info.update(table_s=t3 - t2, solve_s=r["seconds"], iterations=r["iterations"], gap=r["exploitability"],
                        max_violation=r["max_violation"])
        info["seconds"] = time.time() - t0
        return sol

    # ---- acting -------------------------------------------------------------------------------------------------

    def act(self, batch):
        """int64 [k]: one action per decision of `batch` (bbar.match.Decisions: view games, histories, the seat's
        draws, private seeds, obs, legal masks)."""
        k = len(batch.games)
        out = np.full(k, -1, dtype=np.int64)
        u = np.asarray(batch.uniforms, dtype=np.float64)
        base, window = [], []
        for j, (g, hist) in enumerate(zip(batch.games, batch.histories)):
            if perfect_info(g):
                out[j] = minimax_choice(g, np.random.default_rng(int(batch.seeds[j])))
                self.counts["minimax"] += 1
            elif len(hist) < self.F:
                base.append(j)
            else:
                window.append(j)
        if base:
            out[base] = self.model.sample(batch.obs[base], batch.legal[base], u[base])
            self.counts["base"] += len(base)
        if window:
            roots = []
            for j in window:
                g, hist = batch.games[j], batch.histories[j]
                r = first_root(g.start, g.to_move, self.F)
                roots.append(cache_key(g.start, hist[:r], g.to_move))
            todo = [key for key in dict.fromkeys(roots) if key not in self._cache]
            if self.workers > 1 and len(todo) > 1:
                with ThreadPoolExecutor(self.workers) as ex:
                    solved = list(ex.map(lambda key: self._solve(*key), todo))
                for key, sol in zip(todo, solved):
                    self._store(key, sol)
            finished = []
            for j, key in zip(window, roots):
                g, hist = batch.games[j], batch.histories[j]
                sol = self._get(key)
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
                self._cache.pop(key, None)
            self._trim()
        return out

    @classmethod
    def summarize(cls, agents):
        """Statistics of one or more agents (bbar.match.PyPlayer instances): decisions by kind; solves, of which
        re-solves of evicted windows; and over the distinct windows solved, mean / max of sizes, times and gaps."""
        counts = {k: sum(a.counts[k] for a in agents) for k in agents[0].counts}
        log = [x for a in agents for x in a.log]
        first = [x for x in log if not x["resolve"]]
        out = {"decisions": counts, "instances": len(agents), "solves": len(log),
               "resolves": sum(a.resolves for a in agents), "windows": len(first)}
        for key in ("internal", "entries", "belief_s", "table_s", "solve_s", "seconds", "iterations", "gap"):
            xs = [x[key] for x in first if key in x]
            if xs:
                out[key] = {"mean": float(np.mean(xs)), "max": float(np.max(xs))}
        out["impossible_opponent_decisions"] = sum(len(x["impossible"][1]) for x in first)
        return out

    def summary(self):
        return self.summarize([self])

    # ---- the agent's behaviour as tables (bbar.delta) -------------------------------------------------------------

    def window_table(self, tree, seat, deal, actions):
        """`seat`'s part of this agent's strategy table on the window `tree` rooted after `actions` (from the start
        position `deal`): base network before its first own decision at ply >= F, its solved windows from there."""
        T = len(actions)
        start = deal.start
        r = first_root(start, seat, self.F)
        movers = tree.entry_movers()
        mine = movers == seat
        if r <= T:
            sol = self.solution(start, tuple(actions[:r]), seat)
            n = sol.tree.find(list(actions[r:]))
            if n is None:
                raise ValueError(f"window at ply {T} is not inside the solved window at ply {r}")
            node = sol.tree.node(n)
            prof = sol.profile[node["entry"]:node["entry_end"]]
            assert len(prof) == tree.num_entries
            return np.where(mine, prof, 0.0)
        table = self.model.window_table(tree, seat)
        for n in tree.nodes_at(r - T, mover=seat):
            node = tree.node(n)
            sol = self.solution(start, tuple(actions) + tuple(node["path"]), seat)
            a, b = node["entry"], node["entry_end"]
            assert b - a == sol.window.num_entries
            table[a:b] = np.where(mine[a:b], sol.profile, table[a:b])
        return table

    def solved_root(self, deal, actions, seat):
        """(Window, WindowTree, seat's solved table) of the window `seat` plays from after `actions` (from the start
        position `deal`) if its root is at or before ply len(actions), else None (bbar.delta safety margins)."""
        r = first_root(deal.start, seat, self.F)
        if r > len(actions):
            return None
        sol = self.solution(deal.start, tuple(actions[:r]), seat)
        return sol.window, sol.tree, np.where(sol.tree.entry_movers() == seat, sol.profile, 0.0)

    def likelihoods(self, game, actions, player):
        """Per-row likelihoods (rows of bbar._core.belief_rows) of `player`'s actions under this agent: the base
        network before its first own decision at ply >= F, its solved window from there."""
        from bbar._core import belief_rows
        lik = self.model.likelihoods(game, actions, player)
        ply, ranks, _ = belief_rows(game, list(actions), player)
        r = first_root(game.start, player, self.F)
        late = np.flatnonzero(ply >= r)
        if len(late):
            if ply[late].max() >= PERFECT_INFO_PLY:
                raise ValueError("likelihoods at perfect-information plies are not implemented")
            sol = self.solution(game.start, tuple(actions[:r]), player)
            pols = {}
            for i in late:
                t = int(ply[i])
                if t not in pols:
                    pols[t] = sol.window.policy(list(actions[r:t]))
                lik[i] = pols[t][ranks[i], actions[t]]
        return lik
