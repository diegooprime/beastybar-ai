"""Independent explicit-infoset tabular CFR for tiny Beasty Bar variants: the cross-check of the window solver.

The tree, its CFR and its best responses use nothing from the Rust core (only `compare` calls the window solver for
the other side of the comparison). The rules come from the pure-Python reference engine (bbar/ref.py); the game tree
is built explicitly with
  - a chance node for the deal (each player's hand: a uniform k-subset of their animals),
  - a chance node for every draw (uniform over the mover's deck),
  - perfect-recall information sets keyed by (player, own private observations = dealt hand and every own draw in
    order, public action history). Nothing is merged by "current hand": that is exactly the claim being checked.
It is solved by vanilla CFR (simultaneous updates, uniform averaging) or CFR+ (simultaneous updates, regret
matching+, linear averaging), vectorised with numpy over the explicit tree (nodes in breadth-first order, so each
depth is a contiguous range).

Exact best responses are computed infoset by infoset (all histories of an information set lie at the same depth,
checked), deepest first. Exploitability = (BR_0 + BR_1 - 1) / 2 in win probability (win 1, draw .5, loss 0).

    python -m bbar.xcfr --animals chameleon,monkey,giraffe,lion --k 2 --iters 4000 --plus-iters 2000

compares with the window solver (bbar._core.Window, leaves="terminal" from the initial position with uniform
deal ranges): the exploitability of uniform play and of a fixed hand-dependent strategy (both exact quantities),
and the game value.
"""
import argparse
import itertools
import time
import zlib
from collections import deque

import numpy as np

from bbar.actions import ANIMALS, POINTS, decode
from bbar.ref import Game as Ref

ORDER = {a: i for i, a in enumerate(ANIMALS)}


class Rules:
    """Queue dynamics and legality from the reference engine, memoised (they depend only on the queue, the mover
    and the move, or on the hand and the queue)."""

    def __init__(self):
        self._legal = {}
        self._play = {}

    @staticmethod
    def _game(mover, hand, queue):
        g = Ref(mover, [[], []], [[], []])
        g._hands[mover] = sorted(hand, key=ORDER.get)
        g._decks[1 - mover] = ["skunk"] * 5  # padding so that the game is not over (never drawn)
        g._queue = list(queue)
        return g

    def legal(self, mover, hand, queue):
        key = (mover, hand, queue)
        if key not in self._legal:
            self._legal[key] = tuple(sorted(self._game(mover, hand, queue).legal_actions()))
        return self._legal[key]

    def play(self, mover, queue, action):
        """(new queue, cards entering the bar) after `action` (the mover's draw is handled by the caller)."""
        key = (mover, queue, action)
        if key not in self._play:
            animal, params = decode(action)
            g = self._game(mover, [animal], queue)
            g.play(animal, params)
            self._play[key] = (tuple(g.queue()), tuple(g.bar()))
        return self._play[key]


class Tree:
    """The explicit game tree of a variant (breadth-first arrays)."""

    def __init__(self, animals, k, start=0):
        animals = sorted(animals, key=ORDER.get)
        assert len(set(animals)) == len(animals) and 1 <= k <= len(animals)
        self.animals, self.k, self.start = animals, k, start
        rules = Rules()
        # Per node: kind (0/1 decision of that player, 2 chance, 3 terminal), parent, depth, edge (action slot for a
        # decision parent, probability for a chance parent), infoset (decision nodes), u0 (terminals).
        kind, parent, depth, slot, prob, infoset, u0 = [], [], [], [], [], [], []
        infosets = {}
        iset_player, iset_actions, iset_depth = [], [], []
        # A frontier item: (parent, slot, prob, depth, payload); payload describes what to create.
        # State = (mover, hands, decks, queue, scores, obs, pub); hands/decks: frozensets per player.
        hands_all = [frozenset(h) for h in itertools.combinations(animals, k)]
        rest = frozenset(animals)
        todo = deque()
        kind.append(2), parent.append(-1), depth.append(0), slot.append(-1), prob.append(1.0)
        infoset.append(-1), u0.append(0.0)
        p_deal = 1.0 / len(hands_all) ** 2
        for h0 in hands_all:
            for h1 in hands_all:
                state = (start, (h0, h1), (rest - h0, rest - h1), (), (0, 0),
                         ((tuple(sorted(h0, key=ORDER.get)),), (tuple(sorted(h1, key=ORDER.get)),)), ())
                todo.append((0, -1, p_deal, 1, ("state", state)))
        while todo:
            par, sl, pr, d, (what, data) = todo.popleft()
            n = len(kind)
            parent.append(par), slot.append(sl), prob.append(pr), depth.append(d)
            if what == "draw":
                # Chance over the mover's draw; data = (state after the play, before the draw)
                mover, hands, decks, queue, scores, obs, pub = data
                kind.append(2), infoset.append(-1), u0.append(0.0)
                deck = sorted(decks[mover], key=ORDER.get)
                for c in deck:
                    nh = list(hands)
                    nd = list(decks)
                    no = list(obs)
                    nh[mover] = hands[mover] | {c}
                    nd[mover] = decks[mover] - {c}
                    no[mover] = obs[mover] + (c,)
                    todo.append((n, -1, 1.0 / len(deck), d + 1,
                                 ("state", (1 - mover, tuple(nh), tuple(nd), queue, scores, tuple(no), pub))))
                continue
            mover, hands, decks, queue, scores, obs, pub = data
            cards = sum(len(h) for h in hands) + sum(len(x) for x in decks) + len(queue)
            if cards < 5:
                kind.append(3), infoset.append(-1)
                u0.append(1.0 if scores[0] > scores[1] else 0.5 if scores[0] == scores[1] else 0.0)
                continue
            legal = rules.legal(mover, hands[mover], queue)
            assert legal, f"no legal action: {data}"
            key = (mover, obs[mover], pub)
            if key not in infosets:
                infosets[key] = len(iset_player)
                iset_player.append(mover), iset_actions.append(legal), iset_depth.append(d)
            i = infosets[key]
            assert iset_actions[i] == legal and iset_depth[i] == d, "infoset histories disagree"
            kind.append(mover), infoset.append(i), u0.append(0.0)
            for j, a in enumerate(legal):
                animal, _ = decode(a)
                nq, entered = rules.play(mover, queue, a)
                sc = list(scores)
                for an, owner in entered:
                    sc[owner] += POINTS[an]
                nh = list(hands)
                nh[mover] = hands[mover] - {animal}
                after = (mover, tuple(nh), decks, nq, tuple(sc), obs, pub + (a,))
                left = sum(len(h) for h in nh) + sum(len(x) for x in decks) + len(nq)
                if decks[mover] and left >= 5:
                    todo.append((n, j, 1.0, d + 1, ("draw", after)))
                else:
                    todo.append((n, j, 1.0, d + 1, ("state", (1 - mover,) + after[1:])))
        self.kind = np.array(kind, dtype=np.int8)
        self.parent = np.array(parent, dtype=np.int64)
        self.depth = np.array(depth, dtype=np.int64)
        self.slot = np.array(slot, dtype=np.int64)
        self.prob = np.array(prob)
        self.infoset = np.array(infoset, dtype=np.int64)
        self.u0 = np.array(u0)
        assert np.all(np.diff(self.depth) >= 0), "breadth-first order"
        self.n = len(kind)
        self.n_infosets = len(iset_player)
        self.A = max(len(a) for a in iset_actions)
        self.iset_player = np.array(iset_player, dtype=np.int64)
        self.iset_actions = iset_actions
        self.mask = np.zeros((self.n_infosets, self.A), dtype=bool)
        for i, acts in enumerate(iset_actions):
            self.mask[i, :len(acts)] = True
        # A representative history per infoset (own reach is equal across an infoset under perfect recall).
        self.rep = np.full(self.n_infosets, -1, dtype=np.int64)
        dec = np.nonzero(self.infoset >= 0)[0]
        self.rep[self.infoset[dec][::-1]] = dec[::-1]
        self.keys = {i: k for k, i in infosets.items()}
        D = int(self.depth.max())
        self.levels = [np.nonzero(self.depth == d)[0] for d in range(D + 1)]
        self.starts = [int(lv[0]) for lv in self.levels]
        self.terminal = self.kind == 3

    def uniform(self):
        return self.mask / self.mask.sum(1, keepdims=True)

    def factors(self, sigma):
        """Edge factor of every node from its parent: sigma for decision parents, the probability for chance."""
        f = self.prob.copy()
        par = self.parent
        dec = (par >= 0) & (self.kind[np.maximum(par, 0)] < 2)
        f[dec] = sigma[self.infoset[par[dec]], self.slot[dec]]
        return f

    def reaches(self, f):
        """Reach of player 0, player 1 and chance at every node."""
        pi = np.ones((3, self.n))
        for lv in self.levels[1:]:
            par = self.parent[lv]
            pi[:, lv] = pi[:, par]
            who = self.kind[par].astype(np.int64)
            pi[who, lv] *= f[lv]
        return pi

    def values(self, f):
        """Player 0's expected payoff at every node under the edge factors `f`."""
        u = np.where(self.terminal, self.u0, 0.0)
        for d in range(len(self.levels) - 1, 0, -1):
            lv = self.levels[d]
            par = self.parent[lv]
            base = self.starts[d - 1]
            n_par = len(self.levels[d - 1])
            u[base:base + n_par] += np.bincount(par - base, weights=f[lv] * u[lv], minlength=n_par)
        return u

    def best_response(self, sigma, i):
        """Exact best-response value of player `i` against `sigma` (for the other player)."""
        f = self.factors(sigma)
        pi = self.reaches(f)
        cf = pi[1 - i] * pi[2]
        v = np.where(self.terminal, self.u0 if i == 0 else 1.0 - self.u0, 0.0)
        Q = np.zeros(self.n_infosets * self.A)
        for d in range(len(self.levels) - 1, 0, -1):
            lv = self.levels[d]
            par = self.parent[lv]
            own = self.kind[par] == i
            # Others: expectation.
            other = ~own
            base = self.starts[d - 1]
            n_par = len(self.levels[d - 1])
            v[base:base + n_par] += np.bincount(par[other] - base, weights=f[lv[other]] * v[lv[other]],
                                                minlength=n_par)
            if not own.any():
                continue
            # Own: the best action per infoset, from the counterfactual values summed over its histories.
            c, p = lv[own], par[own]
            I = self.infoset[p]
            idx = I * self.A + self.slot[c]
            Q[:] = 0.0
            Q += np.bincount(idx, weights=cf[p] * v[c], minlength=Q.size)
            Qm = np.where(self.mask, Q.reshape(self.n_infosets, self.A), -np.inf)
            best = Qm.argmax(1)
            take = self.slot[c] == best[I]
            v[p[take]] = v[c[take]]
        return v[0]

    def exploitability(self, sigma):
        br0 = self.best_response(sigma, 0)
        br1 = self.best_response(sigma, 1)
        value = self.values(self.factors(sigma))[0]
        return dict(exploitability=(br0 + br1 - 1) / 2, nashconv=br0 + br1 - 1, br=(br0, br1), value=value)

    def cfr(self, iters, plus=False, log_every=0):
        """Vanilla CFR (or CFR+ with linear averaging), simultaneous updates. Returns the average strategy."""
        R = np.zeros((self.n_infosets, self.A))
        S = np.zeros_like(R)
        dec_child = np.nonzero((self.parent >= 0) & (self.kind[np.maximum(self.parent, 0)] < 2))[0]
        par = self.parent[dec_child]
        p = self.kind[par].astype(np.int64)
        I = self.infoset[par]
        idx = I * self.A + self.slot[dec_child]
        sign = np.where(p == 0, 1.0, -1.0)
        for t in range(1, iters + 1):
            pos = np.maximum(R, 0.0) * self.mask
            tot = pos.sum(1, keepdims=True)
            sigma = np.where(tot > 0, pos / np.where(tot > 0, tot, 1.0), self.uniform())
            f = self.factors(sigma)
            pi = self.reaches(f)
            u = self.values(f)
            cf = pi[1 - p, par] * pi[2, par]
            r = np.bincount(idx, weights=cf * sign * (u[dec_child] - u[par]), minlength=R.size)
            R += r.reshape(R.shape)
            if plus:
                np.maximum(R, 0.0, out=R)
            own = pi[self.iset_player, self.rep]
            S += (t if plus else 1.0) * own[:, None] * sigma
            if log_every and t % log_every == 0:
                print(f"  iter {t}: exploitability {self.exploitability(self.average(S))['exploitability']:.3e}",
                      flush=True)
        return self.average(S)

    def average(self, S):
        tot = S.sum(1, keepdims=True)
        return np.where(tot > 0, S / np.where(tot > 0, tot, 1.0), self.uniform())

    def current_hand(self, i):
        """The current hand (animal names, sorted) at infoset i: dealt hand + own draws - own plays."""
        player, obs, pub = self.keys[i]
        hand = set(obs[0]) | set(obs[1:])
        for j, a in enumerate(pub):
            if (j % 2 == 0) == (player == self.start):
                hand.discard(decode(a)[0])
        return tuple(sorted(hand, key=ORDER.get))

    def table_strategy(self, weight):
        """The strategy sigma(a | public history, current hand) ~ weight(history, hand, a) over the legal actions."""
        sig = np.zeros((self.n_infosets, self.A))
        for i in range(self.n_infosets):
            pub = self.keys[i][2]
            hand = self.current_hand(i)
            w = np.array([weight(pub, hand, a) for a in self.iset_actions[i]], dtype=float)
            sig[i, :len(w)] = w / w.sum()
        return sig


def hashed_weight(history, hand, action):
    """A fixed, deterministic weight in 1..4 of (public history, current hand, action)."""
    return 1 + zlib.crc32(repr((tuple(history), tuple(hand), int(action))).encode()) % 4


def window_profile(window, weight):
    """The strategy sigma(a | public history, current hand) ~ weight(history, hand, a) as a window profile
    (bbar._core.Window entry layout)."""
    prof = np.zeros(window.num_entries)
    for history, mover, offset, hands, actions in window.nodes():
        for i, hand in enumerate(hands):
            ws = [weight(history, hand, a) if decode(a)[0] in hand else 0.0 for a in actions]
            tot = sum(ws)
            for j, x in enumerate(ws):
                prof[offset + i * len(actions) + j] = x / tot
    return prof


def compare(animals, k, iters, plus_iters=0, start=0, quiet=False):
    """Builds the explicit tree and the window of a variant from its initial position and compares them.
    Returns a dict of the numbers."""
    from bbar._core import Game, Window

    out = dict(animals=list(animals), k=k)
    t = time.time()
    tree = Tree(animals, k, start)
    out.update(nodes=tree.n, infosets=tree.n_infosets, build_seconds=time.time() - t)
    sorted_animals = sorted(animals, key=ORDER.get)
    g = Game(start, [sorted_animals, sorted_animals], [[], []])
    w = Window(g, k=k, leaves="terminal")
    out.update(window_internal=w.sizes()["internal"], window_entries=w.num_entries)

    # 1. Uniform play: an exact quantity in both.
    xu = tree.exploitability(tree.uniform())
    wu = w.exploitability("uniform", "uniform")
    out.update(uniform_xcfr=xu["exploitability"], uniform_window=wu["exploitability"],
               uniform_value_xcfr=xu["value"], uniform_value_window=wu["value"])

    # 2. A fixed hand-dependent strategy sigma(a | public history, current hand): exact in both.
    xs = tree.exploitability(tree.table_strategy(hashed_weight))
    prof = window_profile(w, hashed_weight)
    ws = w.exploitability(prof, prof)
    out.update(fixed_xcfr=xs["exploitability"], fixed_window=ws["exploitability"],
               fixed_value_xcfr=xs["value"], fixed_value_window=ws["value"])

    # 3. Equilibrium: vanilla CFR (and CFR+) on the explicit tree vs the window solver.
    t = time.time()
    xe = tree.exploitability(tree.cfr(iters))
    out.update(vanilla_iters=iters, vanilla_exploitability=xe["exploitability"], vanilla_value=xe["value"],
               vanilla_bounds=(1 - xe["br"][1], xe["br"][0]), vanilla_seconds=time.time() - t)
    if plus_iters:
        t = time.time()
        xp = tree.exploitability(tree.cfr(plus_iters, plus=True))
        out.update(plus_iters=plus_iters, plus_exploitability=xp["exploitability"], plus_value=xp["value"],
                   plus_bounds=(1 - xp["br"][1], xp["br"][0]), plus_seconds=time.time() - t)
    r = w.solve(target=1e-10, max_iters=20000)
    out.update(window_exploitability=r["exploitability"], window_value=r["value"],
               window_bounds=(1 - r["br"][1], r["br"][0]), window_iters=r["iterations"])
    wf = Window(g, k=k, leaves="frontier")
    rf = wf.solve(target=1e-10, max_iters=20000)
    out.update(frontier_internal=wf.sizes()["internal"], frontier_value=rf["value"],
               frontier_exploitability=rf["exploitability"])
    if not quiet:
        for key, v in out.items():
            print(f"  {key:<24} {v}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--animals", default="chameleon,monkey,giraffe,lion")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--iters", type=int, default=4000, help="vanilla CFR iterations")
    ap.add_argument("--plus-iters", type=int, default=0, help="CFR+ iterations (0: skip)")
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    print(f"variant {a.animals} k={a.k} start={a.start}")
    compare(a.animals.split(","), a.k, a.iters, a.plus_iters, a.start)


if __name__ == "__main__":
    main()
