"""PIMC-gap experiment on exactly solvable late-game subgames.

Root: a public state at ply S (default 13) taken from a golden game, with uniform prior ranges over each player's
possible hands (4-subsets of their unplayed animals). The subgame runs to ply 17, after which all hands are public and
the game is deterministic; leaves are solved exactly by alpha-beta (win 1 / draw .5 / loss 0 for seat 0).

Infoset of a player = (public node, current hand): lossless here because draws depend only on the hand
(deck = unplayed - hand), terminal payoffs depend only on public info, and hands of the two players are independent.

Computes:
  - CFR+ (vector form, private draws as sparse range transitions) and its exact exploitability in this subgame
  - PIMC (uniform beliefs, determinized expectimax with chance draws) and its exact exploitability
  - PIMC with Bayesian beliefs from the CFR equilibrium (fixes non-locality only)
Exploitability = (BR0 + BR1 - 1) / 2 in win-probability units, averaged over the two seats.
"""
import itertools
import sys
import time

import numpy as np

from bbar._core import Game
from bbar.golden import load
from bbar.actions import ANIMALS, decode

ALL = list(ANIMALS)
AID = {a: i for i, a in enumerate(ANIMALS)}


def replay_public(g, upto):
    G = Game(g["start"], [ALL, ALL], [[], []])
    for s in g["steps"][:upto]:
        a, p = s["move"]
        G.play(a, list(p))
    return G


def pkey(G):
    return (G.to_move, tuple(G.queue()), tuple(sorted(G.bar())), tuple(sorted(G.thats_it())))


def unplayed(G, p):
    return frozenset(AID[a] for a in G.hand(p))  # all-cards game: hand == unplayed


TT = {}


def u0(G):
    s = G.score()
    return 1.0 if s[0] > s[1] else (0.5 if s[0] == s[1] else 0.0)


def tail(G, alpha=-1.0, beta=2.0):
    """Exact perfect-information value (seat-0 win prob) by alpha-beta with a transposition table."""
    if G.is_over():
        return u0(G)
    k = pkey(G)
    hit = TT.get(k)
    if hit is not None:
        return hit
    a0, b0 = alpha, beta
    maxing = G.to_move == 0
    best = -1.0 if maxing else 2.0
    for a in G.legal_actions():
        y = G.copy(); y.play_action(a)
        v = tail(y, alpha, beta)
        if maxing:
            if v > best: best = v
            if best > alpha: alpha = best
        else:
            if v < best: best = v
            if best < beta: beta = best
        if alpha >= beta:
            break
    if a0 < best < b0 or (best <= a0 and not maxing) or (best >= b0 and maxing):
        pass
    # store only exact values (window not cut)
    if a0 < best < b0:
        TT[k] = best
    return best


def tail_exact(G):
    return tail(G, -1.0, 2.0)


class Node:
    __slots__ = ("G", "ply", "p", "U", "hands", "hidx", "kids", "leaf_u")


def hands_of(U):
    U = sorted(U)
    if len(U) <= 4:
        return [frozenset(U)]
    return [frozenset(c) for c in itertools.combinations(U, 4)]


def build(G, ply, end_ply):
    n = Node()
    n.G, n.ply, n.p = G, ply, G.to_move
    n.U = (unplayed(G, 0), unplayed(G, 1))
    n.hands = (hands_of(n.U[0]), hands_of(n.U[1]))
    n.hidx = tuple({h: i for i, h in enumerate(n.hands[q])} for q in (0, 1))
    n.kids = []
    n.leaf_u = None
    if ply == end_ply or G.is_over():
        n.leaf_u = tail_exact(G)
        return n
    p = n.p
    for a in G.legal_actions():
        card = AID[decode(a)[0]]
        y = G.copy(); y.play_action(a)
        child = build(y, ply + 1, end_ply)
        # transition matrix T[h, h'] for the actor's hand, and legality mask over actor hands
        H, Hc = n.hands[p], child.hands[p]
        T = np.zeros((len(H), len(Hc)))
        legal = np.zeros(len(H), dtype=bool)
        for i, h in enumerate(H):
            if card not in h:
                continue
            legal[i] = True
            deck = n.U[p] - h
            rest = h - {card}
            if deck:
                for d in deck:
                    T[i, child.hidx[p][rest | {d}]] += 1.0 / len(deck)
            else:
                T[i, child.hidx[p][rest]] = 1.0
        n.kids.append((a, child, T, legal))
    return n


def nodes(root):
    out, st = [], [root]
    while st:
        n = st.pop(); out.append(n)
        for _, c, _, _ in n.kids:
            st.append(c)
    return out


# ---------- strategy evaluation ----------

def values(n, sigma, reach, br=None):
    """Bottom-up counterfactual values for both seats.
    sigma[id(n)] -> array [hands_of_actor, n_actions] (behavior strategy).
    reach: dict id(n) -> (reach0, reach1) top-down.
    br: None, or seat q that best-responds (max over legal actions at q's nodes instead of sigma).
    Returns (v0, v1) arrays over each seat's hands at n."""
    if n.leaf_u is not None:
        r0, r1 = reach[id(n)]
        return n.leaf_u * r1.sum() * np.ones(len(n.hands[0])), (1 - n.leaf_u) * r0.sum() * np.ones(len(n.hands[1]))
    p = n.p
    s = sigma[id(n)]
    vp = None
    vo = np.zeros(len(n.hands[1 - p]))
    qs = []
    for j, (a, c, T, legal) in enumerate(n.kids):
        cv = values(c, sigma, reach, br)
        q = T @ cv[p]
        qs.append(np.where(legal, q, -np.inf))
        vo += cv[1 - p]
    Q = np.stack(qs, 1)
    if br == p:
        vp = Q.max(1)
    else:
        vp = (np.where(np.isfinite(Q), Q, 0.0) * s).sum(1)
    return (vp, vo) if p == 0 else (vo, vp)


def top_down(root, sigma, r0, r1):
    reach = {}
    st = [(root, r0, r1)]
    while st:
        n, a0, a1 = st.pop()
        reach[id(n)] = (a0, a1)
        if n.leaf_u is not None:
            continue
        s = sigma[id(n)]
        for j, (a, c, T, legal) in enumerate(n.kids):
            if n.p == 0:
                st.append((c, T.T @ (a0 * s[:, j]), a1))
            else:
                st.append((c, a0, T.T @ (a1 * s[:, j])))
    return reach


def exploitability(root, sigma, r0, r1):
    reach = top_down(root, sigma, r0, r1)
    br0 = values(root, sigma, reach, br=0)[0] @ r0
    br1 = values(root, sigma, reach, br=1)[1] @ r1
    ev0 = values(root, sigma, reach)[0] @ r0
    return (br0 + br1 - 1) / 2, ev0, br0, 1 - br1


def legal_mask(n):
    return np.stack([lg for _, _, _, lg in n.kids], 1)


def uniform_sigma(all_nodes):
    sig = {}
    for n in all_nodes:
        if n.leaf_u is None:
            m = legal_mask(n).astype(float)
            sig[id(n)] = m / m.sum(1, keepdims=True)
    return sig


def cfr_plus(root, all_nodes, r0, r1, iters):
    regret = {id(n): np.zeros((len(n.hands[n.p]), len(n.kids))) for n in all_nodes if n.leaf_u is None}
    avg = {k: np.zeros_like(v) for k, v in regret.items()}
    masks = {id(n): legal_mask(n) for n in all_nodes if n.leaf_u is None}
    sigma = uniform_sigma(all_nodes)
    for t in range(1, iters + 1):
        reach = top_down(root, sigma, r0, r1)

        def rec(n):
            if n.leaf_u is not None:
                a0, a1 = reach[id(n)]
                return n.leaf_u * a1.sum() * np.ones(len(n.hands[0])), (1 - n.leaf_u) * a0.sum() * np.ones(len(n.hands[1]))
            p = n.p
            s = sigma[id(n)]
            vo = np.zeros(len(n.hands[1 - p]))
            qs = []
            for j, (a, c, T, legal) in enumerate(n.kids):
                cv = rec(c)
                qs.append(T @ cv[p])
                vo += cv[1 - p]
            Q = np.stack(qs, 1)
            m = masks[id(n)]
            vp = (Q * s).sum(1)
            R = regret[id(n)]
            R += np.where(m, Q - vp[:, None], 0.0)
            np.maximum(R, 0, out=R)  # CFR+
            avg[id(n)] += t * reach[id(n)][p][:, None] * s  # linear averaging
            return (vp, vo) if p == 0 else (vo, vp)

        rec(root)
        for k, R in regret.items():
            pos = R
            tot = pos.sum(1, keepdims=True)
            m = masks[k]
            sigma[k] = np.where(tot > 0, pos / np.where(tot > 0, tot, 1), m / m.sum(1, keepdims=True))
    out = {}
    for k, A in avg.items():
        tot = A.sum(1, keepdims=True)
        m = masks[k]
        out[k] = np.where(tot > 0, A / np.where(tot > 0, tot, 1), m / m.sum(1, keepdims=True))
    return out


# ---------- PIMC ----------

def det_value(n, h0, h1, memo):
    """Perfect-information (both hands known, draws are chance) value for seat 0."""
    if n.leaf_u is not None:
        return n.leaf_u
    key = (id(n), h0, h1)
    if key in memo:
        return memo[key]
    p = n.p
    hp = h0 if p == 0 else h1
    i = n.hidx[p][hp]
    best = -1.0 if p == 0 else 2.0
    for a, c, T, legal in n.kids:
        if not legal[i]:
            continue
        ev = 0.0
        for jn in np.nonzero(T[i])[0]:
            hn = c.hands[p][jn]
            ev += T[i, jn] * (det_value(c, hn, h1, memo) if p == 0 else det_value(c, h0, hn, memo))
        best = max(best, ev) if p == 0 else min(best, ev)
    memo[key] = best
    return best


def pimc_sigma(all_nodes, belief_fn):
    """belief_fn(n) -> weights over opponent hands at n. Deterministic argmax of belief-averaged determinized value."""
    memo = {}
    sig = {}
    for n in all_nodes:
        if n.leaf_u is not None:
            continue
        p = n.p
        b = belief_fn(n)
        S = np.zeros((len(n.hands[p]), len(n.kids)))
        for i, h in enumerate(n.hands[p]):
            best_j, best_v = None, None
            for j, (a, c, T, legal) in enumerate(n.kids):
                if not legal[i]:
                    continue
                v = 0.0
                for jn in np.nonzero(T[i])[0]:
                    hn = c.hands[p][jn]
                    for k, ho in enumerate(n.hands[1 - p]):
                        if b[k] == 0:
                            continue
                        val = det_value(c, hn, ho, memo) if p == 0 else det_value(c, ho, hn, memo)
                        v += T[i, jn] * b[k] * (val if p == 0 else 1 - val)
                if best_v is None or v > best_v + 1e-12:
                    best_j, best_v = j, v
            S[i, best_j] = 1.0
        sig[id(n)] = S
    return sig


def main():
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    n_pos = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    games = load()
    rows = []
    balanced = len(sys.argv) > 4
    skip = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    done = 0
    for gi, g in enumerate(games[skip:], start=skip):
        if done >= n_pos:
            break
        t0 = time.time()
        G = replay_public(g, S - 1)
        root = build(G, S, 17)
        alln = nodes(root)
        r0 = np.ones(len(root.hands[0])) / len(root.hands[0])
        r1 = np.ones(len(root.hands[1])) / len(root.hands[1])
        sig_cfr = cfr_plus(root, alln, r0, r1, iters)
        e_cfr, v_cfr, _, _ = exploitability(root, sig_cfr, r0, r1)
        if balanced and not (0.1 < v_cfr < 0.9):
            continue
        done += 1
        sig_unif = uniform_sigma(alln)
        e_unif = exploitability(root, sig_unif, r0, r1)[0]
        # PIMC with uniform (public-consistent) beliefs
        sig_pimc = pimc_sigma(alln, lambda n: np.ones(len(n.hands[1 - n.p])) / len(n.hands[1 - n.p]))
        e_pimc, v_pimc, _, _ = exploitability(root, sig_pimc, r0, r1)
        # PIMC with Bayesian beliefs from the equilibrium (opponent reach under CFR average strategy)
        reach = top_down(root, sig_cfr, r0, r1)

        def bayes(n):
            r = reach[id(n)][1 - n.p]
            return r / r.sum() if r.sum() > 0 else np.ones(len(r)) / len(r)
        sig_pimc_b = pimc_sigma(alln, bayes)
        e_pimc_b = exploitability(root, sig_pimc_b, r0, r1)[0]
        dec = [n for n in alln if n.leaf_u is None]
        rows.append((e_cfr, e_pimc, e_pimc_b, e_unif))
        print(f"pos {gi}: ply {S} internal {len(dec)} leaves {len(alln)-len(dec)} ranges {len(r0)}x{len(r1)} | "
              f"value {v_cfr:.3f} | expl CFR+{iters} {e_cfr:.4f}  PIMC {e_pimc:.4f}  PIMC-bayes {e_pimc_b:.4f}  "
              f"uniform {e_unif:.4f}  [{time.time()-t0:.1f}s]", flush=True)
    R = np.array(rows)
    print("\nmean exploitability (win-prob units, avg of seats): CFR+ %.4f  PIMC %.4f  PIMC-bayes %.4f  uniform %.4f" % tuple(R.mean(0)))
    print("max: CFR+ %.4f  PIMC %.4f  PIMC-bayes %.4f" % tuple(R.max(0)[:3]))


if __name__ == "__main__":
    main()
