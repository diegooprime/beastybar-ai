"""Beliefs (bbar/belief.py): exact Bayes ranges over a player's hands from the public history and a policy model.

- The filter equals brute-force enumeration of every deal of a mini variant under a fixed hand-dependent policy
  (zero probabilities included), and its range is a product of independent per-player ranges.
- A zero-probability observed action: the true hand survives (floor at eps) and the decision is flagged impossible.
- Network likelihoods are the network's probabilities on the real observations (rows = every candidate hand).
"""
import itertools
import math

import numpy as np
import pytest

core = pytest.importorskip("bbar._core")
torch = pytest.importorskip("torch")

from bbar import belief, policy  # noqa: E402
from bbar._core import Game, OBS_DIM, belief_obs, belief_rows, encode_moves, hand_rank, public_key  # noqa: E402
from bbar.actions import ANIMALS, decode  # noqa: E402
from bbar.rl.net import PolicyNet  # noqa: E402

MINI = ["parrot", "kangaroo", "seal", "giraffe", "crocodile"]   # 5 animals, hands of 2
K = 2


def sigma(game, hand, action, salt):
    """A fixed policy of public information and the mover's hand: weights 0..3 (zeros included), uniform if all 0."""
    legal = [a for a in game.legal_actions() if decode(a)[0] in hand]
    if action not in legal:
        return 0.0
    key = public_key(game)
    w = {a: (hash((key, tuple(sorted(hand)), a, salt)) % 4) for a in legal}
    tot = sum(w.values())
    return 1.0 / len(legal) if tot == 0 else w[action] / tot


def play_history(rng, start, salt, plies):
    order = [list(rng.permutation(MINI)) for _ in (0, 1)]
    g = Game(start, [o[:K] for o in order], [o[K:] for o in order])
    actions = []
    for _ in range(plies):
        if g.is_over():
            break
        legal = g.legal_actions()
        p = np.array([sigma(g, g.hand(g.to_move), a, salt) for a in legal])
        a = legal[rng.choice(len(legal), p=p / p.sum())]
        actions.append(a)
        g.play_action(a)
    return actions


def brute_force(start, actions, salt):
    """Joint posterior over the final hands (H_0, H_1), enumerating every deal (orders of both players' animals)."""
    post = {}
    orders = list(itertools.permutations(MINI))
    for o0 in orders:
        for o1 in orders:
            g = Game(start, [list(o0[:K]), list(o1[:K])], [list(o0[K:]), list(o1[K:])])
            w = 1.0
            for a in actions:
                w *= sigma(g, g.hand(g.to_move), a, salt)
                if w == 0:
                    break
                g.play_action(a)
            if w > 0:
                key = (tuple(g.hand(0)), tuple(g.hand(1)))
                post[key] = post.get(key, 0.0) + w
    tot = sum(post.values())
    return {k: v / tot for k, v in post.items()}


def likelihoods(start, actions, player, salt):
    """sigma(a_t | h) for every row of belief_rows (every candidate hand of every decision), on public positions."""
    deal = belief.start_game(start, MINI)
    ply, _, masks = belief_rows(deal, actions, player, K)
    out = []
    for t, m in zip(ply, masks):
        hand = [ANIMALS[a] for a in range(12) if m >> a & 1]
        out.append(sigma(belief.replay(deal, actions[:t]), hand, actions[t], salt))
    return np.array(out)


def test_sigma_depends_only_on_public_information_and_the_hand():
    # The brute force evaluates sigma on real games, the filter on public positions: they must agree.
    rng = np.random.default_rng(0)
    for _ in range(20):
        start = int(rng.integers(2))
        order = [list(rng.permutation(MINI)) for _ in (0, 1)]
        g = Game(start, [o[:K] for o in order], [o[K:] for o in order])
        pub = belief.start_game(start, MINI)
        while not g.is_over():
            hand = g.hand(g.to_move)
            for b in pub.legal_actions():
                assert sigma(g, hand, b, 1) == sigma(pub, hand, b, 1)
                assert (sigma(g, hand, b, 1) > 0) <= (b in g.legal_actions())
            a = g.legal_actions()[0]
            g.play_action(a)
            pub.play_action(a)


@pytest.mark.parametrize("start,salt", [(0, 1), (1, 2), (0, 3), (1, 4)])
def test_filter_equals_brute_force_enumeration(start, salt):
    rng = np.random.default_rng(salt)
    actions = play_history(rng, start, salt, 6)
    joint = brute_force(start, actions, salt)
    deal = belief.start_game(start, MINI)
    ranges = []
    for p in (0, 1):
        r = belief.bayes_range(deal, actions, p, likelihoods(start, actions, p, salt), k=K, eps=0.0)
        assert abs(r.range.sum() - 1) < 1e-12 and not r.impossible
        ranges.append(r.range)
    # The joint posterior is the product of the two filtered ranges.
    for (h0, h1), prob in joint.items():
        assert abs(prob - ranges[0][hand_rank(list(h0))] * ranges[1][hand_rank(list(h1))]) < 1e-12
    for p in (0, 1):
        marginal = {}
        for hands, prob in joint.items():
            marginal[hands[p]] = marginal.get(hands[p], 0.0) + prob
        assert abs(sum(ranges[p][hand_rank(list(h))] for h in marginal) - 1) < 1e-12
        for h, prob in marginal.items():
            assert abs(ranges[p][hand_rank(list(h))] - prob) < 1e-12


def test_zero_probability_observation_is_floored_and_flagged():
    start, salt = 0, 5
    rng = np.random.default_rng(7)
    actions = play_history(rng, start, salt, 6)
    deal = belief.start_game(start, MINI)
    p = start
    lik = likelihoods(start, actions, p, salt)
    ply, _, _ = belief_rows(deal, actions, p, K)
    # A misspecified model: probability 0 for the first observed action, for every hand.
    t0 = ply[0]
    bad = np.where(ply == t0, 0.0, lik)
    r = belief.bayes_range(deal, actions, p, bad, k=K, eps=1e-9)
    assert r.impossible == [int(t0)]
    assert abs(r.range.sum() - 1) < 1e-12
    # Flooring makes that decision uninformative beyond the exact elimination of hands without the played animal:
    # the same range as a constant likelihood there.
    ref = belief.bayes_range(deal, actions, p, np.where(ply == t0, 1.0, lik), k=K, eps=1e-9)
    assert not ref.impossible
    assert np.allclose(r.range, ref.range, atol=1e-12)
    # The true hand at the end keeps positive mass under the floored filter.
    ok = belief.bayes_range(deal, actions, p, lik, k=K, eps=1e-9)
    assert not ok.impossible and ok.log_likelihood < 0


def tiny_net(seed=0, width=16, sharp=30.0):
    torch.manual_seed(seed)
    net = PolicyNet(width, 1)
    with torch.no_grad():
        net.policy.weight.mul_(sharp)
    return net


def test_network_likelihoods_are_the_networks_probabilities_on_real_observations():
    net = tiny_net()
    model = policy.NetModel(net, "cpu")
    rng = np.random.default_rng(3)
    for d in range(4):
        g0 = core.arena_deal(11, d)
        hands, decks, start = [g0.hand(0), g0.hand(1)], [g0.deck(0), g0.deck(1)], g0.start
        g = g0.copy()
        actions = []
        truth = {}
        for t in range(11):
            truth[t] = g.hand(g.to_move)
            legal = g.legal_actions()
            a = int(legal[rng.integers(len(legal))])
            actions.append(a)
            g.play_action(a)
        deal = belief.start_game(start)
        for p in (0, 1):
            ply, ranks, masks, obs, legal = belief_obs(deal, actions, p)
            assert obs.shape == (len(ply), OBS_DIM) and set(ply.tolist()) == {t for t in range(11) if (start + t) % 2 == p}
            lik = model.likelihoods(deal, actions, p)
            for t in set(ply.tolist()):
                rows = np.flatnonzero(ply == t)
                assert list(ranks[rows]) == sorted(ranks[rows])
                (i,) = [i for i in rows if ranks[i] == hand_rank(truth[t])]
                want_obs, want_legal, _ = encode_moves(start, hands, decks, actions[:t])
                assert np.array_equal(obs[i], want_obs) and np.array_equal(legal[i], want_legal)
                with torch.no_grad():
                    logp = model.agent.logp(want_obs[None], want_legal[None])
                # Equal up to float32 rounding (a batch of rows vs one row).
                assert abs(lik[i] - float(logp.double().exp()[0, actions[t]])) < 1e-6
            r = belief.bayes_range(deal, actions, p, lik)
            assert r.range[hand_rank(g.hand(p))] > 0 and abs(r.range.sum() - 1) < 1e-12
            assert set(np.flatnonzero(r.range)) <= set(core.possible_hands(belief.replay(deal, actions), p))


def test_argmax_model_flags_impossible_observations_and_keeps_the_true_hand():
    model = policy.NetModel(tiny_net(1), "cpu", argmax=True)
    rng = np.random.default_rng(4)
    g = core.arena_deal(12, 0)
    deal = belief.start_game(g.start)
    actions = []
    for _ in range(10):
        legal = g.legal_actions()
        actions.append(int(legal[rng.integers(len(legal))]))
        g.play_action(actions[-1])
    flagged = 0
    for p in (0, 1):
        r = belief.model_range(model, deal, actions, p)
        flagged += len(r.impossible)
        assert r.range[hand_rank(g.hand(p))] > 0
    assert flagged > 0, "random play must contradict an argmax model somewhere"


def test_uniform_range():
    g = core.arena_deal(1, 0)
    deal = belief.start_game(g.start)
    actions = []
    for _ in range(9):
        a = g.legal_actions()[0]
        actions.append(a)
        g.play_action(a)
    for p in (0, 1):
        r = belief.uniform_range(deal, actions, p)
        hs = core.possible_hands(belief.replay(deal, actions), p)
        assert len(r.range) == math.comb(12, 4) and np.allclose(r.range[hs], 1 / len(hs))
        assert abs(r.range.sum() - 1) < 1e-12
