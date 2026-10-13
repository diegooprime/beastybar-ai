"""Exact Bayes ranges over a player's hand given the public move history and a policy model.

The only hidden information is each player's hand (a k-subset of their unplayed animals while their deck lasts, k = 4;
at most C(12, 4) = 495 hands, in the fixed bbar._core.hand_rank layout), and the two players' hands are independent.
A player's range starts from the deal prior (uniform over the hands they could hold) and, at each of their
decisions, is multiplied by sigma(a_t | h) for every candidate hand h and renormalised, then pushed through the
play-and-draw transition: hand h - c + d with d uniform over the deck (core/src/policy_obs.rs, bayes_filter; checked
against brute-force enumeration of deals in core/src/policy_obs.rs and tests/test_belief.py). The opponent's
decisions do not change it.

sigma comes from a model in bbar/policy.py: a network (exact probabilities, one batched forward over every candidate
hand of every decision) or a built-in agent (estimated from sampled runs).

Zero probabilities. A hand that does not hold the played animal is eliminated (exact). For the others the model's
probability is floored at eps (default 1e-9): a model that gives the observed play (almost) zero probability - a
misspecified opponent model, an argmax policy, an unlucky sampled estimate - never wipes out the true hand. Each
decision where the model's probability of the observed action given the public history is <= eps is recorded as
'impossible' (Range.impossible lists their plies): the range is then shaped only by the exact eliminations and by the
floored likelihoods.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from bbar._core import Game, bayes_filter, possible_hands
from bbar.actions import ANIMALS

EPS = 1e-9


def start_game(start, animals=ANIMALS):
    """The public start position: every animal in hand for both players, decks empty, `start` to move."""
    return Game(int(start), [list(animals), list(animals)], [[], []])


def replay(game, actions):
    """A copy of `game` after `actions`."""
    g = game.copy()
    for a in actions:
        g.play_action(int(a))
    return g


@dataclass
class Range:
    player: int
    ply: int
    range: np.ndarray              # float64 [C(12, m)], hand_rank layout, sums to 1
    steps: list = field(default_factory=list)   # [(ply, prob of the observed action, impossible)]

    @property
    def impossible(self):
        return [ply for ply, _, imp in self.steps if imp]

    @property
    def log_likelihood(self):
        """log P(observed actions of this player | public history) under the model (floored at eps)."""
        return float(sum(np.log(max(p, EPS)) for _, p, _ in self.steps))

    def support(self):
        return np.flatnonzero(self.range)


def bayes_range(game, actions, player, likelihood, k=4, eps=EPS):
    """The exact Bayes range from per-row likelihoods (rows of bbar._core.belief_rows)."""
    r, steps = bayes_filter(game, list(actions), player, np.asarray(likelihood, dtype=np.float64), k, eps)
    return Range(player, len(actions), r, steps)


def model_range(model, game, actions, player, eps=EPS, **kw):
    """`player`'s range after `actions` (from the start position `game`) assuming they play `model` (bbar/policy.py)."""
    return bayes_range(game, actions, player, model.likelihoods(game, actions, player, **kw), eps=eps)


def uniform_range(game, actions, player, k=4):
    """Uniform over the hands `player` could hold after `actions` (no inference from their play)."""
    g = replay(game, actions)
    hs = possible_hands(g, player, k)
    unplayed = len(g.hand(player)) + len(g.deck(player))
    r = np.zeros(math.comb(12, min(k, unplayed)))
    r[hs] = 1.0 / len(hs)
    return Range(player, len(actions), r, [])
