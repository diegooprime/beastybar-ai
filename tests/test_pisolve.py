"""Python bindings of the Markov key, hand layout and perfect-information solver (core/src/python_solve.rs).

The Rust tests (core/tests/pisolve_brute.rs) check the solver against minimax on 100k+ positions; these check the
bindings against a pure-Python minimax and the public-form conventions.
"""
import itertools
import random

import pytest

core = pytest.importorskip("bbar._core")
from bbar.actions import ANIMALS  # noqa: E402

Game = core.Game


def random_game(seed, plies):
    """A real game after `plies` random moves (None if it ended), and its moves."""
    rng = random.Random(seed)
    g = Game.random(seed, seed % 2)
    moves = []
    for _ in range(plies):
        if g.is_over():
            return None, moves
        a = rng.choice(g.legal_actions())
        g.play_action(a)
        moves.append(a)
    return (None if g.is_over() else g), moves


def public_form(start, moves):
    """The public position after `moves`: every unplayed animal in hand, decks empty."""
    g = Game(start, [ANIMALS, ANIMALS], [[], []])
    for a in moves:
        g.play_action(a)
    return g


def minimax(g):
    """Value for player 0 in {0, .5, 1}, by plain minimax (exact once both decks are empty)."""
    if g.is_over():
        s0, s1 = g.score()
        return 1.0 if s0 > s1 else 0.5 if s0 == s1 else 0.0
    vals = []
    for a in g.legal_actions():
        c = g.copy()
        c.play_action(a)
        vals.append(minimax(c))
    return max(vals) if g.to_move == 0 else min(vals)


def positions(ply, n, seed=0):
    out = []
    while len(out) < n:
        g, moves = random_game(seed, ply)
        seed += 1
        if g is not None:
            out.append((g, moves, seed - 1))
    return out


def test_pi_value_equals_minimax():
    for ply in (17, 18, 19):
        for g, _, _ in positions(ply, 15, seed=1000 * ply):
            assert core.pi_value(g) == minimax(g)


def test_pi_value_needs_public_hands():
    g, _, _ = positions(15, 1)[0]
    with pytest.raises(ValueError):
        core.pi_value(g)


def test_frontier_values():
    for g, moves, seed in positions(15, 6, seed=7):
        mover = g.to_move
        assert len(g.deck(mover)) == 1 and g.deck(1 - mover) == []
        unplayed = set(g.hand(mover)) | set(g.deck(mover))
        values = core.frontier_values(g)
        public = core.frontier_values(public_form(seed % 2, moves))
        assert values == public
        hands = [tuple(h) for h, _ in values]
        assert sorted(hands, key=core.hand_rank) == hands
        assert {frozenset(h) for h in hands} == {frozenset(c) for c in itertools.combinations(unplayed, 4)}
        # The real hand's value is the real position's exact value (its last deck card is known).
        real = dict((tuple(h), v) for h, v in values)[tuple(g.hand(mover))]
        assert real == minimax(g)


def test_frontier_values_reject_hidden_opponents():
    g, _, _ = positions(13, 1)[0]
    with pytest.raises(ValueError):
        core.frontier_values(g)


def test_markov_keys():
    for ply in (4, 10, 15, 16, 18):
        for g, moves, seed in positions(ply, 5, seed=100 * ply):
            p = public_form(seed % 2, moves)
            assert core.public_key(g) == core.markov_key(p) == core.public_key(p)
            # Once both decks are empty the real position is its public form.
            assert (core.markov_key(g) == core.markov_key(p)) == (ply >= 16)
            assert 0 < core.markov_key(g) < 2**63


def test_hand_layout():
    assert core.hand_rank([]) == 0
    hands = [core.hand_unrank(4, r) for r in range(495)]
    assert len({frozenset(h) for h in hands}) == 495
    assert all(core.hand_rank(h) == r for r, h in enumerate(hands))
    with pytest.raises(ValueError):
        core.hand_unrank(4, 495)
