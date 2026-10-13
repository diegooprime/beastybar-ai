"""The iter949 adapter (bbar/iter949.py): move catalog mapping and hand order, with a stand-in for the network."""
import numpy as np
import pytest

core = pytest.importorskip("bbar._core")

from bbar import iter949, match  # noqa: E402
from bbar.actions import decode  # noqa: E402


class FirstLegal:
    """Stands in for iter949's Iter949: picks the smallest legal move index and records what it was shown."""

    def __init__(self):
        self.seen = []

    def pick(self, views, legal):
        self.seen += list(zip(views, legal))
        return [min(m) for m in legal]


def test_move_catalog_is_one_to_one_on_legal_moves():
    rng = np.random.default_rng(0)
    for seed in range(200):
        g = core.Game.random(seed, seed % 2)
        while not g.is_over():
            order = list(rng.permutation(g.hand(g.to_move)))
            idx = [iter949.move_index(order.index(decode(a)[0]), decode(a)[1]) for a in g.legal_actions()]
            assert len(set(idx)) == len(idx) and all(0 <= i < 124 for i in idx)
            g.play_action(int(rng.choice(g.legal_actions())))


def test_hand_order_keeps_arrival_order_across_a_game():
    agent = iter949.Iter949Agent(FirstLegal(), "iter949:test", {})
    rng = np.random.default_rng(1)
    for seed in range(20):
        agent.prepare(1)
        g, hist, last = core.Game.random(seed, 0), [], None
        while not g.is_over():
            me = g.to_move
            if me == 0:
                order = agent.hand_order(0, g, hist, 0, seed)
                assert sorted(order) == sorted(g.hand(0))
                if last is not None:
                    kept = [a for a in last[0] if a != last[1]]
                    assert order[:len(kept)] == kept, "earlier cards keep their order; drawn cards go last"
            a = int(rng.choice(g.legal_actions()))
            if me == 0:
                last = (order, decode(a)[0])
            g.play_action(a)
            hist.append(a)


def test_plays_legal_moves_in_a_match():
    fake = FirstLegal()
    agent = match.PyPlayer(lambda: iter949.Iter949Agent(fake, "iter949:test", {}))
    raw = match.play(agent, match.make_player("random"), 6, seed=3, batch=3)
    assert len(raw["deal"]) == 12 and fake.seen
    for view, legal in fake.seen:
        assert view["active"] == view["me"] and len(view["hand"]) == view["hand_counts"][view["me"]]
        assert legal == sorted(legal)
