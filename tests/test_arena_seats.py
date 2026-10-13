"""Seat and perspective checks of the arena through the Python API, with an independent Python evaluation.

In every game of a duplicate pair, the moves made from A's seat (per the "a_seat" column) must be the greedy agent's
(1-ply argmax of the static evaluation, which hidden information cannot change), and A's points must be that seat's.
"""
import pytest

core = pytest.importorskip("bbar._core")

from bbar.actions import POINTS  # noqa: E402

POS_WEIGHT = [0.6, 0.45, 0.25, 0.1]


def value(game, me):
    """core/src/eval.rs `evaluate`, reimplemented: the mover's static value."""
    s0, s1 = game.score()
    margin = (s0 - s1) if me == 0 else (s1 - s0)
    if game.is_over():
        return 1000.0 * ((margin > 0) - (margin < 0)) + margin
    queue = sum(POS_WEIGHT[i] * POINTS[a] * (1 if owner == me else -1) for i, (a, owner) in enumerate(game.queue()))
    return margin + queue


def is_greedy(game, action):
    me = game.to_move

    def after(a):
        g = game.copy()
        g.play_action(a)
        return value(g, me)

    best = max(after(a) for a in game.legal_actions())
    return after(action) >= best - 1e-4


@pytest.mark.parametrize("a,b,greedy_is_a", [("greedy", "random", True), ("random", "greedy", False)])
def test_greedy_moves_come_from_the_recorded_seat(a, b, greedy_is_a):
    seed = 13
    r = core.play_match(a, b, 40, seed=seed, threads=3)
    random_deviations = 0
    wins_by_seat = [0, 0]
    for k, moves in enumerate(r["moves"]):
        a_seat = r["a_seat"][k]
        game = core.arena_deal(seed, r["deal"][k])
        for m in moves:
            mover_is_a = game.to_move == a_seat
            if mover_is_a == greedy_is_a:
                assert is_greedy(game, m), f"game {k}: greedy is not in the seat the record says"
            elif not is_greedy(game, m):
                random_deviations += 1
            game.play_action(m)
        assert game.is_over()
        assert (game.score()[a_seat], game.score()[1 - a_seat]) == (r["a_points"][k], r["b_points"][k])
        wins_by_seat[a_seat] += r["a_points"][k] > r["b_points"][k]
    assert random_deviations > 50
    # Greedy wins most games in both seats, so points are A's in both games of every pair.
    for seat in (0, 1):
        if greedy_is_a:
            assert wins_by_seat[seat] > 25, wins_by_seat
        else:
            assert wins_by_seat[seat] < 15, wins_by_seat
