"""Unit tests of the subtle rules in the reference engine. The golden games are the full test (test_golden.py)."""
import random

import pytest

from bbar.actions import ANIMALS, encode
from bbar.ref import Game


def position(queue, hand, to_move=0):
    """A game with the given queue (front first) and the mover's hand; every other card is in a deck."""
    used = set(queue) | {(a, to_move) for a in hand}
    hands = [[], []]
    hands[to_move] = list(hand)
    game = Game(to_move, hands, [[a for a in ANIMALS if (a, p) not in used] for p in (0, 1)])
    game._queue = list(queue)
    return game


def state(game):
    return (game.to_move, game.queue(), game.bar(), game.thats_it(),
            [game.hand(p) for p in (0, 1)], [game.deck(p) for p in (0, 1)])


def after(queue, move):
    """The game after the player to move plays `move` onto `queue`."""
    animal, params = move
    game = position(queue, [animal])
    game.play(animal, params)
    return game


# ---- legal moves ----------------------------------------------------------------------------------

def test_empty_queue_parrot_and_chameleon_illegal_kangaroo_without_hop():
    game = position([], ["parrot", "chameleon", "kangaroo", "zebra"])
    assert game.legal_moves() == [("kangaroo", ()), ("zebra", ())]


def test_chameleon_needs_a_non_chameleon_to_copy():
    game = position([("chameleon", 1)], ["chameleon", "parrot", "kangaroo"])
    assert game.legal_moves() == [("parrot", (0,)), ("kangaroo", (1,))]


def test_chameleon_takes_copied_parameter():
    game = position([("parrot", 1), ("kangaroo", 1)], ["chameleon"])
    assert game.legal_moves() == [("chameleon", (0, 0)), ("chameleon", (0, 1)),
                                  ("chameleon", (1, 1)), ("chameleon", (1, 2))]


def test_legal_actions_match_moves_and_illegal_move_raises():
    game = position([("seal", 1), ("zebra", 1)], ["parrot", "kangaroo", "chameleon", "lion"])
    assert game.legal_actions() == [encode(a, p) for a, p in game.legal_moves()]
    assert game.legal_actions() == sorted(game.legal_actions())
    with pytest.raises(ValueError):
        game.play("parrot", (2,))
    with pytest.raises(ValueError):
        game.play("hippo", ())


# ---- actions --------------------------------------------------------------------------------------

def test_chameleon_copying_lion_is_bounced():
    game = after([("lion", 1), ("seal", 0)], ("chameleon", (0,)))
    assert game.queue() == [("lion", 1), ("seal", 0)]
    assert game.thats_it() == [("chameleon", 0)]


def test_chameleon_copying_hippo_is_strength_5_again_for_recurring():
    # Golden game 0, step 3: it passes the crocodile as a hippo, then the crocodile bounces it.
    game = after([("hippo", 1), ("hippo", 0), ("crocodile", 0)], ("chameleon", (0,)))
    assert game.queue() == [("hippo", 1), ("hippo", 0), ("crocodile", 0)]
    assert game.thats_it() == [("chameleon", 0)]


def test_chameleon_copying_monkey_goes_front_other_monkeys_reversed():
    game = after([("monkey", 1), ("monkey", 0), ("crocodile", 1), ("seal", 1)], ("chameleon", (0,)))
    assert game.thats_it() == [("crocodile", 1)]
    assert game.queue() == [("chameleon", 0), ("monkey", 0), ("monkey", 1), ("seal", 1)]


def test_monkey_bounces_hippos_and_crocodiles_front_to_back():
    game = after([("hippo", 1), ("monkey", 1), ("crocodile", 1)], ("monkey", ()))
    assert game.thats_it() == [("hippo", 1), ("crocodile", 1)]
    assert game.queue() == [("monkey", 0), ("monkey", 1)]


def test_chameleon_copying_skunk_is_not_bounced():
    game = after([("skunk", 1), ("lion", 1), ("hippo", 0), ("zebra", 1)], ("chameleon", (0,)))
    assert game.thats_it() == [("lion", 1), ("hippo", 0)]
    assert game.queue() == [("skunk", 1), ("zebra", 1), ("chameleon", 0)]


def test_skunk_bounces_two_highest_strengths_never_skunks():
    game = after([("seal", 1), ("skunk", 1), ("zebra", 0), ("seal", 0)], ("skunk", ()))
    assert game.thats_it() == [("seal", 1), ("zebra", 0), ("seal", 0)]
    assert game.queue() == [("skunk", 1), ("skunk", 0)]


def test_chameleon_copying_snake_sorts_as_9_ties_keep_order():
    game = after([("snake", 1), ("seal", 1), ("lion", 0)], ("chameleon", (0,)))
    assert game.queue() == [("lion", 0), ("snake", 1), ("chameleon", 0), ("seal", 1)]


def test_hippo_stops_behind_zebra_and_crocodile_does_not_pass_it():
    game = after([("seal", 1), ("zebra", 1), ("seal", 0)], ("hippo", ()))
    assert game.queue() == [("seal", 1), ("zebra", 1), ("hippo", 0), ("seal", 0)]
    game = after([("seal", 1), ("zebra", 1), ("seal", 0)], ("crocodile", ()))
    assert game.queue() == [("seal", 1), ("zebra", 1), ("crocodile", 0)]


def test_kangaroo_hops_from_the_back():
    game = after([("seal", 1), ("zebra", 1), ("snake", 1)], ("kangaroo", (2,)))
    assert game.queue() == [("seal", 1), ("kangaroo", 0), ("zebra", 1), ("snake", 1)]


# ---- turn structure -------------------------------------------------------------------------------

def test_played_giraffe_does_not_jump_again():
    game = after([("monkey", 1), ("kangaroo", 1)], ("giraffe", ()))
    assert game.queue() == [("monkey", 1), ("giraffe", 0), ("kangaroo", 1)]


def test_giraffe_in_queue_jumps_once_per_turn():
    game = after([("seal", 1), ("monkey", 1), ("giraffe", 1)], ("zebra", ()))
    assert game.queue() == [("seal", 1), ("giraffe", 1), ("monkey", 1), ("zebra", 0)]


def test_animal_pushed_back_by_hippo_does_not_act_again():
    # Seal reverses; the giraffe jumps the kangaroo, then the hippo passes everything. The giraffe now
    # faces the seal again but has already acted this turn.
    game = after([("hippo", 1), ("snake", 1), ("giraffe", 1), ("kangaroo", 1)], ("seal", ()))
    assert game.bar() == [("hippo", 1), ("seal", 0)]
    assert game.thats_it() == [("snake", 1)]
    assert game.queue() == [("giraffe", 1), ("kangaroo", 1)]


def test_recurring_acts_front_to_back():
    # The kangaroo lands in front of the hippo. The hippo acts first and passes it, which leaves the
    # kangaroo directly ahead of the crocodile (back to front, the crocodile would face the hippo).
    game = after([("zebra", 1), ("hippo", 1), ("crocodile", 1)], ("kangaroo", (2,)))
    assert game.thats_it() == [("kangaroo", 0)]
    assert game.queue() == [("zebra", 1), ("hippo", 1), ("crocodile", 1)]


def test_five_animals_front_two_to_bar_last_bounced():
    game = after([("lion", 1), ("zebra", 1), ("snake", 1), ("seal", 0)], ("monkey", ()))
    assert game.bar() == [("lion", 1), ("zebra", 1)]
    assert game.thats_it() == [("monkey", 0)]
    assert game.queue() == [("snake", 1), ("seal", 0)]
    assert game.score() == (0, 6)


def test_game_ends_with_fewer_than_five_cards_left():
    assert Game(0, [["lion", "seal"], ["lion", "seal"]], [[], []]).is_over()
    game = Game(0, [["lion", "seal", "zebra"], ["lion", "seal"]], [[], []])
    assert not game.is_over()
    assert game.legal_moves()


# ---- random play ----------------------------------------------------------------------------------

def test_random_games_keep_invariants_and_copy_is_independent():
    rng = random.Random(0)
    for n in range(200):
        decks = [rng.sample(ANIMALS, 12) for _ in range(2)]
        game = Game(n % 2, [d[:4] for d in decks], [d[4:] for d in decks])
        while not game.is_over():
            assert game.legal_actions(), "a position with no legal moves before the end"
            action = rng.choice(game.legal_actions())
            before, branch = state(game), game.copy()
            branch.play_action(action)
            assert state(game) == before
            game.play_action(action)
            assert state(game) == state(branch)
            assert len(game.queue()) <= 4
            cards = (game.queue() + game.bar() + game.thats_it()
                     + [(a, p) for p in (0, 1) for a in game.hand(p) + game.deck(p)])
            assert sorted(cards) == sorted((a, p) for a in ANIMALS for p in (0, 1))
