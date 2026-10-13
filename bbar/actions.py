"""Animals and the fixed action index layout shared by every engine, agent and network.

Animal id = strength - 1. A card is (animal, owner); packed card id = owner * 12 + animal.

Action indices (NUM_ACTIONS = 36). Parameters are queue positions before the move (0 = front):
    0           skunk []
    1..4        parrot [t]            t = 0..3
    5           kangaroo []           (queue empty)
    6, 7        kangaroo [1], [2]
    8           monkey []
    9..12       chameleon [t]         copies a parameterless animal at t
    13..28      chameleon [t, x]      13 + 4*t + x; copies the parrot at t (x = target) or kangaroo at t (x = hop)
    29..35      seal, zebra, giraffe, snake, crocodile, hippo, lion  []
"""

ANIMALS = ["skunk", "parrot", "kangaroo", "monkey", "chameleon", "seal",
           "zebra", "giraffe", "snake", "crocodile", "hippo", "lion"]
ANIMAL_ID = {a: i for i, a in enumerate(ANIMALS)}
STRENGTH = {a: i + 1 for i, a in enumerate(ANIMALS)}
POINTS = {"lion": 2, "hippo": 2, "crocodile": 3, "snake": 2, "giraffe": 3, "zebra": 4,
          "seal": 2, "chameleon": 3, "monkey": 3, "kangaroo": 4, "parrot": 4, "skunk": 4}

NUM_ACTIONS = 36
_PLAIN = {"skunk": 0, "monkey": 8, "seal": 29, "zebra": 30, "giraffe": 31, "snake": 32,
          "crocodile": 33, "hippo": 34, "lion": 35}


def encode(animal, params):
    """(animal, params) -> action index."""
    params = tuple(params)
    if animal in _PLAIN:
        assert params == (), (animal, params)
        return _PLAIN[animal]
    if animal == "parrot":
        (t,) = params
        return 1 + t
    if animal == "kangaroo":
        return 5 if params == () else 5 + params[0]
    if animal == "chameleon":
        if len(params) == 1:
            return 9 + params[0]
        t, x = params
        return 13 + 4 * t + x
    raise ValueError(animal)


def decode(index):
    """action index -> (animal, params)."""
    for animal, i in _PLAIN.items():
        if i == index:
            return animal, ()
    if 1 <= index <= 4:
        return "parrot", (index - 1,)
    if index == 5:
        return "kangaroo", ()
    if index in (6, 7):
        return "kangaroo", (index - 5,)
    if 9 <= index <= 12:
        return "chameleon", (index - 9,)
    if 13 <= index <= 28:
        return "chameleon", divmod(index - 13, 4)
    raise ValueError(index)


assert all(encode(*decode(i)) == i for i in range(NUM_ACTIONS))
