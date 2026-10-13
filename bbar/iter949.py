"""iter949 as an opponent in bbar.match: the previous attempt's best policy network (17M parameters, greedy), kept
outside this repository as a directory holding iter949.py and iter949_weights.pt (its README: 95.5% wins against
uniformly random play).

    python -m bbar.match iter949:DIR OTHER --deals N [--device cpu]

The adapter turns the player's view (bbar.match.Decisions: view Game and public history) into iter949's view dict and
maps its 124-move catalog (hand slot * 31 + parameter code) onto this project's 36 actions. iter949 sees its hand as
its engine kept it: the opening four in dealt (random) order, then each played card removed and each drawn card
appended; the adapter tracks that order per game (Decisions.envs) and draws the opening order from the decision's
private seed.
"""
import functools
import importlib.util
import sys
from pathlib import Path

import numpy as np

from bbar.actions import decode

PREFIX = "iter949:"


def load(directory, device="cpu"):
    """The reference module's Iter949 player from DIR (iter949.py, iter949_weights.pt)."""
    directory = Path(directory)
    spec = importlib.util.spec_from_file_location("iter949_reference", directory / "iter949.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module   # its dataclasses look their module up while it loads
    spec.loader.exec_module(module)
    return module.Iter949(str(directory / "iter949_weights.pt"), device=str(device))


def move_index(slot, params):
    """iter949's move index of playing hand slot `slot` with `params` (queue positions before the move, kangaroo hop,
    or a chameleon's target followed by the copied animal's parameter)."""
    if len(params) == 0:
        sub = 0
    elif len(params) == 1:
        sub = 1 + 6 * params[0]
    else:
        sub = 2 + 6 * params[0] + params[1]
    return slot * 31 + sub


def view_dict(game, history, me, hand):
    """iter949's view of `game` (a bbar._core.Game seen by `me`) with its hand in order `hand`."""
    return {"me": me, "active": me, "turn": len(history), "queue": game.queue(), "bar": game.bar(),
            "thats_it": game.thats_it(), "hand": list(hand),
            "hand_counts": (len(game.hand(0)), len(game.hand(1))),
            "deck_counts": (len(game.deck(0)), len(game.deck(1)))}


class Iter949Agent:
    """bbar.match Python agent (PyPlayer) playing iter949 greedily."""
    needs_obs = False

    def __init__(self, player, name, checkpoints):
        self.player = player
        self.name = name
        self.checkpoints = dict(checkpoints)
        self._hands = {}   # env index -> (history at our last decision, hand order then)

    @classmethod
    def factory(cls, body, device="cpu"):
        """A zero-argument callable making fresh agents for the spec body 'DIR' that share one loaded network."""
        from bbar.match import sha256_short
        weights = Path(body) / "iter949_weights.pt"
        if not weights.is_file():
            raise ValueError(f"iter949: no {weights}")
        player = load(body, device)
        return functools.partial(cls, player, PREFIX + body, {str(weights): sha256_short(weights.read_bytes())})

    def prepare(self, live):
        """A new batch of games: env indices start over."""
        self._hands = {}

    def hand_order(self, env, game, history, me, seed):
        hand = game.hand(me)
        prev = self._hands.get(env)
        order = None
        if prev is not None and len(history) > len(prev[0]) and tuple(history[:len(prev[0])]) == prev[0]:
            played = decode(history[len(prev[0])])[0]
            kept = [a for a in prev[1] if a != played]
            order = kept + sorted(set(hand) - set(kept))
        if order is None or sorted(order) != sorted(hand):
            order = list(np.random.default_rng(int(seed)).permutation(hand))   # a new game: dealt order
        self._hands[env] = (tuple(history), order)
        return order

    def act(self, batch):
        k = len(batch.games)
        envs = batch.envs if batch.envs is not None else -1 - np.arange(k)
        views, legal, maps = [], [], []
        for j, (game, history) in enumerate(zip(batch.games, batch.histories)):
            me = game.to_move
            order = self.hand_order(int(envs[j]), game, history, me, batch.seeds[j])
            to_action = {}
            for a in game.legal_actions():
                animal, params = decode(a)
                to_action[move_index(order.index(animal), params)] = a
            views.append(view_dict(game, history, me, order))
            legal.append(sorted(to_action))
            maps.append(to_action)
        picks = self.player.pick(views, legal)
        return np.array([maps[j][m] for j, m in enumerate(picks)], dtype=np.int64)
