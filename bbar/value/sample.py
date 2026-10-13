"""Public states at given plies from games played by a mix of policies (each seat of each game draws its policy).

    sampler = StateSampler(seed, mix={"random": 0.2, "pimc": 0.4, "bp": 0.4},
                           blueprint="checkpoints/bp-v1/step_8000.pt")
    for x in sampler.batch(plies=(10, 11, 12, 13)):   # Sample(ply, game, policies, start, actions)
        ...                                            # game: the public position (bbar._core.Game)

Policies: "random" (uniform), "pimc" (`pimc:w=8,d=4`, built-in agents act through VecEnv.act with the arena's views),
"bp" (the blueprint network, sampling its mixed strategy on the CPU). With probability `explore` any move is
replaced by a uniformly random one (more varied states). Ply k = k moves played: the state at ply k is the position
before the (k+1)-th move.
"""
from typing import NamedTuple

import numpy as np

from bbar._core import Game, VecEnv
from bbar.actions import ANIMALS

SPECS = {"random": "random", "pimc": "pimc:w=8,d=4"}
DEFAULT_MIX = {"random": 0.2, "pimc": 0.4, "bp": 0.4}


def parse_mix(text):
    """'random:0.2,pimc:0.4,bp:0.4' -> dict."""
    out = {}
    for part in text.split(","):
        name, _, w = part.partition(":")
        if name not in (*SPECS, "bp"):
            raise ValueError(f"unknown policy {name!r} (random, pimc, bp)")
        out[name] = float(w or 1)
    return out


class Sample(NamedTuple):
    ply: int
    game: Game          # the public position at that ply
    policies: tuple     # policy names of seats 0 and 1
    start: int
    actions: list       # the moves played (action indices)


def public_game(start, actions):
    """The public position (every unplayed animal in hand) after `actions` from a game started by `start`."""
    g = Game(int(start), [ANIMALS, ANIMALS], [[], []])
    for a in actions:
        g.play_action(int(a))
    return g


class StateSampler:
    def __init__(self, seed, mix=None, blueprint=None, envs=64, explore=0.05, threads=1, pimc=None):
        mix = dict(DEFAULT_MIX if mix is None else mix)
        if mix.get("bp", 0) > 0 and not blueprint:
            mix.pop("bp")
        self.names = [k for k, v in mix.items() if v > 0]
        p = np.array([mix[k] for k in self.names], dtype=float)
        self.p = p / p.sum()
        self.rng = np.random.default_rng(seed)
        self.env = VecEnv(envs, seed=int(self.rng.integers(2 ** 63)), threads=threads)
        self.n = envs
        self.explore = explore
        self.specs = dict(SPECS, **({"pimc": pimc} if pimc else {}))
        self.agent = None
        if "bp" in self.names:
            import torch
            from bbar.rl.agent import NetAgent
            torch.manual_seed(int(self.rng.integers(2 ** 62)))
            self.agent = NetAgent(blueprint, "cpu")

    def batch(self, plies, keep=None):
        """[Sample] from one batch of fresh games: each game gives its state at each ply of `plies`, kept with
        probability keep[ply] (default 1). In ply order."""
        plies = sorted(set(int(x) for x in plies))
        keep = keep or {}
        n, env, rng = self.n, self.env, self.rng
        env.reset_deals(rng.integers(0, 2 ** 63, size=n, dtype=np.uint64), rng.integers(0, 2, size=n))
        pol = rng.choice(len(self.names), size=(n, 2), p=self.p)
        out = []
        for step in range(plies[-1] + 1):
            if self.agent is not None:
                obs, mask, mover, ply = env.observe()
            else:
                mover, ply = env.movers()
            assert (ply == step).all(), "games advance together"
            if step in plies:
                for i in np.flatnonzero(rng.random(n) < keep.get(step, 1.0)):
                    hist = env.history(int(i))
                    start = int(hist[0][0] if hist else mover[i])
                    actions = [int(a) for _, a in hist]
                    out.append(Sample(step, public_game(start, actions), (self.names[pol[i, 0]],
                                                                           self.names[pol[i, 1]]), start, actions))
            if step == plies[-1]:
                break
            who = pol[np.arange(n), mover]
            actions = np.full(n, -1, dtype=np.int64)
            for k, name in enumerate(self.names):
                idx = np.flatnonzero(who == k)
                if not len(idx):
                    continue
                if name == "bp":
                    actions[idx] = self.agent.act(obs[idx], mask[idx])
                else:
                    actions[idx] = env.act(self.specs[name], idx)
            wild = np.flatnonzero(rng.random(n) < self.explore)
            if len(wild):
                actions[wild] = env.act("random", wild)
            env.step(actions)
        return out
