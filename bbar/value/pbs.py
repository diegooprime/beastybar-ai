"""Public belief states (PBS): what the value network sees, and the root-range distributions of the experiments.

A PBS is a public state plus both players' beliefs over their possible hands. Ply k = k moves played. Hands are the
4-subsets of a player's unplayed animals while their deck is non-empty, in the fixed 495-slot layout of
`bbar._core.hand_rank` (colex rank: animals a1 < a2 < a3 < a4 have rank C(a1,1) + C(a2,2) + C(a3,3) + C(a4,4)).
Value leaves of the window solver (plies 10..14 of the real game) always have hands of 4 for both players.

Encoding (PBS_VERSION 1), relative to the player to move ("mover"), "opp" the other:

    offset  size  content
       0     56   queue, front first: 4 slots x (animal one-hot 12, mover's, opp's); empty slots 0
      56     12   mover's unplayed animals (hand + deck)
      68     12   opp's unplayed animals
      80      2   deck sizes / 8: mover, opp (unplayed - 4)
      82      1   score difference (mover - opp) / 20
      83     25   score difference one-hot, clamped to -12..12
     108     25   ply one-hot (0..24)
     133      1   the mover started the game
     134          PUB_DIM
    then 495 mover's belief, 495 opp's belief (normalised): IN_DIM = 1124

The public part is decoded from the public Markov key (`bbar._core.public_key`, layout in core/src/markov.rs) and
the starting player, so label shards store only (key, start). Targets: per-hand values of both players (each the
win probability of that hand's owner against the other's belief), masked to the feasible hands.
"""
import functools
from itertools import combinations

import numpy as np

from bbar.actions import ANIMALS

PBS_VERSION = 1
ANIMAL_NAMES = tuple(ANIMALS)   # by animal id
NUM_ANIMALS = 12
HAND = 4
SLOTS = 495
QUEUE_SLOT = NUM_ANIMALS + 2
PUB_DIM = 134
IN_DIM = PUB_DIM + 2 * SLOTS
DIFF_CLAMP = 12

_C = [[0] * 13 for _ in range(13)]
for _n in range(13):
    _C[_n][0] = 1
    for _m in range(1, _n + 1):
        _C[_n][_m] = _C[_n - 1][_m - 1] + (_C[_n - 1][_m] if _m < _n else 0)


def rank(animals):
    """Slot of a set of animal ids (colex rank among sets of its size)."""
    return sum(_C[a][i + 1] for i, a in enumerate(sorted(animals)))


SLOT_ANIMALS = [None] * SLOTS
for _h in combinations(range(NUM_ANIMALS), HAND):
    SLOT_ANIMALS[rank(_h)] = _h
SLOT_MASK = np.array([sum(1 << a for a in h) for h in SLOT_ANIMALS], dtype=np.int64)
SLOT_ONEHOT = np.array([[a in h for a in range(NUM_ANIMALS)] for h in SLOT_ANIMALS], dtype=np.float32)  # [495, 12]


def feasible(unplayed):
    """bool [..., 495]: hands (4-subsets) of the unplayed animal masks `unplayed` (int array, any shape)."""
    u = np.asarray(unplayed, dtype=np.int64)
    return (SLOT_MASK & ~u[..., None]) == 0


@functools.lru_cache(maxsize=4096)
def feasible_slots(unplayed):
    """Slots of the 4-subsets of the unplayed mask (int), increasing."""
    return np.flatnonzero(feasible(int(unplayed)))


def decode_keys(keys):
    """Fields of public Markov keys (uint64 array): queue_len, queue cards [n, 4], unplayed [n, 2], diff (s0 - s1),
    to_move."""
    k = np.asarray(keys, dtype=np.uint64)
    qlen = (k & np.uint64(7)).astype(np.int64)
    queue = np.stack([((k >> np.uint64(3 + 5 * i)) & np.uint64(31)).astype(np.int64) for i in range(4)], -1)
    u = np.stack([((k >> np.uint64(s)) & np.uint64(0xFFF)).astype(np.int64) for s in (23, 35)], -1)
    diff = ((k >> np.uint64(55)) & np.uint64(127)).astype(np.int64) - 64
    to_move = ((k >> np.uint64(62)) & np.uint64(1)).astype(np.int64)
    return qlen, queue, u, diff, to_move


def popcount(x):
    x = np.asarray(x, dtype=np.int64)
    return np.array([bin(int(v)).count("1") for v in x.ravel()], dtype=np.int64).reshape(x.shape)


def public_features(keys, start):
    """float32 [n, PUB_DIM]: mover-relative public features of public keys (uint64 [n]) with the starting player
    (int [n])."""
    keys = np.atleast_1d(np.asarray(keys, dtype=np.uint64))
    start = np.broadcast_to(np.asarray(start, dtype=np.int64), keys.shape)
    n = len(keys)
    qlen, queue, u, diff, m = decode_keys(keys)
    out = np.zeros((n, PUB_DIM), dtype=np.float32)
    rows = np.arange(n)
    for i in range(4):
        live = qlen > i
        c = queue[:, i]
        animal, owner = c % NUM_ANIMALS, (c >= NUM_ANIMALS).astype(np.int64)
        base = i * QUEUE_SLOT
        out[rows[live], base + animal[live]] = 1
        out[rows[live], base + NUM_ANIMALS + (owner[live] != m[live])] = 1
    um = np.where(m == 0, u[:, 0], u[:, 1])
    uo = np.where(m == 0, u[:, 1], u[:, 0])
    bits = 1 << np.arange(NUM_ANIMALS)
    out[:, 56:68] = (um[:, None] & bits) != 0
    out[:, 68:80] = (uo[:, None] & bits) != 0
    nm, no = popcount(um), popcount(uo)
    out[:, 80] = np.maximum(nm - HAND, 0) / 8
    out[:, 81] = np.maximum(no - HAND, 0) / 8
    d = np.where(m == 0, diff, -diff)
    out[:, 82] = d / 20
    out[rows, 83 + np.clip(d, -DIFF_CLAMP, DIFF_CLAMP) + DIFF_CLAMP] = 1
    ply = 2 * NUM_ANIMALS - nm - no
    out[rows, 108 + np.clip(ply, 0, 24)] = 1
    out[:, 133] = start == m
    return out


def key_info(keys):
    """(mover [n], unplayed of mover and opp [n, 2], ply [n]) of public keys."""
    _, _, u, _, m = decode_keys(np.atleast_1d(np.asarray(keys, dtype=np.uint64)))
    um = np.where(m == 0, u[:, 0], u[:, 1])
    uo = np.where(m == 0, u[:, 1], u[:, 0])
    ply = 2 * NUM_ANIMALS - popcount(um) - popcount(uo)
    return m, np.stack([um, uo], -1), ply


def masks_of_keys(keys):
    """bool [n, 2, 495]: feasible hands of the mover and the opp."""
    _, uu, _ = key_info(keys)
    return feasible(uu)


# ---- root ranges ---------------------------------------------------------------------------------------------------

RANGE_KINDS = ("uniform", "dirichlet0.1", "dirichlet0.3", "dirichlet1", "sparse", "peaked", "mixture",
               "bayes", "bayes-mix")   # the last two: Bayes ranges under a policy model (bbar/value/gen.py --bayes)
RANGE_WEIGHTS = (0.10, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15)
SYNTHETIC_KINDS = RANGE_KINDS[:len(RANGE_WEIGHTS)]   # what sample_range draws


def _draw(kind, n, slots, rng):
    if kind == "uniform":
        return np.ones(n)
    if kind.startswith("dirichlet"):
        return rng.dirichlet(np.full(n, float(kind[len("dirichlet"):])))
    if kind == "sparse":
        size = int(rng.integers(1, min(n, 8) + 1))
        w = np.zeros(n)
        w[rng.choice(n, size, replace=False)] = rng.dirichlet(np.ones(size))
        return w
    if kind == "peaked":
        center = SLOT_ONEHOT[slots[rng.integers(n)]]
        dist = HAND - SLOT_ONEHOT[slots] @ center          # animals not shared with the center hand
        return np.exp(-rng.uniform(0.5, 4.0) * dist)
    raise ValueError(kind)


def sample_range(unplayed, rng, kind=None):
    """(range float64 [495] normalised on the feasible hands of `unplayed`, kind): a root range from the experiments'
    family (RANGE_KINDS with RANGE_WEIGHTS; 'mixture' = a random convex combination of two other kinds)."""
    slots = feasible_slots(int(unplayed))
    n = len(slots)
    if kind is None:
        kind = SYNTHETIC_KINDS[rng.choice(len(SYNTHETIC_KINDS), p=RANGE_WEIGHTS)]
    if kind == "mixture":
        a, b = rng.choice(len(SYNTHETIC_KINDS) - 1, 2, replace=False)
        lam = rng.uniform()
        w = lam * _normalized(_draw(RANGE_KINDS[a], n, slots, rng)) + (1 - lam) * _normalized(
            _draw(RANGE_KINDS[b], n, slots, rng))
    else:
        w = _draw(kind, n, slots, rng)
    r = np.zeros(SLOTS)
    r[slots] = _normalized(w)
    return r, kind


def _normalized(w):
    w = np.maximum(np.asarray(w, dtype=np.float64), 0)
    s = w.sum()
    return w / s if s > 0 else np.full(len(w), 1 / len(w))


def mover_relative(x, mover):
    """[..., 2, S] arrays over absolute players -> (mover, opp) order (mover int or [n] array)."""
    x = np.asarray(x)
    mover = np.asarray(mover)
    if mover.ndim == 0:
        return x if int(mover) == 0 else x[..., ::-1, :]
    idx = np.stack([mover, 1 - mover], -1)
    return np.take_along_axis(x, idx[..., None], axis=-2)
