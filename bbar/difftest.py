"""Differential tester: plays random games on two engines in lockstep and compares everything they expose.

    python -m bbar.difftest --games 100000 --seed 0            # uniformly random legal moves
    python -m bbar.difftest --games 100000 --seed 0 --biased   # over-sample rare rule branches

Game i uses seed S + i: random.Random(seed) deals (random start player; each player's 12 animals shuffled, the
first 4 the hand, the rest the deck, top first) and then picks every move from the sorted legal list. A game is
a pure function of (seed, mode), so `--games 1 --seed <seed>` (plus `--biased`) replays a reported discrepancy.

Before the first move and after every move, the engines must agree on to_move, is_over, legal moves and actions,
queue, bar, THAT'S IT, score, both hands and both decks. Each engine is also checked on its own: invariants,
copy() independence, play_action(encode(m)) == play(m), and a ValueError for an illegal action.
"""
import argparse
import multiprocessing
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field

from bbar.actions import ANIMAL_ID, ANIMALS, NUM_ACTIONS, POINTS, encode
from bbar.golden import engine

ALL_CARDS = sorted((a, p) for a in ANIMALS for p in (0, 1))
FIELDS = ("to_move", "is_over", "legal_moves", "legal_actions", "queue", "bar", "thats_it", "score", "hands",
          "decks")

# Biased mode weighs each legal move by the rule branch it exercises (branch()); unlisted branches weigh 1.
# Every card is played at most once per game, so the bias works by playing a card when it hits a rare branch
# (the rarest under uniform play get RARE), and by putting a lone monkey down early to wait for the other.
RARE = 25
BIAS = {"monkey!": RARE, "lion!": RARE, "kangaroo[]": RARE, "monkey": 5,
        **{f"chameleon>{a}": RARE for a in ("monkey", "lion", "snake", "hippo", "giraffe", "zebra", "seal", "skunk")}}


def deal(rng):
    start = rng.randrange(2)
    hands, decks = [], []
    for _ in range(2):
        animals = list(ANIMALS)
        rng.shuffle(animals)
        hands.append(animals[:4])
        decks.append(animals[4:])
    return start, hands, decks


def observe(game):
    """Everything the engine API exposes about a position, normalised for comparison (hand order is free)."""
    return {
        "to_move": game.to_move,
        "is_over": game.is_over(),
        "legal_moves": sorted((a, tuple(p)) for a, p in game.legal_moves()),
        "legal_actions": list(game.legal_actions()),
        "queue": [tuple(c) for c in game.queue()],
        "bar": [tuple(c) for c in game.bar()],
        "thats_it": [tuple(c) for c in game.thats_it()],
        "score": tuple(game.score()),
        "hands": [sorted(game.hand(p), key=ANIMAL_ID.get) for p in (0, 1)],
        "decks": [list(game.deck(p)) for p in (0, 1)],
    }


def invariant_errors(obs):
    errors = []
    if obs["legal_actions"] != sorted(encode(a, p) for a, p in obs["legal_moves"]):
        errors.append("legal_actions is not the sorted encoding of legal_moves")
    if obs["is_over"] == bool(obs["legal_moves"]):
        errors.append("there must be legal moves exactly while the game is not over")
    if len(obs["queue"]) > 4:
        errors.append("more than 4 animals in the queue between turns")
    held = [(a, p) for p in (0, 1) for a in obs["hands"][p] + obs["decks"][p]]
    if sorted(obs["queue"] + obs["bar"] + obs["thats_it"] + held) != ALL_CARDS:
        errors.append("cards are not conserved")
    if obs["score"] != tuple(sum(POINTS[a] for a, o in obs["bar"] if o == p) for p in (0, 1)):
        errors.append("score is not the points in the bar")
    return errors


def branch(move, queue):
    """The rule branch a move exercises: the animal, a chameleon's copy, a triggered monkey, a doomed lion."""
    animal, params = move
    if animal == "chameleon":
        return "chameleon>" + queue[params[0]][0]
    if animal in ("monkey", "lion") and any(a == animal for a, _ in queue):
        return animal + "!"
    if animal == "kangaroo":
        return f"kangaroo{list(params)}"
    return animal


def choose(rng, legal, queue, biased):
    if not biased:
        return rng.choice(legal)
    weights = [BIAS.get(branch(m, queue), 1) for m in legal]
    return rng.choices(legal, weights)[0]


def new_game(Game, start, hands, decks):
    game = Game(start, hands, decks)
    return game, observe(game), []


def step(game, before, move, illegal):
    """Plays `move` on `game`, and by action index on a copy. Returns (game, observation after, problems)."""
    problems = []
    try:
        game.copy().play_action(illegal)
        problems.append(f"illegal action {illegal} was accepted")
    except ValueError:
        pass
    branch_game = game.copy()
    game.play(*move)
    if observe(branch_game) != before:
        return game, observe(game), problems + ["playing the original changed its copy"]
    branch_game.play_action(encode(*move))
    after = observe(game)
    if observe(branch_game) != after:
        problems.append("play_action(encode(m)) on a copy differs from play(m) (or leaked into the original)")
    return game, after, problems


@dataclass
class Discrepancy:
    seed: int
    biased: bool
    deal: tuple
    moves: list           # moves played, the last one (if any) being the move after which things went wrong
    before: dict          # first engine's observation before that move (None at the deal)
    problems: list

    def __str__(self):
        start, hands, decks = self.deal
        where = f"after move {len(self.moves)} {_fmt(self.moves[-1])}" if self.moves else "at the deal"
        lines = [f"seed {self.seed}{' (biased)' if self.biased else ''}, {where}:"]
        lines += [f"  {p}" for p in self.problems]
        lines.append(f"  replay: python -m bbar.difftest --games 1 --seed {self.seed}"
                     + (" --biased" if self.biased else ""))
        lines.append(f"  Game({start}, {hands}, {decks})")
        lines.append(f"  moves: [{', '.join(_fmt(m) for m in self.moves)}]")
        if self.before:
            b = self.before
            lines.append(f"  before the last move: player {b['to_move']} to move, hand {b['hands'][b['to_move']]}, "
                         f"queue {b['queue']}")
        return "\n".join(lines)


def _fmt(move):
    return f"[{move[0]!r}, {list(move[1])}]"


def _guarded(fn, *args):
    """(result, None), or (None, error text). A Rust panic surfaces as pyo3's PanicException, a BaseException."""
    try:
        return fn(*args), None
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        return None, f"{type(e).__name__}: {e}"


def play_game(engines, seed, biased=False, coverage=None):
    """Plays one game on every engine class in lockstep. Returns (moves played, Discrepancy or None)."""
    names = [f"{E.__module__.removeprefix('bbar.')}.{E.__name__}" for E in engines]
    rng = random.Random(seed)
    start, hands, decks = deal(rng)
    moves, before = [], None
    results = [_guarded(new_game, Game, start, hands, decks) for Game in engines]
    while True:
        problems = []
        for name, (result, err) in zip(names, results):
            if err:
                problems.append(f"{name}: {err}")
            else:
                problems += [f"{name}: {p}" for p in result[2] + invariant_errors(result[1])]
        if not any(err for _, err in results):
            games, obs, _ = zip(*(result for result, _ in results))
            for key in FIELDS:
                if any(o[key] != obs[0][key] for o in obs[1:]):
                    problems.append(f"{key} differs:" + "".join(f"\n    {n:<14} {o[key]}" for n, o in zip(names, obs)))
        if problems:
            return len(moves), Discrepancy(seed, biased, (start, hands, decks), moves, before, problems)
        if obs[0]["is_over"]:
            return len(moves), None

        before = obs[0]
        move = choose(rng, before["legal_moves"], before["queue"], biased)
        if coverage is not None:
            coverage[branch(move, before["queue"])] += 1
        moves.append(move)
        not_legal = sorted(set(range(NUM_ACTIONS)) - set(before["legal_actions"]))
        illegal = not_legal[len(moves) % len(not_legal)]
        results = [_guarded(step, game, o, move, illegal) for game, o in zip(games, obs)]


@dataclass
class Summary:
    games: int = 0
    moves: int = 0
    discrepancies: list = field(default_factory=list)
    coverage: Counter = field(default_factory=Counter)

    def add(self, other):
        self.games += other.games
        self.moves += other.moves
        self.discrepancies += other.discrepancies
        self.coverage += other.coverage


def _run_chunk(args):
    names, seeds, biased = args
    engines = [engine(n) for n in names]
    summary = Summary(games=len(seeds))
    for seed in seeds:
        n, discrepancy = play_game(engines, seed, biased, summary.coverage)
        summary.moves += n
        if discrepancy:
            summary.discrepancies.append(discrepancy)
    return summary


def run(games, seed=0, biased=False, names=("ref", "core"), workers=None, chunk=250):
    """Plays games with seeds seed .. seed + games - 1; discrepancies come back in seed order."""
    jobs = [(names, range(s, min(s + chunk, seed + games)), biased) for s in range(seed, seed + games, chunk)]
    total = Summary()
    if workers == 1:
        for job in jobs:
            total.add(_run_chunk(job))
    else:
        with multiprocessing.Pool(workers) as pool:
            for summary in pool.imap(_run_chunk, jobs):
                total.add(summary)
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--biased", action="store_true", help="over-sample rare rule branches")
    ap.add_argument("--engines", nargs=2, default=["ref", "core"])
    ap.add_argument("--workers", type=int, default=None, help="processes (default: all cores)")
    ap.add_argument("--report", type=int, default=3, help="discrepancies to print in full")
    args = ap.parse_args()

    t = time.time()
    s = run(args.games, args.seed, args.biased, tuple(args.engines), args.workers)
    for d in s.discrepancies[: args.report]:
        print(d, end="\n\n")
    rare = ", ".join(f"{k} {v}" for k, v in sorted(s.coverage.items(), key=lambda kv: kv[1])[:8])
    print(f"rarest branches: {rare}")
    print(f"{' vs '.join(args.engines)}, {'biased' if args.biased else 'uniform'}: {s.games} games "
          f"(seeds {args.seed}..{args.seed + args.games - 1}), {s.moves} moves, "
          f"{len(s.discrepancies)} discrepancies, {time.time() - t:.1f}s")
    sys.exit(1 if s.discrepancies else 0)


if __name__ == "__main__":
    main()
