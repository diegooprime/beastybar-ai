"""Golden-game checker. An engine's results count only if it reproduces every game in golden/ exactly.

    python -m bbar.golden --engine core      # or: ref
"""
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent.parent / "golden" / "games.jsonl.gz"


def load(path=GOLDEN):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def engine(name):
    """Engine Game class by name. Every engine exposes the same Python API:

    Game(start, hands, decks)            # hands/decks: [[animal, ...], [animal, ...]]; decks top card first
    .to_move -> int
    .legal_moves() -> [(animal, (params...)), ...]
    .play(animal, params)                # raises ValueError on an illegal move
    .queue() / .bar() / .thats_it() -> [(animal, owner), ...]   # queue front first; others in order of arrival
    .score() -> (int, int)
    .is_over() -> bool
    .hand(player) -> [animal, ...]; .deck(player) -> [animal, ...] (top first)
    """
    if name == "ref":
        from bbar.ref import Game
    elif name == "core":
        from bbar._core import Game
    else:
        raise ValueError(f"unknown engine {name!r}")
    return Game


def _cards(xs):
    return [[a, o] for a, o in xs]


def check_game(Game, g):
    """None if the engine reproduces golden game g exactly, else a description of the first difference."""
    game = Game(g["start"], g["hands"], g["decks"])
    player = g["start"]
    for i, s in enumerate(g["steps"]):
        if game.is_over():
            return f"step {i}: engine says the game is over"
        if game.to_move != player:
            return f"step {i}: to_move {game.to_move}, expected {player}"
        legal = sorted([a, list(p)] for a, p in game.legal_moves())
        if legal != s["legal"]:
            return f"step {i}: legal\n  got      {legal}\n  expected {s['legal']}"
        a, p = s["move"]
        game.play(a, tuple(p))
        for key, got in (("queue", game.queue()), ("bar", game.bar()), ("thats_it", game.thats_it())):
            if _cards(got) != s[key]:
                return f"step {i} move {s['move']}: {key}\n  got      {_cards(got)}\n  expected {s[key]}"
        if list(game.score()) != s["score"]:
            return f"step {i}: score {list(game.score())}, expected {s['score']}"
        player = 1 - player
    if not game.is_over():
        return "engine says the game continues after the last step"
    if list(game.score()) != g["final_score"]:
        return f"final score {list(game.score())}, expected {g['final_score']}"
    return None


def check(Game, games, max_report=5):
    failures = []
    for n, g in enumerate(games):
        err = check_game(Game, g)
        if err:
            failures.append((n, err))
    for n, err in failures[:max_report]:
        print(f"game {n}: {err}")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="core")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    games = load()[: args.limit]
    t = time.time()
    failures = check(engine(args.engine), games)
    moves = sum(len(g["steps"]) for g in games)
    print(f"{args.engine}: {len(games) - len(failures)}/{len(games)} golden games exact "
          f"({moves} moves, {time.time() - t:.1f}s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
