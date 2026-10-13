"""Duplicate-deal matches between agents; every match is appended to runs/results.jsonl, the evidence trail.

    python -m bbar.arena A B [--deals N] [--seed S] [--threads T]
    python -m bbar.arena --ladder SPEC1 SPEC2 ... [--deals N] [--seed S] [--threads T]   # round robin

Agent specs (core/src/agents.rs): random, greedy, pimc:w=W,d=D (W sampled worlds searched D >= 1 plies; D = 24
reaches the end of the game). Oracles that see hidden information, only for measuring its value: pimc-hand:w=W,d=D
(true opponent hand), cheat:d=D (true full state). Ctrl-C stops a match at once, even mid-search.

Deal i depends only on (seed, i), so matches with the same seed play the same deals; each deal is played twice with
the seats swapped. Results do not depend on the thread count. Without --seed, a random 64-bit seed is drawn (and
logged): an explicit seed is reproducible but guessable, and whoever knows the seed can recompute every deal, so use
explicit seeds only with agents that follow the trust model in core/src/agents.rs (all built-in agents do).

Each match appends one JSON line to --log: specs, deals, seed, score and its 95% interval, W/D/L, point margin and
CI, seconds, provenance (git commit, code hash and engine build, bbar/provenance.py), hostname and the command line.
Nothing is played or logged if the engine extension is stale (built from other core/ sources); and the evidence
trail (the default --log) only takes results of committed code, unless --allow-dirty.
"""
import argparse
import json
import math
import secrets
import shlex
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

from bbar import provenance, stats
from bbar._core import parse_spec, play_match

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "runs" / "results.jsonl"


def random_seed():
    """A fresh 64-bit match seed: no agent can guess it, so no agent can recompute the deals."""
    return secrets.randbits(64)


def run(a, b, deals, seed=None, threads=0, progress=False):
    """Plays a match (seed None: random_seed()); returns (stats.MatchResult, raw per-game results from
    bbar._core.play_match)."""
    seed = random_seed() if seed is None else seed
    raw = play_match(a, b, deals, seed=seed, threads=threads, progress=progress)
    result = stats.summarize(raw["a"], raw["b"], seed, raw["a_points"], raw["b_points"], raw["deal"])
    return result, raw


def record(result, raw, command, revision):
    """The JSON log line of a match; revision: provenance.revision() of the code that played it."""
    def num(x, digits=5):
        return None if math.isnan(x) else round(x, digits)
    return {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "a": result.a, "b": result.b, "deals": result.deals, "seed": result.seed,
        "score": num(result.score), "score_lo": num(result.score_lo), "score_hi": num(result.score_hi),
        "score_ci": "wilson-pairs-95",
        "wins": result.wins, "draws": result.draws, "losses": result.losses,
        "margin": num(result.margin, 4), "margin_ci95": num(result.margin_ci95, 4),
        "seconds": round(raw["seconds"], 3), "threads": raw["threads"],
        "commit": revision["commit"], "code": revision["code"], "build": raw["build"],
        "hostname": socket.gethostname(), "command": command,
    }


def append(log, line):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(json.dumps(line) + "\n")


def ladder_table(specs, scores):
    """Round-robin table: row's score (%) against each column with its 95% interval, and the row's mean."""
    w = max(len(s) for s in specs) + 4
    lines = [" " * w + "".join(f"{j + 1:>19}" for j in range(len(specs))) + f"{'mean':>8}"]
    for i, s in enumerate(specs):
        cells, mean = [], []
        for j in range(len(specs)):
            if i == j:
                cells.append(f"{'-':>19}")
                continue
            score, lo, hi = scores[i, j]
            mean.append(score)
            cells.append(f"{100 * score:>7.1f} [{100 * lo:4.1f},{100 * hi:5.1f}]")
        avg = 100 * sum(mean) / len(mean) if mean else float("nan")
        lines.append(f"{i + 1:>2}  {s:<{w - 4}}" + "".join(cells) + f"{avg:>8.1f}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bbar.arena", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", nargs="?", help="agent A spec")
    ap.add_argument("b", nargs="?", help="agent B spec")
    ap.add_argument("--ladder", nargs="+", metavar="SPEC", help="round robin between these specs")
    ap.add_argument("--deals", type=int, default=1000, help="deal pairs per match (2 games each)")
    ap.add_argument("--seed", type=int, help="match seed, 0 to 2^64 - 1 (default: random, logged)")
    ap.add_argument("--threads", type=int, default=0, help="0 = all cores")
    ap.add_argument("--log", default=str(LOG), help="JSON-lines log to append to (default runs/results.jsonl)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="log to the evidence trail although the code is not committed (the line's code hash "
                         "still identifies it)")
    ap.add_argument("--quiet", action="store_true", help="no progress lines on stderr")
    args = ap.parse_args(argv)
    if args.ladder:
        if args.a or args.b:
            ap.error("give either A B or --ladder SPEC ..., not both")
        if len(args.ladder) < 2:
            ap.error("--ladder needs at least two specs")
        pairs = [(i, j) for i in range(len(args.ladder)) for j in range(i + 1, len(args.ladder))]
        specs = args.ladder
    elif args.a and args.b:
        pairs, specs = [(0, 1)], [args.a, args.b]
    else:
        ap.error("give two agent specs A B, or --ladder SPEC ...")
    if args.deals < 1:
        ap.error("--deals must be at least 1")
    if args.seed is None:
        args.seed = random_seed()
    elif not 0 <= args.seed < 2 ** 64:
        ap.error("--seed must be in 0 .. 2^64 - 1")
    try:
        specs = [parse_spec(s) for s in specs]  # canonical, and fail before playing anything
    except ValueError as e:
        ap.error(str(e))
    # Provenance, before playing anything: results must be attributable to the code that produced them.
    try:
        provenance.check_build()
    except provenance.StaleBuild as e:
        ap.error(str(e))
    revision = provenance.revision()
    if (Path(args.log).resolve() == LOG.resolve() and not provenance.is_clean(revision["commit"])
            and not args.allow_dirty):
        ap.error(f"the code is not committed ({revision['commit']}): commit it before adding results to the "
                 f"evidence trail {LOG.relative_to(ROOT)}, or pass --allow-dirty (the line records code hash "
                 f"{revision['code']})")
    # The interpreter as "python" (its resolved path is machine-specific), then the arguments exactly as given.
    command = shlex.join(["python", *sys.orig_argv[1:]] if argv is None else ["python", "-m", "bbar.arena", *argv])

    scores = {}
    for i, j in pairs:
        result, raw = run(specs[i], specs[j], args.deals, args.seed, args.threads, progress=not args.quiet)
        append(args.log, record(result, raw, command, revision))
        games = 2 * result.deals
        print(f"{result}  [{raw['seconds']:.1f}s, {games / max(raw['seconds'], 1e-9):.0f} games/s]", flush=True)
        scores[i, j] = (result.score, result.score_lo, result.score_hi)
        scores[j, i] = (1 - result.score, 1 - result.score_hi, 1 - result.score_lo)
    if args.ladder:
        print()
        print(ladder_table(specs, scores))
    return scores


if __name__ == "__main__":
    main()
