"""Pool bbar.delta runs of the same agent (several seeds, several pods) and compare agents position by position.

    python3 scripts/delta_pool.py FILE.jsonl [...] [--T 8] [--append runs/delta-summary.jsonl]

Each FILE is a bbar.delta output (a "run" line, then one "position" line per position, unfinished runs included).
Prints, per agent, the pooled mean delta_T with its 95% interval (normal, over positions); for every pair of agents,
the paired difference over the positions both were measured on (same seed, same deal, same moves before T: agents
that play the same base network before T reach the same positions). --append writes one "pooled" line per agent
(its runs' seeds, commits and hosts) to the evidence trail.
"""
import argparse
import itertools
import json
import math
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

Z = 1.959963984540054


def ci(xs):
    xs = np.asarray(xs, dtype=float)
    half = Z * xs.std(ddof=1) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan")
    return float(xs.mean()), float(xs.mean() - half), float(xs.mean() + half)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--append", default=None)
    a = ap.parse_args()
    pos = defaultdict(dict)    # agent -> {(seed, deal, moves): delta}
    runs = defaultdict(list)
    for f in a.files:
        lines = [json.loads(x) for x in open(f) if x.strip()]
        if not lines or lines[0].get("type") != "run":
            continue
        run = lines[0]
        rows = [r for r in lines[1:] if r.get("type") == "position" and r["T"] == a.T]
        if not rows:
            continue
        agent = run["agent"]
        runs[agent].append({"seed": run["seed"], "positions": len(rows), "commit": run.get("commit"),
                            "code": run.get("code"), "hostname": run.get("hostname"), "time": run.get("time")})
        for r in rows:
            pos[agent][(run["seed"], r["deal"], tuple(r["actions"][:a.T]))] = r["delta"]
    names = sorted(pos)
    for i, agent in enumerate(names):
        m, lo, hi = ci(list(pos[agent].values()))
        seeds = sorted({r["seed"] for r in runs[agent]})
        print(f"[{i}] delta_{a.T} {m:.4f} [{lo:.4f}, {hi:.4f}]  n={len(pos[agent])}  seeds {seeds}\n    {agent}")
        if a.append:
            line = {"type": "pooled", "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "agent": agent, "T": a.T, "n": len(pos[agent]),
                    "delta": {"mean": m, "ci95": [lo, hi]}, "runs": runs[agent]}
            with open(a.append, "a") as out:
                out.write(json.dumps(line) + "\n")
    for i, j in itertools.combinations(range(len(names)), 2):
        common = sorted(set(pos[names[i]]) & set(pos[names[j]]))
        if len(common) < 2:
            continue
        d = [pos[names[i]][k] - pos[names[j]][k] for k in common]
        m, lo, hi = ci(d)
        print(f"[{i}] - [{j}] paired on {len(common)} positions: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
