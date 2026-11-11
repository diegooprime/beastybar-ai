#!/usr/bin/env python3
"""Builds site/data.js from the evidence trail (runs/results.jsonl, runs/delta-summary.jsonl).

Every number on the site comes from a line of those files, kept with its command and commit. The few numbers that
were measured but whose run records are not in the trail (README "What we have learned") are listed in NOTED below
with that caveat. Run: python3 scripts/site_data.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Players in display order (weakest first). `match` recognises a spec string from the evidence trail.
PLAYERS = [
    ("random", "Random", "baseline", "Uniform over legal moves.", r"^random$"),
    ("greedy", "Greedy", "baseline", "One ply: the move that looks best right after it.", r"^greedy$"),
    ("pimc4", "Search, 4 plies", "search", "Samples 8 hidden worlds, searches each 4 plies deep (pimc:w=8,d=4).",
     r"^pimc:w=8,d=4$"),
    ("iter949", "iter949", "network", "The previous attempt's best network (PPO self-play, January 2026).",
     r"^iter949:"),
    ("bp8k", "Blueprint, step 8000", "network", "The blueprint network early in training.", r"step_8000\.pt$"),
    ("pimc8", "Search, 8 plies", "search",
     "Samples 128 hidden worlds, searches each 8 plies deep (pimc:w=128,d=8). Runs in your browser.",
     r"^pimc:w=128,d=8$"),
    ("bp", "Blueprint", "network", "3.4M-parameter policy trained by MMD-PPO self-play. Plays plies 0-7 for every "
     "re-solver.", r"^nn:checkpoints/bp-v1/step_30518\.pt$"),
    ("v2c-safe", "Re-solver v2c, safe", "resolver", "First value network, exact from ply 10, gadget re-solve.",
     r"^resolve:V=checkpoints/value/v2c/.*F=10.*mode=safe"),
    ("v2c", "Re-solver v2c", "resolver", "First value network, exact from ply 10.",
     r"^resolve:V=checkpoints/value/v2c/.*F=10.*mode=plain"),
    ("v3h", "Re-solver v3h", "resolver", "Option features in the value network, exact from ply 10.",
     r"^resolve:V=checkpoints/value/v3h/.*P0=8,F=10"),
    ("c5m", "Re-solver c5m", "resolver", "Cascade labels at plies 4-9, exact from ply 9.",
     r"^resolve:V=checkpoints/value/c5m/.*F=9"),
    ("c5-2", "Re-solver c5-2", "resolver", "Best player. Value network on all 931k labels, exact from ply 9.",
     r"^resolve:V=checkpoints/value/c5-2/.*F=9"),
]
# Variants that appear in the trail but are not ladder rows (kept for the matrix tooltips).
EXTRA = [
    ("pimc-hand6", "Search knowing your hand", "oracle", r"^pimc-hand:w=32,d=6$"),
    ("cheat6", "Search knowing everything", "oracle", r"^cheat:d=6$"),
    ("pimc6", "Search, 6 plies", "search", r"^pimc:w=32,d=6$"),
    ("pimc6w", "Search, 6 plies, 128 worlds", "search", r"^pimc:w=128,d=6$"),
    ("pimc8n", "Search, 8 plies, 32 worlds", "search", r"^pimc:w=32,d=8$"),
    ("pimc4w", "Search, 4 plies, 32 worlds", "search", r"^pimc:w=32,d=4$"),
    ("c5m-10", "Re-solver c5m, exact from ply 10", "resolver", r"^resolve:V=checkpoints/value/c5m/.*F=10"),
    ("v3h-9", "Re-solver v3h, exact from ply 9", "resolver", r"^resolve:V=checkpoints/value/v3h/.*P0=8,F=9"),
    ("c1-6", "Re-solver, search from ply 6", "resolver", r"^resolve:.*P0=6"),
    ("ex-pimc", "PPO exploiter of search d=4", "exploiter", r"^nn:checkpoints/ex-pimc/final\.pt(,argmax)?$"),
    ("ex-bp", "PPO exploiter of the blueprint", "exploiter", r"^nn:checkpoints/ex-bp/final\.pt(,argmax)?$"),
]

# Measured, reported in README.md, run record not in the trail.
NOTED = {
    "value_error": [  # held-out value-network error by number of labelled public states
        {"labels": 62_000, "error": 0.070, "net": "v1"},
        {"labels": 139_000, "error": 0.063, "net": "v2"},
        {"labels": 253_000, "error": 0.057, "net": "v3"},
        {"labels": 152_000, "error": 0.0555, "net": "v3h", "note": "option features, 60% of the data"},
        {"labels": 931_000, "error": 0.0444, "net": "c5-2", "note": "test set includes the new plies"},
    ],
    "window": [  # exact window solves by root ply
        {"ply": 10, "seconds": "0.4 s"}, {"ply": 9, "seconds": "1-5 s"}, {"ply": 8, "seconds": "2 min"},
    ],
    "bp8k_delta": {"T": 9, "mean": 0.127, "lo": 0.119, "hi": 0.135},
}


def ident(spec):
    for row in PLAYERS:
        if re.search(row[4], spec):
            return row[0]
    for pid, _, _, pat in EXTRA:
        if re.search(pat, spec):
            return pid
    return None


def matches():
    seen = {}
    for line in (ROOT / "runs/results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "a" not in r:
            continue
        a, b = ident(r["a"]), ident(r["b"])
        if a is None or b is None:
            continue
        lo = r.get("score_lo", r["score"] - r.get("ci95", 0))
        hi = r.get("score_hi", r["score"] + r.get("ci95", 0))
        row = {"a": a, "b": b, "score": round(r["score"], 4), "lo": round(lo, 4), "hi": round(hi, 4),
               "pairs": r["deals"], "margin": r.get("margin"), "time": r["time"][:16], "commit": r["commit"],
               "command": r["command"], "argmax": r["a"].endswith(",argmax")}
        key = (a, b, row["argmax"])
        # Later runs of the same pair replace earlier ones unless they are smaller (the trail is append-only).
        if key not in seen or row["pairs"] >= seen[key]["pairs"]:
            seen[key] = row
    return list(seen.values())


def exploitability():
    out, run_agent = [], None
    for line in (ROOT / "runs/delta-summary.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d["type"] == "run":
            run_agent = d["agent"]
        if d["type"] not in ("summary", "pooled"):
            continue
        pid = ident(d.get("agent") or run_agent)   # a summary without its own agent belongs to the run above it
        if pid is None:
            continue
        out.append({"player": pid, "T": d["T"], "mean": round(d["delta"]["mean"], 5),
                    "lo": round(d["delta"]["ci95"][0], 5), "hi": round(d["delta"]["ci95"][1], 5),
                    "n": d["n"], "kind": d["type"]})
    return out


def paired():
    out = []
    for line in (ROOT / "runs/delta-summary.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d["type"] == "paired":
            out.append({"a": ident(d["a"]), "b": ident(d["b"]), "T": d["T"], "n": d["n"],
                        "diff": round(d["a_minus_b"]["mean"], 5), "lo": round(d["a_minus_b"]["ci95"][0], 5),
                        "hi": round(d["a_minus_b"]["ci95"][1], 5)})
    return out


def e2():
    out = []
    for line in (ROOT / "runs/results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("experiment") != "E2":
            continue
        m = r["metrics"]
        out.append({"ply": r["ply"], "positions": r["positions"],
                    "metrics": {k: round(v["mean"], 6) for k, v in m.items() if isinstance(v, dict) and "mean" in v}})
    return out


def main():
    data = {
        "players": [{"id": p[0], "name": p[1], "family": p[2], "note": p[3]} for p in PLAYERS]
        + [{"id": p[0], "name": p[1], "family": p[2], "extra": True} for p in EXTRA],
        "matches": matches(),
        "exploitability": exploitability(),
        "paired": paired(),
        "e2": e2(),
        "noted": NOTED,
    }
    text = ("// Generated by scripts/site_data.py from runs/results.jsonl and runs/delta-summary.jsonl. Do not edit.\n"
            "export const DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
    (ROOT / "site/data.js").write_text(text)
    print(f"site/data.js: {len(data['matches'])} matches, {len(data['exploitability'])} exploitability rows")


if __name__ == "__main__":
    main()
