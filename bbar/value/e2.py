"""E2: can a PBS value network replace exact solving? Exact exploitability of depth-limited play in full windows.

    python -m bbar.value.e2 --net checkpoints/value/NAME/best.pt [--positions 64] [--ply 9] [--leaf-plies 10,11]
        [--iters 200] [--oracle-positions 8] [--workers 4] [--threads 4] [--seed 0] [--out runs/e2/NAME.jsonl]
        [--log runs/results.jsonl] [--allow-dirty]

Positions: public states at ply S (--ply; 8 is possible on a big machine) from games of the policy mix of
bbar/value/sample.py, root ranges drawn per player from the label family (bbar/value/pbs.py). For each:
  - exact: the ply-S window solved by DCFR to --exact-target (the reference; its exploitability and time);
  - uniform: exploitability of uniform play (scale);
  - for each leaf ply L (--leaf-plies): the depth-limited window from S with value leaves at L, solved by --iters
    DCFR iterations whose leaf values come from the network (CFR-D: the iterate's beliefs; bbar/value/dl.py);
    then the COMPOSITE profile in the full window — plies [S, L) from the depth-limited average strategy, below each
    leaf both players' safe re-solves (gadget with the opponent cfvs the solve implies there and our average belief)
    — and its exact exploitability in the full window (`net`). Also: the exploitability of the same trunk with the
    exact solution below the leaves (`trunk_eq`, the trunk's own error: the composite differs from it by what the
    safe re-solves add, as they must respect the averaged opponent values, only upper bounds of the opponent's
    best-response values, or remove, as they answer the trunk; examples both ways in tests/test_value.py), the
    network's leaf error at the solve's final average beliefs against exact leaf values (on-path range-weighted MAE,
    reach-weighted), and the root value of the depth-limited solve against the exact one;
  - for the first --oracle-positions positions, the same with exact leaf values (`oracle`: each leaf's window solved
    with the query's beliefs every half-iteration): what a perfect network would give with the same iterations.
  The blueprint network would need a table filler (its policy written into window profiles) that this tree does not
  have: skipped.
Exploitability = (BR_0 + BR_1 - 1) / 2 in win probability, exact, for the full window with its root ranges.

Cost (seconds, wall clock with --threads): `exact_seconds` = full window build + exact solve; `L*_net_seconds` = what
a player using the network spends: depth-limited window build + solve (network calls included); `L*_oracle_seconds`
= the same plus building the leaf windows the oracle solves. Evaluation-only work is logged apart
(`*_eval_overhead_seconds`: the composite's re-solves below every leaf for both players, the exact scoring and, for
the network, the leaf windows and the leaf-error check); the leaf windows are built once per position and leaf ply
and their build time is recorded in both rows.

Output: --out (default runs/e2/<net name>-S<ply>.jsonl): a "config" line (provenance: commit, code hash, engine
build, host, command, network SHA-256, arguments), one "position" line per position (as finished), and a "summary"
line: per metric, n, mean, median, p90 (each with a 95% bootstrap interval over positions) and max. The summary also
goes to the evidence trail (--log, default runs/results.jsonl: per metric n, mean and median with their intervals,
max; command, seed, commit, code hash, build, network SHA-256), which like bbar.arena's only takes committed code
unless --allow-dirty.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import shlex
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bbar import arena, match, provenance
from bbar._core import Window, build_id, public_key
from bbar.value import dl as D
from bbar.value import pbs
from bbar.value.sample import StateSampler, parse_mix, public_game

ROOT = Path(__file__).resolve().parents[2]


def sample_positions(n, ply, seed, mix, blueprint):
    """n positions: dicts with start, actions, ranges (2 x 495, absolute players), kinds, policies."""
    rng = np.random.default_rng([seed, 1])
    sampler = StateSampler(int(rng.integers(2 ** 63)), mix, blueprint, envs=max(1, n))
    out = []
    while len(out) < n:
        for x in sampler.batch((ply,)):
            key = public_key(x.game)
            _, _, u, _, _ = pbs.decode_keys([key])
            rk = [pbs.sample_range(u[0, p], rng) for p in (0, 1)]
            out.append(dict(index=len(out), start=x.start, actions=x.actions, key=int(key), ply=ply,
                            ranges=[rk[0][0].tolist(), rk[1][0].tolist()], kinds=[rk[0][1], rk[1][1]],
                            policies=list(x.policies)))
            if len(out) == n:
                break
    return out


def depth_limited(full, eq, game, ranges, leaf_ply, leaf_fn_factory, iters, threads, gadget_target, prior, leaf,
                  leaf_in_cost):
    """One depth-limited solve and its composite: dict of results, and the solved window and leaf function.

    leaf: {"windows", "seconds"}, the leaf windows of this leaf ply and the time their build took, filled here on
    first use (built once per position and leaf ply, shared by the network and oracle rows: both record the same
    build time). Cost columns: `seconds` = what a player using the method spends (the depth-limited window build and
    the solve, plus the leaf windows' build when leaf_in_cost: the oracle solves them); the composite (re-solves
    below every leaf, for both players), the scoring and, for the network, the leaf windows are evaluation overhead,
    logged separately. eq: the full window's exact solution, to score the same trunk with it below the leaves."""
    t0 = time.perf_counter()
    dl = Window(game, ranges[0], ranges[1], leaf_ply=leaf_ply, threads=threads)
    t_dl = time.perf_counter() - t0
    if leaf["windows"] is None:
        t = time.perf_counter()
        leaf["windows"] = D.leaf_windows(dl, threads)
        leaf["seconds"] = time.perf_counter() - t
    ws = leaf["windows"]
    fn = leaf_fn_factory(dl, ws)
    t1 = time.perf_counter()
    res = dl.solve_external(fn, iters=iters)
    t_solve = time.perf_counter() - t1
    t2 = time.perf_counter()
    prof, info = D.composite(full, dl, ws, gadget_target=gadget_target, prior=prior, threads=threads)
    t_comp = time.perf_counter() - t2
    assert info["entries"] == info["full_entries"], info
    t3 = time.perf_counter()
    e = full.exploitability(prof, prof)
    # The same trunk with the exact solution below the leaves: the trunk's own error, apart from what the safe
    # re-solves add (the averaged opponent values they respect are upper bounds) or remove (they answer the trunk).
    te, _ = full.embed(eq, dl)
    e_te = full.exploitability(te, te)["exploitability"]
    t_score = time.perf_counter() - t3
    out = dict(exploitability=e["exploitability"], value=e["value"], trunk_eq_exploitability=e_te,
               value_leaves=dl.num_value_leaves, trunk_nodes=dl.sizes()["internal"],
               seconds=t_dl + t_solve + (leaf["seconds"] if leaf_in_cost else 0.0),
               dl_build_seconds=t_dl, solve_seconds=t_solve, callback_seconds=res["callback_seconds"],
               leaf_build_seconds=leaf["seconds"], composite_seconds=t_comp, score_seconds=t_score,
               gadget_exploitability=info["gadget_exploitability"],
               gadget_max_violation=info["gadget_max_violation"])
    return out, dl, fn


def evaluate_position(args):
    pos, cfg = args
    import torch
    torch.set_num_threads(1)
    threads = cfg["threads"]
    t0 = time.time()
    game = public_game(pos["start"], pos["actions"])
    assert public_key(game) == pos["key"]
    ranges = [np.array(r) for r in pos["ranges"]]
    full = Window(game, ranges[0], ranges[1], threads=threads)
    sz = full.sizes()
    t1 = time.time()
    exact = full.solve(target=cfg["exact_target"], max_iters=cfg["exact_iters"])
    eq = full.profile_array()
    row = dict(kind="position", index=pos["index"], ply=pos["ply"], key=pos["key"], start=pos["start"],
               kinds=pos["kinds"], policies=pos["policies"], internal=sz["internal"], entries=sz["entries"],
               build_seconds=t1 - t0, exact=dict(exploitability=exact["exploitability"], value=exact["value"],
                                                 iterations=exact["iterations"], seconds=exact["seconds"]),
               uniform=full.exploitability("uniform", "uniform")["exploitability"])
    net = None
    if cfg["net"]:
        from bbar.value.net import NetValues, load_value_net
        net = NetValues(load_value_net(cfg["net"], cfg["device"])[0], cfg["device"])
    for L in cfg["leaf_plies"]:
        res = {}
        leaf = dict(windows=None, seconds=None)
        common = (full, eq, game, ranges, L)
        solve = (cfg["iters"], threads, cfg["gadget_target"], cfg["prior"], leaf)
        if net is not None:
            r, dl, fn = depth_limited(*common, lambda dl, ws: D.NetLeaf(dl, net), *solve, leaf_in_cost=False)
            # The network's leaf values at the solve's final average beliefs, against exact ones.
            bel, _ = dl.leaf_query()
            t2 = time.perf_counter()
            exact_vals = D.LeafOracle(dl, target=cfg["oracle_target"], windows=leaf["windows"],
                                      threads=threads).values(bel, warm=False)
            r["leaf_mae"], r["leaf_max"] = D.leaf_error(dl, fn.values(bel), exact_vals)
            r["leaf_check_seconds"] = time.perf_counter() - t2
            r["dl_value"] = dl.exploitability(None, None, leaf_values=dl.leaf_values_avg())["value"]
            res["net"] = r
        if pos["index"] < cfg["oracle_positions"]:
            stats = {}

            def oracle(dl, ws):
                o = D.LeafOracle(dl, target=cfg["oracle_target"], windows=ws, threads=threads)
                stats["o"] = o
                return o
            r, dl, _ = depth_limited(*common, oracle, *solve, leaf_in_cost=True)
            r["leaf_solves"] = stats["o"].stats["solves"]
            r["leaf_iterations"] = stats["o"].stats["iterations"]
            res["oracle"] = r
        row[f"L{L}"] = res
    row["seconds"] = time.time() - t0
    return row


def bootstrap(x, stat, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    vals = stat(x[idx], axis=1)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def p90(x, axis=None):
    return np.quantile(x, 0.9, axis=axis)


def describe(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return dict(n=0)
    out = dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)), p90=float(p90(x)), max=float(x.max()))
    if len(x) > 1:
        out.update(mean_ci=bootstrap(x, np.mean), median_ci=bootstrap(x, np.median), p90_ci=bootstrap(x, p90))
    return out


def summarize(rows, leaf_plies):
    metrics = {"exact_exploitability": [r["exact"]["exploitability"] for r in rows],
               "exact_seconds": [r["exact"]["seconds"] + r["build_seconds"] for r in rows],
               "uniform_exploitability": [r["uniform"] for r in rows]}
    for L in leaf_plies:
        for kind in ("net", "oracle"):
            rs = [r[f"L{L}"][kind] for r in rows if kind in r[f"L{L}"]]
            if not rs:
                continue
            metrics[f"L{L}_{kind}_exploitability"] = [x["exploitability"] for x in rs]
            metrics[f"L{L}_{kind}_trunk_eq_exploitability"] = [x["trunk_eq_exploitability"] for x in rs]
            metrics[f"L{L}_{kind}_seconds"] = [x["seconds"] for x in rs]
            metrics[f"L{L}_{kind}_eval_overhead_seconds"] = [
                x["composite_seconds"] + x["score_seconds"] +
                (x["leaf_build_seconds"] + x["leaf_check_seconds"] if kind == "net" else 0.0) for x in rs]
            if kind == "net":
                metrics[f"L{L}_net_leaf_mae"] = [x["leaf_mae"] for x in rs]
                metrics[f"L{L}_net_root_value_error"] = [abs(x["dl_value"] - r["exact"]["value"])
                                                         for x, r in zip(rs, [r for r in rows if "net" in r[f"L{L}"]])]
    return {k: describe(v) for k, v in metrics.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bbar.value.e2", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", default="", help="value network checkpoint ('' = oracle, exact and uniform only)")
    ap.add_argument("--positions", type=int, default=64)
    ap.add_argument("--ply", type=int, default=9, help="root ply S of the positions")
    ap.add_argument("--leaf-plies", default="10,11", help="value-leaf plies L (each > S)")
    ap.add_argument("--iters", type=int, default=200, help="DCFR iterations of each depth-limited solve")
    ap.add_argument("--oracle-positions", type=int, default=8,
                    help="run the exact-leaf (oracle) solve on the first K positions (expensive)")
    ap.add_argument("--oracle-target", type=float, default=1e-6, help="exploitability of each oracle leaf solve")
    ap.add_argument("--exact-target", type=float, default=1e-6)
    ap.add_argument("--exact-iters", type=int, default=50_000)
    ap.add_argument("--gadget-target", type=float, default=1e-6)
    ap.add_argument("--prior", default="uniform", choices=("uniform", "belief"), help="re-solving gadget prior")
    ap.add_argument("--mix", default="random:0.2,pimc:0.4,bp:0.4")
    ap.add_argument("--blueprint", default="checkpoints/bp-v1/step_8000.pt")
    ap.add_argument("--workers", type=int, default=4, help="positions evaluated in parallel (processes)")
    ap.add_argument("--threads", type=int, default=4, help="solver threads per position")
    ap.add_argument("--device", default="cpu", help="network device in the workers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--log", default=str(arena.LOG),
                    help="evidence trail the summary line is appended to (default runs/results.jsonl; '' = none)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="append to the evidence trail from uncommitted code (the line records its code hash)")
    a = ap.parse_args(argv)
    try:   # results must come from the engine these sources build (and, in the trail, from committed code)
        if a.log:
            revision = match.check_trail(a.log, a.allow_dirty)
        else:
            provenance.check_build()
            revision = provenance.revision()
    except (provenance.StaleBuild, ValueError) as e:
        ap.error(str(e))
    leaf_plies = [int(x) for x in a.leaf_plies.split(",")]
    if any(L <= a.ply or L > 14 for L in leaf_plies):
        ap.error("--leaf-plies must lie in (ply, 14]")
    mix = parse_mix(a.mix)
    blueprint = a.blueprint if a.blueprint and mix.get("bp", 0) > 0 else ""
    if blueprint and not Path(blueprint).is_file():
        ap.error(f"no blueprint checkpoint {blueprint} (--blueprint '' drops bp from the mix)")
    if a.net and not Path(a.net).is_file():
        ap.error(f"no network {a.net}")
    out = Path(a.out or ROOT / "runs" / "e2" / f"{Path(a.net).parent.name if a.net else 'oracle'}-S{a.ply}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(vars(a), leaf_plies=leaf_plies, blueprint=blueprint)
    header = dict(kind="config", experiment="E2", revision=revision, build=build_id(),
                  host=socket.gethostname(), time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  command=shlex.join(["python", *sys.orig_argv[1:]]) if argv is None else
                  shlex.join(["python", "-m", "bbar.value.e2", *argv]),
                  net_sha256=hashlib.sha256(Path(a.net).read_bytes()).hexdigest()[:16] if a.net else None,
                  blueprint_sha256=hashlib.sha256(Path(blueprint).read_bytes()).hexdigest()[:16] if blueprint else None,
                  config=cfg)
    t0 = time.time()
    positions = sample_positions(a.positions, a.ply, a.seed, mix, blueprint)
    print(f"E2: {len(positions)} positions at ply {a.ply}, leaves at {leaf_plies}, {a.iters} iterations, "
          f"oracle on {min(a.oracle_positions, len(positions))}; {a.workers} workers x {a.threads} threads -> {out}",
          flush=True)
    rows = []
    with open(out, "a") as f:
        f.write(json.dumps(header) + "\n")
        jobs = [(p, cfg) for p in positions]
        it = map(evaluate_position, jobs) if a.workers <= 1 else \
            mp.get_context("spawn").Pool(a.workers).imap_unordered(evaluate_position, jobs)
        for row in it:
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            parts = [f"#{row['index']} exact {row['exact']['exploitability']:.1e} uniform {row['uniform']:.3f}"]
            for L in leaf_plies:
                for kind, r in row[f"L{L}"].items():
                    parts.append(f"L{L} {kind} {r['exploitability']:.2e}")
            print("  " + ", ".join(parts) + f" [{row['seconds']:.0f}s]", flush=True)
        summary = summarize(sorted(rows, key=lambda r: r["index"]), leaf_plies)
        seconds = time.time() - t0
        f.write(json.dumps(dict(kind="summary", positions=len(rows), seconds=seconds, metrics=summary,
                                blueprint="skipped: no table filler (policy -> window profile) in this tree")) + "\n")
    if a.log:
        arena.append(a.log, trail_record(header, summary, len(rows), seconds, out))
    print(f"summary ({len(rows)} positions, {seconds:.0f}s){f'; appended to {a.log}' if a.log else ''}:")
    for k, v in summary.items():
        if v["n"]:
            ci = v.get("mean_ci", (float("nan"),) * 2)
            print(f"  {k:<36} n={v['n']:<4} mean {v['mean']:.3e} [{ci[0]:.3e}, {ci[1]:.3e}]  median "
                  f"{v['median']:.3e}  p90 {v['p90']:.3e}  max {v['max']:.3e}")
    return summary


def trail_record(header, summary, positions, seconds, out):
    """The evidence-trail line of a run (runs/results.jsonl, as bbar.arena's match lines): per metric n, mean and
    median with their 95% bootstrap intervals over positions, max; the run's configuration and provenance."""
    c = header["config"]
    rnd = lambda x: float(f"{x:.4g}")  # noqa: E731
    metrics = {}
    for k, v in summary.items():
        if v["n"]:
            m = dict(n=v["n"], mean=rnd(v["mean"]), median=rnd(v["median"]), max=rnd(v["max"]))
            for q in ("mean", "median"):
                if f"{q}_ci" in v:
                    m[f"{q}_ci95"] = [rnd(x) for x in v[f"{q}_ci"]]
            metrics[k] = m
    try:
        rel = str(Path(out).resolve().relative_to(ROOT))
    except ValueError:
        rel = str(out)
    return {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "experiment": "E2", "out": rel,
        "positions": positions, "ply": c["ply"], "leaf_plies": c["leaf_plies"], "iters": c["iters"],
        "oracle_positions": min(c["oracle_positions"], positions), "seed": c["seed"], "mix": c["mix"],
        "net": c["net"] or None, "net_sha256": header["net_sha256"], "blueprint_sha256": header["blueprint_sha256"],
        "metrics": metrics, "ci": "bootstrap-95 over positions", "seconds": round(seconds, 1),
        "workers": c["workers"], "threads": c["threads"],
        "commit": header["revision"]["commit"], "code": header["revision"]["code"], "build": header["build"],
        "hostname": header["host"], "command": header["command"],
    }


if __name__ == "__main__":
    main()
