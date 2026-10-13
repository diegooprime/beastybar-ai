"""Exact labels at the beliefs a depth-limited solve actually queries (DAgger for the value network).

    python -m bbar.value.harvest --net checkpoints/value/NAME/best.pt --out data/value/NAME-q [--labels 100000]
        [--root-plies 9,10] [--depth 1] [--iters 200] [--leaves-per-state 12] [--workers 16] [--seed S]

The network is trained on root beliefs from a fixed family (bbar/value/gen.py); inside a depth-limited solve it is
queried at the beliefs CFR's iterates induce at the value leaves, which can look different (E2: leaf error 0.11 at
two plies deep against 0.075 held out). Each worker repeatedly:
  1. samples a public state at a root ply S with root beliefs from gen.py's family (synthetic and, with --bayes, the
     blueprint's Bayes beliefs), cuts its window at ply L = S + depth and solves it for --iters CFR-D iterations with
     the network at the value leaves (bbar/value/dl.py), recording the leaf beliefs at a few iterations (the current
     iterate's, as the solver queries them) and the final average beliefs;
  2. picks up to --leaves-per-state recorded (leaf, beliefs) queries, half the most reached ones, half at random;
  3. labels each exactly like gen.py: the leaf's own window solved with those beliefs as root ranges, per-hand
     best-response values of both players.
Shards are gen.py's format (kind "dl-query" for both players; ply = L), so bbar.value.train mixes them in directly.
"""
import argparse
import json
import multiprocessing as mp
import os
import shlex
import socket
import sys
import time
from pathlib import Path

import numpy as np

from bbar import provenance
from bbar._core import Window, build_id, public_key
from bbar.value import pbs
from bbar.value.dl import leaf_windows
from bbar.value.gen import POLICIES, bayes_pair, write_shard
from bbar.value.sample import StateSampler, parse_mix

KIND = "dl-query"


def record_solve(dl, net_values, iters, record_at):
    """Solve `dl` with network leaves; returns [(leaf j, beliefs [2, 495], weight)] recorded at iterations
    `record_at` (current iterate) and at the end (average beliefs). weight = product of the reach masses."""
    from bbar.value.dl import NetLeaf
    leaf = NetLeaf(dl, net_values)
    seen = []

    def fn(player, t, beliefs, mass):
        if player == 0 and t in record_at:
            seen.extend((j, beliefs[j].copy(), float(mass[j, 0] * mass[j, 1])) for j in range(len(mass)))
        return leaf(player, t, beliefs, mass)

    dl.solve_external(fn, iters)
    bel, mass = dl.leaf_query()
    seen.extend((j, bel[j].copy(), float(mass[j, 0] * mass[j, 1])) for j in range(len(mass)))
    return seen


def pick(queries, k, rng):
    """Up to k queries: half the most reached, half uniformly at random among the rest."""
    if len(queries) <= k:
        return list(queries)
    order = np.argsort([-q[2] for q in queries])
    top = list(order[:k // 2])
    rest = rng.choice(order[k // 2:], size=k - len(top), replace=False)
    return [queries[i] for i in [*top, *rest]]


def label_query(window, beliefs, target, max_iters, algo="dcfr"):
    """gen.py's label of the leaf window with `beliefs` (absolute players) as root ranges."""
    t0 = time.perf_counter()
    window.set_ranges(beliefs[0], beliefs[1])
    res = window.solve(target=target, max_iters=max_iters, check_every=10, algo=algo)
    vals = np.stack([window.br_values(0), window.br_values(1)])
    return res, vals, np.stack(window.ranges()), time.perf_counter() - t0


def worker(args):
    wid, cfg, meta = args
    import torch
    torch.set_num_threads(1)
    from bbar.value.net import NetValues, load_value_net
    net, _ = load_value_net(cfg["net"], "cpu")
    nv = NetValues(net, "cpu")
    rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], wid]))
    sampler = StateSampler(int(rng.integers(2 ** 63)), parse_mix(cfg["mix"]), cfg["blueprint"], envs=cfg["envs"],
                           explore=cfg["explore"])
    model = None
    if cfg["bayes"] > 0:
        from bbar.policy import NetModel
        model = NetModel(cfg["blueprint"], "cpu")
    out = Path(cfg["out"])
    wmeta = dict(meta, worker=wid, worker_seed=[cfg["seed"], wid])
    rows, shard, done, states = [], 0, 0, 0
    deadline = time.time() + 60 * cfg["minutes"] if cfg["minutes"] else float("inf")
    record_at = set(int(x) for x in np.linspace(cfg["iters"] // 4, cfg["iters"] - 1, 3))

    def flush():
        nonlocal rows, shard
        if rows:
            write_shard(out / f"s{cfg['seed']}-w{wid:03d}-{shard:05d}.npz", rows, wmeta)
            print(f"  worker {wid}: shard {shard} ({len(rows)} labels, {done} in all, {states} states)",
                  file=sys.stderr, flush=True)
            rows, shard = [], shard + 1

    while done < cfg["labels_per_worker"] and time.time() < deadline:
        for x in sampler.batch(cfg["root_plies"]):
            u = pbs.decode_keys([public_key(x.game)])[2][0]
            if model is not None and rng.random() < cfg["bayes"]:
                rk = bayes_pair(model, x, rng)
            else:
                rk = [pbs.sample_range(u[p], rng) for p in (0, 1)]
            leaf_ply = x.ply + cfg["depth"]
            dl = Window(x.game, rk[0][0], rk[1][0], leaf_ply=leaf_ply)
            if dl.num_value_leaves == 0:
                continue
            states += 1
            queries = pick(record_solve(dl, nv, cfg["iters"], record_at), cfg["leaves_per_state"], rng)
            windows = leaf_windows(dl)
            for j, bel, _ in queries:
                g = dl.leaf_game(j)
                mover = g.to_move
                res, vals, ranges, secs = label_query(windows[j], bel, cfg["target"], cfg["max_iters"])
                gv = res["value"] if mover == 0 else 1 - res["value"]
                rows.append(dict(
                    key=public_key(g), start=g.start, ply=leaf_ply, ranges=pbs.mover_relative(ranges, mover),
                    values=pbs.mover_relative(vals, mover), game_value=gv, exploitability=res["exploitability"],
                    iterations=res["iterations"], kinds=[pbs.RANGE_KINDS.index(KIND)] * 2,
                    policies=[POLICIES.index(p) for p in x.policies], seconds=secs))
                done += 1
            if len(rows) >= cfg["shard_size"]:
                flush()
            if done >= cfg["labels_per_worker"] or time.time() >= deadline:
                break
    flush()
    return wid, done, states


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bbar.value.harvest", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", required=True, help="value network the depth-limited solves query")
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", type=int, default=100_000)
    ap.add_argument("--minutes", type=float, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--root-plies", default="9,10")
    ap.add_argument("--depth", type=int, default=1, help="leaf ply = root ply + depth")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--leaves-per-state", type=int, default=12)
    ap.add_argument("--target", type=float, default=1e-5)
    ap.add_argument("--max-iters", type=int, default=20_000)
    ap.add_argument("--mix", default="random:0.2,pimc:0.3,bp:0.5")
    ap.add_argument("--blueprint", default="checkpoints/bp-v1/step_30518.pt")
    ap.add_argument("--bayes", type=float, default=0.5)
    ap.add_argument("--explore", type=float, default=0.05)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    try:
        provenance.check_build()
    except provenance.StaleBuild as e:
        ap.error(str(e))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if sorted(out.glob(f"s{a.seed}-w*.npz")):
        ap.error(f"{out} already has shards of seed {a.seed}: use another --seed")
    workers = max(1, a.workers)
    cfg = dict(net=a.net, out=str(out), root_plies=[int(x) for x in a.root_plies.split(",")], depth=a.depth,
               iters=a.iters, leaves_per_state=a.leaves_per_state, target=a.target, max_iters=a.max_iters,
               mix=a.mix, blueprint=a.blueprint, bayes=a.bayes, explore=a.explore, envs=a.envs,
               shard_size=a.shard_size, seed=a.seed, minutes=a.minutes, labels_per_worker=-(-a.labels // workers))
    meta = dict(kind="value-labels-dl-query", pbs_version=pbs.PBS_VERSION, revision=provenance.revision(),
                build=build_id(), host=socket.gethostname(), config=cfg,
                command=shlex.join(["python", "-m", "bbar.value.harvest", *(argv if argv is not None else
                                                                             sys.argv[1:])]),
                range_kinds=list(pbs.RANGE_KINDS), policies=list(POLICIES),
                created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    t0 = time.time()
    jobs = [(i, cfg, meta) for i in range(workers)]
    if workers == 1:
        results = list(map(worker, jobs))
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = list(pool.imap_unordered(worker, jobs))
    n, states = sum(r[1] for r in results), sum(r[2] for r in results)
    wall = time.time() - t0
    print(f"{n} labels from {states} depth-limited solves in {wall:.1f}s", flush=True)
    summary = dict(meta, wall_seconds=wall, labels=n, states=states)
    (out / f"summary-s{a.seed}.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
