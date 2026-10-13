"""Exact labels for the public-belief-state value network: per-hand values of both players from exact window solves.

    python -m bbar.value.gen --out data/value/NAME [--labels 100000 | --minutes 60] [--workers 16]
        [--plies 10,11,12,13] [--ply-weights 0.4,0.3,0.2,0.1] [--ranges-per-state 4] [--target 1e-5]
        [--mix random:0.2,pimc:0.4,bp:0.4] [--blueprint checkpoints/bp-v1/step_8000.pt] [--seed S]

Each worker process (one solver thread each) repeatedly:
  1. plays a batch of games with the policy mix (bbar/value/sample.py) and takes their public states at the plies
     (each kept with its ply weight, the largest weight = 1);
  2. builds the exact window of each state (frontier leaves at ply 15) once, then for each of --ranges-per-state root
     range pairs drawn independently per player from the family of bbar/value/pbs.py (uniform, Dirichlet 0.1 / 0.3 /
     1, sparse supports, peaked around a hand, mixtures) solves it with DCFR until the exploitability of the average
     strategy is at most --target;
  3. labels it: per hand of each player, the best-response counterfactual value against the other's equilibrium
     strategy with the other's range normalised (= the hand's win probability; for hands in the range's support it
     equals the equilibrium value up to the solve's exploitability, and it is the right value for the others), plus
     the game value (the mover's expected win probability under the solved profile).
Shards: OUT/s<seed>-w<worker>-<k>.npz (compressed), arrays per label:
    key uint64 (public Markov key), start int8, ply int8, ranges float32 [2, 495] and values float32 [2, 495] (mover,
    opp; normalised ranges; values 0 outside the feasible hands), game_value float32 (mover's), exploitability
    float32, iterations int32, kinds int8 [2] (index in pbs.RANGE_KINDS), policies int8 [2] (seat policies of the
    game, index in POLICIES), seconds float32 (solve time of the label)
plus `meta` (JSON: provenance, configuration, worker seed). OUT/summary-s<seed>.json: labels and labels/sec per
thread by ply (window build time shared over the state's labels). One run per seed in OUT: another run of the same
seed needs --force, which deletes that seed's shards and summary first (bbar/value/train.py loads every shard of OUT).
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
from bbar.value.sample import StateSampler, parse_mix

POLICIES = ("random", "pimc", "bp")


def label_state(game, n_ranges, rng, target, max_iters, algo="dcfr"):
    """Labels of one public state: a list of dicts (see the module docstring) and the seconds spent."""
    t0 = time.perf_counter()
    key = public_key(game)
    mover = game.to_move
    _, _, u, _, _ = pbs.decode_keys([key])
    w = Window(game, threads=1)
    out = []
    for _ in range(n_ranges):
        rk = [pbs.sample_range(u[0, p], rng) for p in (0, 1)]
        w.set_ranges(rk[0][0], rk[1][0])
        t1 = time.perf_counter()
        res = w.solve(target=target, max_iters=max_iters, check_every=10, algo=algo)
        vals = np.stack([w.br_values(0), w.br_values(1)])
        ranges = np.stack(w.ranges())
        gv = res["value"] if mover == 0 else 1 - res["value"]
        out.append(dict(
            key=key, start=game.start, ranges=pbs.mover_relative(ranges, mover),
            values=pbs.mover_relative(vals, mover), game_value=gv, exploitability=res["exploitability"],
            iterations=res["iterations"], kinds=[pbs.RANGE_KINDS.index(rk[p][1]) for p in
                                                 ((mover, 1 - mover))],
            seconds=time.perf_counter() - t1))
    return out, time.perf_counter() - t0


def write_shard(path, rows, meta):
    arr = lambda k, dt: np.array([r[k] for r in rows], dtype=dt)  # noqa: E731
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp, key=arr("key", np.uint64), start=arr("start", np.int8), ply=arr("ply", np.int8),
        ranges=arr("ranges", np.float32), values=arr("values", np.float32),
        game_value=arr("game_value", np.float32), exploitability=arr("exploitability", np.float32),
        iterations=arr("iterations", np.int32), kinds=arr("kinds", np.int8), policies=arr("policies", np.int8),
        seconds=arr("seconds", np.float32), meta=np.array(json.dumps(meta)))
    os.replace(tmp, path)


def worker(args):
    wid, cfg, meta = args
    if cfg["blueprint"]:
        import torch
        torch.set_num_threads(1)
    seed = np.random.SeedSequence([cfg["seed"], wid])
    rng = np.random.default_rng(seed)
    sampler = StateSampler(int(rng.integers(2 ** 63)), parse_mix(cfg["mix"]), cfg["blueprint"], envs=cfg["envs"],
                           explore=cfg["explore"])
    plies = cfg["plies"]
    wmax = max(cfg["ply_weights"])
    keep = {p: w / wmax for p, w in zip(plies, cfg["ply_weights"])}
    quota = cfg["labels_per_worker"]
    deadline = time.time() + 60 * cfg["minutes"] if cfg["minutes"] else float("inf")
    stats = {p: {"labels": 0, "states": 0, "seconds": 0.0, "iterations": 0} for p in plies}
    rows, shard, done = [], 0, 0
    wmeta = dict(meta, worker=wid, worker_seed=[cfg["seed"], wid])
    out = Path(cfg["out"])

    def flush():
        nonlocal rows, shard
        if rows:
            write_shard(out / f"s{cfg['seed']}-w{wid:03d}-{shard:05d}.npz", rows, wmeta)
            print(f"  worker {wid}: shard {shard} ({len(rows)} labels, {done} in all)", file=sys.stderr, flush=True)
            rows, shard = [], shard + 1

    while done < quota and time.time() < deadline:
        batch = sampler.batch(plies, keep)
        for i in rng.permutation(len(batch)):   # interleave the plies (the batch comes in ply order)
            ply, game, pol = batch[i].ply, batch[i].game, batch[i].policies
            labels, secs = label_state(game, cfg["ranges_per_state"], rng, cfg["target"], cfg["max_iters"],
                                       cfg["algo"])
            st = stats[ply]
            st["states"] += 1
            st["seconds"] += secs
            for lab in labels:
                lab.update(ply=ply, policies=[POLICIES.index(x) for x in pol])
                st["labels"] += 1
                st["iterations"] += lab["iterations"]
            rows += labels
            done += len(labels)
            if len(rows) >= cfg["shard_size"]:
                flush()
            if done >= quota or time.time() >= deadline:
                break
    flush()
    return wid, stats


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bbar.value.gen", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory (data/value/NAME)")
    ap.add_argument("--labels", type=int, default=100_000, help="labels in all (split over the workers)")
    ap.add_argument("--minutes", type=float, default=0, help="also stop after this many minutes (0 = no limit)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--plies", default="10,11,12,13")
    ap.add_argument("--ply-weights", default="0.4,0.3,0.2,0.1",
                    help="relative probability of keeping a game's state at each ply")
    ap.add_argument("--ranges-per-state", type=int, default=4)
    ap.add_argument("--target", type=float, default=1e-5, help="exploitability of each exact solve")
    ap.add_argument("--max-iters", type=int, default=20_000)
    ap.add_argument("--algo", default="dcfr")
    ap.add_argument("--mix", default="random:0.2,pimc:0.4,bp:0.4", help="seat policy mix (random, pimc, bp)")
    ap.add_argument("--blueprint", default="checkpoints/bp-v1/step_8000.pt",
                    help="blueprint checkpoint for 'bp' ('' = drop bp from the mix)")
    ap.add_argument("--explore", type=float, default=0.05, help="probability of a uniformly random move")
    ap.add_argument("--envs", type=int, default=32, help="games per sampling batch in a worker")
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="replace the run of the same seed in --out (its shards and summary are deleted first)")
    a = ap.parse_args(argv)
    try:
        provenance.check_build()   # results must come from the engine these sources build
    except provenance.StaleBuild as e:
        ap.error(str(e))
    plies = [int(x) for x in a.plies.split(",")]
    weights = [float(x) for x in a.ply_weights.split(",")]
    if len(weights) != len(plies) or min(weights) <= 0:
        ap.error("--ply-weights: one positive weight per ply")
    if any(not 1 <= p <= 14 for p in plies):
        ap.error("--plies must be in 1..14 (ply 15 on is a frontier: exact by the perfect-information solver)")
    blueprint = a.blueprint if a.blueprint and parse_mix(a.mix).get("bp", 0) > 0 else ""
    if blueprint and not Path(blueprint).is_file():
        ap.error(f"no blueprint checkpoint {blueprint} (pass --blueprint '' to drop bp from the mix)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    old = sorted(out.glob(f"s{a.seed}-w*.npz"))       # shards of this seed (and unfinished .tmp.npz ones)
    summary_path = out / f"summary-s{a.seed}.json"
    if old and not a.force:
        ap.error(f"{out} already has shards of seed {a.seed}: use another --seed or --force")
    if a.force:   # a run replaces the previous one of its seed whole: no stale shard is left for train.py to load
        for f in old:
            f.unlink()
        summary_path.unlink(missing_ok=True)
        if old:
            print(f"--force: deleted {len(old)} shards of seed {a.seed} in {out}", flush=True)
    workers = max(1, a.workers)
    cfg = dict(out=str(out), plies=plies, ply_weights=weights, ranges_per_state=a.ranges_per_state, target=a.target,
               max_iters=a.max_iters, algo=a.algo, mix=a.mix, blueprint=blueprint, explore=a.explore, envs=a.envs,
               shard_size=a.shard_size, seed=a.seed, minutes=a.minutes,
               labels_per_worker=-(-a.labels // workers))
    meta = dict(kind="value-labels", pbs_version=pbs.PBS_VERSION, revision=provenance.revision(), build=build_id(),
                host=socket.gethostname(), config=cfg,
                command=shlex.join(["python", *sys.orig_argv[1:]]) if argv is None else
                shlex.join(["python", "-m", "bbar.value.gen", *argv]),
                blueprint_sha256=_sha(blueprint) if blueprint else None,
                range_kinds=list(pbs.RANGE_KINDS), policies=list(POLICIES),
                created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    print(f"generating {a.labels} labels at plies {plies} with {workers} workers into {out}", flush=True)
    t0 = time.time()
    total = {p: {"labels": 0, "states": 0, "seconds": 0.0, "iterations": 0} for p in plies}
    jobs = [(i, cfg, meta) for i in range(workers)]
    if workers == 1:
        results = map(worker, jobs)
    else:
        pool = mp.get_context("spawn").Pool(workers)
        results = pool.imap_unordered(worker, jobs)
    for wid, stats in results:
        for p, st in stats.items():
            for k, v in st.items():
                total[int(p)][k] += v
    wall = time.time() - t0
    rows = []
    for p in plies:
        st = total[p]
        rate = st["labels"] / st["seconds"] if st["seconds"] else float("nan")
        rows.append(dict(ply=p, labels=st["labels"], states=st["states"], thread_seconds=st["seconds"],
                         labels_per_sec_per_thread=rate,
                         mean_iterations=st["iterations"] / max(st["labels"], 1)))
        print(f"  ply {p}: {st['labels']} labels from {st['states']} states, {rate:.2f} labels/s per thread, "
              f"{st['iterations'] / max(st['labels'], 1):.0f} DCFR iterations per label", flush=True)
    n = sum(r["labels"] for r in rows)
    print(f"{n} labels in {wall:.1f}s ({n / wall:.1f}/s with {workers} workers)", flush=True)
    summary = dict(meta, wall_seconds=wall, labels=n, by_ply=rows)
    summary_path.write_text(json.dumps(summary, indent=1))
    return summary


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


if __name__ == "__main__":
    main()
