"""Cascade labels: value labels at earlier plies from depth-limited solves whose leaves the current network values,
and the driver that trains one network on all plies, level by level down to ply 0.

    python -m bbar.value.cascade labels --net V.pt --out data/value/NAME --plies 8,9 [--d 2] [--max-leaf-ply 10]
        [--iters 200] [--labels N | --minutes M] [--workers W] [--jobs B] [--threads T] [--device cuda] [--stub-net]
        [bbar.value.gen's position and range flags: --mix --blueprint --bayes --bayes-model --ranges-per-state ...]
    python -m bbar.value.cascade run --name NAME --data data/value/v1 [--net checkpoints/value/v1/best.pt]
        --levels 8,9:200000 6,7:200000 4,5:200000 2,3:200000 0,1:200000 [--d 2] [--iters 200] [--device cuda]
        [--workers W --jobs B --threads T] [--train-args "..."] [--label-args "..."] [--resume]

Label. A PBS is a public state at ply S and both players' root ranges (the family and positions of
bbar/value/gen.py, --bayes included). Its depth-limited window has value leaves at ply L = min(S + d, max-leaf-ply)
(terminal and frontier leaves exact); `iters` CFR-D iterations with the network at the value leaves (the iterate's
beliefs; bbar/value/dl.py) give the average strategy sigma and the averaged leaf values v~ (Window.leaf_values_avg:
at each leaf, each player's values against the other's average strategy there, in units where the other's reach
sums to 1). Player p's label for hand h is its best-response counterfactual value against sigma_-p in the
depth-limited game whose value leaves pay p v~_p, the other's root range normalised (Window.br_values(p,
leaf_values=v~)): gen.py's label (the best-response cfv against the other's equilibrium strategy, other range
normalised) with the window below L replaced by the value function. game_value: the mover's expected value following
sigma with leaf payoffs v~; exploitability: (BR_0 + BR_1 - 1) / 2 of sigma in that depth-limited game; iterations:
iters.

How close to gen.py's label (exact leaf values, dl.ExactValues / dl.LeafOracle; tests/test_cascade.py). Measured on
28 PBSs of gen.py's range family (golden games 0..39 step 3, plies 11-12, d=2, 200 iterations, solves to 1e-8):
  - game value: within the solve's tolerance (error <= 1.9e-4, of the order of the depth-limited exploitability);
  - per hand on the ranges' support: equal where the equilibrium is unique (full-support ranges: 1e-6 and below;
    range-weighted |diff| <= 1.8e-3 at 27 of the 28). Where it is not (sparse ranges), equilibria with the same game
    value give the hands different values, and the label is the one this solve picks: golden game 3, ply 12, uniform vs
    sparse: range-weighted |diff| 2.3e-2 (max 0.11) at a game-value error of 1.6e-7. gen.py's labels have the same
    freedom: its own DCFR, PCFR+ and CFR+ labels of that PBS differ by 2.5e-3 range-weighted (1.2e-2 max);
  - per hand OFF the support (zero or tiny range mass): not gen.py's, and more iterations do not close the gap (up to
    0.33; golden game 1, ply 12, sparse ranges: 0.25 at both 200 and 800 iterations while the hands with mass agree
    to 5e-7 and 6e-9). The averaged values v~ are only upper bounds for such hands and nothing pins the opponent's
    play where the range does not go. Valuing the leaves with V at the average strategy's beliefs instead of v~ is
    no better per hand (same 28 PBSs) and up to 4.5x worse range-weighted. So the shards' meta says
    all_hands_exact: false and bbar.value.train leaves these labels out of its all-feasible-hands term
    (--cascade-all-weight, default 0): only their values on the ranges train.

Shards: gen.py's format (same arrays, `meta` kind "cascade-labels" with the network's hash, d, max-leaf-ply, iters and
all_hands_exact false), so bbar.value.train trains one network on exact and cascade labels together.

Throughput. Each worker process runs --jobs solves in lockstep (bbar/value/dl.solve_many): every half-iteration one
batched network call over all value leaves of all its jobs (on --device; with --device cuda worker k uses GPU k mod
the GPU count), the solver work in one Rust call on --threads threads (GIL released). The summary reports labels/s
per worker, the network's share of the time and the cost of a half-iteration of a batch: value function (the
network's forward passes plus NetValues.session's copies, scatter and gather) and solver; --device cpu vs cuda on the
same flags gives the GPU's gain, --stub-net (a do-nothing network: no shards) the value function's cost without the
forward passes. Measured on an M4 (128 jobs at plies 8-9, 9132 value leaves at ply 10, 150 iterations, 4 workers x 5
threads at once), the same setup before and after WindowBatch: NetValues with a do-nothing network on MPS, 91-129 ->
19-23 ms per half-iteration per worker (3.3-4.6 -> 18-22 labels/s per worker; 15-18 ms of the 19-23 is the session
call, 3-7 ms the solver); constant values instead (the solver alone) 3.6 ms, 107 labels/s per worker. Before, the
per-window loop was GIL-bound (the 1-2 labels/s per worker with idle GPUs seen on a 2xA40 pod). The forward passes
come on top: 2 x parameters FLOP per value leaf per half-iteration (1024x6 MLP, v3m: ~280 GFLOP per half-iteration
of these 128 jobs); scripts/cascade.sh has the pod estimate.

Driver (run): trains V0 on --data (exact labels, plies >= 10) unless --net is given; then for each level, generates
its labels with the current network into data/value/NAME-pA-B, retrains on everything so far (checkpoints/value/
NAME-<k>/best.pt), and moves down. --resume skips a training only if it ran to its end (bbar.value.train.is_complete:
best.pt is also written during training) and a level only if its summary exists and names the current network
(net_sha256); once a step is redone, every later step is redone too.

--labels: exactly that many labels in all (each worker its share), fewer only if --minutes stops it first.
"""
import argparse
import json
import shlex
import sys
import time
from pathlib import Path

import numpy as np

from bbar._core import Window, public_key
from bbar.value import dl as D
from bbar.value import gen, pbs
from bbar.value.sample import StateSampler, parse_mix

ROOT = Path(__file__).resolve().parents[2]


def leaf_ply(S, d, max_leaf_ply):
    return min(S + d, max_leaf_ply)


def label_solved(w, game, kinds, iters, seconds):
    """The label (gen.py's dict) of a depth-limited window `w` solved by CFR-D (module docstring)."""
    mover = game.to_move
    avg = w.leaf_values_avg()
    vals = np.stack([w.br_values(0, None, leaf_values=avg), w.br_values(1, None, leaf_values=avg)])
    ranges = np.stack(w.ranges())
    e = w.exploitability(None, None, leaf_values=avg)
    gv = float(ranges[mover] @ w.cfvs(mover, None, leaf_values=avg))
    return dict(key=public_key(game), start=game.start, ranges=pbs.mover_relative(ranges, mover),
                values=pbs.mover_relative(vals, mover), game_value=gv, exploitability=e["exploitability"],
                iterations=iters, kinds=[pbs.RANGE_KINDS.index(kinds[p]) for p in (mover, 1 - mover)],
                seconds=seconds)


def label_many(items, values, d, max_leaf_ply, iters, algo="dcfr", threads=1):
    """Labels of PBSs `items` = [(game, ply, [(range, kind), (range, kind)])] (absolute players), solved in lockstep
    with the value function `values` (NetValues, dl.ExactValues). Returns (labels, stats of solve_many)."""
    t0 = time.perf_counter()
    jobs = [D.DLJob(Window(g, rk[0][0], rk[1][0], leaf_ply=leaf_ply(S, d, max_leaf_ply), threads=1))
            for g, S, rk in items]
    st = D.solve_many(jobs, values, iters, algo, threads)

    def one(x):
        job, (g, S, rk) = x
        return label_solved(job.w, g, [rk[0][1], rk[1][1]], iters, 0.0)
    labels = D._map(one, list(zip(jobs, items)), threads)
    secs = (time.perf_counter() - t0) / max(len(items), 1)
    for lab in labels:
        lab["seconds"] = secs
    st["total_seconds"] = time.perf_counter() - t0
    return labels, st


def make_values(net, device, chunk=None):
    from bbar.value.net import NetValues, load_value_net
    import torch
    m, _ = load_value_net(net, device)
    return NetValues(m, torch.device(device), chunk=chunk)


def stub_values(device):
    """--stub-net: NetValues over a network that returns the beliefs it is given (NetValues clamps them to [0, 1]):
    everything the value function costs but the forward passes (NetValues.session: beliefs to the device, the
    scatter into (n, 2, 495), the gather, values back)."""
    from bbar.value.net import NetValues
    import torch

    class Stub(torch.nn.Module):
        def forward(self, pub, ranges):
            return ranges
    return NetValues(Stub().to(device), torch.device(device))


def worker_device(device, wid, gpus):
    """Worker `wid`'s network device: --device cuda spreads the workers over the machine's `gpus` GPUs (cuda:wid %
    gpus); any other device (cpu, mps, cuda:K) is every worker's."""
    return f"cuda:{wid % gpus}" if device == "cuda" and gpus > 1 else device


def worker(args):
    wid, cfg, meta = args
    import torch
    torch.set_num_threads(cfg["torch_threads"])
    seed = np.random.SeedSequence([cfg["seed"], wid])
    rng = np.random.default_rng(seed)
    sampler = StateSampler(int(rng.integers(2 ** 63)), parse_mix(cfg["mix"]), cfg["blueprint"], envs=cfg["envs"],
                           explore=cfg["explore"])
    model = None
    if cfg["bayes"] > 0:
        from bbar.policy import NetModel
        model = NetModel(cfg["bayes_model"], "cpu")
    device = worker_device(cfg["device"], wid, torch.cuda.device_count() if cfg["device"] == "cuda" else 0)
    values = stub_values(device) if cfg["stub_net"] else make_values(cfg["net"], device)
    plies = cfg["plies"]
    wmax = max(cfg["ply_weights"])
    keep = {p: w / wmax for p, w in zip(plies, cfg["ply_weights"])}
    quota = worker_quota(cfg["labels"], cfg["workers"], wid)
    deadline = time.time() + 60 * cfg["minutes"] if cfg["minutes"] else float("inf")
    stats = {p: {"labels": 0, "states": 0, "seconds": 0.0, "iterations": 0, "net_seconds": 0.0,
                 "solve_seconds": 0.0, "half_iterations": 0.0} for p in plies}
    rows, shard, done, pending = [], 0, 0, []
    wmeta = dict(meta, worker=wid, worker_seed=[cfg["seed"], wid])
    out = Path(cfg["out"])

    def flush():
        nonlocal rows, shard
        if rows:
            gen.write_shard(out / f"s{cfg['seed']}-w{wid:03d}-{shard:05d}.npz", rows, wmeta)
            print(f"  worker {wid}: shard {shard} ({len(rows)} labels, {done} in all)", file=sys.stderr, flush=True)
            rows, shard = [], shard + 1

    def run(items):
        nonlocal rows, done
        labels, st = label_many([(x.game, x.ply, rk) for x, rk in items], values, cfg["d"], cfg["max_leaf_ply"],
                                cfg["iters"], cfg["algo"], cfg["threads"])
        n = len(items)
        for (x, _), lab in zip(items, labels):
            lab.update(ply=x.ply, policies=[gen.POLICIES.index(p) for p in x.policies])
            s = stats[x.ply]
            s["labels"] += 1
            s["iterations"] += lab["iterations"]
            s["seconds"] += st["total_seconds"] / n
            s["net_seconds"] += st["callback_seconds"] / n
            s["solve_seconds"] += (st["seconds"] - st["callback_seconds"]) / n
            s["half_iterations"] += 2 * cfg["iters"] / n     # (of a batch: per-half-iteration costs below)
        if not cfg["stub_net"]:                             # (stub values: nothing to keep)
            rows += labels
        done += n
        if len(rows) >= cfg["shard_size"]:
            flush()

    while done < quota and time.time() < deadline:
        batch = sampler.batch(plies, keep)
        for i in rng.permutation(len(batch)):
            x = batch[i]
            stats[x.ply]["states"] += 1
            if model is not None and rng.random() < cfg["bayes"]:
                pending.append((x, gen.bayes_pair(model, x, rng)))
            _, _, u, _, _ = pbs.decode_keys([public_key(x.game)])
            for _ in range(cfg["ranges_per_state"]):
                pending.append((x, [pbs.sample_range(u[0, p], rng) for p in (0, 1)]))
            # Full batches of --jobs, the last one cut to the quota.
            while pending and done < quota and time.time() < deadline and \
                    len(pending) >= min(cfg["jobs"], quota - done):
                n = min(cfg["jobs"], quota - done)
                run(pending[:n])
                pending = pending[n:]
            if done >= quota or time.time() >= deadline:
                break
    flush()
    return wid, stats


def worker_quota(labels, workers, wid):
    """Labels worker `wid` of `workers` writes: `labels` split as evenly as possible."""
    return labels // workers + (wid < labels % workers)


def labels_main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bbar.value.cascade labels", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    gen.add_sampling_args(ap, plies="8,9", ply_weights="1,1")
    ap.add_argument("--net", default="", help="value network checkpoint valuing the leaves")
    ap.add_argument("--stub-net", action="store_true",
                    help="throughput check instead of labels: the leaves are valued by a do-nothing network on "
                         "--device through the same path (NetValues.session: copies, scatter, gather) and no shard "
                         "is written, only the summary: the cost of a half-iteration without the forward passes")
    ap.add_argument("--d", type=int, default=2, help="value leaves at ply min(S + d, --max-leaf-ply)")
    ap.add_argument("--max-leaf-ply", type=int, default=10)
    ap.add_argument("--iters", type=int, default=200, help="CFR-D iterations per label")
    ap.add_argument("--jobs", type=int, default=64, help="solves run in lockstep per worker (one network call each "
                                                         "half-iteration for all of them)")
    ap.add_argument("--threads", type=int, default=1, help="solver threads per worker")
    ap.add_argument("--torch-threads", type=int, default=1, help="torch CPU threads per worker")
    ap.add_argument("--device", default="cpu", help="network device in the workers: cpu, cuda, mps")
    a = ap.parse_args(argv)
    if not a.stub_net and not Path(a.net).is_file():
        ap.error(f"no network {a.net!r} (--net, or --stub-net to time the rest)")
    if a.d < 1 or a.iters < 1 or a.jobs < 1:
        ap.error("--d, --iters and --jobs must be at least 1")
    cfg, meta, summary_path = gen.setup_run(
        ap, a, "cascade-labels", "bbar.value.cascade", None if argv is None else ["labels", *argv], min_ply=0,
        max_ply=a.max_leaf_ply - 1,
        plies_error=f"--plies must be in 0..{a.max_leaf_ply - 1} (below --max-leaf-ply)")
    cfg.update(net="" if a.stub_net else a.net, stub_net=a.stub_net, d=a.d, max_leaf_ply=a.max_leaf_ply,
               iters=a.iters, jobs=a.jobs, threads=a.threads, torch_threads=a.torch_threads, device=a.device,
               labels=a.labels)
    meta.update(net_sha256=None if a.stub_net else gen._sha(a.net), stub_net=a.stub_net,
                leaf_plies={p: leaf_ply(p, a.d, a.max_leaf_ply) for p in cfg["plies"]}, all_hands_exact=False)
    plies, workers, out = cfg["plies"], cfg["workers"], cfg["out"]
    print(f"cascade: {a.labels} labels at plies {plies} (value leaves at {meta['leaf_plies']}), {a.iters} CFR-D "
          f"iterations, {workers} workers x {a.jobs} solves in lockstep, "
          f"{'do-nothing network (--stub-net: no shards)' if a.stub_net else 'network'} on {a.device} -> {out}",
          flush=True)
    total, wall = gen.run_workers(worker, cfg, meta)
    rows = []
    for p in plies:
        st = dict(dict(labels=0, states=0, seconds=0.0, iterations=0, net_seconds=0.0, solve_seconds=0.0,
                       half_iterations=0.0), **total[p])
        rate = st["labels"] / st["seconds"] if st["seconds"] else float("nan")
        halves = st["half_iterations"]
        rows.append(dict(ply=p, leaf_ply=leaf_ply(p, a.d, a.max_leaf_ply), labels=st["labels"], states=st["states"],
                         worker_seconds=st["seconds"], labels_per_sec_per_worker=rate,
                         net_share=st["net_seconds"] / st["seconds"] if st["seconds"] else float("nan"),
                         net_ms_per_half_iteration=1e3 * st["net_seconds"] / halves if halves else float("nan"),
                         solver_ms_per_half_iteration=1e3 * st["solve_seconds"] / halves if halves else float("nan")))
        print(f"  ply {p} (leaves at {rows[-1]['leaf_ply']}): {st['labels']} labels, {rate:.2f} labels/s per worker "
              f"({a.device}, {a.jobs} jobs, {a.threads} threads), network {100 * rows[-1]['net_share']:.0f}% of "
              f"the time; per half-iteration of a batch: value function {rows[-1]['net_ms_per_half_iteration']:.1f} "
              f"ms, solver {rows[-1]['solver_ms_per_half_iteration']:.1f} ms", flush=True)
    n = sum(r["labels"] for r in rows)
    print(f"{n} labels in {wall:.1f}s ({n / wall:.2f}/s with {workers} workers)", flush=True)
    summary = dict(meta, wall_seconds=wall, labels=n, labels_per_sec=n / wall, by_ply=rows)
    summary_path.write_text(json.dumps(summary, indent=1))
    return summary


def parse_levels(items):
    """['8,9:200000', ...] -> [([8, 9], 200000), ...] (each level strictly below the previous)."""
    out = []
    for x in items:
        plies, _, n = x.partition(":")
        ps = sorted(int(p) for p in plies.split(","))
        out.append((ps, int(n) if n else 100_000))
    for (a, _), (b, _) in zip(out, out[1:]):
        if max(b) >= min(a):
            raise ValueError(f"levels must go down: {a} then {b}")
    return out


def run_main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bbar.value.cascade run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="names data/value/NAME-pA-B and checkpoints/value/NAME-<k>")
    ap.add_argument("--data", nargs="+", required=True, help="exact label shards (plies >= 10): V0's data")
    ap.add_argument("--net", default="", help="start from this network instead of training V0 on --data")
    ap.add_argument("--levels", nargs="+", default=["8,9:200000", "6,7:200000", "4,5:200000", "2,3:200000",
                                                    "0,1:200000"], help="PLIES:LABELS per level, going down")
    ap.add_argument("--d", type=int, default=2)
    ap.add_argument("--max-leaf-ply", type=int, default=10)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--minutes", type=float, default=0, help="per level: stop generating after this many minutes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label-args", default="", help="extra flags for every labels step (quoted)")
    ap.add_argument("--train-args", default="", help="extra flags for every training (quoted), e.g. '--steps 50000'")
    ap.add_argument("--resume", action="store_true",
                    help="skip trainings that ran to their end and levels generated by the current network (until "
                         "one step is redone)")
    a = ap.parse_args(argv)
    from bbar.value import train
    try:
        levels = parse_levels(a.levels)
    except ValueError as e:
        ap.error(str(e))
    data = list(a.data)
    log = []
    redo = not a.resume     # once a step is (re)done, every later one depends on it

    def fit(k):
        nonlocal redo
        name = f"{a.name}-{k}"
        best = ROOT / "checkpoints" / "value" / name / "best.pt"
        if not redo and train.is_complete(best):
            print(f"[cascade] {name}: trained, skipped", flush=True)
        else:
            why = ""
            if a.resume:
                why = " (an earlier step was redone)" if redo else " (unfinished)" if best.is_file() else ""
            print(f"[cascade] training {name}{why} on {data}", flush=True)
            train.main(["--name", name, "--data", *data, "--device", "auto" if a.device != "cpu" else "cpu",
                        "--force", *shlex.split(a.train_args)])
            redo = True
        return str(best)

    def generated_by(summary, net):
        try:
            return json.loads(summary.read_text()).get("net_sha256") == gen._sha(net)
        except (OSError, ValueError):
            return False

    net = a.net or fit(0)
    for k, (plies, n) in enumerate(levels, 1):
        out = ROOT / "data" / "value" / f"{a.name}-p{'-'.join(map(str, plies))}"
        done = out / f"summary-s{a.seed}.json"
        if not redo and generated_by(done, net):
            print(f"[cascade] level {plies}: {out} generated by {net}, skipped", flush=True)
        else:
            if a.resume and not redo and done.is_file():
                print(f"[cascade] level {plies}: {out} was generated by another network, redone", flush=True)
            redo = True
            args = ["--net", net, "--out", str(out), "--plies", ",".join(map(str, plies)),
                    "--ply-weights", ",".join("1" for _ in plies), "--labels", str(n), "--d", str(a.d),
                    "--max-leaf-ply", str(a.max_leaf_ply), "--iters", str(a.iters), "--workers", str(a.workers),
                    "--jobs", str(a.jobs), "--threads", str(a.threads), "--device", a.device,
                    "--seed", str(a.seed), "--force", *(["--minutes", str(a.minutes)] if a.minutes else []),
                    *shlex.split(a.label_args)]
            print(f"[cascade] level {plies}: python -m bbar.value.cascade labels {shlex.join(args)}", flush=True)
            s = labels_main(args)
            log.append(dict(level=plies, labels=s["labels"], labels_per_sec=s["labels_per_sec"], net=net))
        data.append(str(out))
        net = fit(k)
    print(f"[cascade] done: {net} (trained on {data})", flush=True)
    return net, log


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("labels", "run"):
        print(__doc__)
        raise SystemExit("usage: python -m bbar.value.cascade {labels,run} ...")
    return (labels_main if argv[0] == "labels" else run_main)(argv[1:])


if __name__ == "__main__":
    main()
