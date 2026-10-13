"""Cross-checks and benchmarks of the exact window solver (bbar._core.Window, core/src/window.rs).

    python -m bbar.window_check proto --ply 12 --games 0,4,5 [--pimc] [--cfr-iters 300]
    python -m bbar.window_check bench --plies 13,12,11,10 --games 20 [--target 1e-4] [--threads 1] [--algos dcfr,pcfr+]
    python -m bbar.window_check xcfr [--animals chameleon,monkey,giraffe,lion] [--k 2] [--iters 4000]

Ply convention: ply = moves already played (root of a window at ply P = the public state after the first P moves
of a golden game). research/pimc_gap.py numbers its plies from 1: its "ply 13" is our ply 12.

proto: research/pimc_gap.py (an independent numpy prototype) on golden-game windows with uniform root ranges.
  - Exploitability of uniform play (an exact quantity) on the prototype's tree (leaves at ply 16, both decks empty):
    ours with leaves="public" must agree to 1e-9.
  - --pimc: the prototype's PIMC strategy (deterministic, uniform beliefs) imported as an external profile: its
    exact exploitability must agree to 1e-9.
  - --cfr-iters N: the prototype's CFR+ (N iterations); the game value from our frontier window (solved to 1e-9)
    must lie in the prototype's bounds [1 - BR_1, BR_0].
bench: window sizes and solve times on golden-game windows (uniform root ranges, frontier leaves).
xcfr: the explicit-infoset CFR comparison of bbar/xcfr.py.
"""
import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from bbar._core import Game, Window
from bbar.actions import ANIMALS
from bbar.golden import load

ROOT = Path(__file__).resolve().parent.parent


def prototype():
    """research/pimc_gap.py as a module."""
    spec = importlib.util.spec_from_file_location("pimc_gap", ROOT / "research" / "pimc_gap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def public_game(g, ply):
    """The public position after the first `ply` moves of golden game `g`."""
    G = Game(g["start"], [ANIMALS, ANIMALS], [[], []])
    for s in g["steps"][:ply]:
        a, p = s["move"]
        G.play(a, list(p))
    return G


def import_profile(pg, proto_root, sigma, window):
    """A prototype strategy (dict id(node) -> [hands, actions] array, hands in itertools.combinations order) as a
    window profile. The window must have leaves="public" (the prototype's tree)."""
    prof = np.zeros(window.num_entries)
    by_history = {tuple(h): (off, hands, acts) for h, _, off, hands, acts in window.nodes()}
    stack = [(proto_root, ())]
    while stack:
        n, hist = stack.pop()
        if n.leaf_u is not None:
            continue
        off, hands, acts = by_history[hist]
        assert list(acts) == [a for a, _, _, _ in n.kids], "same public actions in the same order"
        s = sigma[id(n)]
        row = {frozenset(ANIMALS.index(a) for a in h): i for i, h in enumerate(hands)}
        for i, h in enumerate(n.hands[n.p]):
            r = row[frozenset(h)]
            prof[off + r * len(acts):off + (r + 1) * len(acts)] = s[i]
        for a, c, _, _ in n.kids:
            stack.append((c, hist + (a,)))
    return prof


def check_proto(ply, games, pimc=False, cfr_iters=0, quiet=False):
    """Returns a list of per-position dicts; raises AssertionError on a mismatch."""
    pg = prototype()
    data = load()
    rows = []
    for gi in games:
        G = public_game(data[gi], ply)
        t = time.time()
        root = pg.build(G, ply + 1, 17)
        alln = pg.nodes(root)
        r0 = np.ones(len(root.hands[0])) / len(root.hands[0])
        r1 = np.ones(len(root.hands[1])) / len(root.hands[1])
        e_proto = pg.exploitability(root, pg.uniform_sigma(alln), r0, r1)
        W = Window(G, leaves="public")
        e = W.exploitability("uniform", "uniform")
        internal = sum(1 for n in alln if n.leaf_u is None)
        row = dict(game=gi, ply=ply, internal=internal, leaves=len(alln) - internal,
                   uniform=e["exploitability"], uniform_diff=e["exploitability"] - e_proto[0],
                   uniform_value_diff=e["value"] - e_proto[1])
        assert (W.sizes()["internal"], W.sizes()["leaves"]) == (internal, len(alln) - internal), "same tree"
        assert abs(row["uniform_diff"]) < 1e-9 and abs(row["uniform_value_diff"]) < 1e-9, row
        if pimc:
            sig = pg.pimc_sigma(alln, lambda n: np.ones(len(n.hands[1 - n.p])) / len(n.hands[1 - n.p]))
            e_p = pg.exploitability(root, sig, r0, r1)
            prof = import_profile(pg, root, sig, W)
            e_w = W.exploitability(prof, prof)
            row.update(pimc=e_w["exploitability"], pimc_diff=e_w["exploitability"] - e_p[0])
            assert abs(row["pimc_diff"]) < 1e-9, row
        if cfr_iters:
            sig = pg.cfr_plus(root, alln, r0, r1, cfr_iters)
            e_c, v_c, br0, lo = pg.exploitability(root, sig, r0, r1)
            Wf = Window(G)
            r = Wf.solve(target=1e-9, max_iters=50_000)
            row.update(value=r["value"], value_exploitability=r["exploitability"], proto_value=v_c,
                       proto_exploitability=e_c, proto_bounds=(lo, br0))
            assert lo - 1e-9 <= r["value"] <= br0 + 1e-9, row
        row["seconds"] = time.time() - t
        rows.append(row)
        if not quiet:
            print(" ".join(f"{k}={v:.3e}" if isinstance(v, float) and "diff" in k else
                           f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    return rows


def bench(plies, n_games, target, threads, algos, check_every, skip):
    data = load()
    for ply in plies:
        rows = []
        for gi in range(skip, skip + n_games):
            G = public_game(data[gi], ply)
            t = time.time()
            W = Window(G, threads=threads)
            build = time.time() - t
            sz = W.sizes()
            row = dict(internal=sz["internal"], leaves=sz["leaves"], distinct=sz["frontier_positions"],
                       entries=sz["entries"], build=build)
            for algo in algos:
                r = W.solve(target=target, check_every=check_every, algo=algo)
                row[algo] = (r["seconds"], r["iterations"], r["exploitability"])
            rows.append(row)
        print(f"ply {ply}: {n_games} golden games from #{skip}, uniform ranges, target {target:g}, threads {threads}")

        def q(xs):
            xs = sorted(xs)
            return (f"median {statistics.median(xs):>10.4g}  p90 {xs[min(len(xs) - 1, int(0.9 * len(xs)))]:>10.4g}"
                    f"  max {xs[-1]:>10.4g}")
        for key in ("internal", "leaves", "distinct", "entries", "build"):
            print(f"  {key:<18} {q([r[key] for r in rows])}")
        for algo in algos:
            print(f"  {algo + ' seconds':<18} {q([r[algo][0] for r in rows])}")
            print(f"  {algo + ' iterations':<18} {q([r[algo][1] for r in rows])}")
            total = [r["build"] + r[algo][0] for r in rows]
            print(f"  {'build+' + algo:<18} {q(total)}")
        sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("proto")
    p.add_argument("--ply", type=int, default=12)
    p.add_argument("--games", default="0,4,5")
    p.add_argument("--pimc", action="store_true")
    p.add_argument("--cfr-iters", type=int, default=0)
    b = sub.add_parser("bench")
    b.add_argument("--plies", default="13,12,11,10")
    b.add_argument("--games", type=int, default=20)
    b.add_argument("--skip", type=int, default=0)
    b.add_argument("--target", type=float, default=1e-4)
    b.add_argument("--threads", type=int, default=1)
    b.add_argument("--algos", default="dcfr,pcfr+")
    b.add_argument("--check-every", type=int, default=10)
    x = sub.add_parser("xcfr")
    x.add_argument("--animals", default="chameleon,monkey,giraffe,lion")
    x.add_argument("--k", type=int, default=2)
    x.add_argument("--iters", type=int, default=4000)
    x.add_argument("--plus-iters", type=int, default=2000)
    x.add_argument("--start", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "proto":
        check_proto(a.ply, [int(x) for x in a.games.split(",")], a.pimc, a.cfr_iters)
    elif a.cmd == "bench":
        bench([int(x) for x in a.plies.split(",")], a.games, a.target, a.threads, a.algos.split(","),
              a.check_every, a.skip)
    else:
        from bbar.xcfr import compare
        print(f"variant {a.animals} k={a.k} start={a.start}")
        compare(a.animals.split(","), a.k, a.iters, a.plus_iters, a.start)


if __name__ == "__main__":
    main()
