"""E1: exact exploitability of an agent in late-game windows, by root ply T (delta_T).

    python -m bbar.delta X [--T 12,11,10,9] [--N 200] [--seed S] [--ranges bayes|uniform] [--device D] [--out PATH]

X: nn:PATH[,argmax] (a policy network), a built-in spec such as pimc:w=8,d=4, tail:nn:PATH,... (bbar/tail.py) or
resolve:V=PATH,base=nn:PATH,... (bbar/resolve.py: at T in [P0, F) its table is the root strategy of its depth-limited
solve at each of its nodes in [T, F) and its exact window from each ply-F node, for every distinct history).

For each T, N positions from self-play games of X against itself (deal i of the arena's deals for --seed, game 0;
every seat plays X with the arena's seat RNGs; a Python agent gets a fresh instance per seat, bbar.match.PyPlayer),
stopped after T moves (ply T). Games are self-played --batch at a time, and each game is evaluated at every T before
the next (a tail agent reuses its self-play windows, then releases them). At each position:
  1. ranges: each player's Bayes range over their hands given their own past decisions, assuming they play X
     (bbar/belief.py: exact for networks; for built-in specs sigma is estimated from --samples runs per (decision,
     hand), smoothed by --alpha; for tail and resolve agents from their base network and their solves). --ranges
     uniform skips the inference (uniform over the feasible hands: cheaper, and not X's real beliefs - the output says
     so).
  2. the window at ply T (bbar._core.Window: every public continuation to the perfect-information frontier, ply 15,
     whose leaves are valued by exact play - so X's own play from ply 15 on is replaced by perfect play);
  3. X's full strategy table on it, for both players (every node, every hand: bbar/policy.py, bbar._core.WindowTree);
     built-in specs are played once per (node, hand) (--table-samples), a sampled pure strategy whose expected
     exploitability bounds the agent's from above;
  4. exact best responses: delta_T = (BR_0 + BR_1 - 1) / 2 (win probability units; 0 = unexploitable here);
  5. the window's equilibrium value v* (DCFR to --target) and each seat's loss vs equilibrium: player 0 is guaranteed
     1 - BR_1 with X where an equilibrium guarantees v*, so loss_0 = v* - (1 - BR_1) and loss_1 = (1 - v*) - (1 - BR_0)
     (delta_T is their mean; each is known to the equilibrium gap).
  6. agents that re-solve on top of a base network (tail, resolve agents): the per-hand safety margin of each seat p,
     max over the opponent's feasible hands h of cfv_X(h) - cfv_base(h), where cfv(h) is the opponent's best-response
     value with hand h against seat p's strategy (X's, or the base network's) on X's own solved window for p if X
     already plays from it at ply T (rooted at ply F or F + 1, with its ranges), else on the window at ply T. > 0
     means some opponent hand wins more against X than against its base network ("safety", per seat;
     "safety_weighted": the same gains weighted by the opponent's range there).
Per T: mean with a 95% normal interval over positions, median, p90 and max. Every position and summary is a JSON line
in --out (default runs/delta/<agent>-<time>.jsonl), after a header line with provenance: commit, code hash, engine
build, checkpoint hashes, seeds, the command.

Reading delta_T. It best-responds against the ranges induced by X's own play. A re-solving agent that solves its
window for exactly those ranges (tail mode=plain: an equilibrium for the base network's posterior) has delta_T near 0
by construction, yet can be exploited per opponent hand by anyone whose real range differs (an opponent that played
differently before F, as an exploiter does): the safety column shows it; mode=gadget keeps it <= its gap. Do not rank
re-solving modes by delta_T alone: compare their safety margins, --ranges uniform, and exploiter or match results.
"""
import argparse
import json
import math
import os
import re
import shlex
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bbar import arena, belief, match, provenance
from bbar._core import VecEnv, Window, WindowTree, arena_deal, build_id, hand_rank
from bbar.policy import NetModel, RustModel

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "delta"
MAX_T = 14
PY_BATCH = 32


def derive(*xs):
    """A 64-bit seed from integers (numpy SeedSequence)."""
    return int(np.random.SeedSequence([int(x) % 2 ** 64 for x in xs]).generate_state(1, np.uint64)[0])


class Subject:
    """The agent X under test: plays self-play games (`player`), gives per-hand likelihoods of its past decisions and
    its strategy table on a window."""

    def __init__(self, spec, device="cpu", samples=8, alpha=0.5, table_samples=1, threads=0):
        self.spec = spec
        kind, body, argmax = match.parse_agent(spec)
        self.kind = kind
        self.base = None    # the base network of a re-solving agent (safety margins)
        if kind == "nn":
            self.model = NetModel(body, device, argmax=argmax)
            self.player = match.NetPlayer(body, device, argmax=argmax)
        elif kind == "py":
            self.player = match.make_player(spec, device)
            self.model = None   # the self-play instance of each seat (seat_model)
            self.base = getattr(self.player.instance(0), "model", None)
            for p in (0, 1):
                if hasattr(self.player.instance(p), "drop_finished"):
                    self.player.instance(p).drop_finished = False   # the evaluation reuses the self-play windows
        else:
            self.model = RustModel(body, samples=samples, alpha=alpha, table_samples=table_samples, threads=threads)
            self.player = match.RustPlayer(body)
        self.name = self.player.name
        self.checkpoints = dict(self.player.checkpoints)
        self.threads = threads

    def seat_model(self, player):
        """The model answering for `player`: a Python agent's self-play instance of that seat (its solved windows)."""
        return self.player.instance(player) if self.kind == "py" else self.model

    def describe(self):
        d = {"kind": self.kind, "name": self.name}
        if self.kind == "rust":
            d.update(self.model.describe())
        return d

    def likelihoods(self, deal, actions, player, seed):
        if self.kind == "rust":
            return self.model.likelihoods(deal, actions, player, seed=seed)
        return self.seat_model(player).likelihoods(deal, actions, player)

    def table(self, tree, player, deal, actions, seed):
        if self.kind == "py":
            return self.seat_model(player).window_table(tree, player, deal, actions)
        if self.kind == "rust":
            return self.model.window_table(tree, player, seed=seed)
        return self.model.window_table(tree, player)

    def solved_root(self, deal, actions, player):
        """(Window, WindowTree, table) of the solved window a re-solving agent plays `player`'s decisions at ply
        len(actions) from, if its root is at or before that ply; else None."""
        model = self.seat_model(player)
        return model.solved_root(deal, actions, player) if hasattr(model, "solved_root") else None

    def release(self, game, Ts):
        """Done with `game`: a Python agent may forget the windows it solved for it."""
        if self.kind != "py":
            return
        prefix = game["actions"][:min(min(Ts), getattr(self.seat_model(0), "F", 0))]
        for agent in self.player.instances.values():
            if hasattr(agent, "release"):
                agent.release(game["start"], prefix)


def self_play(player, n, plies, seed, threads=0, batch=256, first=0):
    """n games of `player` against itself on the arena's deals first .. first + n - 1 for `seed` (game 0 of each
    deal), stopped after `plies` moves: [{deal, start, actions}]. A Python agent (bbar.match.PyPlayer) plays each seat
    with its own instance (stream = seat), so no instance sees both hands of a game."""
    if plies > 15:
        raise ValueError("stop at ply 15 at the latest (a game can end from ply 16)")
    out = []
    py = isinstance(player, match.PyPlayer)
    for d0 in range(first, first + n, batch):
        d = np.arange(d0, min(first + n, d0 + batch))
        k = len(d)
        env = VecEnv(k, seed=0, threads=threads)
        env.reset_arena(seed, d, np.zeros(k, dtype=np.int64))
        start = env.movers()[0].copy()
        acts = [[] for _ in range(k)]
        if py:
            player.prepare(k)
        with match.full_fp32_matmuls():
            while True:
                if player.needs_obs:
                    obs, mask, mover, ply = env.observe()
                else:
                    mover, ply = env.movers()
                    obs = mask = None
                live = np.flatnonzero(ply < plies)
                if not len(live):
                    break
                actions = np.full(k, -1, dtype=np.int64)
                if py:
                    actions[live] = player.act(env, live, obs, mask, streams=mover[live])
                else:
                    actions[live] = player.act(env, live, obs, mask)
                for j in live:
                    acts[j].append(int(actions[j]))
                done, _, _ = env.step(actions)
                assert not done.any(), "a game ended before the stop ply"
        out += [{"deal": int(i), "start": int(start[j]), "actions": acts[j]} for j, i in enumerate(d)]
    return out


def evaluate(subject, game, T, seed, ranges="bayes", target=1e-4, max_iters=100_000, threads=0, true_hands=None):
    """delta_T and friends at the position after game['actions'][:T] (see the module docstring)."""
    actions = game["actions"][:T]
    deal = belief.start_game(game["start"])
    t0 = time.time()
    if ranges == "bayes":
        rs = [belief.bayes_range(deal, actions, p, subject.likelihoods(deal, actions, p, derive(seed, 1, T, p)))
              for p in (0, 1)]
    else:
        rs = [belief.uniform_range(deal, actions, p) for p in (0, 1)]
    t1 = time.time()
    root = belief.replay(deal, actions)
    w = Window(root, rs[0].range, rs[1].range, threads=threads or os.cpu_count() or 1)
    tree = WindowTree(deal, actions)
    sz = w.sizes()
    if tree.num_entries != w.num_entries or tree.num_nodes != sz["internal"]:
        raise RuntimeError("WindowTree and Window layouts differ")
    t2 = time.time()
    tables = [subject.table(tree, p, deal, actions, derive(seed, 2, T, p)) for p in (0, 1)]
    table = tables[0] + tables[1]
    t3 = time.time()
    if sz["internal"]:
        ex = w.exploitability(table, table)
    else:
        ex = w.exploitability("uniform", "uniform")
    t4 = time.time()
    safety = safety_margins(subject, deal, actions, w, tree, tables) if subject.base is not None else None
    t4b = time.time()
    eq = w.solve(target=target, max_iters=max_iters)
    t5 = time.time()
    br0, br1 = ex["br"]
    v = eq["value"]
    loss = [v - (1 - br1), (1 - v) - (1 - br0)]
    start = game["start"]
    row = {
        "type": "position", "T": T, "deal": game["deal"], "start": start, "to_move": root.to_move,
        "actions": actions,
        "internal": sz["internal"], "entries": sz["entries"], "hands": list(sz["hands"]),
        "frontier_positions": sz["frontier_positions"],
        "delta": ex["exploitability"], "br": [br0, br1], "value": ex["value"],
        "eq_value": v, "eq_gap": eq["exploitability"], "eq_iterations": eq["iterations"],
        "loss": loss, "loss_start": loss[start], "loss_second": loss[1 - start],
        "ranges": ranges,
        "impossible": [r.impossible for r in rs],
        "log_likelihood": [r.log_likelihood for r in rs],
        "range_support": [int(np.count_nonzero(r.range)) for r in rs],
        "seconds": {"belief": t1 - t0, "build": t2 - t1, "table": t3 - t2, "exploitability": t4 - t3,
                    "solve": t5 - t4b},
    }
    if safety is not None:
        row["safety"], row["safety_weighted"] = safety
        row["seconds"]["safety"] = t4b - t4
    if true_hands is not None:
        row["true_hand_prob"] = [float(rs[p].range[hand_rank(true_hands[p])]) for p in (0, 1)]
    return row


def safety_margin(w, tree, table, base, p):
    """(max over the opponent's feasible hands h of cfv_X(h) - cfv_base(h), the same gains weighted by the opponent's
    range): cfv(h) is the opponent's best-response value with hand h against seat p's strategy on window `w`
    (layout `tree`), X's (`table`) or the base network's. 0 if p does not move in the window."""
    if tree.num_rows(p) == 0:
        return 0.0, 0.0
    q = 1 - p
    gain = w.br_values(q, table) - w.br_values(q, base.window_table(tree, p))
    hands = w.hands(q)
    return float(np.max(gain[hands])), float(np.dot(w.ranges()[q][hands], gain[hands]))


def safety_margins(subject, deal, actions, w, tree, tables):
    """Per seat p, the safety margin of X against its base network (module docstring, item 6): on X's own solved
    window for p if X plays from it at this position (where a safe re-solve guarantees margin <= its gap), else on
    the window at this position (X plays the base network before it, then its solved windows). Returns
    ([worst per seat], [range-weighted per seat])."""
    worst, weighted = [], []
    for p in (0, 1):
        own = subject.solved_root(deal, actions, p)
        m = safety_margin(*own, subject.base, p) if own is not None else safety_margin(w, tree, tables[p],
                                                                                          subject.base, p)
        worst.append(m[0])
        weighted.append(m[1])
    return worst, weighted


def true_hands(seed, game, T):
    g = arena_deal(seed, game["deal"])
    for a in game["actions"][:T]:
        g.play_action(a)
    return [g.hand(0), g.hand(1)]


def summarize(rows):
    """Summary of the positions of one T."""
    d = np.array([r["delta"] for r in rows])
    n = len(d)
    half = 1.959963984540054 * d.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")

    def stats(xs):
        xs = np.asarray(xs, dtype=float)
        return {"mean": float(xs.mean()), "median": float(np.median(xs)), "p90": float(np.quantile(xs, 0.9)),
                "max": float(xs.max())}
    secs = {k: stats([r["seconds"][k] for r in rows]) for k in rows[0]["seconds"]}
    return {
        "type": "summary", "T": rows[0]["T"], "n": n,
        "delta": {"mean": float(d.mean()), "ci95": [float(d.mean() - half), float(d.mean() + half)],
                  "median": float(np.median(d)), "p90": float(np.quantile(d, 0.9)), "max": float(d.max())},
        "loss_start": stats([r["loss_start"] for r in rows]),
        "loss_second": stats([r["loss_second"] for r in rows]),
        "value": stats([r["value"] for r in rows]),
        "eq_value": stats([r["eq_value"] for r in rows]),
        "eq_gap_max": float(max(r["eq_gap"] for r in rows)),
        "internal": stats([r["internal"] for r in rows]),
        "entries": stats([r["entries"] for r in rows]),
        "impossible_decisions": int(sum(len(x) for r in rows for x in r["impossible"])),
        "seconds": secs,
        **({"safety": stats([max(r["safety"]) for r in rows])} if all("safety" in r for r in rows) else {}),
    }


def slug(spec):
    s = re.sub(r"[^A-Za-z0-9=.,_-]+", "_", spec.replace("/", "_"))
    return s[-80:]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bbar.delta", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("agent", help="X: nn:PATH[,argmax], a built-in spec (pimc:w=8,d=4) or tail:nn:PATH,...")
    ap.add_argument("--T", default="12,11,10,9", help="root plies, comma-separated (each <= 14)")
    ap.add_argument("--N", type=int, default=200, help="positions per T")
    ap.add_argument("--seed", type=int, help="self-play seed (default: random, logged)")
    ap.add_argument("--ranges", choices=("bayes", "uniform"), default="bayes")
    ap.add_argument("--samples", type=int, default=8, help="built-in specs: runs per (decision, hand) for ranges")
    ap.add_argument("--alpha", type=float, default=0.5, help="built-in specs: likelihood smoothing")
    ap.add_argument("--table-samples", type=int, default=1, help="built-in specs: runs per (node, hand) for tables")
    ap.add_argument("--target", type=float, default=1e-4, help="equilibrium solve: exploitability target")
    ap.add_argument("--max-iters", type=int, default=100_000)
    ap.add_argument("--threads", type=int, default=0, help="0 = all cores")
    ap.add_argument("--device", default="cpu", help="network device: cpu, cuda, mps, auto")
    ap.add_argument("--batch", type=int, help=f"self-play games at once, each evaluated at every T before the next "
                                               f"batch (default 256; {PY_BATCH} for Python agents, which keep the "
                                               f"batch's solved windows)")
    ap.add_argument("--out", help="JSON-lines output (default runs/delta/<agent>-<time>.jsonl)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    try:
        Ts = sorted({int(x) for x in args.T.split(",")}, reverse=True)
    except ValueError:
        ap.error("--T: comma-separated integers")
    if not Ts or min(Ts) < 0 or max(Ts) > MAX_T:
        ap.error(f"--T must be in 0..{MAX_T}")
    if args.N < 1:
        ap.error("--N must be at least 1")
    if args.batch is not None and args.batch < 1:
        ap.error("--batch must be at least 1")
    seed = arena.random_seed() if args.seed is None else args.seed
    try:
        provenance.check_build()
    except provenance.StaleBuild as e:
        ap.error(str(e))
    revision = provenance.revision()
    device = match.resolve_device(args.device) if args.device == "auto" else args.device
    try:
        subject = Subject(args.agent, str(device), args.samples, args.alpha, args.table_samples, args.threads)
    except ValueError as e:
        ap.error(str(e))
    out = Path(args.out) if args.out else OUT / (
        f"{slug(subject.name)}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    command = shlex.join(["python", *sys.orig_argv[1:]] if argv is None else ["python", "-m", "bbar.delta", *argv])
    header = {
        "type": "run", "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agent": subject.name, "subject": subject.describe(), "T": Ts, "N": args.N, "seed": seed,
        "ranges": args.ranges if args.ranges == "uniform" else (
            "bayes" if subject.kind != "rust" else f"bayes-sampled(samples={args.samples},alpha={args.alpha})"),
        "table": "exact" if subject.kind != "rust" else f"sampled(samples={args.table_samples})",
        "frontier": "ply-15 frontier leaves valued by exact play",
        "target": args.target, "device": str(device),
        "commit": revision["commit"], "code": revision["code"], "build": build_id(),
        "checkpoints": subject.checkpoints, "hostname": socket.gethostname(), "command": command,
    }
    say = (lambda *a: None) if args.quiet else (lambda *a: print(*a, file=sys.stderr, flush=True))
    batch = args.batch or (PY_BATCH if subject.kind == "py" else 256)
    rows = {T: [] for T in Ts}
    with open(out, "a") as f:
        f.write(json.dumps(header) + "\n")
        play_s = 0.0
        for first in range(0, args.N, batch):
            t0 = time.time()
            games = self_play(subject.player, min(batch, args.N - first), max(Ts), seed, args.threads, batch, first)
            play_s += time.time() - t0
            for game in games:
                for T in Ts:
                    row = evaluate(subject, game, T, seed, args.ranges, args.target, args.max_iters, args.threads,
                                   true_hands(seed, game, T))
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    rows[T].append(row)
                subject.release(game, Ts)
                done = len(rows[Ts[0]])
                if done % 10 == 0 or done == args.N:
                    say(f"  {done}/{args.N} positions per T (self-play {play_s:.0f}s): delta mean " + ", ".join(
                        f"T={T} {np.mean([r['delta'] for r in rows[T]]):.4f}" for T in Ts))
        summaries = []
        for T in Ts:
            s = summarize(rows[T])
            f.write(json.dumps(s) + "\n")
            f.flush()
            summaries.append(s)
    print(f"{subject.name}: delta_T over {args.N} self-play positions per T (seed {seed}, ranges {header['ranges']})")
    safe = all("safety" in s for s in summaries)
    print(f"{'T':>3} {'mean':>8} {'95% CI':>19} {'median':>8} {'p90':>8} {'max':>8} {'loss1st':>8} {'loss2nd':>8}"
          f" {'nodes':>8} {'table s':>8} {'solve s':>8}" + (f" {'safe max':>8}" if safe else ""))
    for s in summaries:
        d = s["delta"]
        print(f"{s['T']:>3} {d['mean']:>8.4f} [{d['ci95'][0]:>7.4f}, {d['ci95'][1]:>7.4f}] {d['median']:>8.4f} "
              f"{d['p90']:>8.4f} {d['max']:>8.4f} {s['loss_start']['mean']:>8.4f} {s['loss_second']['mean']:>8.4f} "
              f"{s['internal']['mean']:>8.0f} {s['seconds']['table']['mean']:>8.3f} "
              f"{s['seconds']['solve']['mean']:>8.3f}" + (f" {s['safety']['max']:>8.4f}" if safe else ""))
    if safe:
        print("safe max: the largest per-hand gain of an opponent best response against X over the base network "
              "(> 0: X is less safe than its base there; delta_T alone cannot rank re-solving modes)")
    print(f"written to {out}")
    return summaries


if __name__ == "__main__":
    main()
