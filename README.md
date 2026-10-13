# Beasty Bar AI

Start with [MISSION.md](MISSION.md): the goal, how to decide, and the constraints. The approach we are taking and
why: [docs/PLAN.md](docs/PLAN.md).

This file is the project's state: what exists, how to run it, the current best player and its measured strength.

## Status

- **Engine**: Rust core `bbar._core` (`core/`), exact on all 2000 [golden](golden/) games (~550k random-vs-random
  arena games/s per thread); independent pure-Python reference `bbar/ref.py`.
- **Agents** (`core/src/agents.rs`), from spec strings: `random`; `greedy` (1 ply); `pimc:w=W,d=D` (perfect-information
  Monte Carlo: W re-randomized views of the position, alpha-beta to depth D >= 1 (24 = to the end), exact solve once
  the position is perfect information, from ply 15). Agents only ever see `view()`: hidden cards freshly randomized.
  Oracles that see hidden information, for measuring its value only: `pimc-hand:w=W,d=D` (true opponent hand),
  `cheat:d=D` (true state). Trust model: agent code never touches the arena or `unsafe` (linted in
  `core/tests/info_leaks.rs`).
- **Arena** (`bbar/arena.py`): duplicate deal pairs, reproducible for any thread count, Ctrl-C stops it even
  mid-search. Random 64-bit seed unless `--seed`: anyone who knows the seed can recompute the deals, so explicit seeds
  are only for agents that keep the trust model. Every match appends a line to `runs/results.jsonl`, the evidence
  trail: score with its 95% interval over deal pairs (Wilson), commit, code hash, engine build, host, command
  (`bbar/provenance.py`). It refuses a stale engine build, and uncommitted code unless `--allow-dirty`.
- **Any agent against any other** (`bbar/match.py`): built-in specs and network checkpoints (`nn:PATH` samples the
  policy, `nn:PATH,argmax` plays its argmax) in the arena's duplicate deals, with the same evidence trail plus the
  SHA-256 of every checkpoint. Built-in agents act inside the batched env (`VecEnv.act`, `core/src/env_act.rs`) with the
  arena's views and seat RNGs: two built-in agents replay `bbar.arena` move for move.
- **Beliefs** (`bbar/belief.py`, `core/src/policy_obs.rs`): exact Bayes ranges over a player's 495 possible hands from
  the public history and a policy model (`bbar/policy.py`: a network, one batched forward over every candidate hand;
  or a built-in spec, sampled), checked against brute-force enumeration of deals. **Window tables**: a network's or a
  built-in agent's full strategy on a `Window` (`bbar._core.WindowTree`), with the real history (ply-10 windows:
  ~0.3 s for both players on the laptop CPU).
- **Exact-tail player** (`bbar/tail.py`, spec `tail:nn:PATH,F=10,mode=gadget|plain`): the network before ply F
  (move for move: it samples with the network's own draws), then one solved window (Bayes ranges under the network;
  safe gadget re-solve or plain DCFR) followed to ply 14, exact minimax from the perfect-information frontier. Python
  agents plug into `bbar.match` through `VecEnv.views` (the seat RNG's draw, the mover's view and a private seed,
  nothing else), with a fresh instance per stream of games so none sees both games of a duplicate pair. Keeps one
  solved window per game in play (`cache` >= `--batch`; ~12 MB per F=10 window).
- **E1 / delta_T** (`bbar/delta.py`; pods: `scripts/e1.sh`): exact exploitability of an agent's own play in the window
  at ply T of its self-play games, with both players' Bayes ranges, plus the equilibrium value and each seat's loss;
  for tail agents also the per-hand safety margin against the base network (plain DCFR has delta_T ~ 0 by
  construction, so delta_T alone does not rank re-solving modes).
- **Exploiter** (`bbar/rl/exploit.py`; pods: `scripts/exploit.sh`): PPO best response to a frozen target (built-in spec
  or checkpoint), optionally started from a policy (`--init`). Its final match score - 0.5, logged with its interval,
  is a lower bound on the target's exploitability.

## Run

    VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release   # build bbar._core (pods: scripts/build_core.sh)
    .venv/bin/python -m bbar.golden --engine core                 # rules check: 2000/2000 exact
    .venv/bin/python -m pytest -q && (cd core && cargo test --release)
    .venv/bin/python -m bbar.arena A B --deals N [--seed S]       # or: --ladder SPEC1 SPEC2 ...
    .venv/bin/python -m bbar.match A B --deals N [--device D]     # A, B: a spec or nn:PATH[,argmax]
    .venv/bin/python -m bbar.rl.exploit --name NAME --target T [--init PATH]   # exploiter (pods: scripts/exploit.sh)
    .venv/bin/python -m bbar.delta X --T 12,11,10,9 --N 200       # E1: delta_T of X (pods: scripts/e1.sh)
    .venv/bin/python -m bbar.provenance                           # commit and code hash of this tree
    (cd core && cargo run --release --example bench_search)       # search speed

## Current best player

`pimc:w=128,d=8` (search only, no learning). Round robin, 1000 deal pairs each, seed 11, 95% intervals
(`--ladder greedy pimc:w=8,d=4 pimc:w=32,d=4 pimc:w=32,d=6 pimc:w=128,d=6 pimc:w=32,d=8 pimc:w=128,d=8`, commit 598db4f):

| vs | greedy | pimc w=8,d=4 | w=32,d=4 | w=32,d=6 | w=128,d=6 | w=32,d=8 |
|---|---|---|---|---|---|---|
| pimc:w=128,d=8 | 96.7 [96.0, 97.3] | 70.7 [69.1, 72.3] | 67.5 [65.8, 69.1] | 57.4 [55.8, 59.0] | 57.0 [55.5, 58.6] | 51.7 [50.2, 53.2] |

Search depth buys most of the strength (d=4 -> 6 -> 8); more sampled worlds buy little. Exploitability not measured
yet: fixed opponents only.

Value of information (2000 deal pairs, seed 12, depth 6): knowing the opponent's hand scores 66.8% [65.6, 67.9]
against the same PIMC without it; also knowing deck orders adds nothing (50.8% [49.6, 51.9]). Hidden-hand play is
where the strength is.
