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
    .venv/bin/python -m bbar.provenance                           # commit and code hash of this tree
    (cd core && cargo run --release --example bench_search)       # search speed

## Current best player

`pimc:w=16,d=4`: 88.2% [85.1, 90.8] against `greedy`, 99.2% [98.2, 99.7] against `random` (200 deal pairs, seed 1;
95% intervals). Fixed opponents only; exploitability not measured yet.

Value of information (500 deal pairs, seed 2): knowing the opponent's hand scores 65.5% [63.0, 68.0] against the same
PIMC without it; also knowing the deck orders adds nothing measurable (`pimc-hand` vs `cheat`: 49.9% [47.4, 52.4]).

These come from uncommitted code (log lines with commit `7c1c91c-dirty`, code `08cad7cf04c9e56b`); re-run both
ladders (`--ladder random greedy pimc:w=16,d=4 --deals 200 --seed 1`, `--ladder pimc:w=16,d=4 pimc-hand:w=16,d=4
cheat:d=4 --deals 500 --seed 2`) once it is committed. Earlier log lines predate the current seed derivation and
interval.
