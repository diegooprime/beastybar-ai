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
- **Depth-limited solving** (`core/src/window.rs`, `bbar._core.Window(..., leaf_ply=L)`): windows cut at ply L, value
  leaves valued from Python every half-iteration with the current iterate's beliefs (CFR-D); composite profile = trunk
  + safe re-solves below the leaves (`bbar/value/dl.py`), scored by exact exploitability in the full window. With exact
  leaf values, 200 iterations, 8 ply-11 roots (leaves at 12): mean 2.8e-5 [3.8e-6, 7.2e-5], max 1.8e-4
  (`runs/results.jsonl`, smoke-S11). Mostly the trunk's slow CFR-D convergence; on slow positions about a third comes
  from the safe re-solves (the averaged opponent values they respect are only upper bounds), none from leaf-solve or
  gadget accuracy (`tests/test_value.py`). E2 logs `trunk_eq` (same trunk, exact solution below) to separate them.
- **PBS value network, experiment E2** (`bbar/value/`; pods: `scripts/value_gen.sh`, `value_train.sh`, `e2.sh`): exact
  labels (per-hand values of both players from exact window solves, ranges from Dirichlet / sparse / peaked /
  mixture families, states from random / PIMC / blueprint games), residual-MLP trainer (best.pt picked on a
  validation split, reported on a separate test split), and the E2 evaluator (network leaves vs exact leaves vs exact
  solving vs uniform, ply 9 roots; cost = deployable work only; summaries appended to `runs/results.jsonl`). Results
  pending a pod run.
- **Continual re-solving player, M4** (`bbar/resolve.py`, spec
  `resolve:V=PATH,base=nn:PATH,P0=8,F=10,d=2,iters=200,mode=safe|plain,prior=mix|uniform|bayes`): the base network
  before ply P0; from P0 to F, at each own decision a depth-limited window (value leaves at min(ply + d, F), valued by
  the value network V, `iters` CFR-D iterations), playing its root average strategy for the actual hand; from F one
  exact window followed to the end, as the tail agent (P0 >= F is the tail agent, move for move). Beliefs: `plain`
  (ReBeL) Bayes ranges under the policies the solves give both players; `safe` (DeepStack) the opponent's values
  carried from the previous solve (averaged leaf values, or best-response cfvs at an internal node) and the re-solving
  gadget (`Window.resolve_external`; its opponent prior half Bayes, half uniform, so no hand loses its weight),
  base-network values before the first solve. Our range: exact Bayes under the strategy actually used for every hand.
  The depth-limited solves of all games in play run in lockstep, one batched value-network call per half-iteration
  (`bbar/value/dl.solve_many`); fixed-size forward passes keep a game's play independent of the batch. `V=exact`
  (exact leaf values) is the reference. Checked with exact leaves: the carried values are nearly achievable per
  opponent hand. Largest gadget violation over all opponent hands (base step_30518, P0=11, F=14, 100 iterations, seeds
  1/7/21; 36 re-solves on the played path / 288 at random own ply-13 nodes off it): 0.0011 / 0.014 with the mixed
  prior, 0.0005 / 0.010 uniform, 0.17 / 0.090 with the Bayes prior alone (an earlier "<= 2e-4" was one game's played
  path). Plain mode's delta_T <= 1e-4 at T = F-2, F-1 (F = 13, 3 positions, 200 iterations); bbar.delta and bbar.match
  take it like tail agents. Local smoke runs only (a 128x2 V trained on 4k labels at plies 11-13: plumbing, not
  strength); pod runs pending the plies 10-13 value network.
- **Training cascade** (`bbar/value/cascade.py`, pods: `scripts/cascade.sh`): labels at plies 8-9 from depth-limited
  solves (leaves at ply 10) valued by the current network, gen.py's shard format and range family (--bayes
  included), retrain on all plies, then 6-7, ... 0-1. A label is each hand's best-response value against the solve's
  average strategy in the depth-limited game with the averaged leaf values. With exact leaves (`tests/test_cascade.py`)
  its game value is gen.py's within solver tolerance, and so are its per-hand values where the ranges reach (up to
  the choice among equilibria, which gen.py's own solvers also make differently); off the ranges' support it is not
  gen.py's (up to 0.33, and more iterations do not close it), so training leaves cascade labels out of its all-hands
  term. `--resume` skips only trainings that ran to their end and levels generated by the current network.

## Run

    VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release   # build bbar._core (pods: scripts/build_core.sh)
    .venv/bin/python -m bbar.golden --engine core                 # rules check: 2000/2000 exact
    .venv/bin/python -m pytest -q && (cd core && cargo test --release)
    .venv/bin/python -m bbar.arena A B --deals N [--seed S]       # or: --ladder SPEC1 SPEC2 ...
    .venv/bin/python -m bbar.match A B --deals N [--device D]     # A, B: a spec or nn:PATH[,argmax]
    .venv/bin/python -m bbar.rl.exploit --name NAME --target T [--init PATH]   # exploiter (pods: scripts/exploit.sh)
    .venv/bin/python -m bbar.delta X --T 12,11,10,9 --N 200       # E1: delta_T of X (pods: scripts/e1.sh)
    .venv/bin/python -m bbar.value.gen --out data/value/NAME      # exact value labels (pods: scripts/value_gen.sh)
    .venv/bin/python -m bbar.value.train --name NAME --data data/value/NAME   # value network (scripts/value_train.sh)
    .venv/bin/python -m bbar.value.e2 --net checkpoints/value/NAME/best.pt     # E2 (scripts/e2.sh)
    .venv/bin/python -m bbar.value.cascade run --name NAME --data data/value/v1  # nets to ply 0 (scripts/cascade.sh)
    .venv/bin/python -m bbar.match resolve:V=V.pt,base=nn:BP.pt,P0=8,F=10 nn:BP.pt   # M4 player (scripts/match.sh)
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
