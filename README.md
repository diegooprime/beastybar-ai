# Beasty Bar AI

Start with [MISSION.md](MISSION.md): the goal, how to decide, and the constraints. The approach and why:
[docs/PLAN.md](docs/PLAN.md). This file is the project's state: what exists, how to run it, the best player and its
measured strength. Every number below is a line in [runs/results.jsonl](runs/results.jsonl) (command, commit, code
hash, seeds, 95% interval) or a `bbar.delta` run noted with it.

## Best player and measured strength

No player is yet strong in the mission's sense (worst case). Two are about equal head to head:

| Player | vs greedy | vs `pimc:w=32,d=6` | vs `pimc:w=128,d=8` | Exploitability (lower bounds) |
|---|---|---|---|---|
| `pimc:w=128,d=8` (search) | 96.7 [96.0, 97.3] | 57.4 [55.8, 59.0] | - | not measured |
| blueprint `bp-v1` final (network) | - | 53.2 [51.9, 54.4] | 48.4 [47.2, 49.6] | exact best response from ply 9: running |
| blueprint `bp-v1` step 8000 | 92.9 [92.1, 93.6] | 48.6 [47.3, 49.9] | - | >= 0.127 [0.119, 0.135] from ply 9; >= 0.022 [0.018, 0.026] by a PPO exploiter |
| `pimc:w=8,d=4` | 90.1 [88.8, 91.2] | 34.4 [32.7, 36.1] | 29.3 [27.7, 30.9] | >= 0.161 [0.157, 0.165] (PPO exploiter) |

Scores: win 1, draw 0.5, loss 0, duplicate deal pairs (2000 for blueprint rows, 1000 for PIMC, 20,000 for
exploiters). Exploitability = best responder's score - 0.5.

What we have learned:
- **Hidden-hand play is where the strength is.** Knowing the opponent's hand scores 66.8% [65.6, 67.9] against the same
  PIMC without it; knowing deck orders adds nothing (50.8% [49.6, 51.9]).
- **PIMC strength comes from depth** (d=4 -> 6 -> 8), barely from more sampled worlds.
- **The blueprint is far more exploitable than RL exploiters show.** An exact best response that copies it to ply T and
  then plays perfectly gains (200 self-play positions per T, `bbar.delta`, seed 1001): 0.072 from ply 12, 0.092 from
  11, 0.110 from 10, 0.127 from 9 (step 8000). A PPO exploiter finds only 0.022. Longer training (final checkpoint)
  beats step 8000 57.3% [56.2, 58.4] but stays about as exploitable late.
- **The late game can be solved exactly**: exact windows from ply 10 take ~0.4 s, from ply 9 ~1-5 s, from ply 8 ~2 min.
  Depth-limited solving with exact leaf values is nearly unexploitable (1e-6..2e-4, E2 oracle). The plan is to learn
  those leaf values (PBS value network) and re-solve every decision (docs/PLAN.md, M4).

## What exists

- **Engine** `bbar._core` (Rust, `core/`): exact on all 2000 [golden](golden/) games, ~700k random games/s per thread;
  independent Python reference `bbar/ref.py`; differential tester `bbar/difftest.py`.
- **Agents and arena**: `random`, `greedy`, `pimc:w=W,d=D` (+ oracles `pimc-hand`, `cheat` for value-of-information
  experiments only); duplicate-deal arena `bbar/arena.py` and `bbar/match.py` (any built-in spec, network `nn:PATH`, or
  Python agent such as `tail:`). Agents see only re-randomized views; evidence trail with provenance; stale builds
  and uncommitted code refused.
- **Blueprint** (`bbar/rl/mmd.py`): MMD-PPO self-play, 3.4M-parameter policy, oracle critic; `scripts/train_blueprint.sh`.
- **Exact solvers** (`core/src/{hands,markov,pisolve,window}.rs`): perfect-information alpha-beta (from ply 15),
  window solver over the hidden-information public tree (DCFR/PCFR+, private draws, exact best response, resolve
  gadget, external strategy tables, depth-limited cuts with external leaf values). Cross-checked against an
  independent explicit-infoset CFR (`bbar/xcfr.py`) and brute force.
- **Beliefs and exact tail**: exact Bayes ranges under a policy (`bbar/belief.py`); `tail:nn:PATH,F=10,mode=gadget|plain`
  plays the network, then an exact window from ply F (`bbar/tail.py`).
- **Measurement**: `bbar.delta` (exact exploitability from ply T, E1); `bbar.rl.exploit` (PPO best response);
  `bbar.value.e2` (depth-limited solving with a value net vs exact).
- **Value network pipeline** (`bbar/value/`): exact labels from window solves (`gen.py`, synthetic and Bayes ranges),
  trainer, E2 evaluator.

## Run

    VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release   # build bbar._core (pods: scripts/build_core.sh)
    .venv/bin/python -m bbar.golden --engine core                 # rules check: 2000/2000 exact
    .venv/bin/python -m pytest -q && (cd core && cargo test --release)
    .venv/bin/python -m bbar.arena A B --deals N [--seed S]       # built-in agents; --ladder SPEC1 SPEC2 ...
    .venv/bin/python -m bbar.match A B --deals N [--device D]     # any agents: spec, nn:PATH[,argmax], tail:...
    .venv/bin/python -m bbar.delta X --T 12,11,10,9 --N 200       # exact exploitability from ply T (scripts/e1.sh)
    .venv/bin/python -m bbar.rl.exploit --name NAME --target T    # PPO exploiter (scripts/exploit.sh)
    .venv/bin/python -m bbar.value.gen --out data/value/NAME      # exact value labels (scripts/value_gen.sh)
    .venv/bin/python -m bbar.value.train --name NAME --data data/value/NAME   # value net (scripts/value_train.sh)
    .venv/bin/python -m bbar.value.e2 --net checkpoints/value/NAME/best.pt     # E2 (scripts/e2.sh)

Pods: `scripts/pod.sh` (up, setup, sync, push, bg, pull, down). Networks and data are not in git: checkpoints/ and
data/ live on the pods and the laptop.
