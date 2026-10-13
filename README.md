# Beasty Bar AI

Start with [MISSION.md](MISSION.md): the goal, how to decide, and the constraints. The approach and why:
[docs/PLAN.md](docs/PLAN.md). This file is the project's state: what exists, how to run it, the best player and its
measured strength. Every number below is a line in [runs/results.jsonl](runs/results.jsonl) (command, commit, code
hash, seeds, 95% interval) or a `bbar.delta` run noted with it.

## Best player and measured strength

**Best player: the re-solving player** `resolve:V=checkpoints/value/v2c/best.pt,base=nn:checkpoints/bp-v1/step_30518.pt,P0=8,F=10,d=2,iters=200,mode=plain`
(blueprint network for plies 0-7, a depth-limited CFR solve with value-network leaves at every own decision from
ply 8, the exact window from ply 10). It is not yet strong in the mission's sense: plies 0-7 are still the
blueprint's, and the exploitability below covers only play from ply 9 on.

| Player | vs blueprint final | vs `pimc:w=128,d=8` | Exploitability (lower bounds) |
|---|---|---|---|
| **resolve, plain** (v2c) | 59.6 [57.4, 61.7] | 62.6 [59.7, 65.3] | >= 0.015 [0.012, 0.019] from ply 9 |
| resolve, safe (v2c) | 58.8 [56.6, 60.9] | - | >= 0.023 [0.020, 0.027] from ply 9 |
| `pimc:w=128,d=8` (search) | 51.6 [50.4, 52.8] | - | not measured |
| blueprint `bp-v1` final (network) | - | 48.4 [47.2, 49.6] | >= 0.111 [0.104, 0.118] from ply 9 |
| blueprint `bp-v1` step 8000 | 42.7 [41.6, 43.8] | - | >= 0.127 [0.119, 0.135] from ply 9; >= 0.022 [0.018, 0.026] by a PPO exploiter |
| `pimc:w=8,d=4` | - | 29.3 [27.7, 30.9] | >= 0.161 [0.157, 0.165] (PPO exploiter) |

"From ply 9": an exact best response that copies the player until ply 9 and then plays perfectly gains this much over
0.5 (`bbar.delta`, 100-200 self-play positions, seed 1001). Resolve matches: 400 deal pairs vs the blueprint, 300 vs
PIMC; other rows 1000-2000.

Scores: win 1, draw 0.5, loss 0, duplicate deal pairs (2000 for blueprint rows, 1000 for PIMC, 20,000 for
exploiters). Exploitability = best responder's score - 0.5.

What we have learned:
- **Hidden-hand play is where the strength is.** Knowing the opponent's hand scores 66.8% [65.6, 67.9] against the same
  PIMC without it; knowing deck orders adds nothing (50.8% [49.6, 51.9]).
- **PIMC strength comes from depth** (d=4 -> 6 -> 8), barely from more sampled worlds.
- **The blueprint is far more exploitable than RL exploiters show.** An exact best response that copies it to ply T and
  then plays perfectly gains (200 self-play positions per T, `bbar.delta`, seed 1001): 0.072 from ply 12, 0.092 from
  11, 0.110 from 10, 0.127 from 9 (step 8000); 0.069 / 0.087 / 0.100 / 0.111 for the final checkpoint, which beats
  step 8000 57.3% [56.2, 58.4]. A PPO exploiter finds only 0.022: learned exploiters understate the worst case.
- **The late game can be solved exactly**: exact windows from ply 10 take ~0.4 s, from ply 9 ~1-5 s, from ply 8 ~2 min.
  Depth-limited solving with exact leaf values is nearly unexploitable (1e-6..2e-4, E2 oracle). With a learned PBS
  value network at the leaves (held-out error 0.068), depth-limited play from ply 10 is 0.020 [0.014, 0.025]
  exploitable (E2, 48 positions) against the blueprint's 0.100 there. Value-net error falls slowly with data
  (0.070 / 0.063 / 0.057 at 62k / 139k / 253k public states); option features (what playing each animal now
  would do) reach 0.0555 with 60% of the data (v3h).
- **Re-solving works end to end.** With that value network the re-solving player cuts exploitability from ply 9 from
  0.111 (blueprint) to 0.015 and beats every earlier player. Next: the cascade (value labels for plies 8 down to 0) so
  the search starts at ply 0 instead of 8.

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
  labels at the beliefs depth-limited solves query (`harvest.py`), trainer (`--arch mlp|hand|opt`), E2 evaluator, and
  the cascade (`cascade.py`, `scripts/cascade.sh`): labels at earlier plies from depth-limited solves with the current
  net, retrain, repeat down to ply 0. PBS v2 inputs add option features (`core/src/options.rs`: for every unplayed
  animal of either player, what playing it now would do, best and worst parameter); `--arch opt` builds each hand's
  value from its animals' features. Version-1 networks still load and run everywhere.
- **Re-solving player** (`bbar/resolve.py`, spec `resolve:V=PATH,base=nn:PATH,P0=8,F=10,d=2,iters=200,mode=safe|plain`):
  the base network before ply P0, a depth-limited CFR-D solve with value-net leaves at every own decision from P0
  (safe: carried opponent values in a gadget re-solve; plain: common-knowledge Bayes beliefs), the exact window from
  ply F, minimax once perfect-information. Not yet measured with a real value net.

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
    scripts/cascade.sh NAME DATA --net checkpoints/value/NAME/best.pt          # value labels down to ply 0 (pods)

Pods: `scripts/pod.sh` (up, setup, sync, push, bg, pull, down). Networks and data are not in git: checkpoints/ and
data/ live on the pods and the laptop.
