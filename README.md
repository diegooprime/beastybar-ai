# Beasty Bar AI

Start with [MISSION.md](MISSION.md): the goal, how to decide, and the constraints. The approach and why:
[docs/PLAN.md](docs/PLAN.md). This file is the project's state: what exists, how to run it, the best player and its
measured strength. Every number below is a line in [runs/results.jsonl](runs/results.jsonl) (command, commit, code
hash, seeds, 95% interval) or a `bbar.delta` run noted with it.

## Best player and measured strength

**Best player: the re-solving player** `resolve:V=checkpoints/value/c5-2/best.pt,base=nn:checkpoints/bp-v1/step_30518.pt,P0=8,F=9,d=2,iters=200,mode=plain`
(blueprint network for plies 0-7, a depth-limited CFR solve with value-network leaves at its ply-8 decision, the
exact hidden-information window from ply 9, minimax once perfect-information). It is not yet strong in the
mission's sense: plies 0-7 are still the blueprint's, and exploitability is measured only from ply 8 on.

| Player | vs blueprint final | vs `pimc:w=128,d=8` | Exploitability (lower bounds) |
|---|---|---|---|
| **resolve c5-2, exact from ply 9** | 62.9 [59.8, 65.9] | - | >= 0.0077 [0.0057, 0.0097] from ply 8 |
| resolve c5m, exact from ply 9 | 62.6 [60.8, 64.4] | 61.3 [58.6, 64.0] | >= 0.0090 [0.0066, 0.0114] from ply 8 |
| resolve v3h, exact from ply 10 | - | - | >= 0.0187 [0.0153, 0.0221] from ply 8; >= 0.0106 [0.0087, 0.0126] from ply 9 |
| resolve v2c, exact from ply 10 | 59.6 [57.4, 61.7] | 62.6 [59.7, 65.3] | >= 0.015 [0.012, 0.018] from ply 9 |
| resolve v2c, safe | 58.8 [56.6, 60.9] | - | >= 0.023 [0.020, 0.027] from ply 9 |
| `pimc:w=128,d=8` (search) | 51.6 [50.4, 52.8] | - | not measured |
| blueprint `bp-v1` final (network) | - | 48.4 [47.2, 49.6] | >= 0.126 [0.112, 0.139] from ply 8; >= 0.111 [0.104, 0.117] from ply 9 |
| blueprint `bp-v1` step 8000 | 42.7 [41.6, 43.8] | - | >= 0.127 [0.119, 0.135] from ply 9 (run record lost in the balance outage); >= 0.022 [0.018, 0.026] by a PPO exploiter |
| `pimc:w=8,d=4` | - | 29.3 [27.7, 30.9] | >= 0.161 [0.157, 0.165] (PPO exploiter) |
| iter949 (previous attempt's best network) | 14.8 [13.5, 16.1] | - | not measured |

"From ply T": an exact best response that copies the player until ply T and then plays perfectly gains this much over
0.5 (`bbar.delta`, self-play positions with Bayes ranges). Every player that plays the blueprint before ply 8 reaches
the same positions for the same seed, so the ply-8 numbers compare players position by position
(`scripts/delta_pool.py`, "pooled" and "paired" lines in [runs/delta-summary.jsonl](runs/delta-summary.jsonl)): c5-2 is
0.0013 [0.0004, 0.0023] less exploitable than c5m (138 positions, seeds 1001-1024), 0.0105 [0.0039, 0.0172] less than
v3h exact from ply 10 (10 positions) and 0.125 less than the blueprint.

Head to head the resolve variants are close: c5-2 vs c5m 52.1% [47.0, 57.2] (48 pairs); c5m vs v3h exact from ply 10
49.9% [48.2, 51.6] (320 pairs); v3h vs v2c 51.4% [49.5, 53.3]; exact from ply 9 vs from ply 10 50.7% [48.8, 52.5];
search from ply 6 (first cascade net) vs from ply 8 49.0% [46.8, 51.2]. Between strong players, exploitability is
the discriminating measure.

Scores: win 1, draw 0.5, loss 0, duplicate deal pairs: resolve rows 300-500 (c5-2 vs the blueprint 192, c5-2 vs c5m
48 and c5m vs v3h 320: stopped at the compute deadline), 2000 for blueprint rows and PIMC d=8 vs the blueprint, 1000
for `pimc:w=8,d=4` and iter949, 20,000 for exploiters. Exploitability = best responder's score - 0.5.

What we have learned:
- **Hidden-hand play is where the strength is.** Knowing the opponent's hand scores 66.8% [65.6, 67.9] against the same
  PIMC without it; knowing the deck orders as well adds nothing (the full-state player scores 49.2% [48.1, 50.4]
  against the hand-only one).
- **PIMC strength comes from depth** (d=4 -> 6 -> 8), barely from more sampled worlds.
- **The blueprint is far more exploitable than RL exploiters show.** An exact best response that copies it to ply T and
  then plays perfectly gains (200 self-play positions per T, `bbar.delta`, seed 1001): 0.072 from ply 12, 0.092 from
  11, 0.110 from 10, 0.127 from 9 (step 8000; that run's record was lost); 0.069 / 0.087 / 0.100 / 0.111 for the
  final checkpoint, which beats step 8000 57.3% [56.2, 58.4]. A PPO exploiter finds only 0.022: learned exploiters
  understate the worst case.
- **The late game can be solved exactly**: exact windows from ply 10 take ~0.4 s, from ply 9 ~1-5 s, from ply 8 ~2 min.
  Depth-limited solving with exact leaf values is nearly unexploitable (1e-6..2e-4, E2 oracle). With a learned PBS
  value network at the leaves (held-out error 0.068), depth-limited play from ply 10 is 0.020 [0.014, 0.025]
  exploitable (E2, 48 positions) against the blueprint's 0.100 there. Value-net error falls slowly with data
  (0.070 / 0.063 / 0.057 at 62k / 139k / 253k public states); option features (what playing each animal now
  would do) reach 0.0555 with 60% of the data (v3h).
- **Re-solving works end to end.** With that value network the re-solving player cuts exploitability from ply 9 from
  0.111 (blueprint) to 0.015 and beats every earlier player.
- **Better value labels keep paying, measured exactly.** The one-day push (2026-10-13): 106k exact labels at plies
  9-10, 356k depth-limited cascade labels at plies 8-9 and 6-7 from the batched solver (~10x faster: one level in
  ~20 min on 8 pods), 931k labels in all. Retraining took held-out value error from 0.0555 to 0.0444 (test sets
  differ: c5-2's includes the new plies) and the player's exploitability from ply 8 from 0.0187 (v3h, exact from
  ply 10) to 0.0077 (c5-2, exact from ply 9), against the blueprint's 0.126. Match scores barely move (vs the
  blueprint 59.6 -> 62.6-62.9; vs PIMC d=8 62.6 -> 61.3): they no longer separate these players.
- **Next** (in order of expected gain): start the search before ply 8 (P0 = 6, 4, ... 0) once the cascade's labels
  at those plies are as plentiful as at plies 8-9 (a first try with thin labels was level, 49.0%); measure
  exploitability from earlier plies (exact from ply 8 is the limit of the window solver; earlier needs the value net
  as a certified bound); keep growing labels and nets, which has lowered error at every step so far.

## Play

    .venv/bin/python -m bbar.play --open        # http://127.0.0.1:8765, local only

A page in your browser against the strongest player available on this machine (default: the best player above, value
net `checkpoints/value/c5-2/best.pt`, on the CPU; without checkpoints it falls back to the blueprint network or PIMC
and says so). You see the queue from Heaven's Gate (front) to THAT'S IT (back), the bar and bounced piles, both scores
and deck sizes, your hand, your deck as a set and the AI's unplayed animals as a set. Click a card (then its target,
hop or animal to copy), check the preview, confirm. The move log tells what every move did; the AI thinks in the
background with its time shown (on the M4: a few seconds at plies 8-10, about 9 s per game in all, instant
otherwise; it precomputes what it can while you think). Rules cheat sheet, end screen, opponent picker (re-solver, blueprint,
PIMC d=8 / d=4, greedy, random). Options: `--opponent SPEC|KEY --seat 0|1|random --seed S --port P --device cpu|mps`.
The AI plays through bbar.match's code path (a one-game VecEnv: its own view and the public history only). Finished
games go to `runs/play/games.jsonl`.

## What exists

- **Engine** `bbar._core` (Rust, `core/`): exact on all 2000 [golden](golden/) games, ~700k random games/s per thread;
  independent Python reference `bbar/ref.py`; differential tester `bbar/difftest.py`.
- **Agents and arena**: `random`, `greedy`, `pimc:w=W,d=D` (+ oracles `pimc-hand`, `cheat` for value-of-information
  experiments only); duplicate-deal arena `bbar/arena.py` and `bbar/match.py` (any built-in spec, network `nn:PATH`, or
  Python agent such as `tail:`, `resolve:`, or `iter949:DIR`, the previous attempt's best network kept outside the
  repository, through `bbar/iter949.py`). Agents see only re-randomized views; evidence trail with provenance; stale
  builds and uncommitted code refused.
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
  net, retrain, repeat down to ply 0. Many depth-limited solves run in lockstep (`bbar/value/dl.solve_many`): one
  multithreaded Rust call (`bbar._core.WindowBatch`) and one network call per half-iteration. PBS v2 inputs add option features (`core/src/options.rs`: for every unplayed
  animal of either player, what playing it now would do, best and worst parameter); `--arch opt` builds each hand's
  value from its animals' features. Version-1 networks still load and run everywhere.
- **Re-solving player** (`bbar/resolve.py`, spec `resolve:V=PATH,base=nn:PATH,P0=8,F=10,d=2,iters=200,mode=safe|plain`):
  the base network before ply P0, a depth-limited CFR-D solve with value-net leaves at every own decision from P0
  (safe: carried opponent values in a gadget re-solve; plain: common-knowledge Bayes beliefs), the exact window from
  ply F, minimax once perfect-information (best player above).
- **Pod tools**: `scripts/pod.sh` (create, set up, sync, run, back up, guard the balance), `scripts/podqueue.py`
  (start queued jobs on pods as they go idle), `scripts/gather.sh` (collect label shards onto a hub pod),
  `scripts/delta_pool.py` (pool `bbar.delta` runs across seeds and pods, paired comparisons).

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
data/ are on the laptop; the full label sets and per-pod outputs are in backups/<pod id>/ (gitignored).
