# Session state (training day, 2026-10-13)

Working notes for whoever continues (human or agent). README.md is the project's public state; this is the live
operations log for the one-day $50 compute push (deadline ~23:30 UTC, then no more compute).

## Pods (scripts/pod.sh status shows load and jobs; addresses cached in $TMPDIR/bbar-pod-targets)

| id | name | GPU | vCPU | $/hr | job |
|---|---|---|---|---|---|
| ulq9kbd5vy7dux | bb-cascade | 2xA40 | 21 | 1.18 | exact ply-9 labels (x9b, seed 55); hub for training |
| hqxg7gb65gwyn7 | bb-eval | 4090 | 12 | 0.89 | matches: resolve F=9 vs F=10 (v3h, P0=8, 300 deals, seed 71) |
| 59uyapmvn5edyp | bb-x9 | 2xA40 | 19 | 1.18 | exact ply-9 labels (seed 51) |
| 5e1w9a89lkgyni | bb-h1 | A40 | 9 | 0.59 | exact ply-9 labels (seed 52) |
| l2g2inejf282uk | bb-h2 | 4090 | 16 | 0.89 | exact ply-9 labels (seed 53) |
| yx303cn3c0gitu | bb-h3 | L40S | 32 | 1.09 | exact ply-9 labels (seed 54) |

Exact ply-9 generators (restarted ~19:00 UTC): `bbar.value.gen --out data/value/x9 --plies 9 --ranges-per-state 2
--bayes 0.5 --shard-size 20 --target 1e-4 --minutes 70` -> each pod's data/value/x9. Laptop runs `pod.sh guard 2.5`
(backs up and deletes all pods if balance < $2.5) and `pod.sh autobackup` per pod into backups/<id>/.

## Data and networks (local copies in data/ and checkpoints/, all gitignored)

- Exact labels plies 10-13: data/value/{v2,v2q,v3,v3q} (~430k). Cascade labels: c3-p8-9, c4-p8-9 (~30k), c4-p6-7 (6.5k),
  c4-p4-5 (~14k + hub's) on the hub (backups/ulq9kbd5vy7dux).
- Value nets (PBS v2 unless noted): v2c (MLP, v1 features, test 0.0568), v3m (MLP 1024x6, 0.0568), v3h (hand 768x4,
  0.0555, best), c4-1 (cascade, plies 8-13), c4-2.
- Blueprint: checkpoints/bp-v1/step_30518.pt.

## Results so far (evidence: runs/results.jsonl, runs/delta-summary.jsonl)

- resolve plain (v2c, P0=8, F=10): 59.6% vs blueprint, 62.6% vs pimc:w=128,d=8; delta_9 0.0151. safe: 58.8%, 0.0233.
- resolve plain v3h: delta_9 0.0106 [0.0087, 0.0126]; vs v2c-resolve 51.4% [49.5, 53.3] (head to head flat).
- resolve P0=6 vs P0=8 (c4-1): 49.0% [46.8, 51.2]: early search with thin cascade labels did not help.
- resolve F=9 (v3h) vs blueprint, 16 deals: 65.6% (timing only); F=9 vs F=10 match running.

## In flight on the laptop (workflows)

- play-ui (worktree .worktrees/ui): local web UI `python -m bbar.play`, in fix phase.
- dl-speed (worktree .worktrees/fast): batched Rust WindowBatch + NetValues.session(): ~10x faster depth-limited
  solving, bit-identical; in fix phase. Merge, then rerun the cascade at 10x volume.

## Plan for the rest of the day

1. Merge dl-speed and play-ui; sync pods.
2. ~20:10 UTC: x9 generators stop; gather x9 shards on the hub; train v4 (hand + mlp) on exact 9-13 data.
3. Rerun cascade levels 8-9 and 6-7 with the fast code across all pods; retest P0=6 vs P0=8.
4. Pick the final player (F=9 vs 10, P0, mode) by head-to-head + delta_9; final matches vs blueprint and pimc:w=128,d=8.
5. README, results, push; UI default opponent = final player; delete all pods.

## Gotchas learned today

- The laptop shell is zsh: `set -- $var` does not split words; write commands explicitly.
- `pgrep -f`/`pkill -f` patterns match their own command line (and later commands in the same `bash -c`): use `[b]bar`
  and never wait on a pattern that appears in the waiting command itself.
- `ssh pod 'nohup cmd &'` can hang; use `scripts/pod.sh bg`.
- RunPod API lookups get rate-limited with many concurrent pod.sh calls: targets are now cached.
- Two community 2xL40S hosts had unusable CUDA (driver 550 / broken devices): use SECURE; setup now checks CUDA.
- Generators keep labels in memory until a shard fills: use small --shard-size so work survives restarts.
