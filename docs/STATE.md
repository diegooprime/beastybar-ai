# Session state (training day, 2026-10-13)

Working notes for whoever continues (human or agent). README.md is the project's public state; this is the live
operations log for the one-day $50 compute push (2026-10-13; ended ~22:55 UTC).

## Pods

None: the training day ended at ~22:55 UTC and every pod was deleted (budget spent). Their data and checkpoints were
backed up to the laptop (backups/<pod id>/, gitignored; `scripts/pod.sh autobackup`), the final networks are in
checkpoints/value/ (c5-2 = best, c5m, c5h), and every result is in runs/results.jsonl and runs/delta-summary.jsonl.

## Data and networks (local, gitignored)

- Exact labels: plies 10-13 data/value/{v2,v2q,v3,v3q} (~430k); plies 9-10 x9 (15k) and x10 (91k) in
  backups/ulq9kbd5vy7dux/data/value/.
- Cascade labels (depth-limited solves with the previous net): c5-p8-9 (278k), c5-p6-7 (79k), older c3/c4 sets,
  same backup directory.
- Value nets (PBS v2 except v2c, v1 features): c5-2 (MLP 1024x6, all 931k labels, test 0.0444; the best player's), c5m
  (MLP, before x10 and c5-p6-7, 0.0459), c5h (hand 768x4, 0.0458), v3h (0.0555), older v2c, v3m, c4-1, c4-2.
- Blueprint: checkpoints/bp-v1/step_30518.pt.

## Final results of the day

- Best player: resolve c5-2, P0=8, F=9 (README). delta_8 pooled over seeds 1001-1024 for c5-2 and c5m, seed 1001 for the
  blueprint and v3h (scripts/delta_pool.py); c5-2 beats c5m position by position; matches of the c5m variant: 62.6% vs
  blueprint (500 pairs), 61.3% vs PIMC d=8.
- Not done for lack of time: P0 < 8 with the c5 nets; exploitability before ply 8; a cascade level at plies 4-5 with
  the fast solver.

## Next session, in order

1. Cascade levels 4-5, 2-3, 0-1 with c5-2 and the fast solver; retrain; test P0=6 and P0=4 head to head and by
   delta_8 (pairs).
2. More exact labels at plies 9-10 (they lowered the error at the leaves the ply-8 solve queries).
3. A certified exploitability bound from earlier plies (value-net leaves with their error as slack).

## Gotchas learned today

- The laptop shell is zsh: `set -- $var` does not split words; write commands explicitly.
- `pgrep -f`/`pkill -f` patterns match their own command line (and later commands in the same `bash -c`): use `[b]bar`
  and never wait on a pattern that appears in the waiting command itself.
- `ssh pod 'nohup cmd &'` can hang; use `scripts/pod.sh bg`.
- RunPod API lookups get rate-limited with many concurrent pod.sh calls: targets are now cached.
- Two community 2xL40S hosts had unusable CUDA (driver 550 / broken devices): use SECURE; setup now checks CUDA.
- Generators keep labels in memory until a shard fills: use small --shard-size so work survives restarts.
- Cascade label runs can leave their multiprocessing workers behind (parent pid 1, spinning on CPU): after a run
  ends, `ps -eo pid,ppid,args | grep '[s]pawn_main' | awk '$2==1'` must be empty (they halved five pods' speed).
- Matches only log at the end: run them with small --batch and stop them with SIGTERM (`pkill -TERM -o -f
  'python -m [b]bar.match'`) to log the finished batches (`stopped_early`).
- scripts/podqueue.py starts queued jobs on pods as they go idle; scripts/delta_pool.py pools delta runs.
