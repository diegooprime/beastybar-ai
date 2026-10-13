# Plan

How we intend to reach the [mission](../MISSION.md). Decided 2026-10-12 after an independent design panel (three
competing proposals, two judges). Revise when measurements say otherwise; [README.md](../README.md) holds the state.

## What makes this game tractable

- The only hidden information is each player's hand: a K-subset (K = 4) of their unplayed animals, at most 495
  possibilities. Hands of the two players are independent (each owns their own 12 animals), so a public belief state
  is just two independent ranges. Terminal payoffs depend only on public information.
- Ply k = k moves played (0-indexed). Both decks are empty after ply 16; at ply 15 the second mover already knows
  everything. From there the game is perfect-information and solved exactly by alpha-beta in microseconds.
- The hidden-information part of the public tree is small late: ~3e3 nodes from ply 12, ~2e5 from ply 10,
  ~2e7 from ply 8 (estimates; to be re-measured). Late decisions can be solved exactly.

## Player architecture (target)

1. **Perfect-information tail** (ply >= 15): exact minimax.
2. **Exact window** (ply >= F, F = 12 now, earlier as compute allows): vector CFR over the whole remaining
   hidden-information public tree with private draws, re-solved safely (gadget) once at our first decision in the
   window; exact best response gives a certified per-move gap.
3. **Before F**: continual re-solving (DeepStack / ReBeL family): full-width depth-limited public trees, leaves valued
   by a public-belief-state value network trained bottom-up from exact window solves, then by self-play. Moves are
   sampled from the mixed strategy with a private RNG; never argmax outside perfect-information states.
4. **Blueprint pi** (MMD-PPO self-play): first learned champion, CFR warm start, source of realistic ranges, and the
   opponent exploiters pretrain against.

More compute buys strength through: CFR iterations, search depth, value-net data/size, and moving F earlier
(F = 0 would be solving the game).

## Measurement

- Score: win 1, draw 0.5, loss 0, duplicate deal pairs (seats swapped), 95% CI over pairs ([bbar/stats.py](../bbar/stats.py)).
  Every claim logs command, commit, seeds and CI to `runs/results.jsonl`.
- Exploitability e = best responder's score - 0.5.
  - Exact: per-move solver gap; exact best response in windows rooted at ply >= 10 (offline 8); mini variants.
  - Lower bounds (full game): per-ply attribution exploiter (copy the agent before ply t, exact best response after),
    white-box local best response, learned best response (PPO) pretrained on pi, all fixed opponents and past
    champions. Headline = minimum score over the suite, with CI. Full-game numbers are lower bounds only.
- One fixed evaluation compute budget is what exploiters play against ("evaluated == deployed").

## Milestones

- M0 Rust exact core: hand space, Markov key, perfect-information solver, window solver (CFR, exact best response,
  gadget), cross-checked against an independent explicit-infoset CFR and a Python prototype.
- M1 Exact-tail player + harness: first champion with a CI; exact late-game exploitability tool.
- M2 Blueprint pi (MMD-PPO) on a GPU pod.
- M3 Gate experiments: E1 = exact exploitability of cheap play (PIMC, pi, pi + exact tail) by root ply 12..8 with
  realistic ranges; E2 = can a value net trained on exact labels replace exact solving at ply 10-11. Decides how much
  to invest in the value-net machinery.
- M4 Value net + continual re-solving; M5 self-play rounds + exploiter suite; M6 scaling curves.

## Status and learnings (2026-10-13)

- M0, M1, M2 done; E1 done for the blueprint; E2 done at small scale; M4 code done (`bbar/resolve.py`,
  `bbar/value/cascade.py`), not yet measured with a real value net.
- Exact windows are much cheaper than estimated (ply 10: ~0.4 s, ply 9: 1-5 s, ply 8: ~2 min, 3 GB), so F = 10 is
  cheap at play time and F = 9 affordable; only plies 0-8 need the value net.
- The blueprint is exploitable late (exact best response from ply 9: 0.111) while a PPO exploiter finds 0.022:
  learned exploiters badly understate the worst case. Exact per-ply best responses (`bbar.delta`) are the headline
  lower bounds.
- Value-net error is the bottleneck: 0.070 / 0.063 / 0.057 held-out at 62k / 139k / 253k public states (slow
  slope); option features cut it another 5-7%. Even so, depth-1 solving with network leaves from ply 10 is 0.020
  exploitable against the blueprint's 0.100. Depth-2 trees query beliefs unlike the training ranges (leaf error
  0.11): `bbar/value/harvest.py` labels the beliefs solves actually query (DAgger).
- Next, in order: regenerate labels (single range per state, Bayes ranges, harvested queries, PBS v2); train;
  measure the resolve player (delta_T at T = 9, 8; matches vs blueprint, PIMC d=8, tail); run the cascade toward
  ply 0; iterate harvest/train. Back up pods continuously (`scripts/pod.sh autobackup`): a balance outage lost
  ~500k labels.

