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
