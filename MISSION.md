# Mission

*Written by the owner. Read it first; don't edit it.*

**Goal: the strongest Beasty Bar player possible: one that beats any opponent playing any strategy.**

Two players; rules and scoring in [RULES.md](RULES.md) (house points scoring, fixed).

Strongest means:
- **No weak matchup.** It must beat humans, programs, and opponents built specifically to exploit it. Its worst
  result against anything is what counts, not its average against a fixed set.
- **It keeps getting stronger.** More compute (training or thinking time) should buy more strength, at any point.
  Solving the game outright is the end state, if you can get there.

## How to decide

Every choice serves the goal. Before building, running or keeping anything, ask: does it make the player measurably
stronger, or make strength cheaper and more reliable to measure? If neither, don't do it.

- **Measurement is the ground truth.** A strength claim needs reproducible evidence: command, seeds, games,
  confidence interval. Trust nothing unmeasured, including this file.
- **Go where the expected gain is largest.** Run the experiment that settles the biggest open question fastest.
  Drop whatever doesn't move the number.
- **Work autonomously and in parallel:** subagents, several pods, competing ideas, all judged by the same yardstick.
- **Commit often.** [README.md](README.md) is the project's state: what exists, how to run it, the current best
  player and its measured strength. Keep it true and short.

## Constraints

- **Rules:** any engine must reproduce every game in [golden/](golden/) exactly before its results count.
- **Language:** Python; compiled code where it pays off; PyTorch for neural networks.
- **Compute:** RunPod for all computation (`runpodctl`, SSH key `~/.runpod/ssh/RunPod-Key-Go`). The laptop (M4,
  16 GB) is for editing and driving pods. Stop pods when idle.
- **Scale:** hobby project. No fixed budget, but spend deliberately; single runs take hours to days, not weeks.

## Starting kit

- [RULES.md](RULES.md): the rules, with edge-case rulings.
- [golden/](golden/): 2,000 complete games with every legal move and position; the correctness test for engines.
