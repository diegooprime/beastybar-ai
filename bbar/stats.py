"""Match statistics. Every strength claim in this project is a MatchResult: who, how many deals, seed, CI.

Matches are played as duplicate pairs: each deal is played twice with the seats swapped (same cards per seat, same
starting seat), so card luck cancels within a pair. Game score for A: win 1, draw 0.5, loss 0. The confidence
interval is over deal pairs (pairs are independent; the two games of a pair are not).
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class MatchResult:
    a: str
    b: str
    deals: int
    seed: int
    score: float          # A's mean game score (win 1, draw 0.5, loss 0)
    ci95: float           # half-width of the 95% CI on score
    wins: int
    draws: int
    losses: int
    margin: float         # A's mean point margin per game
    margin_ci95: float

    def __str__(self):
        return (f"{self.a} vs {self.b}: {100 * self.score:.1f}% ± {100 * self.ci95:.1f} "
                f"(W{self.wins} D{self.draws} L{self.losses}, margin {self.margin:+.2f} ± {self.margin_ci95:.2f}, "
                f"{self.deals} deal pairs, seed {self.seed})")


def summarize(a, b, seed, a_points, b_points, deal):
    """a_points/b_points: points of A and B in each game; deal: the deal index of each game (two games per deal)."""
    a_points, b_points, deal = map(np.asarray, (a_points, b_points, deal))
    game_score = np.where(a_points > b_points, 1.0, np.where(a_points == b_points, 0.5, 0.0))
    margin = (a_points - b_points).astype(float)
    n = int(deal.max()) + 1 if len(deal) else 0
    counts = np.bincount(deal, minlength=n)
    pair_score = np.bincount(deal, weights=game_score, minlength=n) / np.maximum(counts, 1)
    pair_margin = np.bincount(deal, weights=margin, minlength=n) / np.maximum(counts, 1)
    return MatchResult(
        a=a, b=b, deals=n, seed=seed,
        score=float(game_score.mean()), ci95=_ci(pair_score),
        wins=int((game_score == 1).sum()), draws=int((game_score == 0.5).sum()),
        losses=int((game_score == 0).sum()),
        margin=float(margin.mean()), margin_ci95=_ci(pair_margin),
    )


def _ci(x):
    return 1.96 * float(np.std(x, ddof=1)) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
