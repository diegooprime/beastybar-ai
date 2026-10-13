"""Match statistics. Every strength claim in this project is a MatchResult: who, how many deals, seed, CI.

Matches are played as duplicate pairs: each deal is played twice with the seats swapped (same cards per seat, same
starting seat), so card luck cancels within a pair. Game score for A: win 1, draw 0.5, loss 0. Confidence intervals
are over deal pairs (pairs are independent; the two games of a pair are not).

The score interval is a Wilson score interval on the pair-mean scale (`score_interval`): it stays calibrated near 0%
and 100%, where lopsided matches (worst cases, exploiters) live, and never has zero width. A plain normal interval
there shrinks to ±0 and covers the true score far less than 95% of the time (tests/test_review_stats.py).
"""
import math
from dataclasses import dataclass

import numpy as np

Z95 = 1.959963984540054


@dataclass
class MatchResult:
    a: str
    b: str
    deals: int
    seed: int
    score: float          # A's mean game score (win 1, draw 0.5, loss 0)
    score_lo: float       # 95% confidence interval on score (score_interval)
    score_hi: float
    wins: int
    draws: int
    losses: int
    margin: float         # A's mean point margin per game
    margin_ci95: float    # half-width of the 95% normal interval on margin

    def __str__(self):
        return (f"{self.a} vs {self.b}: {100 * self.score:.1f}% [{100 * self.score_lo:.1f}, {100 * self.score_hi:.1f}] "
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
    lo, hi = score_interval(pair_score)
    return MatchResult(
        a=a, b=b, deals=n, seed=seed,
        score=float(game_score.mean()), score_lo=lo, score_hi=hi,
        wins=int((game_score == 1).sum()), draws=int((game_score == 0.5).sum()),
        losses=int((game_score == 0).sum()),
        margin=float(margin.mean()), margin_ci95=_normal_ci(pair_margin),
    )


def score_interval(x, z=Z95):
    """Confidence interval (lo, hi) for the mean of scores x in [0, 1] (one per deal pair).

    Wilson score interval with n replaced by the effective number of pairs n / phi, where phi = var(x) / (m (1 - m))
    is the variance relative to a win/loss coin with the same mean m (phi = 1 for pure win/loss pairs, where this is
    exactly Wilson's interval; draws and half-won pairs make phi < 1 and the interval narrower). phi is shrunk
    towards 1 by one pseudo-pair, so it is never 0: no interval has zero width, even when every pair scored the same.
    Simulated coverage for 10 to 1000 pairs and true scores from 50% to 99.5%: 0.94-0.98 in most settings, 0.90 at
    worst (99.5% with 10 or 50 pairs); the normal interval it replaces covered as little as 0.10 there.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    m = float(x.mean())
    pq = m * (1 - m)
    phi = float(x.var()) / pq if pq > 0 else 1.0
    phi = (n * phi + 1) / (n + 1)
    n_eff = n / phi
    z2 = z * z
    center = (m + z2 / (2 * n_eff)) / (1 + z2 / n_eff)
    half = z * math.sqrt(pq / n_eff + z2 / (4 * n_eff * n_eff)) / (1 + z2 / n_eff)
    return max(0.0, center - half), min(1.0, center + half)


def _normal_ci(x):
    return Z95 * float(np.std(x, ddof=1)) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
