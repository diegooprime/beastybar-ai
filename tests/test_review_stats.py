"""The score interval of bbar/stats.py on lopsided matches: never zero width, calibrated near 0% and 100%.

(Review finding: the plain normal interval reported "100.0% ± 0.0" for pimc:w=16,d=4 vs random over 20 pairs, and
covered the true score 56% of the time at 20 pairs, 78% at 50, 91% at 200.)
"""
import numpy as np
import pytest

from bbar import stats


def _summarize(pair_scores):
    """A MatchResult from per-pair scores in {0, .25, .5, .75, 1} (two games per deal, margins irrelevant)."""
    game = {0.0: (0, 0), 0.25: (0, 0.5), 0.5: (0.5, 0.5), 0.75: (1, 0.5), 1.0: (1, 1)}
    a, b, deal = [], [], []
    for i, p in enumerate(pair_scores):
        for g in game[p]:
            a.append({1: 2, 0.5: 1, 0: 0}[g])
            b.append(1)
            deal.append(i)
    return stats.summarize("A", "B", 0, a, b, deal)


def test_ci_never_zero_width():
    # pimc:w=16,d=4 vs random, 20 deal pairs, seed 1 was exactly this: 100.0% ± 0.0 (true score ~98.4%).
    r = _summarize([1.0] * 20)
    assert r.score == 1.0
    assert r.score_lo < 0.9 and r.score_hi == 1.0
    for same in ([0.0] * 20, [0.5] * 20, [0.75] * 200):
        r = _summarize(same)
        assert r.score_lo < r.score < r.score_hi or r.score in (r.score_lo, r.score_hi)
        assert r.score_hi - r.score_lo > 0, str(r)
    assert str(_summarize([1.0] * 20)).startswith("A vs B: 100.0% [83.9, 100.0]")


# Pair-score distributions: values (0, .25, .5, .75, 1) and their weights.
LOPSIDED = [0, 0, 5, 3, 192]       # pimc:w=16,d=4 vs random (200 deal pairs, seed 1)
DISTRIBUTIONS = {
    "pimc-vs-random": LOPSIDED,
    "mirror": LOPSIDED[::-1],
    "win-loss 97%": [3, 0, 0, 0, 97],
    "with draws 90%": [1, 2, 10, 20, 167],
    "even with draws": [10, 25, 30, 25, 10],
    "mostly draws": [1, 5, 88, 5, 1],
}


@pytest.mark.parametrize("name", DISTRIBUTIONS)
@pytest.mark.parametrize("n", [20, 50, 200])
def test_ci_covers_the_true_score(name, n):
    values = np.array([0, 0.25, 0.5, 0.75, 1])
    weights = np.array(DISTRIBUTIONS[name], dtype=float)
    p = weights / weights.sum()
    truth = float(values @ p)
    rng = np.random.default_rng(n)
    trials = 3000
    covered = 0
    for x in rng.choice(values, size=(trials, n), p=p):
        lo, hi = stats.score_interval(x)
        covered += lo <= truth <= hi
    # Nominal 0.95; the normal interval measured 0.56 / 0.77 / 0.91 on pimc-vs-random at 20 / 50 / 200 pairs.
    assert covered / trials >= 0.925, covered / trials


def test_interval_is_wilson_for_win_loss_pairs_and_symmetric():
    # Pure win/loss pairs: exactly the Wilson score interval.
    x = np.array([1.0] * 37 + [0.0] * 13)
    n, m, z = 50, 0.74, stats.Z95
    center = (m + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(m * (1 - m) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    assert np.allclose(stats.score_interval(x), (center - half, center + half))
    # Swapping A and B mirrors the interval.
    y = np.array([0.75, 1, 0.5, 1, 1, 0.25, 1, 1])
    lo, hi = stats.score_interval(y)
    assert np.allclose(stats.score_interval(1 - y), (1 - hi, 1 - lo))
