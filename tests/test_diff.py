"""Differential tests: the two engines agree move by move on random games (bbar/difftest.py runs many more)."""
import copy

import pytest

from bbar import difftest, golden, ref


@pytest.fixture(scope="module")
def engines():
    pytest.importorskip("bbar._core")
    return [golden.engine("ref"), golden.engine("core")]


@pytest.mark.parametrize("biased", [False, True])
def test_engines_agree_on_random_games(engines, biased):
    summary = difftest.run(300, seed=1_000_000, biased=biased, workers=1)
    assert summary.games == 300
    assert not summary.discrepancies, str(summary.discrepancies[0])


def first_discrepancy(engines):
    for seed in range(100):
        _, discrepancy = difftest.play_game(engines, seed)
        if discrepancy:
            return discrepancy
    return None


def test_a_rule_bug_is_caught_and_replays(engines, monkeypatch):
    monkeypatch.setattr(ref.Game, "_seal", lambda self, actor: None)
    d = first_discrepancy(engines)
    assert d is not None and d.moves[-1] == ("seal", ())
    assert any(p.startswith("queue differs") for p in d.problems)
    assert difftest.play_game(engines, d.seed)[1].problems == d.problems
    assert f"--seed {d.seed}" in str(d)


def test_a_copy_sharing_state_is_caught(engines, monkeypatch):
    monkeypatch.setattr(ref.Game, "copy", lambda self: copy.copy(self))
    d = first_discrepancy(engines)
    assert d is not None and len(d.moves) == 1
    assert "ref.Game: playing the original changed its copy" in d.problems
