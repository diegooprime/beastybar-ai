import pytest

from bbar import golden

GAMES = golden.load()


@pytest.mark.parametrize("name", ["ref", "core"])
def test_golden(name):
    try:
        Game = golden.engine(name)
    except ImportError:
        pytest.skip(f"engine {name} not built")
    assert golden.check(Game, GAMES) == []
