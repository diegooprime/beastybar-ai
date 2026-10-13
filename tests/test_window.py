"""The exact window solver (bbar._core.Window, core/src/window.rs): cross-checks and the Python API.

Fast versions of the cross-checks in bbar/window_check.py and bbar/xcfr.py:
  - research/pimc_gap.py (independent numpy prototype): exploitability of uniform play and of its PIMC strategy agree
    to 1e-9; our game value lies within the bounds of the prototype's CFR+.
  - bbar/xcfr.py (explicit-infoset tabular CFR on the reference engine, no Rust): exploitability of uniform play and
    of a fixed hand-dependent strategy agree to 1e-9; the game values agree within the solvers' bounds.
The Rust tests (core/tests/window.rs) check evaluation against brute force, determinism and the gadget in Rust.
"""
import numpy as np
import pytest

core = pytest.importorskip("bbar._core")
from bbar import window_check, xcfr  # noqa: E402
from bbar.actions import ANIMALS, decode  # noqa: E402
from bbar.golden import load  # noqa: E402

Window = core.Window


@pytest.fixture(scope="module")
def games():
    return load()


@pytest.fixture(scope="module")
def solved(games):
    """A ply-11 golden window with non-uniform ranges, solved to 1e-6."""
    G = window_check.public_game(games[3], 11)
    w0 = Window(G)
    rng = np.random.default_rng(0)
    ranges = []
    for p in range(2):
        r = np.zeros(495)
        hands = w0.hands(p)
        r[hands] = rng.random(len(hands)) * (rng.random(len(hands)) > 0.3)
        ranges.append(r)
    w = Window(G, ranges[0], ranges[1])
    res = w.solve(target=1e-6)
    return G, w, res


# ---- cross-checks ------------------------------------------------------------------------------------------------

def test_prototype_uniform_and_pimc_exploitability():
    rows = window_check.check_proto(12, [4, 5], pimc=True, quiet=True)
    assert [r["internal"] for r in rows] == [223, 259]
    assert all(abs(r["uniform_diff"]) < 1e-9 and abs(r["pimc_diff"]) < 1e-9 for r in rows)
    assert abs(rows[0]["uniform"] - 0.3888888888888889) < 1e-12  # research/gap13.txt pos 4: 0.3889
    assert abs(rows[0]["pimc"] - 0.0066666666666667) < 1e-9  # research/gap13.txt pos 4: 0.0067


def test_prototype_ply11_uniform():
    rows = window_check.check_proto(11, [0], quiet=True)
    assert rows[0]["internal"] == 2513 and abs(rows[0]["uniform_diff"]) < 1e-9


def test_prototype_equilibrium_value():
    (row,) = window_check.check_proto(12, [4], cfr_iters=100, quiet=True)
    lo, hi = row["proto_bounds"]
    assert lo - 1e-9 <= row["value"] <= hi + 1e-9


@pytest.mark.parametrize("animals,start", [("skunk,kangaroo,zebra,crocodile", 0), ("chameleon,monkey,giraffe,lion", 1)])
def test_explicit_infoset_cfr(animals, start):
    out = xcfr.compare(animals.split(","), 2, iters=300, plus_iters=300, start=start, quiet=True)
    assert abs(out["uniform_xcfr"] - out["uniform_window"]) < 1e-9
    assert abs(out["uniform_value_xcfr"] - out["uniform_value_window"]) < 1e-9
    assert abs(out["fixed_xcfr"] - out["fixed_window"]) < 1e-9
    assert abs(out["fixed_value_xcfr"] - out["fixed_value_window"]) < 1e-9
    v = out["window_value"]
    for key in ("vanilla_bounds", "plus_bounds"):
        lo, hi = out[key]
        assert lo - 1e-9 <= v <= hi + 1e-9, (key, out[key], v)
    assert out["window_exploitability"] <= 1e-10
    assert abs(out["frontier_value"] - v) < 1e-9


# ---- API ---------------------------------------------------------------------------------------------------------

def test_sizes_hands_and_ranges(games):
    G = window_check.public_game(games[0], 12)
    w = Window(G)
    s = w.sizes()
    assert s["internal"] > 0 and s["leaves"] == s["terminal_leaves"] + s["frontier_leaves"]
    assert s["hands"] == (15, 15) and s["entries"] == w.num_entries
    assert len(w.hands(0)) == 15 and all(0 <= h < 495 for h in w.hands(0))
    r0, r1 = w.ranges()
    assert r0.shape == (495,) and abs(r0.sum() - 1) < 1e-12 and np.count_nonzero(r0) == 15
    bad = np.zeros(495)
    bad[[h for h in range(495) if h not in set(w.hands(0))][0]] = 1.0
    with pytest.raises(ValueError):
        Window(G, bad)
    with pytest.raises(ValueError):
        Window(G, np.ones(10))
    with pytest.raises(ValueError):
        w.exploitability()  # nothing solved yet
    with pytest.raises(ValueError):
        Window(G, leaves="nope")


def test_solve_and_exploitability(solved):
    G, w, res = solved
    assert res["exploitability"] <= 1e-6 and res["iterations"] > 0 and res["seconds"] >= 0
    e = w.exploitability()
    assert abs(e["exploitability"] - res["exploitability"]) < 1e-15
    assert abs(e["value"] - res["value"]) < 1e-15
    assert 1 - e["br"][1] - 1e-12 <= e["value"] <= e["br"][0] + 1e-12
    # The average profile passed back as an external table gives the same numbers.
    avg = w.profile_array()
    assert avg.shape == (w.num_entries,)
    e2 = w.exploitability(avg, avg)
    assert all(abs(e2[k] - e[k]) < 1e-14 for k in ("nashconv", "value"))
    # Uniform play is much more exploitable.
    assert w.exploitability("uniform", "uniform")["exploitability"] > 0.05


def test_cfvs_and_best_responses(solved):
    G, w, res = solved
    r0, r1 = w.ranges()
    e = w.exploitability()
    assert abs(w.br_values(0) @ r0 - e["br"][0]) < 1e-12
    assert abs(w.br_values(1) @ r1 - e["br"][1]) < 1e-12
    assert abs(w.cfvs(0) @ r0 - e["value"]) < 1e-12
    assert abs(w.cfvs(1) @ r1 - (1 - e["value"])) < 1e-12
    assert np.all(w.br_values(1) >= w.cfvs(1) - 1e-12)
    # A pure best response achieves the best-response value.
    br = w.best_response(0)
    assert abs(w.exploitability(br, None)["value"] - e["br"][0]) < 1e-12
    br1 = w.best_response(1)
    assert abs(1 - w.exploitability(None, br1)["value"] - e["br"][1]) < 1e-12


def test_policies_and_nodes(solved):
    G, w, res = solved
    pol = w.root_policy()
    assert pol.shape == (495, 36)
    mover = G.to_move
    hands = set(w.hands(mover))
    nodes = w.nodes()
    history, m, offset, node_hands, actions = nodes[0]
    assert history == [] and m == mover and offset == 0
    assert sorted(G.legal_actions()) == list(actions)
    for r in range(495):
        row = pol[r]
        if r in hands:
            hand = core.hand_unrank(4, r)
            assert abs(row.sum() - 1) < 1e-12
            assert all(decode(a)[0] in hand for a in np.nonzero(row)[0])
        else:
            assert not row.any()
    # A deeper node: the public game there, its policy and its profile entries agree.
    history, m, offset, node_hands, actions = nodes[len(nodes) // 2]
    g = w.node_game(history)
    assert g.to_move == m and sorted(g.legal_actions()) == list(actions)
    pol = w.policy(history)
    avg = w.profile_array()
    for i, hand in enumerate(node_hands):
        r = core.hand_rank(list(hand))
        assert np.allclose(pol[r, list(actions)], avg[offset + i * len(actions): offset + (i + 1) * len(actions)])
    with pytest.raises(ValueError):
        w.policy([255])


def test_external_profile_layout(games):
    """An external table built from nodes(): a deterministic 'play your weakest animal' strategy."""
    G = window_check.public_game(games[1], 12)
    w = Window(G)
    prof = np.zeros(w.num_entries)
    for history, mover, offset, hands, actions in w.nodes():
        for i, hand in enumerate(hands):
            legal = [j for j, a in enumerate(actions) if decode(a)[0] in hand]
            j = min(legal, key=lambda j: ANIMALS.index(decode(actions[j])[0]))
            prof[offset + i * len(actions) + j] = 1.0
    e = w.exploitability(prof, prof)
    assert 0 <= e["exploitability"] <= 0.5
    # Any positive scaling of a row is the same strategy; illegal entries are ignored.
    assert w.exploitability(3 * prof, 3 * prof) == e
    with pytest.raises(ValueError):
        w.exploitability(-prof, prof)
    with pytest.raises(ValueError):
        w.exploitability(prof[:-1], prof)


def test_resolve_gadget(solved):
    G, w, res = solved
    star = w.profile_array()
    for player in (0, 1):
        o = 1 - player
        cfvs = w.br_values(o, star)
        out = w.resolve(player, cfvs, target=1e-7)
        assert out["exploitability"] <= 1e-7 and out["max_violation"] <= 1e-7 * 2 * len(w.hands(o))
        resolved = w.profile_array()
        assert np.all(w.br_values(o, resolved) <= cfvs + 2e-7 * len(w.hands(o)))
        mixed = (resolved, star) if player == 0 else (star, resolved)
        r = w.ranges()[o]
        bound = 1e-6 + 1e-7 * r.max() * len(w.hands(o))
        assert w.exploitability(*mixed)["exploitability"] <= bound + 1e-12


def test_deterministic_and_threads(games):
    G = window_check.public_game(games[7], 11)
    outs = []
    for threads in (1, 1, 4):
        w = Window(G, threads=threads)
        w.solve(target=0, max_iters=40, algo="pcfr+")
        outs.append((w.profile_array().tobytes(), w.exploitability()))
    assert outs[0] == outs[1] == outs[2]


def test_leaf_rules_agree(games):
    G = window_check.public_game(games[2], 12)
    values = []
    for leaves in ("frontier", "public"):
        w = Window(G, leaves=leaves)
        r = w.solve(target=1e-8, algo="dcfr")
        values.append((r["value"], r["nashconv"]))
    (a, ea), (b, eb) = values
    assert abs(a - b) <= ea + eb + 1e-12


def test_leaf_roots(games):
    g = games[0]
    for ply in (15, 16, len(g["steps"])):
        w = Window(window_check.public_game(g, ply))
        assert w.sizes()["internal"] == 0 and w.sizes()["leaves"] == 1
        r = w.solve()
        assert r["iterations"] == 0 and r["exploitability"] == 0.0
        assert w.exploitability("uniform", "uniform")["nashconv"] == 0.0
        # Re-solving at a leaf root is exact without iterations: each opponent hand takes the better of follow and
        # terminate (it used to stay 50/50, reporting a gadget exploitability of 0.05 for shifts of 0.2).
        for player in (0, 1):
            o = 1 - player
            br = w.br_values(o, "uniform")
            hands = w.hands(o)
            for shift in (0.2, -0.2):
                cfvs = br.copy()
                cfvs[hands] = np.clip(br[hands] + shift, 0, 1)
                out = w.resolve(player, cfvs, target=1e-9)
                assert out["iterations"] == 0 and abs(out["exploitability"]) <= 1e-12, out
                assert out["max_violation"] == (br[hands] - cfvs[hands]).max()
        with pytest.raises(ValueError):
            w.root_policy()
