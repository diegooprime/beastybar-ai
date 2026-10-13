"""Option features (core/src/options.rs) and PBS version 2 (bbar/value/pbs.py), with networks of both versions.

- The Rust features equal an independent reimplementation on the pure-Python reference engine (bbar/ref.py): every
  legal parameter choice of every unplayed animal of both players played on the public position, best and worst by
  the documented order, on random positions and golden positions. (core/src/options.rs's own tests check them
  against State::apply on real positions with hidden hands and decks.)
- They read only what the public key says: the same for a position's full Markov key and its public key.
- PBS v2 = v1 block unchanged + flattened option features; every architecture works with the inputs it was built for;
  checkpoints written by the code before PBS v2 (its networks, copied here) still load and, from v2 features, give
  that code's values: bit-identical for mlp, for hand up to float rounding on feasible hands (0.5 before the zero-sum
  shift on the others, where no belief has mass); and the pipeline's call sites (NetValues, depth-limited solves,
  cascade labels, the re-solving player's value model, E2, harvest, the trainer) run with networks of both versions.
"""
import json

import numpy as np
import pytest

core = pytest.importorskip("bbar._core")
torch = pytest.importorskip("torch")
from bbar import ref, window_check  # noqa: E402
from bbar.actions import ANIMALS, POINTS, encode  # noqa: E402
from bbar.golden import load  # noqa: E402
from bbar.value import dl as D  # noqa: E402
from bbar.value import gen, pbs  # noqa: E402
from bbar.rl.net import Trunk  # noqa: E402
from bbar.value.net import NetValues, load_value_net, make_value_net  # noqa: E402

F = pbs.OPTION_F
C = (F - 2) // 2


def ref_features(key):
    """Option features of one public key from the reference engine, written from the documentation of options.rs."""
    qlen, queue, u, _, m = pbs.decode_keys([key])
    q = [(ANIMALS[c % 12], int(c >= 12)) for c in queue[0][:qlen[0]]]
    names = [[ANIMALS[a] for a in range(12) if u[0, x] >> a & 1] for x in (0, 1)]
    out = np.zeros((2, 12, F), dtype=np.float32)
    mover = int(m[0])
    for rel, p in ((0, mover), (1, 1 - mover)):
        g = ref.Game(p, names, [[], []])
        g._queue = list(q)
        choices = {}
        for animal, params in g.legal_moves():
            h = g.copy()
            h.play(animal, params)
            card = (animal, p)
            bar, gone, after = h.bar(), h.thats_it(), h.queue()

            def pts(cards, who):
                return sum(POINTS[a] for a, o in cards if o == who)
            front = after[:2]
            pos = after.index(card) + 1 if card in after else 0
            fields = [pts(bar, p) / 8, pts(bar, 1 - p) / 8, pts(gone, p) / 8, pts(gone, 1 - p) / 8,
                      float(len(bar) > 0), len(after) / 4, pos / 4, float(card in bar), float(card in gone),
                      pts(front, p) / 8, pts(front, 1 - p) / 8]
            rank = (pts(bar, p) - pts(bar, 1 - p), pts(gone, 1 - p) - pts(gone, p), pts(front, p) - pts(front, 1 - p))
            choices.setdefault(animal, []).append((encode(animal, params), rank, fields))
        for a in range(12):
            if not u[0, p] >> a & 1:
                continue
            row = out[rel, a]
            row[0] = 1
            cs = choices.get(ANIMALS[a], [])
            row[1] = len(cs) / 8
            if cs:
                row[2:2 + C] = max(cs, key=lambda c: (c[1], -c[0]))[2]     # ties: the lower action index
                row[2 + C:] = min(cs, key=lambda c: (c[1], c[0]))[2]
    return out


def random_positions(n, seed, max_ply=20):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        g = core.Game.random(int(rng.integers(2 ** 63)), int(rng.integers(2)))
        for _ in range(int(rng.integers(0, max_ply + 1))):
            if g.is_over():
                break
            acts = g.legal_actions()
            g.play_action(acts[rng.integers(len(acts))])
        out.append(g)
    return out


def test_feature_names_agree():
    assert list(core.option_feature_names()) == list(pbs.OPTION_NAMES) and pbs.OPTION_F == 24
    assert pbs.PUB_DIMS == {1: 134, 2: 134 + 2 * 12 * 24} and pbs.PUB_DIM == pbs.PUB_DIMS[pbs.PBS_VERSION]


def test_option_features_match_the_reference_engine():
    games = random_positions(1500, 0) + [window_check.public_game(g, ply) for g in load()[:12] for ply in
                                         range(0, 20, 3)]
    keys = np.array([core.public_key(g) for g in games], dtype=np.uint64)
    got = pbs.option_features(keys)
    assert got.shape == (len(keys), 2, 12, F) and got.dtype == np.float32
    nonzero = 0
    for k, x in zip(keys, got):
        want = ref_features(int(k))
        assert np.array_equal(x, want), (int(k), np.argwhere(x != want)[:5])
        nonzero += int((x[:, :, 2:] != 0).any())
    assert nonzero > 0.9 * len(keys)
    # Every field takes several values somewhere (the comparison has teeth).
    flat = got.reshape(-1, F)[got.reshape(-1, F)[:, 1] > 0]
    assert all(len(np.unique(flat[:, j])) >= 2 for j in range(1, F))   # (held is 1 there)


def test_option_features_read_only_the_public_key():
    games = random_positions(300, 1)
    full = np.array([core.markov_key(g) for g in games], dtype=np.uint64)     # with hand sizes, hidden splits
    pub = np.array([core.public_key(g) for g in games], dtype=np.uint64)
    assert (full != pub).any()
    assert np.array_equal(pbs.option_features(full), pbs.option_features(pub))
    # Threads change nothing.
    assert np.array_equal(pbs.option_features(pub, threads=1), pbs.option_features(pub, threads=3))
    assert pbs.option_features(np.zeros(0, dtype=np.uint64)).shape == (0, 2, 12, F)


def test_pbs_v2_is_v1_plus_option_features():
    games = random_positions(200, 2, max_ply=16)
    keys = np.array([core.public_key(g) for g in games], dtype=np.uint64)
    start = np.array([g.start for g in games])
    v1 = pbs.public_features(keys, start, 1)
    v2 = pbs.public_features(keys, start)
    assert v1.shape == (len(keys), 134) and v2.shape == (len(keys), pbs.PUB_DIMS[2])
    assert np.array_equal(v2[:, :134], v1)
    assert np.array_equal(v2[:, 134:].reshape(-1, 2, 12, F), pbs.option_features(keys))
    # Mover-relative: the mover's row is the player to move's; held = unplayed in the v1 block.
    opt = v2[:, 134:].reshape(-1, 2, 12, F)
    assert np.array_equal(opt[:, 0, :, 0], v1[:, 56:68]) and np.array_equal(opt[:, 1, :, 0], v1[:, 68:80])
    with pytest.raises(ValueError):
        pbs.public_features(keys, start, 3)
    assert [pbs.version_of(d) for d in (134, pbs.PUB_DIMS[2])] == [1, 2]


# ---- networks of both versions ----------------------------------------------------------------------------------

ARCHS = [("mlp", 1, {}), ("mlp", 2, {}), ("hand", 1, {"hand_width": 8}), ("hand", 2, {"hand_width": 8}),
         ("opt", 2, {"hand_width": 8, "tok_width": 8})]


def real_batch(n=6, seed=0, plies=(10, 11, 12)):
    """(games, keys, starts, mover-relative ranges [n, 2, 495]) at golden positions of the value net's plies."""
    rng = np.random.default_rng(seed)
    gs = load()
    games = [window_check.public_game(gs[i], plies[i % len(plies)]) for i in range(n)]
    keys = np.array([core.public_key(g) for g in games], dtype=np.uint64)
    start = np.array([g.start for g in games])
    _, uu, _ = pbs.key_info(keys)
    r = np.stack([[pbs.sample_range(uu[i, p], rng)[0] for p in (0, 1)] for i in range(n)])
    return games, keys, start, torch.as_tensor(r, dtype=torch.float32)


def save_net(net, path, version):
    torch.save(dict(model=net.state_dict(), hparams=net.hparams, pbs_version=version), path)
    return path


@pytest.mark.parametrize("arch,version,kw", ARCHS)
def test_architectures_round_trip_with_their_inputs(tmp_path, arch, version, kw):
    torch.manual_seed(0)
    net = make_value_net(arch, width=16, blocks=1, pub_dim=pbs.PUB_DIMS[version], **kw)
    assert net.pub_dim == pbs.PUB_DIMS[version] and net.hparams["arch"] == arch
    _, keys, start, r = real_batch()
    pub = torch.as_tensor(pbs.public_features(keys, start))         # current version: every network reads its part
    v = net(pub, r)
    assert v.shape == (len(keys), 2, pbs.SLOTS)
    assert torch.allclose((r * v).sum((1, 2)), torch.ones(len(keys)), atol=1e-5)
    if version == 1:   # a version-1 network sees only the v1 block
        assert torch.equal(v, net(pub[:, :134].clone(), r))
    path = save_net(net, tmp_path / "n.pt", version)
    back, ck = load_value_net(path)
    assert ck["pbs_version"] == version and torch.allclose(back(pub, r), v.detach())
    with pytest.raises(ValueError):
        net(pub[:, :pbs.PUB_DIMS[version] - 1], r)


def test_hand_and_opt_values_depend_only_on_feasible_hands():
    """HandNet and OptNet evaluate only feasible hands (gathered); HandNet's values there are its dense formula's."""
    torch.manual_seed(0)
    _, keys, start, r = real_batch(9)
    pub = torch.as_tensor(pbs.public_features(keys, start))
    mask = torch.as_tensor(pbs.masks_of_keys(keys))
    net = make_value_net("hand", width=16, blocks=1, hand_width=8, zero_sum=False)
    b, hw = len(pub), 8
    x = torch.cat([pub, r.flatten(1), (r @ net.animals).flatten(1)], 1)
    c = net.ctx(net.trunk(x)).view(b, 2, 1, hw)
    e = net.embed(net.animals).view(pbs.SLOTS, 2, hw).transpose(0, 1)
    dense = torch.sigmoid(net.out(c + e).squeeze(-1))
    v = net(pub, r)
    assert torch.allclose(v[mask], dense[mask], atol=1e-6) and torch.all(v[~mask] == 0.5)
    opt = make_value_net("opt", width=16, blocks=1, hand_width=8, tok_width=8, zero_sum=False)
    v = opt(pub, r)
    assert torch.all(v[~mask] == 0.5) and torch.all((v[mask] > 0) & (v[mask] < 1))
    # Each row's values do not depend on the other rows (padding to the batch's largest count).
    for i in range(b):
        assert torch.allclose(opt(pub[i:i + 1], r[i:i + 1]), v[i:i + 1], atol=1e-6)


def test_version_errors(tmp_path):
    with pytest.raises(ValueError):
        make_value_net("opt", width=16, blocks=1, pub_dim=134)
    net = make_value_net("mlp", width=16, blocks=1)
    with pytest.raises(ValueError):
        load_value_net(save_net(net, tmp_path / "a.pt", 3))
    with pytest.raises(ValueError):
        load_value_net(save_net(net, tmp_path / "b.pt", 1))    # v2 hparams (pub_dim 710) saved as version 1


# The value networks as the code before PBS v2 had them (bbar/value/net.py at commit bbf087e, verbatim but for the
# names and IN_DIM): checkpoints they wrote must still load and give their values.
OLD_IN_DIM = 134 + 2 * pbs.SLOTS


class OldValueNet(torch.nn.Module):
    def __init__(self, width=512, blocks=3, expansion=1, zero_sum=True):
        super().__init__()
        self.hparams = dict(width=width, blocks=blocks, expansion=expansion, zero_sum=zero_sum)
        self.zero_sum = zero_sum
        self.trunk = Trunk(OLD_IN_DIM, width, blocks, expansion)
        self.head = torch.nn.Linear(width, 2 * pbs.SLOTS)

    def forward(self, pub, ranges):
        x = torch.cat([pub, ranges.flatten(1)], 1)
        v = torch.sigmoid(self.head(self.trunk(x)).float()).view(-1, 2, pbs.SLOTS)
        if self.zero_sum:
            s = (ranges.float() * v).sum((1, 2))
            v = v - ((s - 1) / 2)[:, None, None]
        return v


class OldHandNet(torch.nn.Module):
    def __init__(self, width=512, blocks=3, expansion=1, zero_sum=True, hand_width=256):
        super().__init__()
        nn = torch.nn
        self.hparams = dict(arch="hand", width=width, blocks=blocks, expansion=expansion, zero_sum=zero_sum,
                            hand_width=hand_width)
        self.zero_sum = zero_sum
        self.register_buffer("animals", torch.as_tensor(pbs.SLOT_ONEHOT))          # [495, 12]
        self.trunk = Trunk(OLD_IN_DIM + 24, width, blocks, expansion)
        self.ctx = nn.Linear(width, 2 * hand_width)
        self.embed = nn.Linear(12, 2 * hand_width)
        self.out = nn.Sequential(nn.GELU(), nn.Linear(hand_width, hand_width), nn.GELU(), nn.Linear(hand_width, 1))

    def forward(self, pub, ranges):
        b, hw = len(pub), self.hparams["hand_width"]
        marginals = ranges @ self.animals.to(ranges.dtype)                      # [B, 2, 12]
        x = torch.cat([pub, ranges.flatten(1), marginals.flatten(1)], 1)
        c = self.ctx(self.trunk(x)).view(b, 2, 1, hw)
        e = self.embed(self.animals).view(pbs.SLOTS, 2, hw).transpose(0, 1)    # [2, 495, hw]
        v = torch.sigmoid(self.out(c + e).squeeze(-1).float())                 # [B, 2, 495]
        if self.zero_sum:
            s = (ranges.float() * v).sum((1, 2))
            v = v - ((s - 1) / 2)[:, None, None]
        return v


def old_v1_checkpoint(path, arch="mlp"):
    """A checkpoint written by the code before PBS v2: its network, its hparams (no pub_dim), pbs_version 1."""
    torch.manual_seed(1)
    net = OldValueNet(width=16, blocks=1) if arch == "mlp" else OldHandNet(width=16, blocks=1, hand_width=8)
    with torch.no_grad():
        for w in net.parameters():     # away from initialisation: values spread over (0, 1)
            w += 0.3 * torch.randn_like(w)
    torch.save(dict(model=net.state_dict(), hparams=net.hparams, pbs_version=1), path)
    return path, net.eval()


@pytest.mark.parametrize("chunk", [None, 4])
@pytest.mark.parametrize("arch", ["mlp", "hand"])
def test_version_1_checkpoints_still_load(tmp_path, arch, chunk):
    """Through NetValues (every call site's path) with version-2 features: an mlp's values are bit-identical to the old
    code's on version-1 features; a hand network's agree on feasible hands up to float rounding and are 0.5 before the
    zero-sum shift on the others (HandNet evaluates feasible hands only), where no belief has mass."""
    path, old = old_v1_checkpoint(tmp_path / "v1.pt", arch)
    back, ck = load_value_net(path)
    assert back.pub_dim == 134 and ck["pbs_version"] == 1
    games, keys, start, r = real_batch(9)
    v1, v2 = pbs.public_features(keys, start, 1), pbs.public_features(keys, start)
    movers = np.array([g.to_move for g in games])
    bel = pbs.mover_relative(r.numpy().astype(np.float64), movers)    # mover-relative -> absolute (an involution)
    got = NetValues(back, chunk=chunk)(v2, movers, bel)
    want = NetValues(old, chunk=chunk)(v1, movers, bel)
    assert want.std() > 0.01                                             # (not a constant network)
    if arch == "mlp":
        assert np.array_equal(got, want)
        return
    feas = pbs.mover_relative(pbs.masks_of_keys(keys), movers)
    assert np.allclose(got[feas], want[feas], rtol=0, atol=1e-6)
    with torch.no_grad():
        v_old = old(torch.as_tensor(v1), r)
        old.zero_sum = False
        shift = old(torch.as_tensor(v1), r) - v_old                      # the zero-sum shift, one per row
        mask = torch.as_tensor(pbs.masks_of_keys(keys))
        assert torch.allclose(back(torch.as_tensor(v2), r)[~mask], (0.5 - shift)[~mask], atol=1e-6)
    assert np.abs(got[~feas] - want[~feas]).max() > 1e-3                 # (the old dense values there differ)


@pytest.fixture(scope="module")
def two_nets(tmp_path_factory):
    d = tmp_path_factory.mktemp("v12")
    v1, _ = old_v1_checkpoint(d / "v1.pt")
    torch.manual_seed(2)
    v2 = save_net(make_value_net("opt", width=16, blocks=1, hand_width=8, tok_width=8), d / "v2.pt", 2)
    return [str(v1), str(v2)]


def test_call_sites_take_both_versions(two_nets, tmp_path):
    """Depth-limited solves (solve_many, NetLeaf), cascade labels, the re-solving player's value model: same code
    path for a version-1 and a version-2 network."""
    from bbar.resolve import ValueModel
    from bbar.value import cascade
    G = window_check.public_game(load()[4], 11)
    w0 = core.Window(G)
    rng = np.random.default_rng(0)
    r = [pbs.sample_range(int(pbs.decode_keys([core.public_key(G)])[2][0, p]), rng)[0] for p in (0, 1)]
    for path in two_nets:
        net, _ = load_value_net(path)
        nv = NetValues(net)
        dl = core.Window(G, *r, leaf_ply=12)
        res = dl.solve_external(D.NetLeaf(dl, nv), iters=5)
        assert res["iterations"] == 5
        st = D.solve_many([D.DLJob(core.Window(G, *r, leaf_ply=12))], nv, 3)
        assert st["leaves"] == dl.num_value_leaves > 0
        labels, _ = cascade.label_many([(G, 11, [(r[0], "uniform"), (r[1], "uniform")])], nv, 1, 12, 5)
        assert np.isfinite(labels[0]["values"]).all() and abs(labels[0]["ranges"].sum() - 2) < 1e-6
        vm = ValueModel(path)
        pub, movers = D.leaf_features(dl)
        bel, _ = dl.leaf_query("uniform")
        assert pub.shape[1] == pbs.PUB_DIM and vm(pub, movers, bel).shape == bel.shape
    assert w0.num_entries > 0


def test_e2_and_harvest_take_both_versions(two_nets, tmp_path):
    from bbar.value import e2, harvest
    for i, path in enumerate(two_nets):
        out = tmp_path / f"e2-{i}.jsonl"
        e2.main(["--net", path, "--positions", "1", "--ply", "12", "--leaf-plies", "13", "--iters", "5",
                 "--oracle-positions", "0", "--mix", "random:1", "--blueprint", "", "--workers", "1", "--threads", "1",
                 "--out", str(out), "--log", ""])
        rows = [json.loads(x) for x in out.read_text().splitlines()]
        assert rows[-1]["kind"] == "summary" and rows[1]["L13"]["net"]["exploitability"] >= 0
        s = harvest.main(["--net", path, "--out", str(tmp_path / f"q{i}"), "--labels", "2", "--workers", "1",
                          "--root-plies", "12", "--iters", "5", "--leaves-per-state", "2", "--bayes", "0",
                          "--mix", "random:1", "--blueprint", ""])
        assert s["labels"] >= 2


def test_trainer_versions_archs_and_budgets(tmp_path, monkeypatch):
    """--pbs 1 / 2 networks, --arch opt, --train-permille (nested subsets of the training states, same validation and
    test) and --minutes (time-limited, the run still ends with a complete best.pt)."""
    from bbar.value import train
    data = tmp_path / "data"
    gen.main(["--out", str(data), "--workers", "1", "--plies", "12,13", "--ply-weights", "1,1", "--blueprint", "",
              "--mix", "random:1", "--labels", "60", "--ranges-per-state", "1"])
    d, _ = train.load_labels([str(data)], 1)
    assert d["pub"].shape[1] == 134 and train.load_labels([str(data)])[0]["pub"].shape[1] == pbs.PUB_DIMS[2]
    keys = d["key"]
    subs = [train.train_subset(keys, p) for p in (250, 500, 1000)]
    assert np.all(subs[0] <= subs[1]) and np.all(subs[1] <= subs[2]) and subs[2].all()
    assert 0 < subs[0].sum() < subs[1].sum() < len(keys)
    monkeypatch.setattr(train, "ROOT", tmp_path)
    common = ["--data", str(data), "--batch", "8", "--width", "16", "--blocks", "1", "--warmup", "1",
              "--eval-every", "2", "--log-every", "2", "--device", "cpu", "--val-permille", "200",
              "--test-permille", "200", "--hand-width", "8", "--tok-width", "8"]
    for name, extra in (("a", ["--arch", "mlp", "--pbs", "1", "--steps", "4"]),
                        ("b", ["--arch", "opt", "--steps", "4", "--train-permille", "500"]),
                        ("c", ["--arch", "hand", "--steps", "1000000", "--minutes", "0.02"])):
        train.main(["--name", name, *common, *extra])
        lines = [json.loads(x) for x in (tmp_path / f"runs/value/{name}/metrics.jsonl").read_text().splitlines()]
        cfg, done = lines[0], lines[-1]
        net, ck = load_value_net(tmp_path / f"checkpoints/value/{name}/best.pt")
        assert ck["complete"] and ck["pbs_version"] == cfg["pbs_version"] == (1 if name == "a" else 2)
        assert net.pub_dim == pbs.PUB_DIMS[ck["pbs_version"]]
        if name == "b":
            assert cfg["n_train"] == int((subs[1] & ~np.isin(np.arange(len(keys)),
                                                             np.flatnonzero(train.split(keys, 200, 200)[0] |
                                                                            train.split(keys, 200, 200)[1]))).sum())
        if name == "c":
            assert 0 < done["step"] < 1000000 and done["train_seconds"] >= 0.02 * 60
    with pytest.raises(SystemExit):
        train.main(["--name", "d", *common, "--arch", "opt", "--pbs", "1", "--steps", "2"])
