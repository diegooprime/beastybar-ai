"""The PBS value network: public features + both beliefs -> per-hand values of both players.

Input: the PBS encoding of bbar/value/pbs.py (mover-relative public features, the mover's and the opp's belief over
the 495 hand slots). Output [B, 2, 495]: per hand, the win probability of its owner (mover's hands, then the opp's),
sigmoid. With zero_sum (default), both outputs shift by the same amount so that the beliefs' expected values sum to
1, as exact values do (DeepStack's zero-sum layer): b_m . v_m + b_o . v_o = 1 (the exact labels are best-response
values, which sum to 1 + NashConv of the solve, 2e-5 at most). Only feasible hands mean anything.

Every network reads the first `pub_dim` public features it is given (hparams; PBS version 1: 134, version 2: 710 with
the option features), so callers pass the current version's features to networks of either version. Architectures
(--arch of bbar.value.train): mlp (ValueNet), hand (HandNet), opt (OptNet, version 2 only).
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from bbar.rl.net import Trunk
from bbar.value.pbs import (NUM_ANIMALS, OPTION_DIM, OPTION_F, PBS_VERSION, PUB_DIM, PUB_DIM_V1, PUB_DIMS,
                            SLOT_ONEHOT, SLOTS)

UNPLAYED = slice(56, 80)     # mover's and opp's unplayed animals in the version-1 block (bbar/value/pbs.py)


def _pub(net, pub):
    if pub.shape[1] < net.pub_dim:
        raise ValueError(f"{type(net).__name__} reads {net.pub_dim} public features, got {pub.shape[1]}")
    return pub[:, :net.pub_dim]


def _zero_sum(v, ranges):
    s = (ranges.float() * v).sum((1, 2))
    return v - ((s - 1) / 2)[:, None, None]


def _feasible(pub, animals):
    """(idx long [B, 2, K], valid bool [B, 2, K]): each player's feasible hand slots (4-subsets of their unplayed
    animals, from the version-1 block of `pub`) first, in slot order, padded to the batch's largest count K."""
    b = len(pub)
    feas = (pub[:, UNPLAYED].reshape(b, 2, NUM_ANIMALS) @ animals.T) > 3.5      # [B, 2, 495]
    k = int(feas.sum(-1).max()) if b else 0
    idx = torch.sort(feas.float(), stable=True, dim=-1, descending=True).indices[..., :k]
    return idx, feas.gather(-1, idx)


def _scatter(vals, idx, valid):
    """Values float [B, 2, K] of the gathered hands -> [B, 2, 495], 0.5 at every other slot."""
    v = torch.full((len(vals), 2, SLOTS), 0.5, device=vals.device)
    return v.scatter(-1, idx, torch.where(valid, vals, 0.5))


class ValueNet(nn.Module):
    def __init__(self, width=512, blocks=3, expansion=1, zero_sum=True, pub_dim=PUB_DIM):
        super().__init__()
        self.hparams = dict(arch="mlp", width=width, blocks=blocks, expansion=expansion, zero_sum=zero_sum,
                            pub_dim=pub_dim)
        self.zero_sum, self.pub_dim = zero_sum, pub_dim
        self.trunk = Trunk(pub_dim + 2 * SLOTS, width, blocks, expansion)
        self.head = nn.Linear(width, 2 * SLOTS)

    def forward(self, pub, ranges):
        """pub float [B, >= pub_dim], ranges float [B, 2, 495] (mover, opp; normalised) -> values [B, 2, 495]."""
        x = torch.cat([_pub(self, pub), ranges.flatten(1)], 1)
        v = torch.sigmoid(self.head(self.trunk(x)).float()).view(-1, 2, SLOTS)
        return _zero_sum(v, ranges) if self.zero_sum else v


class HandNet(nn.Module):
    """Same input and output as ValueNet, with a head shared across hands: a context per player (from the public
    features, both beliefs and their per-animal marginals) is added to an embedding of each hand's animals, and a small
    MLP maps the sum to that hand's value. Hands with common animals share parameters, which ValueNet's 990-way linear
    head cannot. Only feasible hands are evaluated (as OptNet; the others get 0.5 before the zero-sum shift). Until PBS
    v2 HandNet evaluated all 495: its checkpoints of then give the same values on feasible hands up to float rounding
    (about 2e-7 on CPU, from matmuls of other shapes) and 0.5 on the others, where no belief or label has mass."""

    def __init__(self, width=512, blocks=3, expansion=1, zero_sum=True, hand_width=256, pub_dim=PUB_DIM):
        super().__init__()
        self.hparams = dict(arch="hand", width=width, blocks=blocks, expansion=expansion, zero_sum=zero_sum,
                            hand_width=hand_width, pub_dim=pub_dim)
        self.zero_sum, self.pub_dim = zero_sum, pub_dim
        self.register_buffer("animals", torch.as_tensor(SLOT_ONEHOT))          # [495, 12]
        self.trunk = Trunk(pub_dim + 2 * SLOTS + 24, width, blocks, expansion)
        self.ctx = nn.Linear(width, 2 * hand_width)
        self.embed = nn.Linear(12, 2 * hand_width)
        self.out = nn.Sequential(nn.GELU(), nn.Linear(hand_width, hand_width), nn.GELU(), nn.Linear(hand_width, 1))

    def forward(self, pub, ranges):
        b, hw = len(pub), self.hparams["hand_width"]
        marginals = ranges @ self.animals.to(ranges.dtype)                      # [B, 2, 12]
        x = torch.cat([_pub(self, pub), ranges.flatten(1), marginals.flatten(1)], 1)
        c = self.ctx(self.trunk(x)).view(b, 2, 1, hw)
        idx, valid = _feasible(x, self.animals.to(x.dtype))
        if idx.shape[-1]:
            e = self.embed(self.animals).view(SLOTS, 2, hw).transpose(0, 1)    # [2, 495, hw]
            e = e[torch.arange(2, device=idx.device)[None, :, None], idx]       # [B, 2, K, hw]
            v = _scatter(torch.sigmoid(self.out(c + e).squeeze(-1).float()), idx, valid)
        else:
            v = torch.full((b, 2, SLOTS), 0.5, device=x.device)
        return _zero_sum(v, ranges) if self.zero_sum else v


class OptNet(nn.Module):
    """Per-hand values built from what the hand's animals would do now (PBS version 2: the option features).

    Context per player: a Trunk over the public features (option features included), both beliefs and their
    per-animal marginals, as HandNet. Token per (player, animal): an MLP of that animal's option features (for that
    player: what playing it now would do), its identity, the player (mover / opp) and both players' marginals of
    that animal, plus a projection of the player's context. A hand's embedding is the sum over its 4 animals of one
    projection of their tokens plus the smooth maximum (log-sum-exp of values bounded to [-4, 4]) of another, plus
    the player's context; a small MLP maps it to the hand's value, sigmoid, then the zero-sum shift. Animals are
    shared across hands (as HandNet's embedding, but each animal's token knows what it does on this queue).

    Only feasible hands (4-subsets of the player's unplayed animals, read from the public features) are evaluated,
    padded to the most feasible hands of any row in the batch (35 at plies 11-12 against 495): the others get 0.5,
    which means nothing (no belief or label has mass there)."""

    def __init__(self, width=512, blocks=3, expansion=1, zero_sum=True, hand_width=256, tok_width=128,
                 pub_dim=PUB_DIM):
        super().__init__()
        if pub_dim < PUB_DIMS[2]:
            raise ValueError(f"arch opt reads the option features: needs PBS version 2 ({PUB_DIMS[2]} public "
                             f"features), got pub_dim {pub_dim}")
        self.hparams = dict(arch="opt", width=width, blocks=blocks, expansion=expansion, zero_sum=zero_sum,
                            hand_width=hand_width, tok_width=tok_width, pub_dim=pub_dim)
        self.zero_sum, self.pub_dim = zero_sum, pub_dim
        self.register_buffer("animals", torch.as_tensor(SLOT_ONEHOT))          # [495, 12]
        ident = torch.zeros(2, NUM_ANIMALS, NUM_ANIMALS + 2)                     # animal one-hot, player one-hot
        for p in range(2):
            ident[p, :, :NUM_ANIMALS] = torch.eye(NUM_ANIMALS)
            ident[p, :, NUM_ANIMALS + p] = 1
        self.register_buffer("ident", ident)
        self.trunk = Trunk(pub_dim + 2 * SLOTS + 2 * NUM_ANIMALS, width, blocks, expansion)
        self.tok_in = nn.Linear(OPTION_F + NUM_ANIMALS + 2 + 2, tok_width)
        self.tok_ctx = nn.Linear(width, 2 * tok_width)
        self.tok_mid = nn.Linear(tok_width, tok_width)
        self.tok_sum = nn.Linear(tok_width, hand_width)
        self.tok_max = nn.Linear(tok_width, hand_width)
        self.ctx = nn.Linear(width, 2 * hand_width)
        self.out = nn.Sequential(nn.GELU(), nn.Linear(hand_width, hand_width), nn.GELU(), nn.Linear(hand_width, 1))

    def forward(self, pub, ranges):
        b, hw, tw = len(pub), self.hparams["hand_width"], self.hparams["tok_width"]
        pub = _pub(self, pub)
        animals = self.animals.to(ranges.dtype)
        marginals = ranges @ animals                                             # [B, 2, 12]
        h = self.trunk(torch.cat([pub, ranges.flatten(1), marginals.flatten(1)], 1))
        opt = pub[:, PUB_DIM_V1:PUB_DIM_V1 + OPTION_DIM].reshape(b, 2, NUM_ANIMALS, OPTION_F)
        own, other = marginals[..., None], marginals.flip(1)[..., None]
        x = torch.cat([opt, self.ident.expand(b, -1, -1, -1).to(opt.dtype), own, other], -1)
        t = F.gelu(self.tok_in(x) + self.tok_ctx(h).view(b, 2, 1, tw))
        t = F.gelu(self.tok_mid(t))                                              # [B, 2, 12, tw]
        idx, valid = _feasible(pub, animals)
        if idx.shape[-1]:
            hand = self.animals[idx].to(t.dtype)                                 # [B, 2, K, 12]
            m = 4 * torch.tanh(self.tok_max(t) / 4)
            e = hand @ self.tok_sum(t) + torch.log(hand @ torch.exp(m)) + self.ctx(h).view(b, 2, 1, hw)
            v = _scatter(torch.sigmoid(self.out(e).squeeze(-1).float()), idx, valid)   # from [B, 2, K]
        else:
            v = torch.full((b, 2, SLOTS), 0.5, device=pub.device)
        return _zero_sum(v, ranges) if self.zero_sum else v


ARCHS = {"mlp": ValueNet, "hand": HandNet, "opt": OptNet}


def make_value_net(arch="mlp", **hparams):
    return ARCHS[arch](**hparams)


def load_value_net(path, device="cpu"):
    """(value network in eval mode on `device`, checkpoint dict). Networks of every PBS version load: one without
    pub_dim in its hparams (version 1, before pub_dim existed) reads its version's public features. An mlp saved by the
    code before PBS v2 gives bit-identical values; a hand network the same up to float rounding on feasible hands and
    0.5 (before the zero-sum shift) on infeasible ones (HandNet; tests/test_options.py)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    version = ck.get("pbs_version")
    if version not in PUB_DIMS:
        raise ValueError(f"{path}: PBS encoding v{version}, this code reads {sorted(PUB_DIMS)} (current "
                         f"v{PBS_VERSION})")
    hp = dict(ck["hparams"])
    hp.setdefault("pub_dim", PUB_DIMS[version])
    if hp["pub_dim"] != PUB_DIMS[version]:
        raise ValueError(f"{path}: pub_dim {hp['pub_dim']} is not PBS v{version}'s {PUB_DIMS[version]}")
    net = make_value_net(**hp)
    net.load_state_dict(ck["model"])
    return net.to(device).eval(), ck


class NetValues:
    """Leaf values from a value network for a depth-limited window's value leaves (bbar/value/dl.py).

    `chunk`: evaluate in forward passes of exactly `chunk` rows (the last one padded): every row's values are then the
    same whatever else is in the call (float matmuls can round differently for different batch shapes: one row alone
    takes another kernel), so a player batching many solves' leaves plays as it would solving them one at a time."""

    def __init__(self, net, device="cpu", batch=4096, chunk=None):
        self.net, self.device, self.batch, self.chunk = net, torch.device(device), batch, chunk

    @torch.no_grad()
    def __call__(self, pub, movers, beliefs):
        """pub float32 [n, PUB_DIM] (the current PBS version's features; the network reads its first pub_dim), movers
        int [n], beliefs [n, 2, 495] (absolute players) -> values [n, 2, 495] (absolute players, clamped to [0, 1])."""
        n = len(movers)
        if n == 0:
            return np.zeros((0, 2, SLOTS))
        dev = self.device
        flip = torch.as_tensor(np.asarray(movers) == 1).to(dev)[:, None, None]
        # float32 on the host (MPS has no float64), then mover-relative on the device.
        b = torch.as_tensor(np.asarray(beliefs), dtype=torch.float32).to(dev)
        p = torch.as_tensor(np.asarray(pub), dtype=torch.float32).to(dev)
        rel = torch.where(flip, b.flip(1), b)
        out = torch.empty((n, 2, SLOTS), device=dev)
        step = self.chunk or self.batch
        for i in range(0, n, step):
            pc, rc = p[i:i + step], rel[i:i + step]
            k = len(pc)
            if self.chunk and k < step:
                pc = torch.cat([pc, pc.new_zeros((step - k, pc.shape[1]))])
                rc = torch.cat([rc, rc.new_full((step - k, 2, SLOTS), 1 / SLOTS)])
            out[i:i + k] = self.net(pc, rc)[:k].clamp(0, 1)
        return torch.where(flip, out.flip(1), out).cpu().double().numpy()
