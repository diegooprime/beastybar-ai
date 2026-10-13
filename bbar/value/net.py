"""The PBS value network: public features + both beliefs -> per-hand values of both players.

Input: the PBS encoding of bbar/value/pbs.py (mover-relative public features, the mover's and the opp's belief over
the 495 hand slots). Output [B, 2, 495]: per hand, the win probability of its owner (mover's hands, then the opp's),
sigmoid. With zero_sum (default), both outputs shift by the same amount so that the beliefs' expected values sum to
1, as exact values do (DeepStack's zero-sum layer): b_m . v_m + b_o . v_o = 1 (the exact labels are best-response
values, which sum to 1 + NashConv of the solve, 2e-5 at most). Only feasible hands mean anything.
"""
import numpy as np
import torch
from torch import nn

from bbar.rl.net import Trunk
from bbar.value.pbs import IN_DIM, PBS_VERSION, SLOT_ONEHOT, SLOTS


class ValueNet(nn.Module):
    def __init__(self, width=512, blocks=3, expansion=1, zero_sum=True):
        super().__init__()
        self.hparams = dict(width=width, blocks=blocks, expansion=expansion, zero_sum=zero_sum)
        self.zero_sum = zero_sum
        self.trunk = Trunk(IN_DIM, width, blocks, expansion)
        self.head = nn.Linear(width, 2 * SLOTS)

    def forward(self, pub, ranges):
        """pub float [B, PUB_DIM], ranges float [B, 2, 495] (mover, opp; normalised) -> values [B, 2, 495]."""
        x = torch.cat([pub, ranges.flatten(1)], 1)
        v = torch.sigmoid(self.head(self.trunk(x)).float()).view(-1, 2, SLOTS)
        if self.zero_sum:
            s = (ranges.float() * v).sum((1, 2))
            v = v - ((s - 1) / 2)[:, None, None]
        return v


class HandNet(nn.Module):
    """Same input and output as ValueNet, with a head shared across hands: a context per player (from the public
    features, both beliefs and their per-animal marginals) is added to an embedding of each hand's animals, and a small
    MLP maps the sum to that hand's value. Hands with common animals share parameters, which ValueNet's 990-way linear
    head cannot."""

    def __init__(self, width=512, blocks=3, expansion=1, zero_sum=True, hand_width=256):
        super().__init__()
        self.hparams = dict(arch="hand", width=width, blocks=blocks, expansion=expansion, zero_sum=zero_sum,
                            hand_width=hand_width)
        self.zero_sum = zero_sum
        self.register_buffer("animals", torch.as_tensor(SLOT_ONEHOT))          # [495, 12]
        self.trunk = Trunk(IN_DIM + 24, width, blocks, expansion)
        self.ctx = nn.Linear(width, 2 * hand_width)
        self.embed = nn.Linear(12, 2 * hand_width)
        self.out = nn.Sequential(nn.GELU(), nn.Linear(hand_width, hand_width), nn.GELU(), nn.Linear(hand_width, 1))

    def forward(self, pub, ranges):
        b, hw = len(pub), self.hparams["hand_width"]
        marginals = ranges @ self.animals.to(ranges.dtype)                      # [B, 2, 12]
        x = torch.cat([pub, ranges.flatten(1), marginals.flatten(1)], 1)
        c = self.ctx(self.trunk(x)).view(b, 2, 1, hw)
        e = self.embed(self.animals).view(SLOTS, 2, hw).transpose(0, 1)        # [2, 495, hw]
        v = torch.sigmoid(self.out(c + e).squeeze(-1).float())                 # [B, 2, 495]
        if self.zero_sum:
            s = (ranges.float() * v).sum((1, 2))
            v = v - ((s - 1) / 2)[:, None, None]
        return v


ARCHS = {"mlp": ValueNet, "hand": HandNet}


def make_value_net(arch="mlp", **hparams):
    return ARCHS[arch](**hparams)


def load_value_net(path, device="cpu"):
    """(value network in eval mode on `device`, checkpoint dict)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("pbs_version") != PBS_VERSION:
        raise ValueError(f"{path}: PBS encoding v{ck.get('pbs_version')}, this code has v{PBS_VERSION}")
    net = make_value_net(**ck["hparams"])
    net.load_state_dict(ck["model"])
    return net.to(device).eval(), ck


class NetValues:
    """Leaf values from a value network for a depth-limited window's value leaves (bbar/value/dl.py)."""

    def __init__(self, net, device="cpu", batch=4096):
        self.net, self.device, self.batch = net, torch.device(device), batch

    @torch.no_grad()
    def __call__(self, pub, movers, beliefs):
        """pub float32 [n, PUB_DIM], movers int [n], beliefs [n, 2, 495] (absolute players) -> values [n, 2, 495]
        (absolute players, clamped to [0, 1])."""
        n = len(movers)
        movers = np.asarray(movers)
        flip = movers == 1
        rel = np.where(flip[:, None, None], beliefs[:, ::-1, :], beliefs)
        out = np.empty((n, 2, SLOTS))
        for i in range(0, n, self.batch):
            p = torch.as_tensor(pub[i:i + self.batch], dtype=torch.float32, device=self.device)
            r = torch.as_tensor(np.ascontiguousarray(rel[i:i + self.batch]), dtype=torch.float32,
                                device=self.device)
            out[i:i + self.batch] = self.net(p, r).clamp(0, 1).double().cpu().numpy()
        return np.where(flip[:, None, None], out[:, ::-1, :], out)
