"""Batched agents over trained policies, for evaluation, arenas and exploiter training.

    agent = NetAgent("checkpoints/blueprint/latest.pt", device="cuda")
    obs, mask, mover, ply = env.observe()           # bbar._core.VecEnv
    actions = agent.act(obs, mask)                    # int64 [n], sampled from pi (a mixed strategy)

Play pi by sampling: it approximates an equilibrium, a mixed strategy, and its argmax can be exploitable.
"""
import math

import numpy as np
import torch

from bbar._core import OBS_DIM, OBS_VERSION
from bbar.rl.net import OPP_HAND_PUBLIC, OPP_UNPLAYED, PolicyNet, masked_logits, sample as sample_from

HAND_SIZE = 4   # cards in a hand while the deck lasts; after that, the hand is public


def check_layout(ck, name):
    """Raises ValueError unless checkpoint `ck` was trained on this engine's observation layout."""
    if ck.get("obs_version") != OBS_VERSION or ck.get("obs_dim") != OBS_DIM:
        raise ValueError(f"{name}: observation layout v{ck.get('obs_version')} ({ck.get('obs_dim')} floats), "
                         f"engine has v{OBS_VERSION} ({OBS_DIM})")


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=True)
    check_layout(ck, path)
    return ck


def policy_from_checkpoint(ck):
    net = PolicyNet(**ck["policy_hparams"])
    net.load_state_dict(ck["policy"])
    return net


def _tensor(x, device, dtype=None):
    t = torch.from_numpy(np.ascontiguousarray(x)) if isinstance(x, np.ndarray) else x
    return t.to(device=device, dtype=dtype) if dtype is not None else t.to(device)


class NetAgent:
    """The policy of a checkpoint (or a PolicyNet). Inputs are numpy arrays or tensors: obs float32 [B, OBS_DIM],
    mask uint8/bool [B, 36]."""

    def __init__(self, checkpoint, device="cpu", name=None):
        self.device = torch.device(device)
        if isinstance(checkpoint, PolicyNet):
            self.net, self.name, self.info = checkpoint, name or "policy", {}
        else:
            ck = load_checkpoint(checkpoint, self.device)
            self.net = policy_from_checkpoint(ck)
            self.name = name or str(checkpoint)
            self.info = {k: ck.get(k) for k in ("update", "decisions", "config")}
        self.net.to(self.device).eval()

    @torch.no_grad()
    def logp(self, obs, mask):
        """Masked log-probabilities, float32 tensor [B, 36] on the agent's device."""
        obs = _tensor(obs, self.device, torch.float32)
        mask = _tensor(mask, self.device).bool()
        logits, _ = self.net(obs)
        return torch.log_softmax(masked_logits(logits, mask), dim=-1)

    def probs(self, obs, mask):
        return self.logp(obs, mask).exp().cpu().numpy()

    @torch.no_grad()
    def act(self, obs, mask, sample=True, generator=None):
        """int64 numpy [B]: sampled actions (or the argmax with sample=False, for debugging only)."""
        logp = self.logp(obs, mask)
        a = sample_from(logp, generator) if sample else logp.argmax(-1)
        return a.cpu().numpy()

    @torch.no_grad()
    def opponent_hand(self, obs, exact_size=True):
        """Probability that each animal is in the opponent's hand, float32 numpy [B, 12], from the auxiliary head
        and what the observation already settles: 0 for animals the opponent has played, and the hand itself once
        it is public (opponent's deck empty). While the hand is hidden it holds exactly 4 of the unplayed animals;
        with exact_size, one logit shift per row makes the probabilities sum to 4 (order preserved), otherwise they
        are the head's raw marginals (trained only on unplayed animals)."""
        obs = _tensor(obs, self.device, torch.float32)
        _, aux = self.net(obs)
        unplayed = obs[:, OPP_UNPLAYED] > 0
        public = obs[:, OPP_HAND_PUBLIC].double()
        z = aux.double().masked_fill(~unplayed, -math.inf)
        if exact_size:
            z = z + _logit_shift(z, HAND_SIZE)
        # Hand public, or at most 4 unplayed animals (deck empty, so they are the hand; the encoder sets the public
        # block then): exact.
        known = (public > 0).any(-1, keepdim=True) | (unplayed.sum(-1, keepdim=True) <= HAND_SIZE)
        return torch.where(known, public, torch.sigmoid(z)).float().cpu().numpy()


def _logit_shift(z, k, iters=64):
    """[B, 1] shift t with sum_i sigmoid(z_i + t) = k, by bisection, for rows of z (float64, -inf = excluded) with
    more than k finite entries (other rows get some finite t)."""
    finite = torch.isfinite(z)
    has = finite.any(-1, keepdim=True)
    zmax = torch.where(has, z.masked_fill(~finite, -math.inf).amax(-1, keepdim=True), 0.0)
    zmin = torch.where(has, z.masked_fill(~finite, math.inf).amin(-1, keepdim=True), 0.0)
    lo, hi = -zmax - 50, -zmin + 50       # sum ~ 0 at lo, ~ (number of finite entries) > k at hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        over = torch.sigmoid(z + mid).sum(-1, keepdim=True) > k
        lo, hi = torch.where(over, lo, mid), torch.where(over, mid, hi)
    return (lo + hi) / 2
