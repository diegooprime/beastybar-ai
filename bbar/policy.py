"""Policy models sigma(a | a player's information), queried for EVERY hand a player could hold, in batches.

A model answers two questions (both for one player, both batched):
  - likelihoods(game, actions, player): for each of the player's decisions in a move history and each hand they could
    hold there, the probability of the action actually played (rows of bbar._core.belief_rows) - the input of the
    exact Bayes filter in bbar/belief.py;
  - window_table(tree, player): the player's part of a strategy table on a window (bbar._core.WindowTree, the layout
    of bbar._core.Window profiles): for every node where the player moves and every hand, its action probabilities,
    seeing the real game history up to the window root plus the path inside the window.

Models:
  NetModel   a policy network (checkpoint path or PolicyNet): exact probabilities, one batched forward over all rows
             (observations from the Rust encoders). float64 exp of the float32 log-probabilities, as bbar.match samples.
             With argmax=True, the deterministic policy that plays the network's most probable action.
  RustModel  a built-in agent spec (core/src/agents.rs): each row's position is played `samples` times from fresh
             views (the agent's own world sampling), deterministic given a seed. Likelihoods are smoothed
             frequencies (count + alpha) / (samples + alpha * legal actions); tables are the raw frequencies (one
             sampled pure strategy per row with samples=1: its expected exploitability bounds the agent's from above).
"""
import contextlib
import hashlib
import io
import os
import sys
from pathlib import Path

import numpy as np

from bbar._core import agent_likelihoods, belief_obs, parse_spec

NN = "nn:"
ARGMAX = ",argmax"


def sha256_short(data):
    return hashlib.sha256(data).hexdigest()[:16]


@contextlib.contextmanager
def full_fp32_matmuls():
    """CUDA float32 matmuls in full precision inside the block (see bbar.match.full_fp32_matmuls)."""
    torch = sys.modules.get("torch")
    if torch is None:
        yield
        return
    matmul = torch.backends.cuda.matmul
    old = matmul.allow_tf32
    matmul.allow_tf32 = False
    try:
        yield
    finally:
        matmul.allow_tf32 = old


class NetModel:
    """A policy network as a per-hand policy model. `source`: checkpoint path (hashed as loaded) or a PolicyNet."""

    def __init__(self, source, device="cpu", argmax=False, batch=8192, name=None):
        import torch
        from bbar.rl.agent import NetAgent, check_layout, policy_from_checkpoint
        self.device = torch.device(device)
        self.argmax = argmax
        self.batch = batch
        if isinstance(source, (str, os.PathLike)):
            path = str(source)
            data = Path(path).read_bytes()
            self.checkpoints = {path: sha256_short(data)}
            ck = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            check_layout(ck, path)
            net = policy_from_checkpoint(ck)
            name = name or NN + path
        else:
            net, self.checkpoints = source, {}
            name = name or NN + "<policy>"
        self.agent = NetAgent(net, self.device, name=name)
        self.name = name + (ARGMAX if argmax else "")

    def probs(self, obs, legal):
        """float64 [R, 36]: action probabilities for rows of observations (float32 [R, OBS_DIM]) and legal masks: the
        float64 exp of the network's float32 log-probabilities (as bbar.match samples them), renormalised."""
        n = len(obs)
        out = np.zeros((n, 36))
        with full_fp32_matmuls():
            for a in range(0, n, self.batch):
                logp = self.agent.logp(obs[a:a + self.batch], legal[a:a + self.batch])
                if self.argmax:
                    best = logp.argmax(-1).cpu().numpy()
                    out[np.arange(a, a + len(best)), best] = 1.0
                else:
                    p = logp.cpu().double().exp().numpy()
                    out[a:a + self.batch] = p / p.sum(axis=1, keepdims=True)   # exactly normalised in float64
        return out

    def sample(self, obs, legal, u):
        """int64 [R]: one action per row, exactly as bbar.match.NetPlayer plays this network with uniforms `u`
        (float64 [R]): inverse CDF over the float64 exp of the float32 log-probabilities (the argmax with
        argmax=True), all rows in one forward pass."""
        from bbar.match import sample_inverse_cdf
        with full_fp32_matmuls():
            logp = self.agent.logp(obs, legal)
        if self.argmax:
            return logp.argmax(-1).cpu().numpy()
        return sample_inverse_cdf(logp.cpu().double().exp().numpy(), np.asarray(u, dtype=np.float64))

    def likelihoods(self, game, actions, player):
        ply, _, _, obs, legal = belief_obs(game, list(actions), player)
        if not len(ply):
            return np.zeros(0)
        p = self.probs(obs, legal)
        acts = np.asarray(actions, dtype=np.int64)[ply]
        return p[np.arange(len(ply)), acts]

    def row_probs(self, tree, player, chunk=16384, threads=0):
        """float64 [rows, 36]: the policy of every (node, hand) row of `player` in `tree`."""
        n = tree.num_rows(player)
        out = np.zeros((n, 36))
        for a in range(0, n, chunk):
            obs, legal = tree.encode(player, a, min(n, a + chunk), threads)
            out[a:a + len(obs)] = self.probs(obs, legal)
        return out

    def window_table(self, tree, player, seed=0, threads=0):
        table = np.zeros(tree.num_entries)
        rows, acts, entries = tree.scatter(player)
        if len(rows):
            table[entries] = self.row_probs(tree, player, threads=threads)[rows, acts]
        return table

    def describe(self):
        return {"kind": "nn", "name": self.name, "argmax": self.argmax, "checkpoints": self.checkpoints}


class RustModel:
    """A built-in agent spec as a per-hand policy model, estimated from `samples` runs per row."""

    def __init__(self, spec, samples=8, alpha=0.5, table_samples=1, threads=0):
        self.name = parse_spec(spec)
        self.samples, self.alpha, self.table_samples, self.threads = samples, alpha, table_samples, threads
        self.checkpoints = {}

    def likelihoods(self, game, actions, player, seed=0):
        return agent_likelihoods(self.name, game, list(actions), player, samples=self.samples, alpha=self.alpha,
                                 seed=seed, threads=self.threads)

    def window_table(self, tree, player, seed=0, threads=None):
        return tree.agent_table(self.name, player, samples=self.table_samples, seed=seed,
                                threads=self.threads if threads is None else threads)

    def describe(self):
        return {"kind": "rust", "name": self.name, "samples": self.samples, "alpha": self.alpha,
                "table_samples": self.table_samples}


def make_model(spec, device="cpu", **rust_options):
    """'nn:PATH[,argmax]' -> NetModel; anything else -> RustModel (a built-in spec)."""
    if spec.startswith(NN):
        body = spec[len(NN):]
        argmax = body.endswith(ARGMAX)
        if argmax:
            body = body[:-len(ARGMAX)]
        if not Path(body).is_file():
            raise ValueError(f"{spec!r}: no checkpoint file {body}")
        return NetModel(body, device, argmax=argmax)
    return RustModel(spec, **rust_options)
