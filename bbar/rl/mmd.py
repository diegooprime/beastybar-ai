"""MMD-PPO self-play trainer for the blueprint policy pi: one network plays both seats; the target is an approximate
Nash equilibrium (low exploitability), not the highest score against weak bots.

    python -m bbar.rl.mmd --name NAME [--device cuda] [--n-envs 4096] [...]    # every Config field is a flag
    python -m bbar.rl.mmd --name NAME --resume [--total-decisions 4e9]         # continue checkpoints/NAME/latest.pt
    python -m bbar.rl.mmd --name NAME --force [...]       # start over; the old runs/NAME, checkpoints/NAME move aside
    python -m bbar.rl.mmd --help

Algorithm (magnetic mirror descent, Sokota et al. 2023, implemented as PPO):
- Rollouts: n_envs games in a bbar._core.VecEnv, actions sampled from pi on the device. A game's transitions are
  trained on only once the game is complete; games still running at the end of a rollout carry over to the next
  update (their behaviour log-probs are stored, so the PPO ratio stays correct). Every transition therefore has
  its exact final outcome.
- Advantages: GAE (gamma = 1, lambda) along each player's own decisions in a game, with values from the oracle
  critic (sees the opponent's hand and both next draws) and terminal reward = that player's game score (1/0.5/0).
- Loss: PPO clipped surrogate (normalized advantages) + alpha_t * KL(pi || magnet) (reverse KL, exact over the
  legal actions; magnet = uniform over legal actions, optionally reset to the current policy every M updates,
  R-NaD style; alpha annealed from alpha0 to alpha_min) - entropy_coef * entropy + critic cross-entropy on the
  win/draw/loss outcome + aux_coef * BCE of the opponent-hand head (over the opponent's unplayed animals only).

Outputs: runs/NAME/metrics.jsonl (one "train" line per update, one "eval" line per evaluation match),
runs/NAME/config.json, checkpoints/NAME/{latest.pt, step_<update>.pt} with the config embedded.
Evaluation: duplicate deals (each deal twice, seats swapped), pi sampled (mixed strategy), against uniform random,
the frozen snapshot from the previous evaluation, and any --eval-vs checkpoints; bbar.stats.summarize CIs.
"""
import argparse
import copy
import dataclasses
import itertools
import json
import math
import os
import socket
import subprocess
import sys
import time
import typing
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bbar import stats
from bbar._core import MAX_PLIES, NUM_ACTIONS, OBS_DIM, OBS_VERSION, PRIV_DIM, VecEnv
from bbar.rl.agent import NetAgent
from bbar.rl.net import OPP_UNPLAYED, PRIV_OPP_HAND, Critic, PolicyNet, masked_logits, sample, wdl_value

ROOT = Path(__file__).resolve().parents[2]
ARCH = ("width", "blocks", "critic_width", "critic_blocks", "expansion")


@dataclass
class Config:
    name: str = "blueprint"
    seed: int = 0
    device: str = "auto"              # auto = cuda if available, else cpu
    # environment and rollouts
    n_envs: int = 4096
    rollout_steps: int = 16           # env steps per update; each step is one decision in every env
    env_threads: int = 0              # 0 = all cores
    # networks
    width: int = 512
    blocks: int = 4
    critic_width: int = 512
    critic_blocks: int = 4
    expansion: int = 1                # hidden width of a residual block = expansion * width
    # optimisation
    lr: float = 3e-4
    critic_lr: float = 3e-4
    warmup_updates: int = 20          # linear learning-rate warmup
    weight_decay: float = 1e-4
    adam_eps: float = 1e-6
    max_grad_norm: float = 1.0
    epochs: int = 2
    minibatch: int = 8192
    clip: float = 0.2
    gae_lambda: float = 0.95
    adv_norm: str = "std"             # std: (A - mean) / std per update; none: raw score units
    # magnetic mirror descent
    alpha0: float = 0.1
    alpha_min: float = 0.005
    alpha_schedule: str = "exp"       # exp (geometric), linear, const; progress = decisions / alpha_decay_decisions
    alpha_decay_decisions: float = 0.0  # 0 = total_decisions of the first run (fixed at start; resumes keep it)
    magnet_reset: int = 0             # updates between magnet <- current policy (0: uniform magnet throughout)
    entropy_coef: float = 0.0
    critic_coef: float = 1.0
    aux_coef: float = 0.1
    # performance
    amp: str = "auto"                 # bf16 autocast: auto = on for cuda
    compile: bool = False             # torch.compile the networks
    obs_storage: str = "auto"         # dtype of stored observations: auto = float16 on cuda, float32 elsewhere
    # budget, logging, evaluation, checkpoints
    total_decisions: float = 1e9
    log_every: int = 10               # stdout every N updates (metrics.jsonl gets every update)
    eval_every: int = 100             # updates; 0 = only at the start and the end
    eval_deals: int = 2000            # duplicate deal pairs per evaluation match
    eval_seed: int = 1000
    eval_vs: list = field(default_factory=list)   # extra checkpoints to evaluate against
    ckpt_every: int = 500             # updates between step_<k>.pt (latest.pt is also written at every evaluation)
    runs_dir: str = "runs"
    ckpt_dir: str = "checkpoints"


HELP = {
    "rollout_steps": "env steps per update (decisions per update = n_envs * rollout_steps)",
    "gae_lambda": "GAE lambda along each player's own decisions (gamma = 1)",
    "alpha0": "initial weight of KL(pi || magnet)",
    "magnet_reset": "reset the magnet to the current policy every M updates (0 = uniform magnet throughout)",
    "eval_vs": "checkpoints to evaluate against at every evaluation",
}


# --------------------------------------------------------------------------------------------- configuration / CLI

def build_parser(explicit_only=False):
    """The CLI: one flag per Config field. With explicit_only, absent flags are left out of the namespace."""
    p = argparse.ArgumentParser(prog="python -m bbar.rl.mmd", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                argument_default=argparse.SUPPRESS if explicit_only else None)
    defaults = Config()
    types = typing.get_type_hints(Config)   # the annotation, not type(default): `x: float = 0` must accept 2e9
    for f in dataclasses.fields(Config):
        flag = "--" + f.name.replace("_", "-")
        default = getattr(defaults, f.name)
        kw = {"help": HELP.get(f.name, f"default {default!r}")}
        if not explicit_only:
            kw["default"] = default
        if types[f.name] is bool:
            p.add_argument(flag, action=argparse.BooleanOptionalAction, **kw)
        elif types[f.name] is list:
            p.add_argument(flag, nargs="*", type=str, **kw)
        else:
            p.add_argument(flag, type=types[f.name], **kw)
    p.add_argument("--resume", action="store_true", help="continue from checkpoints/NAME/latest.pt (which must exist)",
                   **({} if explicit_only else {"default": False}))
    p.add_argument("--force", action="store_true",
                   help="start over although runs/NAME or checkpoints/NAME exists: both are renamed to "
                        "NAME.replaced-<time> first", **({} if explicit_only else {"default": False}))
    return p


def git_commit():
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return run("rev-parse", "--short", "HEAD").strip() + ("-dirty" if run("status", "--porcelain").strip() else "")
    except (OSError, subprocess.CalledProcessError):
        rev = ROOT / "REVISION"
        return rev.read_text().strip() if rev.exists() else "unknown"


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def mix_seed(*xs):
    """A 63-bit seed from integers (SplitMix64 finalizer over a running combination)."""
    z = 0x243F6A8885A308D3
    for x in xs:
        z = (z ^ (int(x) & 0xFFFFFFFFFFFFFFFF)) * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        z ^= z >> 31
    return z >> 1


# ------------------------------------------------------------------------------------------------------- learning

def gae(values, outcome, valid, lam):
    """GAE with gamma = 1 along each player's decisions. All inputs [G, MAX_PLIES] in (game, ply) layout: values of
    the mover, the mover's final score, and which plies were played (a prefix of each row). The mover alternates, so a
    player's decisions are the even or the odd plies; after a player's last decision the 'next value' is their
    final score. Returns advantages [G, MAX_PLIES] (0 where not valid)."""
    G = values.shape[0]
    half = MAX_PLIES // 2
    v, z, m = (x.view(G, half, 2) for x in (values, outcome, valid))
    adv = torch.zeros_like(v)
    for j in reversed(range(half)):
        if j + 1 < half:
            has_next = m[:, j + 1]
            next_v = torch.where(has_next, v[:, j + 1], z[:, j])
            next_a = torch.where(has_next, adv[:, j + 1], torch.zeros_like(v[:, j]))
        else:
            next_v, next_a = z[:, j], torch.zeros_like(v[:, j])
        adv[:, j] = torch.where(m[:, j], next_v - v[:, j] + lam * next_a, torch.zeros_like(v[:, j]))
    return adv.view(G, MAX_PLIES)


def wdl_target(score):
    """Mover's game score 1 / 0.5 / 0 -> class 0 (win) / 1 (draw) / 2 (loss)."""
    return torch.round(2 - 2 * score).long()


def calibration(pred, target, bins=10):
    """Expected calibration error of predicted scores against outcomes (equal-width bins, weighted by count)."""
    b = (pred * bins).long().clamp(0, bins - 1)
    count = torch.bincount(b, minlength=bins).float()
    sp = torch.bincount(b, weights=pred, minlength=bins)
    st = torch.bincount(b, weights=target, minlength=bins)
    return float(((sp - st).abs()).sum() / count.sum().clamp(min=1))


@dataclass
class Batch:
    obs: torch.Tensor          # [N, OBS_DIM] storage dtype
    priv: torch.Tensor         # [N, PRIV_DIM]
    mask: torch.Tensor         # [N, 36] bool
    action: torch.Tensor       # [N] int64
    old_logp: torch.Tensor     # [N] behaviour log-prob
    adv: torch.Tensor          # [N] raw advantages
    outcome: torch.Tensor      # [N] mover's final score
    value: torch.Tensor        # [N] critic value before the update
    stats: dict

    @property
    def n(self):
        return self.obs.shape[0]


class Trainer:
    def __init__(self, cfg, checkpoint=None):
        if not cfg.alpha_decay_decisions:
            cfg = dataclasses.replace(cfg, alpha_decay_decisions=cfg.total_decisions)
        self.cfg = cfg
        self.device = dev = resolve_device(cfg.device)
        self.amp = dev.type == "cuda" if cfg.amp == "auto" else cfg.amp in ("on", "true", "1", "bf16")
        self.store_dtype = {"auto": torch.float16 if dev.type == "cuda" else torch.float32,
                            "float16": torch.float16, "bfloat16": torch.bfloat16,
                            "float32": torch.float32}[cfg.obs_storage]
        if dev.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(cfg.seed)

        self.policy = PolicyNet(cfg.width, cfg.blocks, cfg.expansion).to(dev)
        self.critic = Critic(cfg.critic_width, cfg.critic_blocks, cfg.expansion).to(dev)
        self.magnet = None          # None = uniform over legal actions
        self.prev = copy.deepcopy(self.policy).eval()   # frozen snapshot from the previous evaluation
        self.opt = torch.optim.AdamW(
            [{"params": self.policy.parameters(), "lr": cfg.lr, "base_lr": cfg.lr},
             {"params": self.critic.parameters(), "lr": cfg.critic_lr, "base_lr": cfg.critic_lr}],
            weight_decay=cfg.weight_decay, eps=cfg.adam_eps)
        self.update = 0
        self.decisions = 0
        self.games = 0
        self.wall = 0.0
        if checkpoint is not None:
            self._load(checkpoint)
        self.policy_fwd = torch.compile(self.policy, dynamic=True) if cfg.compile else self.policy
        self.critic_fwd = torch.compile(self.critic, dynamic=True) if cfg.compile else self.critic

        # Rollout ring buffer: the last K env steps of every env. A game ending at step t started at t - 23 or later,
        # and steps of the current rollout never overwrite those (K = rollout_steps + MAX_PLIES).
        n, K = cfg.n_envs, cfg.rollout_steps + MAX_PLIES
        self.K = K
        self.env = VecEnv(n, seed=mix_seed(cfg.seed, self.update), threads=cfg.env_threads)
        self.buf_obs = torch.zeros(K, n, OBS_DIM, dtype=self.store_dtype, device=dev)
        self.buf_priv = torch.zeros(K, n, PRIV_DIM, dtype=self.store_dtype, device=dev)
        self.buf_mask = torch.zeros(K, n, NUM_ACTIONS, dtype=torch.bool, device=dev)
        self.buf_action = torch.zeros(K, n, dtype=torch.long, device=dev)
        self.buf_logp = torch.zeros(K, n, dtype=torch.float32, device=dev)
        self.buf_mover = torch.zeros(K, n, dtype=torch.int8, device=dev)
        # Host staging buffers the env encodes into (pinned on CUDA: asynchronous host-to-device copies).
        pin = dev.type == "cuda"
        self.h_obs = torch.empty(n, OBS_DIM, dtype=torch.float32, pin_memory=pin)
        self.h_mask = torch.empty(n, NUM_ACTIONS, dtype=torch.uint8, pin_memory=pin)
        self.h_priv = torch.empty(n, PRIV_DIM, dtype=torch.float32, pin_memory=pin)
        self.h_mover = torch.empty(n, dtype=torch.int64, pin_memory=pin)
        self.h_ply = torch.empty(n, dtype=torch.int64)
        self.t = 0                                      # env steps taken by this process
        # Step of each env's current game's first move. Env i sits out its first steps (staggered over a game
        # length), so games end spread out over time instead of in waves of n_envs at once.
        self.game_start = (np.arange(n, dtype=np.int64) * 23) // n

        self.run_dir = ROOT / cfg.runs_dir / cfg.name
        self.ckpt_dir = ROOT / cfg.ckpt_dir / cfg.name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = open(self.run_dir / "metrics.jsonl", "a")
        meta = {"config": asdict(cfg), "commit": git_commit(), "host": socket.gethostname(), "device": str(dev),
                "torch": torch.__version__, "obs_version": OBS_VERSION, "obs_dim": OBS_DIM,
                "policy_params": sum(p.numel() for p in self.policy.parameters()),
                "critic_params": sum(p.numel() for p in self.critic.parameters()),
                "command": " ".join(sys.argv), "start_update": self.update}
        (self.run_dir / "config.json").write_text(json.dumps(meta, indent=1))
        self.log({"type": "start", **meta})

    # ---------------------------------------------------------------- utilities

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp)

    def log(self, record):
        self.metrics.write(json.dumps(record) + "\n")
        self.metrics.flush()

    def alpha(self):
        c = self.cfg
        p = min(1.0, self.decisions / max(c.alpha_decay_decisions, 1.0))
        if c.alpha_schedule == "const":
            return c.alpha0
        if c.alpha_schedule == "linear":
            return c.alpha0 + (c.alpha_min - c.alpha0) * p
        if c.alpha_schedule == "exp":
            return c.alpha0 * (c.alpha_min / c.alpha0) ** p
        raise ValueError(f"unknown alpha schedule {c.alpha_schedule!r}")

    def policy_logp(self, net, obs, mask):
        with self.autocast():
            logits, _ = net(obs)
        return torch.log_softmax(masked_logits(logits, mask), dim=-1)

    # ---------------------------------------------------------------- rollouts

    @torch.no_grad()
    def rollout(self):
        """rollout_steps env steps in every env. Returns the games completed during them: (env, first step,
        length, player 0's score, player 0's margin), each a numpy array."""
        cfg, dev, K = self.cfg, self.device, self.K
        self.policy.eval()
        ended = []
        for _ in range(cfg.rollout_steps):
            t, slot = self.t, self.t % K
            # The staging buffers are free again: the previous step's copies finished before action.cpu() returned.
            self.env.observe_into(self.h_obs.numpy(), self.h_mask.numpy(), self.h_mover.numpy(), self.h_ply.numpy())
            self.env.privileged_into(self.h_priv.numpy())
            active = self.game_start <= t
            if not np.array_equal(self.h_ply.numpy(), np.where(active, t - self.game_start, 0)):
                raise RuntimeError("ply bookkeeping out of sync with the env")
            obs_t = self.h_obs.to(dev, non_blocking=True)
            mask_t = self.h_mask.to(dev, non_blocking=True).bool()
            self.buf_priv[slot].copy_(self.h_priv, non_blocking=True)
            self.buf_mover[slot].copy_(self.h_mover, non_blocking=True)
            logp = self.policy_logp(self.policy_fwd, obs_t, mask_t)
            action = sample(logp)
            self.buf_obs[slot] = obs_t
            self.buf_mask[slot] = mask_t
            self.buf_action[slot] = action
            self.buf_logp[slot] = logp.gather(1, action[:, None])[:, 0]
            actions = action.cpu().numpy()
            if not active.all():
                actions[~active] = -1                   # not started yet: the env does not move
            done, result, margin = self.env.step(actions)
            idx = np.flatnonzero(done)
            if len(idx):
                ended.append((idx, self.game_start[idx].copy(), t - self.game_start[idx] + 1, result[idx].copy(),
                              margin[idx].copy()))
                self.game_start[idx] = t + 1
            self.t += 1
        self.policy.train()
        if not ended:
            return None
        return tuple(np.concatenate(x) for x in zip(*ended))

    @torch.no_grad()
    def make_batch(self, ended):
        """Gathers the transitions of the completed games, evaluates the critic, computes GAE."""
        cfg, dev, K = self.cfg, self.device, self.K
        env, first, length, p0_score, margin = ended
        G = len(env)
        k = np.arange(MAX_PLIES)
        valid = k[None, :] < length[:, None]                     # [G, 24]
        slots = (first[:, None] + k[None, :]) % K
        envs = np.broadcast_to(env[:, None], valid.shape)
        gi = torch.from_numpy(np.nonzero(valid)[0]).to(dev)    # game of each transition, row-major (game, ply)
        si = torch.from_numpy(slots[valid]).to(dev)
        ei = torch.from_numpy(envs[valid]).to(dev)
        obs, priv = self.buf_obs[si, ei], self.buf_priv[si, ei]
        mover = self.buf_mover[si, ei].long()
        score0 = torch.from_numpy(p0_score).to(dev)[gi]
        outcome = torch.where(mover == 0, score0, 1 - score0)

        value = torch.empty(len(gi), device=dev)
        ce = torch.zeros((), device=dev)
        self.critic.eval()
        for i in range(0, len(gi), cfg.minibatch):
            sl = slice(i, i + cfg.minibatch)
            with self.autocast():
                wdl = self.critic_fwd(obs[sl].float(), priv[sl].float())
            value[sl] = wdl_value(wdl)
            ce += F.cross_entropy(wdl.float(), wdl_target(outcome[sl]), reduction="sum")
        self.critic.train()

        valid_t = torch.from_numpy(valid).to(dev)
        V = torch.zeros(G, MAX_PLIES, device=dev)
        Z = torch.zeros(G, MAX_PLIES, device=dev)
        V[valid_t], Z[valid_t] = value, outcome
        adv = gae(V, Z, valid_t, cfg.gae_lambda)[valid_t]

        starter = mover[torch.from_numpy(np.cumsum(length) - length).to(dev)].cpu().numpy()  # mover at ply 0
        start_score = np.where(starter == 0, p0_score, 1 - p0_score)
        s = {
            "games": G, "transitions": int(len(gi)), "game_len": float(length.mean()),
            "start_score": float(start_score.mean()), "draw_rate": float((p0_score == 0.5).mean()),
            "abs_margin": float(np.abs(margin).mean()),
            "critic_ce_pre": float(ce / len(gi)),
            "critic_brier": float(((value - outcome) ** 2).mean()),
            "critic_ece": calibration(value, outcome),
            "value_mean": float(value.mean()), "adv_std": float(adv.std()),
            "carried": int((first < self.t - cfg.rollout_steps).sum()),
        }
        return Batch(obs, priv, self.buf_mask[si, ei], self.buf_action[si, ei], self.buf_logp[si, ei], adv, outcome,
                     value, s)

    # ---------------------------------------------------------------- update

    def learn(self, batch):
        cfg, dev = self.cfg, self.device
        alpha = self.alpha()
        warm = min(1.0, (self.update + 1) / max(cfg.warmup_updates, 1))
        for g in self.opt.param_groups:
            g["lr"] = g["base_lr"] * warm
        adv = batch.adv
        if cfg.adv_norm == "std":
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        elif cfg.adv_norm != "none":
            raise ValueError(f"unknown adv_norm {cfg.adv_norm!r}")
        n_legal = batch.mask.sum(-1, keepdim=True).float()
        acc = []
        for _ in range(cfg.epochs):
            perm = torch.randperm(batch.n, device=dev)
            for i in range(0, batch.n, cfg.minibatch):
                idx = perm[i:i + cfg.minibatch]
                obs, priv, mask = batch.obs[idx].float(), batch.priv[idx].float(), batch.mask[idx]
                act, old_logp, a = batch.action[idx], batch.old_logp[idx], adv[idx]
                with self.autocast():
                    logits, aux = self.policy_fwd(obs)
                    wdl = self.critic_fwd(obs, priv)
                logp_all = torch.log_softmax(masked_logits(logits, mask), dim=-1)
                p_all = logp_all.exp()
                logp = logp_all.gather(1, act[:, None])[:, 0]
                log_ratio = logp - old_logp
                ratio = log_ratio.exp()
                pg_loss = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
                entropy = -(p_all * logp_all).masked_fill(~mask, 0).sum(-1)
                if self.magnet is None:
                    log_magnet = -torch.log(n_legal[idx])
                else:
                    with torch.no_grad():
                        log_magnet = self.policy_logp(self.magnet, obs, mask)
                kl_magnet = (p_all * (logp_all - log_magnet)).masked_fill(~mask, 0).sum(-1)
                critic_loss = F.cross_entropy(wdl.float(), wdl_target(batch.outcome[idx]))
                opp_hand, aux_mask = priv[:, PRIV_OPP_HAND], obs[:, OPP_UNPLAYED]
                aux_loss = (F.binary_cross_entropy_with_logits(aux.float(), opp_hand, reduction="none")
                            * aux_mask).sum() / aux_mask.sum().clamp(min=1)
                loss = (pg_loss + alpha * kl_magnet.mean() - cfg.entropy_coef * entropy.mean()
                        + cfg.aux_coef * aux_loss + cfg.critic_coef * critic_loss)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gn_pi = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                gn_v = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    acc.append(torch.stack([
                        loss.detach(), pg_loss.detach(), kl_magnet.mean().detach(), entropy.mean().detach(),
                        critic_loss.detach(), aux_loss.detach(), ((ratio - 1) - log_ratio).mean().detach(),
                        ((ratio - 1).abs() > cfg.clip).float().mean(), gn_pi, gn_v]))
        names = ("loss", "pg_loss", "kl_magnet", "entropy", "critic_ce", "aux_bce", "approx_kl", "clip_frac",
                 "grad_norm_pi", "grad_norm_v")
        means = torch.stack(acc).mean(0).tolist()
        return {"alpha": alpha, "lr": self.opt.param_groups[0]["lr"], **dict(zip(names, means))}

    # ---------------------------------------------------------------- evaluation

    @torch.no_grad()
    def evaluate(self, opponent, name, deals=None):
        """Duplicate-deal match: pi (sampled) against `opponent` ('random', a PolicyNet or a NetAgent)."""
        cfg, dev = self.cfg, self.device
        deals = deals or cfg.eval_deals
        gen = torch.Generator(device=dev).manual_seed(mix_seed(cfg.eval_seed, self.update))
        self.policy.eval()

        def logp_of(who, obs, mask):
            if who == "random":
                return torch.log_softmax(masked_logits(torch.zeros_like(mask, dtype=torch.float32), mask), -1)
            if isinstance(who, NetAgent):
                return who.logp(obs, mask)
            return self.policy_logp(who, obs, mask)

        a_pts, b_pts, deal_idx = [], [], []
        chunk = max(1, min(deals, 4096))
        for d0 in range(0, deals, chunk):
            d = np.arange(d0, min(deals, d0 + chunk))
            n = 2 * len(d)
            seeds = np.repeat(np.array([mix_seed(cfg.eval_seed, j) for j in d], dtype=np.uint64), 2)
            starts = np.repeat(d % 2, 2)
            a_seat = np.tile([0, 1], len(d))          # env 2j: pi is player 0; env 2j+1: pi is player 1
            env = VecEnv(n, seed=0, threads=cfg.env_threads)
            env.reset_deals(seeds, starts)
            finished = np.zeros(n, dtype=bool)
            pts = np.zeros((n, 2), dtype=np.int64)
            while not finished.all():
                obs, mask, mover, _ = env.observe()
                obs_t = torch.from_numpy(obs).to(dev)
                mask_t = torch.from_numpy(mask).to(dev).bool()
                actions = np.full(n, -1, dtype=np.int64)
                for who, sel in ((self.policy, ~finished & (mover == a_seat)),
                                 (opponent, ~finished & (mover != a_seat))):
                    if sel.any():
                        rows = torch.from_numpy(np.flatnonzero(sel)).to(dev)
                        actions[sel] = sample(logp_of(who, obs_t[rows], mask_t[rows]), gen).cpu().numpy()
                done, _, _ = env.step(actions)
                newly = done & ~finished
                if newly.any():
                    pts[newly] = env.final_scores()[newly]
                finished |= done
            a_pts.append(pts[np.arange(n), a_seat])
            b_pts.append(pts[np.arange(n), 1 - a_seat])
            deal_idx.append(np.repeat(d, 2))
        self.policy.train()
        return stats.summarize(f"pi@{self.update}", name, cfg.eval_seed, np.concatenate(a_pts),
                               np.concatenate(b_pts), np.concatenate(deal_idx))

    def run_evals(self):
        t = time.time()
        opponents = [("random", "random"), (self.prev, f"pi@{self.prev_update}")]
        for path in self.cfg.eval_vs:
            opponents.append((NetAgent(path, self.device), path))
        for opp, name in opponents:
            r = self.evaluate(opp, name)
            self.log({"type": "eval", "update": self.update, "decisions": self.decisions, "opponent": name,
                      "wall": round(self.wall, 1), **asdict(r)})
            print(f"  eval {r}", flush=True)
        self.prev.load_state_dict(self.policy.state_dict())
        self.prev_update = self.update
        return time.time() - t

    # ---------------------------------------------------------------- checkpoints

    def state(self, resumable=True):
        """Checkpoint contents. Snapshots (resumable=False) keep only the networks: enough for NetAgent and for
        warm starts, about a fifth of the size."""
        st = {
            "config": asdict(self.cfg), "obs_version": OBS_VERSION, "obs_dim": OBS_DIM,
            "policy_hparams": self.policy.hparams, "critic_hparams": self.critic.hparams,
            "policy": self.policy.state_dict(), "critic": self.critic.state_dict(),
            "update": self.update, "decisions": self.decisions, "games": self.games, "wall": self.wall,
            "commit": git_commit(),
        }
        if resumable:
            st.update({
                "magnet": None if self.magnet is None else self.magnet.state_dict(),
                "prev": self.prev.state_dict(), "prev_update": self.prev_update,
                "opt": self.opt.state_dict(), "torch_rng": torch.get_rng_state(),
            })
        return st

    def save(self, step=False):
        """latest.pt (resumable) and, with step, the snapshot step_<update>.pt."""
        for path, resumable in [(self.ckpt_dir / "latest.pt", True)] + (
                [(self.ckpt_dir / f"step_{self.update}.pt", False)] if step else []):
            tmp = path.with_suffix(".tmp")
            torch.save(self.state(resumable), tmp)
            os.replace(tmp, path)

    def _load(self, ck):
        self.policy.load_state_dict(ck["policy"])
        self.critic.load_state_dict(ck["critic"])
        if ck.get("magnet") is not None:
            self.magnet = copy.deepcopy(self.policy).eval()
            self.magnet.load_state_dict(ck["magnet"])
        self.prev.load_state_dict(ck["prev"])
        self.prev_update = ck.get("prev_update", 0)
        self.opt.load_state_dict(ck["opt"])
        for g, lr in zip(self.opt.param_groups, (self.cfg.lr, self.cfg.critic_lr)):
            g["base_lr"] = lr
        self.update, self.decisions, self.games, self.wall = ck["update"], ck["decisions"], ck["games"], ck["wall"]
        torch.set_rng_state(ck["torch_rng"])

    prev_update = 0

    # ---------------------------------------------------------------- main loop

    def train_step(self):
        """One update: rollout, batch of completed games, PPO epochs. Returns the metrics record."""
        cfg = self.cfg
        if cfg.magnet_reset > 0 and self.update > 0 and self.update % cfg.magnet_reset == 0:
            if self.magnet is None:
                self.magnet = copy.deepcopy(self.policy).eval()
            self.magnet.load_state_dict(self.policy.state_dict())
        t0 = time.time()
        ended = self.rollout()
        t1 = time.time()
        rec = {"type": "train", "update": self.update}
        if ended is not None:
            batch = self.make_batch(ended)
            rec.update(batch.stats)
            rec.update(self.learn(batch))
            self.games += batch.stats["games"]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.time()
        self.update += 1
        self.decisions += cfg.n_envs * cfg.rollout_steps
        self.wall += t2 - t0
        rec.update({"decisions": self.decisions, "games_total": self.games, "wall": round(self.wall, 2),
                    "rollout_s": round(t1 - t0, 4), "learn_s": round(t2 - t1, 4),
                    "decisions_per_s": round(cfg.n_envs * cfg.rollout_steps / (t2 - t0), 1)})
        self.log(rec)
        return rec

    def train(self):
        cfg = self.cfg
        print(f"{cfg.name}: device {self.device}, amp {self.amp}, "
              f"pi {sum(p.numel() for p in self.policy.parameters()) / 1e6:.2f}M params, "
              f"critic {sum(p.numel() for p in self.critic.parameters()) / 1e6:.2f}M, "
              f"{cfg.n_envs} envs x {cfg.rollout_steps} steps per update, from update {self.update}", flush=True)
        try:
            if self.update == 0:
                self.wall += self.run_evals()
                self.save()
            while self.decisions < cfg.total_decisions:
                rec = self.train_step()
                if self.update % cfg.log_every == 0:
                    keys = ("decisions", "games_total", "pg_loss", "kl_magnet", "entropy", "critic_ce", "aux_bce",
                            "approx_kl", "clip_frac", "critic_ece", "alpha", "decisions_per_s")
                    print(f"u{self.update} " + " ".join(
                        f"{k}={rec[k]:.4g}" if isinstance(rec.get(k), float) else f"{k}={rec.get(k)}" for k in keys),
                        flush=True)
                if cfg.eval_every and self.update % cfg.eval_every == 0:
                    self.wall += self.run_evals()
                    self.save(step=False)
                if cfg.ckpt_every and self.update % cfg.ckpt_every == 0:
                    self.save(step=True)
            if not (cfg.eval_every and self.update % cfg.eval_every == 0):
                self.wall += self.run_evals()
            self.save(step=True)
        except KeyboardInterrupt:
            print("interrupted: saving latest.pt", flush=True)
            self.save()
            raise
        finally:
            self.metrics.close()


def set_aside(path):
    """Renames the directory `path` to `<name>.replaced-<time>` beside it (never over an existing path); returns the
    new path."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for i in itertools.count():
        dest = path.with_name(f"{path.name}.replaced-{stamp}" + (f"-{i}" if i else ""))
        if not dest.exists():
            path.rename(dest)
            return dest


def main(argv=None):
    args = build_parser().parse_args(argv)
    explicit = vars(build_parser(explicit_only=True).parse_args(argv))
    resume, force = args.resume, args.force
    if resume and force:
        sys.exit("--resume and --force exclude each other")
    cfg = Config(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Config)})
    run_dir, ckpt_dir = ROOT / cfg.runs_dir / cfg.name, ROOT / cfg.ckpt_dir / cfg.name
    latest = ckpt_dir / "latest.pt"
    checkpoint = None
    if resume:
        if not latest.exists():
            sys.exit(f"--resume: {latest} not found; check --name and --ckpt-dir (on a new pod, copy the checkpoint "
                     f"over first; scripts/pod.sh sync skips checkpoints/)")
        checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
        stored = Config(**checkpoint["config"])
        changed = {k: v for k, v in explicit.items() if k in asdict(stored) and getattr(stored, k) != v}
        if any(k in ARCH for k in changed):
            sys.exit(f"cannot change the architecture on resume: {sorted(k for k in changed if k in ARCH)}")
        cfg = dataclasses.replace(stored, **changed)
        print(f"resuming {latest} at update {checkpoint['update']}" + (f", overriding {changed}" if changed else ""))
    else:
        # A fresh run must not mix with an old one: metrics.jsonl is appended to, and old step_<k>.pt files would
        # sit beside the new run's.
        old = [d for d in (run_dir, ckpt_dir) if d.is_dir() and any(d.iterdir())]
        if old and not force:
            sys.exit(f"{' and '.join(map(str, old))} already exist: "
                     + ("pass --resume to continue the run, " if latest.exists() else "")
                     + "pass --force to start over (the old directories are renamed, not deleted) or pick another --name")
        for d in old:
            print(f"--force: moved {d} to {set_aside(d)}", flush=True)
    Trainer(cfg, checkpoint).train()


if __name__ == "__main__":
    main()
