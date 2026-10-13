"""Networks for the blueprint policy pi.

PolicyNet sees only the mover's observation (bbar._core.OBS_DIM, layout in core/src/features.rs) and outputs masked
policy logits over the 36 actions plus an auxiliary prediction of the opponent's hand. Critic additionally sees the
privileged features (opponent hand, both next draws) and predicts win/draw/loss for the mover; it is used only for
training, never to act.
"""
import torch
from torch import nn
import torch.nn.functional as F

from bbar._core import NUM_ACTIONS, OBS_DIM, PRIV_DIM, obs_layout

NUM_ANIMALS = 12
LAYOUT = obs_layout()
OPP_UNPLAYED = slice(LAYOUT["opp_unplayed"][0], LAYOUT["opp_unplayed"][0] + NUM_ANIMALS)
OPP_HAND_PUBLIC = slice(LAYOUT["opp_hand_public"][0], LAYOUT["opp_hand_public"][0] + NUM_ANIMALS)
PRIV_OPP_HAND = slice(LAYOUT["priv_opp_hand"][0], LAYOUT["priv_opp_hand"][0] + NUM_ANIMALS)
NEG = -1e9  # logit of an illegal action: exp() underflows to exactly 0 in float32, and 0 * NEG stays finite


class Block(nn.Module):
    """Pre-LayerNorm residual MLP block: x + W2 GELU(W1 LN(x))."""

    def __init__(self, width, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class Trunk(nn.Module):
    """Linear embedding, `blocks` residual blocks, final LayerNorm."""

    def __init__(self, d_in, width, blocks, expansion=1):
        super().__init__()
        self.embed = nn.Linear(d_in, width)
        self.blocks = nn.Sequential(*[Block(width, expansion * width) for _ in range(blocks)])
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return F.gelu(self.norm(self.blocks(self.embed(x))))


def masked_logits(logits, mask):
    """float32 logits with illegal actions at NEG. `mask`: bool or 0/1 [B, 36]."""
    return logits.float().masked_fill(~mask.bool(), NEG)


class PolicyNet(nn.Module):
    """obs -> (raw policy logits [B, 36], opponent-hand logits [B, 12]). Mask with masked_logits() before use."""

    def __init__(self, width=512, blocks=4, expansion=1):
        super().__init__()
        self.hparams = dict(width=width, blocks=blocks, expansion=expansion)
        self.trunk = Trunk(OBS_DIM, width, blocks, expansion)
        self.policy = nn.Linear(width, NUM_ACTIONS)
        self.opp_hand = nn.Linear(width, NUM_ANIMALS)
        with torch.no_grad():  # start near uniform over legal actions (= the initial magnet)
            self.policy.weight.mul_(0.01)
            self.policy.bias.zero_()

    def forward(self, obs):
        h = self.trunk(obs)
        return self.policy(h), self.opp_hand(h)

    def dist(self, obs, mask):
        """Masked log-probabilities [B, 36] (float32; illegal actions ~ NEG)."""
        logits, _ = self(obs)
        return torch.log_softmax(masked_logits(logits, mask), dim=-1)


class Critic(nn.Module):
    """(obs, privileged) -> win/draw/loss logits [B, 3] for the player to move."""

    def __init__(self, width=512, blocks=4, expansion=1):
        super().__init__()
        self.hparams = dict(width=width, blocks=blocks, expansion=expansion)
        self.trunk = Trunk(OBS_DIM + PRIV_DIM, width, blocks, expansion)
        self.head = nn.Linear(width, 3)

    def forward(self, obs, priv):
        return self.head(self.trunk(torch.cat([obs, priv], dim=-1)))


def wdl_value(wdl_logits):
    """Expected game score of the mover: P(win) + 0.5 P(draw)."""
    p = torch.softmax(wdl_logits.float(), dim=-1)
    return p[..., 0] + 0.5 * p[..., 1]


def sample(logp, generator=None):
    """Samples one action per row from masked log-probabilities [B, 36]."""
    return torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
