"""Duplicate-deal matches between any two agents, built-in or trained, with bbar.arena's protocol and evidence trail.

    python -m bbar.match A B --deals N [--seed S] [--device D] [--log PATH] [--batch B] [--threads T]

Agents:
  SPEC             a built-in agent (core/src/agents.rs): random, greedy, pimc:w=W,d=D, and the oracles
                   pimc-hand:w=W,d=D and cheat:d=D (see bbar.arena)
  nn:PATH          the policy of a checkpoint (bbar/rl/agent.py), sampling its mixed strategy (the default: an
                   approximate equilibrium is a mixed strategy, and its argmax can be exploitable)
  nn:PATH,argmax   the same policy playing its most probable action
  tail:nn:PATH,... the exact-tail player (bbar/tail.py): the network early, a solved window from ply F, exact
                   minimax at the end; e.g. tail:nn:PATH,F=10,mode=gadget. It keeps one solved window per game in
                   play (about 12 MB each at F=10), so --batch defaults to 128 when a Python agent plays (at most
                   its cache, default 256, or windows get re-solved: same moves, more compute)

Protocol, identical to core/src/arena.rs: deal i is the arena's deal (seed, i), started by seat i % 2; game 2i seats
A at 0 and B at 1, game 2i+1 replays the deal with the seats swapped. Each seat of each game has its own RNG, keyed
by the seed (SipHash, non-invertible): built-in agents build their views and act with it, exactly as in the arena;
networks see only their observation (core/src/features.rs) and sample with one uniform draw from it per decision
(inverse CDF over the float64 probabilities). Agents written in Python (PyPlayer: bbar.tail) get what a built-in agent
gets and nothing else (VecEnv.views): the same one draw per decision from the seat's RNG that a network gets (so a
Python agent sampling a network with it replays that network move for move), and, from a stream derived from that
draw, the mover's view (own hand and public state true, hidden cards re-randomized) and a private seed; plus the move
history and the observation. Each stream of games gets its own fresh agent instance (one for the games where A sits
in seat 0, one for those where A sits in seat 1, separately for A and B, fresh every match), so an agent that keeps
state across calls never sees both games of a duplicate pair: its hand in one is the opponent's in the other. The
games run in a bbar._core.VecEnv, --batch deals at a time: built-in agents act through VecEnv.act (multithreaded, GIL
released), networks in batches on the device.

Between two built-in agents, a match replays bbar.arena's games move for move (same seed, same deals, same moves;
tests/test_match.py). Results never depend on --batch or --threads; with networks, the device (and, rarely, the batch
composition) can matter only through float rounding of the probabilities at a sampling boundary. Networks always play
with full-FP32 matmuls (TF32 off, PyTorch's default) whatever the calling process set (the exploiter trains with
TF32 on CUDA), so a match logged from inside a trainer replays the same as `python -m bbar.match` on that device.

Every match appends one JSON line to --log (default runs/results.jsonl, the evidence trail): bbar.arena's record
(score with its 95% interval over deal pairs, W/D/L, margin, seed, commit, code hash, engine build, host, command),
plus "checkpoints" {path: first 16 hex digits of the SHA-256 of the file as loaded}, "device" and "runner". The same
refusals as bbar.arena: a stale engine build, and uncommitted code in the evidence trail unless --allow-dirty.
Without --seed, a random 64-bit seed is drawn and logged.
"""
import argparse
import contextlib
import hashlib
import io
import os
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bbar import arena, provenance, stats
from bbar._core import MAX_PLIES, VecEnv, build_id, parse_spec

NN = "nn:"
ARGMAX = ",argmax"
LOG = arena.LOG
PY_BATCH = 128   # default --batch when a Python agent plays
# Python agents by spec prefix: module with a class whose factory(body, device) returns a zero-argument callable that
# makes fresh agents (PyPlayer).
PY_AGENTS = {"tail:": ("bbar.tail", "TailAgent")}


def sha256_short(data):
    return hashlib.sha256(data).hexdigest()[:16]


def resolve_device(name="auto"):
    """'auto': cuda if available, else mps, else cpu."""
    import torch
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def sample_inverse_cdf(probs, u):
    """One action per row of probs (float64 [k, 36], zero on illegal actions): the first action whose cumulative
    probability exceeds u * total. It always has positive probability."""
    cdf = np.cumsum(probs, axis=1)
    total = cdf[:, -1]
    x = np.minimum(u * total, np.nextafter(total, 0))
    return (cdf <= x[:, None]).sum(axis=1).astype(np.int64)


class RustPlayer:
    """A built-in agent, acting inside the env (VecEnv.act)."""
    needs_obs = False

    def __init__(self, spec):
        self.name = parse_spec(spec)   # canonical; ValueError if invalid
        self.checkpoints = {}

    def act(self, env, idx, obs, mask):
        return env.act(self.name, idx)


class NetPlayer:
    """A policy network: a checkpoint path (hashed as loaded) or a PolicyNet. Samples its mixed strategy with the
    seat RNGs' uniforms (argmax=False) or plays its most probable action."""
    needs_obs = True

    def __init__(self, source, device="cpu", argmax=False, name=None):
        from bbar.rl.agent import NetAgent, check_layout, policy_from_checkpoint
        import torch
        self.device = torch.device(device)
        self.argmax = argmax
        if isinstance(source, (str, os.PathLike)):
            path = str(source)
            name = name or NN + path
            data = Path(path).read_bytes()   # hash exactly the bytes that are loaded
            self.checkpoints = {name[len(NN):] if name.startswith(NN) else path: sha256_short(data)}
            ck = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
            check_layout(ck, path)
            net = policy_from_checkpoint(ck)
        else:
            net, self.checkpoints = source, {}
            name = name or NN + "<policy>"
        self.agent = NetAgent(net, self.device, name=name)
        self.name = name + (ARGMAX if argmax else "")

    def act(self, env, idx, obs, mask):
        logp = self.agent.logp(obs[idx], mask[idx])
        if self.argmax:
            return logp.argmax(-1).cpu().numpy()
        # float64 on the host: MPS has no float64, and the device only ever affects the float32 log-probabilities.
        return sample_inverse_cdf(logp.cpu().double().exp().numpy(), env.uniforms(idx))


@dataclass
class Decisions:
    """What a Python agent gets for k decisions (one per env, each for the player to move there): only that player's
    information (VecEnv.views). uniforms: float64 [k], the seat RNG's one draw per decision, the draw a network in that
    seat samples with (NetPlayer); games: view Games (own hand and public state true, the opponent's hand and every
    deck order freshly randomized); histories: every action played so far (lists of ints); seeds: uint64 [k], a
    private seed per decision (the view and the seed come from a stream derived from the draw); obs, legal: the
    mover's observation (float32 [k, OBS_DIM]) and legal mask (uint8 [k, 36]), or None when no player asked for
    observations."""
    games: list
    histories: list
    seeds: np.ndarray
    uniforms: np.ndarray
    obs: object = None
    legal: object = None


class PyPlayer:
    """An agent written in Python. `make`: a zero-argument callable returning a fresh agent (e.g. its class), with
    `act(Decisions) -> int64 [k]`, `name` and `checkpoints` (optionally `needs_obs`, `summary()`, `prepare(live)`). It
    sees the game only through VecEnv.views (see Decisions).

    Every stream of games gets its own instance, made by `make` on its first decision, so an agent that keeps state
    across calls cannot carry information between games that share a deal: a match (`play`) starts with fresh
    instances, one per player and seat assignment (the games where A sits in seat 0, those where it sits in seat 1),
    so no instance sees both games of a duplicate pair, whose hands and draws are each other's; self-play
    (bbar.delta) uses one per seat. Calls without streams (the exploiter's training env, all deals different) use
    stream 0. `summary()` combines the instances' (the agent class's `summarize(instances)` if it has one)."""

    def __init__(self, make, name=None):
        if not callable(make) or (hasattr(make, "act") and not isinstance(make, type)):
            raise TypeError("PyPlayer takes a factory (a zero-argument callable such as the agent's class) that makes "
                            "fresh agents, not an agent: each stream of games gets its own instance")
        self.make = make
        proto = make()
        self.name = name or proto.name
        self.checkpoints = dict(getattr(proto, "checkpoints", {}))
        self.needs_obs = bool(getattr(proto, "needs_obs", False))
        self.instances = {}
        self.live = None

    def instance(self, stream=0):
        """The agent playing `stream` (made fresh on first use)."""
        agent = self.instances.get(stream)
        if agent is None:
            agent = self.instances[stream] = self.make()
            if self.live is not None and hasattr(agent, "prepare"):
                agent.prepare(self.live)
        return agent

    def reset(self):
        """Forgets every instance: the next decisions get fresh ones."""
        self.instances = {}
        self.live = None

    def prepare(self, live):
        """Tells the instances (present and future) that up to `live` games of each will be in play at once."""
        self.live = live
        for agent in self.instances.values():
            if hasattr(agent, "prepare"):
                agent.prepare(live)

    def act(self, env, idx, obs, mask, streams=None):
        """Actions for the envs `idx`; `streams[k]` (int) is the stream of game idx[k] (default: all stream 0)."""
        idx = np.asarray(idx)
        games, hists, seeds, uniforms = env.views(idx)
        streams = np.zeros(len(idx), dtype=np.int64) if streams is None else np.asarray(streams, dtype=np.int64)
        actions = np.full(len(idx), -1, dtype=np.int64)
        for stream in np.unique(streams):
            sel = np.flatnonzero(streams == stream)
            batch = Decisions([games[j] for j in sel], [hists[j] for j in sel], seeds[sel], uniforms[sel],
                              None if obs is None else obs[idx[sel]], None if mask is None else mask[idx[sel]])
            got = np.asarray(self.instance(int(stream)).act(batch), dtype=np.int64)
            if got.shape != (len(sel),):
                raise ValueError(f"{self.name} returned {got.shape} actions for {len(sel)} decisions")
            actions[sel] = got
        for g, a in zip(games, actions):
            if int(a) not in g.legal_actions():
                raise ValueError(f"{self.name} played illegal action {a}")
        return actions

    def summary(self):
        """The instances' statistics, or None if the agent keeps none."""
        agents = [self.instances[k] for k in sorted(self.instances)]
        if not agents:
            return None
        combine = getattr(type(agents[0]), "summarize", None)
        if combine is not None:
            return combine(agents)
        return [a.summary() for a in agents] if hasattr(agents[0], "summary") else None


def parse_agent(agent):
    """(kind, body, argmax): kind 'nn', 'py' (body: the whole spec) or 'rust'."""
    if isinstance(agent, str) and any(agent.startswith(prefix) for prefix in PY_AGENTS):
        return "py", agent, False
    if isinstance(agent, str) and agent.startswith(NN):
        body = agent[len(NN):]
        argmax = body.endswith(ARGMAX)
        if argmax:
            body = body[:-len(ARGMAX)]
        if not body:
            raise ValueError(f"{agent!r}: expected nn:PATH or nn:PATH,argmax")
        return "nn", body, argmax
    return "rust", agent, False


def make_player(agent, device="cpu"):
    """A player from a spec string ('nn:PATH[,argmax]' or a built-in spec) or an existing player."""
    if isinstance(agent, (RustPlayer, NetPlayer, PyPlayer)):
        return agent
    kind, body, argmax = parse_agent(agent)
    if kind == "py":
        import importlib
        prefix = next(p for p in PY_AGENTS if body.startswith(p))
        module, cls = PY_AGENTS[prefix]
        return PyPlayer(getattr(importlib.import_module(module), cls).factory(body[len(prefix):], device))
    if kind == "nn":
        if not Path(body).is_file():
            raise ValueError(f"{agent!r}: no checkpoint file {body}")
        return NetPlayer(body, device, argmax)
    return RustPlayer(body)


@contextlib.contextmanager
def full_fp32_matmuls():
    """CUDA float32 matmuls in full precision (TF32 off, PyTorch's default) inside the block, restored after: matches
    must not depend on what the calling process set (bbar.rl.exploit turns TF32 on to train). The legacy flag, as the
    trainers set it (mixing it with torch's newer fp32_precision API is an error). No-op if torch is not loaded."""
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


def play(pa, pb, deals, seed, threads=0, batch=1024, progress=False):
    """Plays `deals` duplicate pairs of players pa (A) against pb (B); returns raw per-game results in
    bbar._core.play_match's format, ordered by deal then A's seat. Networks run with full-FP32 matmuls. Python agents
    start the match with fresh instances (PyPlayer)."""
    if deals < 1:
        raise ValueError("deals must be at least 1")
    for p in (pa, pb):
        if isinstance(p, PyPlayer):
            p.reset()
    with full_fp32_matmuls():
        return _play(pa, pb, deals, seed, threads, batch, progress)


def _play(pa, pb, deals, seed, threads, batch, progress):
    t0 = last = time.time()
    need_obs = pa.needs_obs or pb.needs_obs
    cols = {k: [] for k in ("deal", "a_seat", "start", "a_points", "b_points")}
    moves = []
    for d0 in range(0, deals, batch):
        d = np.arange(d0, min(deals, d0 + batch))
        n = 2 * len(d)
        a_seat = np.tile([0, 1], len(d))
        env = VecEnv(n, seed=0, threads=threads)
        env.reset_arena(seed, np.repeat(d, 2), a_seat)
        start = env.movers()[0].copy()
        for p in (pa, pb):
            if isinstance(p, PyPlayer):
                p.prepare(len(d))   # each instance plays one game of every deal of the batch
        played = np.full((n, MAX_PLIES), -1, dtype=np.int64)
        finished = np.zeros(n, dtype=bool)
        pts = np.zeros((n, 2), dtype=np.int64)
        obs = mask = None
        while not finished.all():
            if need_obs:
                obs, mask, mover, ply = env.observe()
            else:
                mover, ply = env.movers()
            actions = np.full(n, -1, dtype=np.int64)
            for role, player, sel in ((0, pa, ~finished & (mover == a_seat)), (1, pb, ~finished & (mover != a_seat))):
                idx = np.flatnonzero(sel)
                if not len(idx):
                    continue
                if isinstance(player, PyPlayer):
                    # Streams: (player, A's seat), so no instance sees both games of a pair, even if pa is pb.
                    actions[idx] = player.act(env, idx, obs, mask, streams=2 * role + a_seat[idx])
                else:
                    actions[idx] = player.act(env, idx, obs, mask)
            live = np.flatnonzero(~finished)
            played[live, ply[live]] = actions[live]
            done, _, _ = env.step(actions)
            newly = done & ~finished
            if newly.any():
                pts[newly] = env.final_scores()[newly]
            finished |= done
        cols["deal"] += np.repeat(d, 2).tolist()
        cols["a_seat"] += a_seat.tolist()
        cols["start"] += start.tolist()
        cols["a_points"] += pts[np.arange(n), a_seat].tolist()
        cols["b_points"] += pts[np.arange(n), 1 - a_seat].tolist()
        moves += [[int(m) for m in row if m >= 0] for row in played]
        if progress and time.time() - last >= 10:
            last = time.time()
            done_deals = d[-1] + 1
            secs = last - t0
            print(f"  {pa.name} vs {pb.name}: {done_deals}/{deals} deals, {secs:.0f}s elapsed, "
                  f"~{secs * (deals - done_deals) / done_deals:.0f}s left", file=sys.stderr, flush=True)
    return {"a": pa.name, "b": pb.name, "seconds": time.time() - t0,
            "threads": threads or os.cpu_count(), "build": build_id(), **cols, "moves": moves}


def run(a, b, deals, seed=None, device="cpu", threads=0, batch=1024, progress=False):
    """Plays a match (seed None: a random 64-bit seed); a and b are spec strings ('nn:PATH[,argmax]' or built-in)
    or players. Returns (stats.MatchResult, raw, (player_a, player_b))."""
    pa, pb = make_player(a, device), make_player(b, device)
    seed = arena.random_seed() if seed is None else seed
    raw = play(pa, pb, deals, seed, threads, batch, progress)
    result = stats.summarize(pa.name, pb.name, seed, raw["a_points"], raw["b_points"], raw["deal"])
    return result, raw, (pa, pb)


def record(result, raw, command, revision, players, device=None):
    """The evidence-trail line of a match: bbar.arena's record, plus checkpoint hashes, device and runner."""
    line = arena.record(result, raw, command, revision)
    nets = [p for p in players if isinstance(p, NetPlayer) or (isinstance(p, PyPlayer) and p.checkpoints)]
    line["checkpoints"] = {k: v for p in players for k, v in p.checkpoints.items()}
    line["device"] = str(device) if nets else None
    line["runner"] = "bbar.match"
    stats_ = {p.name: p.summary() for p in players if isinstance(p, PyPlayer) and p.summary() is not None}
    if stats_:
        line["agent_stats"] = stats_
    return line


def check_trail(log, allow_dirty):
    """Provenance before playing: the engine build must be current, and the evidence trail only takes committed
    code (unless allow_dirty). Returns the revision; raises ValueError with the reason otherwise."""
    try:
        provenance.check_build()
    except provenance.StaleBuild as e:
        raise ValueError(str(e))
    revision = provenance.revision()
    if Path(log).resolve() == LOG.resolve() and not provenance.is_clean(revision["commit"]) and not allow_dirty:
        raise ValueError(
            f"the code is not committed ({revision['commit']}): commit it before adding results to the evidence trail "
            f"{LOG.relative_to(arena.ROOT)}, or pass --allow-dirty (the line records code hash {revision['code']})")
    return revision


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bbar.match", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="agent A: built-in spec, nn:PATH, nn:PATH,argmax or tail:nn:PATH,...")
    ap.add_argument("b", help="agent B")
    ap.add_argument("--deals", type=int, default=1000, help="deal pairs (2 games each)")
    ap.add_argument("--seed", type=int, help="match seed, 0 to 2^64 - 1 (default: random, logged)")
    ap.add_argument("--device", default="auto", help="network device: auto (cuda, else mps, else cpu), cpu, cuda, mps")
    ap.add_argument("--threads", type=int, default=0, help="threads for built-in agents and encoding (0 = all cores)")
    ap.add_argument("--batch", type=int, help="deals played at once (memory, not results; default 1024, or 128 when a "
                                               "Python agent plays: the tail agent keeps a solved window per game)")
    ap.add_argument("--log", default=str(LOG), help="JSON-lines log to append to (default runs/results.jsonl)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="log to the evidence trail although the code is not committed")
    ap.add_argument("--quiet", action="store_true", help="no progress lines on stderr")
    args = ap.parse_args(argv)
    if args.deals < 1:
        ap.error("--deals must be at least 1")
    if args.batch is not None and args.batch < 1:
        ap.error("--batch must be at least 1")
    if args.seed is None:
        args.seed = arena.random_seed()
    elif not 0 <= args.seed < 2 ** 64:
        ap.error("--seed must be in 0 .. 2^64 - 1")
    try:
        revision = check_trail(args.log, args.allow_dirty)
        kinds = [parse_agent(x)[0] for x in (args.a, args.b)]
        device = resolve_device(args.device) if {"nn", "py"} & set(kinds) else None
        players = [make_player(x, device) for x in (args.a, args.b)]   # fail before playing anything
    except ValueError as e:
        ap.error(str(e))
    if args.batch is None:
        args.batch = PY_BATCH if "py" in kinds else 1024
    command = shlex.join(["python", *sys.orig_argv[1:]] if argv is None else ["python", "-m", "bbar.match", *argv])
    result, raw, players = run(*players, args.deals, args.seed, device, args.threads, args.batch,
                               progress=not args.quiet)
    arena.append(args.log, record(result, raw, command, revision, players, device))
    games = 2 * result.deals
    print(f"{result}  [{raw['seconds']:.1f}s, {games / max(raw['seconds'], 1e-9):.0f} games/s]", flush=True)
    return result


if __name__ == "__main__":
    main()
