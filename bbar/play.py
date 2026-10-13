"""Play Beasty Bar against the AI in your browser.

    python -m bbar.play [--open] [--opponent SPEC|KEY] [--seat 0|1|random] [--seed S] [--port 8765] [--device cpu]

A local web app: a standard-library HTTP server on 127.0.0.1 (default port 8765) serving one self-contained page
(bbar/play.html) and a small JSON API. --open opens the browser.

The server holds the true game (a bbar._core.Game) and plays the AI through the same code path as bbar.match: a
one-game bbar._core.VecEnv holding the same deal, where a built-in spec (pimc:..., greedy) acts through VecEnv.act, a
network (nn:PATH) samples its observation with the seat's draw, and a Python agent (tail:..., resolve:...) gets
VecEnv.views: its own hand and the public state true, the human's hand and every deck order re-randomized, plus the
move history. Each game's AI thinks in its own background thread with its own agent instance (the loaded networks
are shared), so a new game never waits for an abandoned one: the abandoned game's search stops at its next value
network call. The page shows "thinking" and the think time.

The human gets only what they may know (human_state): their hand, their deck as a set (never its order), the queue,
the bar, THAT'S IT, both scores and deck sizes, the AI's unplayed animals as a set (public: each player owns the same
12 animals) and its hand size, never which of them are in its hand, and the public move log. Previews of the human's
own moves show only public effects (the card drawn afterwards is not shown).

Opponents (--opponent takes a spec or one of these keys; default: the strongest available):
  resolve    the re-solving player (bbar/resolve.py), the project's best:
             resolve:V=checkpoints/value/v3h/best.pt,base=nn:checkpoints/bp-v1/step_30518.pt,P0=8,F=10,d=2,iters=200,
             mode=plain, on the CPU (faster than MPS here). On an M4: 5-12 s at its first re-solved move (ply 8 or
             9), about 0.3 s at ply 10 or 11, instant otherwise. On your turn it precomputes what its next move
             needs from the public history alone (ponder_resolve: same moves, shorter waits; --no-ponder)
  blueprint  the blueprint policy network alone, nn:checkpoints/bp-v1/step_30518.pt
  pimc8      pimc:w=128,d=8 (search)
  pimc4      pimc:w=8,d=4 (weaker search)
  greedy, random
Missing checkpoints drop the opponents that need them, with a message.

API (JSON): GET /api/state (the human's state, legal moves with previews when it is their turn; "server": this
process's id, "version": grows with every change); POST /api/new {"opponent": KEY, "first": "alternate"|"you"|"ai"};
POST /api/play {"game": N, "ply": P, "action": A, "server": ID (optional)} (A: action index, see bbar/actions.py; P: the
ply of the position the move was chosen in). Refused requests (illegal moves, stale game numbers or plies, another
server process, moves out of turn, wrong types) get HTTP 400 with {"error": message, "code": reason}. Finished games
are appended to --record (default runs/play/games.jsonl, local only).
"""
import argparse
import json
import re
import secrets
import sys
import threading
import time
import traceback
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from bbar import match, ref
from bbar._core import Game, VecEnv
from bbar.actions import ANIMALS, POINTS, STRENGTH, decode

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).resolve().with_name("play.html")
PORT = 8765
BLUEPRINT = "checkpoints/bp-v1/step_30518.pt"
VALUE = "checkpoints/value/v3h/best.pt"
RECORD = "runs/play/games.jsonl"
FIRST = ("alternate", "you", "ai")


class PlayError(ValueError):
    """A request the game refuses (HTTP 400). code: a short machine-readable reason for the page."""

    def __init__(self, message, code="bad_request"):
        super().__init__(message)
        self.code = code


class Abandoned(Exception):
    """Raised inside an abandoned game's AI computation to stop it (see Cancellable)."""


class Cancellable:
    """A value function (a re-solving agent's `values`) that raises Abandoned once `event` is set: an abandoned game's
    depth-limited solve stops at its next value-network call instead of running to the end. Everything else passes
    through."""

    def __init__(self, values, event):
        self._values, self._event = values, event

    def __call__(self, *args, **kwargs):
        if self._event.is_set():
            raise Abandoned
        return self._values(*args, **kwargs)

    def session(self, *args, **kwargs):
        """The fast path of bbar/value/dl.solve_many: its per-call function, cancellable the same way."""
        return Cancellable(self._values.session(*args, **kwargs), self._event)

    def __getattr__(self, name):
        return getattr(self._values, name)


# ---- opponents ----------------------------------------------------------------------------------------------------

@dataclass
class Opponent:
    key: str
    label: str
    spec: str
    note: str = ""
    player: object = None   # a ready player (tests); otherwise made from spec on first use
    short: str = ""         # a few words for the page (strength, speed)

    def public(self):
        return {"key": self.key, "label": self.label, "spec": self.spec, "note": self.note, "short": self.short}


def find(path):
    """`path` if it exists from here, else relative to the repository, else None."""
    if Path(path).is_file():
        return str(path)
    return str(ROOT / path) if (ROOT / path).is_file() else None


def catalogue():
    """(opponents available, strongest first; {key: message} for the catalogue's opponents whose checkpoints are
    missing)."""
    bp, v = find(BLUEPRINT), find(VALUE)
    out, missing = [], {}
    if bp and v:
        out.append(Opponent(
            "resolve", "Re-solver (strongest)",
            f"resolve:V={v},base=nn:{bp},P0=8,F=10,d=2,iters=200,mode=plain",
            "Blueprint network for the first 8 moves, then solves the rest of the game on the fly with a value "
            "network and exact endgame solving. Thinks 5-12 s once per game, at move 9 or 10.",
            short="re-solves as it plays; thinks 5-12 s once per game"))
    else:
        missing["resolve"] = (f"the re-solving player needs {VALUE} and {BLUEPRINT}: "
                              f"{', '.join(p for p, f in ((VALUE, v), (BLUEPRINT, bp)) if not f)} not found")
    if bp:
        out.append(Opponent("blueprint", "Blueprint network", f"nn:{bp}",
                            "The self-play policy network alone (no search). Instant.",
                            short="policy network, no search; instant"))
    else:
        missing["blueprint"] = f"the blueprint network needs {BLUEPRINT} (not found)"
    out += [
        Opponent("pimc8", "Deep search (PIMC d=8)", "pimc:w=128,d=8",
                 "Samples 128 guesses of your hand and searches 8 moves deep.",
                 short="searches 128 guesses of your hand"),
        Opponent("pimc4", "Shallow search (PIMC d=4)", "pimc:w=8,d=4", "A weaker, faster search.",
                 short="a weaker, faster search"),
        Opponent("greedy", "Greedy", "greedy", "Plays the move that looks best one move ahead.",
                 short="best-looking move one ahead; weak"),
        Opponent("random", "Random", "random", "Plays a random legal move.", short="random legal moves"),
    ]
    return out, missing


# ---- cards and words ----------------------------------------------------------------------------------------------

def card(c, human):
    animal, owner = c
    return {"animal": animal, "owner": "you" if owner == human else "ai", "strength": STRENGTH[animal],
            "points": POINTS[animal]}


def by_strength(animals):
    return sorted(animals, key=lambda a: STRENGTH[a])


def names(cards, human, start=False):
    """Rich text naming `cards` ((animal, owner) pairs), consecutive cards of one owner grouped:
    '{{y:lion,hippo}}, and {{a:seal}}' (plain: "your Lion and Hippo, and AI's Seal"). start: at the start of a
    sentence ({{Y:...}}: "Your ...")."""
    groups = []
    for animal, owner in cards:
        tag = "y" if owner == human else "a"
        if groups and groups[-1][0] == tag:
            groups[-1][1].append(animal)
        else:
            groups.append((tag, [animal]))
    parts = ["{{" + (tag.upper() if start and i == 0 else tag) + ":" + ",".join(animals) + "}}"
             for i, (tag, animals) in enumerate(groups)]
    if len(parts) == 2 and all(len(a) == 1 for _, a in groups):
        return " and ".join(parts)
    return _listing(parts, ", and ")


def _listing(words, last=" and "):
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + last + words[-1]


def plain(rich):
    """Rich text as plain text: {{y:lion,hippo}} -> 'your Lion and Hippo', {{a:seal}} -> "AI's Seal"."""
    def sub(m):
        animals = [a.capitalize() for a in m.group(2).split(",")]
        owner = {"y": "your ", "Y": "Your ", "a": "AI's ", "A": "AI's "}[m.group(1)]
        return owner + _listing(animals)
    return re.sub(r"\{\{([yaYA]):([a-z,]+)\}\}", sub, rich)


class Trace(ref.Game):
    """bbar.ref's engine playing one move on a public queue, recording what each animal does (the played animal's
    action, then each recurring animal's): public information only."""

    def __init__(self, queue, mover, animal):
        super().__init__(mover, [[], []], [[], []])
        self._hands[mover] = [animal]
        self._queue = [tuple(c) for c in queue]
        self.steps = []
        self._depth = 0

    def is_over(self):
        return False

    def _act(self, animal, actor, params):
        if self._depth:
            return super()._act(animal, actor, params)
        step = {"actor": actor, "as": animal, "params": tuple(params), "before": list(self._queue),
                "bounced": [], "moves": []}
        self.steps.append(step)
        self._depth += 1
        try:
            super()._act(animal, actor, params)
        finally:
            self._depth -= 1
        step["after"] = list(self._queue)

    def _bounce(self, c):
        if self.steps:
            self.steps[-1]["bounced"].append(c)
        super()._bounce(c)

    def _move_to(self, c, position):
        i = self._queue.index(c)
        passed = self._queue[position:i] if position < i else self._queue[i + 1:position + 1]
        if self.steps and position != i:
            self.steps[-1]["moves"].append((c, passed))
        super()._move_to(c, position)


def _step_effect(step, human, played):
    """What one animal did, as a rich-text verb phrase (past tense, no subject), or '' if nothing."""
    actor, a = step["actor"], step["as"]
    if a == "chameleon":   # acting as the animal it copies
        a = step["before"][step["params"][0]][0]
    others = [c for c in step["bounced"] if c != actor]
    passed = [p for c, p in step["moves"] if c == actor]
    passed = passed[-1] if passed else []
    if actor in step["bounced"]:
        if a == "lion":
            return "was bounced (a lion was already in the queue)"
        return "was bounced"
    if a == "lion":   # it bounces the monkeys, then goes to the front (passing nobody if only monkeys were ahead)
        if step["before"].index(actor) == 0:   # (only into an empty queue)
            return "stayed at the front" + (f" and bounced {names(others, human)}" if others else "")
        return (f"bounced {names(others, human)} and " if others else "") + "went to the front"
    if a == "hippo":
        return f"pushed past {names(passed, human)}" if passed else ("could not move" if played else "")
    if a == "crocodile":
        return f"ate {names(others, human)}" if others else ("ate nothing" if played else "")
    if a == "snake":
        return "sorted the queue by strength"
    if a == "giraffe":
        return f"jumped over {names(passed, human)}" if passed else ("did not jump" if played else "")
    if a == "zebra":
        return ""
    if a == "seal":
        return "reversed the queue"
    if a == "monkey":
        if step["before"] == step["after"] and not others:
            return "no other monkey in the queue, nothing happened"
        s = "led the monkeys to the front"
        return s + (f" and bounced {names(others, human)}" if others else "")
    if a == "kangaroo":
        return f"hopped over {names(passed, human)}" if passed else ""
    if a == "parrot":
        return f"bounced {names(others, human)}"
    if a == "skunk":
        return f"bounced {names(others, human)}" if others else "found nothing to bounce"
    return ""


def narrate(before, after, action, human):
    """The public story of `action` played in `before` (giving `after`): {"rich", "text", "checked", "entered",
    "bounced", "queue"}. Uses only public information: the queue, bar and THAT'S IT before and after, and the move.
    The step-by-step story replays the move in bbar.ref (Trace); checked: it reproduced the engine's result (else the
    story only lists what changed)."""
    animal, params = decode(action)
    mover = before.to_move
    who = "You" if mover == human else "AI"
    entered = after.bar()[len(before.bar()):]
    bounced = after.thats_it()[len(before.thats_it()):]
    trace = Trace(before.queue(), mover, animal)
    trace.play(animal, params)
    exact = (trace.queue() == after.queue() and trace.bar() == entered and trace.thats_it() == bounced)
    lines = []
    if exact and trace.steps:
        first = trace.steps[0]
        head = f"{who} played {animal.capitalize()}"
        if animal == "chameleon":
            copied = before.queue()[params[0]][0]
            head += f" as {copied.capitalize()}"
            if copied == "kangaroo" and len(params) > 1:
                head += f" (hop {params[1]})"
        elif animal == "kangaroo" and params:
            head += f" (hop {params[0]})"
        effect = _step_effect(first, human, played=True)
        lines.append(head + (f": {effect}." if effect else "."))
        for step in trace.steps[1:]:
            effect = _step_effect(step, human, played=False)
            if effect:
                lines.append(f"{names([step['actor']], human, start=True)} {effect}.")
        kicked = bounced[-1:] if len(entered) == 2 else []
    else:   # the reference disagrees with the engine (never seen): just say what changed
        lines.append(f"{who} played {animal.capitalize()}.")
        kicked = bounced[-1:] if len(entered) == 2 else []
        rest = bounced[:-1] if kicked else bounced
        if rest:
            lines.append(f"Bounced: {names(rest, human)}.")
    if len(entered) == 2:
        mine = sum(POINTS[a] for a, o in entered if o == human)
        theirs = sum(POINTS[a] for a, o in entered if o != human)
        pts = ", ".join(s for s in (f"+{mine} you" if mine else "", f"+{theirs} AI" if theirs else "") if s)
        lines.append(f"Five animals: {names(entered, human)} enter the bar ({pts}); "
                     f"{names(kicked, human)} is bounced.")
    rich = " ".join(lines)
    return {"rich": rich, "text": plain(rich), "checked": exact, "entered": [card(c, human) for c in entered],
            "bounced": [card(c, human) for c in bounced], "queue": [card(c, human) for c in after.queue()]}


def move_info(game, action, human):
    """A legal move of the human as the page shows it: action, animal, params and their meaning, preview."""
    animal, params = decode(action)
    info = {"action": int(action), "animal": animal, "params": list(params)}
    q = game.queue()
    if animal == "parrot":
        info["target"] = params[0]
    elif animal == "kangaroo":
        info["hop"] = params[0] if params else 0
    elif animal == "chameleon":
        info["copy"] = q[params[0]][0]
        info["copy_at"] = params[0]
        if len(params) > 1:
            info["target" if info["copy"] == "parrot" else "hop"] = params[1]
    after = game.copy()
    after.play_action(int(action))
    story = narrate(game, after, action, human)
    story.pop("checked")
    you, ai = (after.score()[human], after.score()[1 - human])
    info["preview"] = {**story, "score": {"you": int(you), "ai": int(ai)}, "over": after.is_over()}
    return info


def result_of(game, human):
    you, ai = (int(x) for x in (game.score()[human], game.score()[1 - human]))
    return {"you": you, "ai": ai, "winner": "you" if you > ai else "ai" if ai > you else "draw"}


def human_state(game, human, log):
    """What the human may know: public state, their own cards (deck as a set), the AI's unplayed animals as a set,
    the public log. Pure function of the true game and the public log."""
    ai = 1 - human
    seen_ai = {a for a, o in game.queue() + game.bar() + game.thats_it() if o == ai}
    unplayed_ai = by_strength([a for a in ANIMALS if a not in seen_ai])
    ai_deck = len(game.deck(ai))
    over = game.is_over()
    score = game.score()
    return {
        "ply": len(log),
        "to_move": "you" if game.to_move == human else "ai",
        "start": "you" if game.start == human else "ai",
        "you": {"seat": human, "hand": [card((a, human), human) for a in by_strength(game.hand(human))],
                "deck": by_strength(game.deck(human)), "deck_count": len(game.deck(human)),
                "score": int(score[human])},
        "ai": {"unplayed": unplayed_ai, "hand_count": len(unplayed_ai) - ai_deck, "deck_count": ai_deck,
               "score": int(score[ai])},
        "queue": [card(c, human) for c in game.queue()],
        "bar": [card(c, human) for c in game.bar()],
        "thats_it": [card(c, human) for c in game.thats_it()],
        "over": over,
        "result": result_of(game, human) if over else None,
        "log": log,
    }


# ---- the game in play ---------------------------------------------------------------------------------------------

@dataclass
class Table:
    """One game: the true Game, the AI's one-game VecEnv on the same deal, the public log, and the game's own AI
    worker thread and player (its own agent instance over the shared networks), stopped when the game is abandoned."""
    number: int
    deal: Game
    key: int
    human: int
    opponent: Opponent
    env_seed: int
    game: Game = None
    env: VecEnv = None
    player: object = None
    history: list = field(default_factory=list)
    log: list = field(default_factory=list)
    thinks: list = field(default_factory=list)
    turn_since: float = None    # when the AI's current turn began (after loading the AI): its think time so far
    loading: bool = True
    error: str = None
    recorded: bool = False
    worker: ThreadPoolExecutor = None
    abandoned: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self):
        self.game = self.deal.copy()
        if self.game.to_move == self.bot:
            self.turn_since = time.time()
        self.env = VecEnv(1, seed=self.env_seed, threads=1)
        self.env.reset_games([self.deal], np.array([self.key], dtype=np.uint64))
        self.worker = ThreadPoolExecutor(1, thread_name_prefix=f"ai-game{self.number}")

    @property
    def bot(self):
        return 1 - self.human

    def abandon(self):
        """Stops this game's AI: queued work is dropped, running work stops at its next check."""
        self.abandoned.set()
        self.worker.shutdown(wait=False, cancel_futures=True)


def ponder_resolve(agent, start, hist, seat, stop=None):
    """Solves a ResolveAgent (bbar/resolve.py) playing `seat` will need at its next decision, at ply len(hist) + 1, that
    depend only on the public history `hist` (the human to move at ply len(hist)): plain mode's depth-limited solve at
    this decision point of the human's (outside the agent's previous trees: ply P0), and the exact window if it is
    rooted here (ply F). The agent caches its solves by public history and would compute exactly these when it next
    decides (same inputs, one solve at a time), so its moves do not change; it only answers sooner. stop: an Event; once
    set, raises Abandoned between solves."""
    from bbar.tail import PERFECT_INFO_PLY

    def check():
        if stop is not None and stop.is_set():
            raise Abandoned

    t = len(hist)
    if t < agent.F:   # (later, the depth-limited states are released once the exact window is built)
        reqs = [r for r in agent._deps(start, hist + (0,), seat) if len(r[2]) <= t]
        if reqs:
            check()
            agent._ensure(reqs)
    if agent.F <= t + 1 < PERFECT_INFO_PLY and agent.exact_root(start, seat) == t:
        check()
        agent._get_exact(start, hist, seat)


def thinking_about(spec, ply):
    """What the AI `spec` does at ply `ply` (for the page)."""
    if spec.startswith("resolve:"):
        from bbar.resolve import parse_spec
        try:
            o = parse_spec(spec[len("resolve:"):])
        except ValueError:
            return ""
        if ply >= 15:
            return "searching the endgame to the end"
        if ply < min(o["P0"], o["F"]):
            return "its blueprint network"
        if ply < o["F"]:
            return "re-solving the position with its value network"
        return "solving the endgame exactly"
    if spec.startswith("nn:"):
        return "its policy network"
    if spec.startswith("pimc"):
        return "searching guesses of your hand"
    return ""


def derive(seed, *path):
    """A 63-bit integer from (seed, path)."""
    return int(np.random.SeedSequence([seed, *path]).generate_state(1, np.uint64)[0] >> np.uint64(1))


class App:
    """The game server's state: the current table, the loaded players (networks load once, shared by every game)."""

    def __init__(self, opponents, default=None, device="cpu", seat="random", seed=None, record=None, notes=(),
                 ponder=True):
        if not opponents:
            raise ValueError("no opponents")
        self.opponents = {o.key: o for o in opponents}
        self.default = default or opponents[0].key
        self.device = device
        self.seed = int(np.random.SeedSequence().entropy % 2 ** 63) if seed is None else int(seed)
        self.seat = seat
        self.record = record
        self.notes = list(notes)
        self.ponder = ponder
        self.lock = threading.RLock()
        self.loading = threading.Lock()      # one network load at a time (two games never load the same one twice)
        self.players = {}
        self.table = None
        self.games = 0
        self.version = 0                     # grows with every change the page can see
        self.server_id = secrets.token_hex(6)   # this process: a page left open across a restart notices
        self.last_start = None
        seat_rng = np.random.default_rng(derive(self.seed, 1))
        self.human = int(seat_rng.integers(2)) if seat == "random" else int(seat)

    # -- players --

    def player(self, opp):
        """The shared player for `opp`, made on first use (networks load once)."""
        if opp.player is not None:
            return opp.player
        with self.loading:
            with self.lock:
                p = self.players.get(opp.spec)
            if p is None:
                p = match.make_player(opp.spec, self.device)
                with self.lock:
                    self.players[opp.spec] = p
        return p

    @staticmethod
    def table_player(shared, stop):
        """The player of one game: for a Python agent (bbar.match.PyPlayer), a new PyPlayer over the same factory
        (fresh agent instances sharing the loaded networks) whose agents' value functions stop the game's search
        once `stop` is set (Cancellable); built-in specs and networks keep no state between calls and are shared."""
        if not isinstance(shared, match.PyPlayer):
            return shared
        make = shared.make

        def fresh():
            agent = make()
            values = getattr(agent, "values", None)
            if callable(values):
                agent.values = Cancellable(values, stop)
            return agent

        p = match.PyPlayer(fresh, name=shared.name)
        p.prepare(1)
        return p

    def close(self):
        with self.lock:
            if self.table is not None:
                self.table.abandon()

    # -- games --

    def new_game(self, opponent=None, first="alternate", deal=None):
        """Starts a new game (abandoning the current one). deal: a fresh Game to play (tests); otherwise a random
        deal from the session seed. Returns the state."""
        if opponent is not None and not isinstance(opponent, str):
            raise PlayError("opponent must be a string (a key from /api/opponents)")
        if first is None:
            first = "alternate"
        if not isinstance(first, str) or first not in FIRST:
            raise PlayError(f"first must be one of {', '.join(FIRST)}")
        with self.lock:
            key = opponent or (self.table.opponent.key if self.table else self.default)
            if key not in self.opponents:
                raise PlayError(f"unknown opponent {key!r} (have: {', '.join(self.opponents)})")
            n = self.games
            self.games += 1
            if deal is not None:
                start = deal.start
            elif first == "you":
                start = self.human
            elif first == "ai":
                start = 1 - self.human
            else:
                start = 0 if self.last_start is None else 1 - self.last_start
            self.last_start = start
            dkey = derive(self.seed, 2, n)
            if deal is None:
                deal = Game.random(dkey, start)
            if self.table is not None:
                self.table.abandon()
            opp = self.opponents[key]
            self.table = table = Table(n, deal, dkey, self.human, opp, self.seed % 2 ** 63)
            shared = opp.player if opp.player is not None else self.players.get(opp.spec)
            if shared is not None:   # loaded already: no "Loading the AI" (else the worker loads it)
                table.player, table.loading = self.table_player(shared, table.abandoned), False
            self.version += 1
            table.worker.submit(self._ai_turns, table)
        return self.state()

    def play(self, game, ply, action, server=None):
        """The human plays `action` in game number `game` at ply `ply` (the position they chose it in). server: the
        server id the page saw (optional). Raises PlayError if that is not possible."""
        if server is not None and server != self.server_id:
            raise PlayError("the game server was restarted: this is a new game", "restarted")
        for name, v in (("game", game), ("ply", ply), ("action", action)):
            if isinstance(v, bool) or not isinstance(v, int):
                raise PlayError(f"{name} must be an integer, not {json.dumps(v)}")
        with self.lock:
            t = self.table
            if t is None:
                raise PlayError("no game in progress", "no_game")
            if game != t.number:
                raise PlayError(f"game {game} is over or abandoned (the current game is {t.number})", "stale_game")
            if t.game.is_over():
                raise PlayError("the game is over", "over")
            if t.game.to_move != t.human:
                raise PlayError("it is the AI's turn", "not_your_turn")
            if ply != len(t.log):
                raise PlayError(f"the move was chosen at ply {ply} but the game is at ply {len(t.log)}", "stale_ply")
            if action not in t.game.legal_actions():
                raise PlayError(f"illegal move {action}", "illegal")
            self._apply(t, action)
            if not t.game.is_over():
                t.worker.submit(self._ai_turns, t)
        return self.state()

    def _apply(self, t, action, think=None):
        """Plays `action` in the true game and the AI's env (lock held)."""
        before = t.game.copy()
        mover = before.to_move
        done, _, _ = t.env.step(np.array([action], dtype=np.int64))
        t.game.play_action(action)
        if bool(done[0]) != t.game.is_over():
            raise RuntimeError("the AI's env and the true game disagree")
        t.history.append(int(action))
        story = narrate(before, t.game, action, t.human)
        if not story["checked"]:
            print(f"note: the move log's reference trace disagrees with the engine at ply {len(t.log)}", file=sys.stderr)
        animal, params = decode(action)
        entry = {"ply": len(t.log), "who": "you" if mover == t.human else "ai", "action": int(action),
                 "animal": animal, "params": list(params), "rich": story["rich"], "text": story["text"],
                 "entered": story["entered"], "bounced": story["bounced"],
                 "think": None if think is None else round(think, 2)}
        t.log.append(entry)
        if think is not None:
            t.thinks.append({"ply": entry["ply"], "seconds": round(think, 3)})
        t.turn_since = time.time() if t.game.to_move == t.bot and not t.game.is_over() else None
        self.version += 1
        if t.game.is_over():
            self._record(t)

    def _current(self, t):
        """t is the game in play (lock held)."""
        return t is self.table and not t.abandoned.is_set()

    def _ai_turns(self, t):
        """t's worker thread: plays the AI's moves in table t while it is the AI's turn and t is the current game."""
        try:
            if t.player is None:
                player = self.table_player(self.player(t.opponent), t.abandoned)
                with self.lock:
                    t.player, t.loading = player, False
                    if t.turn_since is not None:
                        t.turn_since = time.time()   # loading is not thinking
                    self.version += 1
            while True:
                with self.lock:
                    if not self._current(t) or t.game.is_over() or t.error:
                        return
                    if t.game.to_move != t.bot:
                        break
                action = self._decide(t)
                with self.lock:
                    # From the start of the AI's turn: what the human waited, including any pondering still running.
                    secs = time.time() - (t.turn_since or time.time())
                    if not self._current(t):
                        return
                    if action not in t.game.legal_actions():
                        raise RuntimeError(f"the AI chose illegal action {action}")
                    self._apply(t, action, think=secs)
        except Abandoned:   # a new game started: this one's search was stopped
            return
        except Exception as e:   # shown on the page; the game cannot continue
            if t.abandoned.is_set():
                return
            traceback.print_exc()
            with self.lock:
                t.error = f"{type(e).__name__}: {e}"
                t.turn_since = None
                t.loading = False
                self.version += 1
            return
        if self.ponder:
            self._ponder(t)

    def _ponder(self, t):
        """t's worker, on the human's turn: a re-solving AI precomputes what its next decision needs that depends only
        on the public history so far (ponder_resolve). Never changes a move; errors are only printed."""
        with self.lock:
            if not self._current(t) or t.game.is_over() or t.game.to_move != t.human or t.error:
                return
            hist, start = tuple(t.history), int(t.game.start)
        p = t.player
        if not isinstance(p, match.PyPlayer):
            return
        try:
            from bbar.resolve import ResolveAgent
            agent = p.instance(0)
            if isinstance(agent, ResolveAgent):
                with match.full_fp32_matmuls():
                    ponder_resolve(agent, start, hist, t.bot, stop=t.abandoned)
        except Abandoned:
            pass
        except Exception:
            if not t.abandoned.is_set():
                traceback.print_exc()

    def _decide(self, t):
        """The AI's action, exactly as bbar.match asks a player: built-in specs act inside the env, networks from the
        observation with the seat's draw, Python agents from VecEnv.views (own view + public history)."""
        p = t.player
        obs = mask = None
        if p.needs_obs:
            obs, mask, _, _ = t.env.observe()
        idx = np.array([0])
        with match.full_fp32_matmuls():
            if isinstance(p, match.PyPlayer):
                a = p.act(t.env, idx, obs, mask, streams=np.zeros(1, dtype=np.int64))
            else:
                a = p.act(t.env, idx, obs, mask)
        return int(np.asarray(a)[0])

    def _record(self, t):
        if not self.record or t.recorded:
            return
        t.recorded = True
        d = t.deal
        line = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "opponent": t.opponent.spec, "human_seat": t.human,
                "start": d.start, "hands": [d.hand(0), d.hand(1)], "decks": [d.deck(0), d.deck(1)],
                "moves": t.history, "score": list(t.game.score()), "result": result_of(t.game, t.human)["winner"],
                "think_seconds": t.thinks, "device": str(self.device)}
        try:
            path = Path(self.record)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(line) + "\n")
        except OSError as e:
            print(f"could not record the game to {self.record}: {e}", file=sys.stderr)

    # -- what the human sees --

    def state(self):
        with self.lock:
            t = self.table
            out = {"server": self.server_id, "version": self.version,
                   "opponents": [o.public() for o in self.opponents.values()], "notes": self.notes}
            if t is None:
                return {**out, "game": None, "status": "none"}
            s = human_state(t.game, t.human, [dict(e) for e in t.log])
            if t.error:
                status = "error"
            elif s["over"]:
                status = "over"
            elif s["to_move"] == "you":
                status = "your_turn"
            else:
                status = "loading" if t.loading else "thinking"
            legal = []
            if status == "your_turn":
                legal = [move_info(t.game, a, t.human) for a in t.game.legal_actions()]
            think = round(time.time() - t.turn_since, 1) if status == "thinking" and t.turn_since else None
            about = thinking_about(t.opponent.spec, len(t.log)) if status == "thinking" else ""
            return {**out, **s, "game": t.number, "opponent": t.opponent.public(), "status": status,
                    "thinking_for": think, "thinking_about": about, "legal": legal, "error": t.error}


# ---- HTTP ---------------------------------------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "bbar-play/1"

    def log_message(self, fmt, *args):   # quiet
        pass

    def _send(self, code, body, ctype):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj), "application/json")

    def _host_ok(self):
        """Only requests addressed to this machine by name or loopback address (no DNS rebinding)."""
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        if host in ("127.0.0.1", "localhost", "::1"):
            return True
        self._json(403, {"error": "forbidden host"})
        return False

    def _fail(self):
        """A handler bug: a JSON 500 instead of a dropped connection (the server keeps serving)."""
        traceback.print_exc()
        e = sys.exc_info()[1]
        try:
            self._json(500, {"error": f"server error: {type(e).__name__}: {e}", "code": "server"})
        except OSError:
            pass

    def do_GET(self):
        try:
            self._get()
        except Exception:
            self._fail()

    def do_POST(self):
        try:
            self._post()
        except Exception:
            self._fail()

    def _get(self):
        if not self._host_ok():
            return
        app = self.server.app
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._json(200, app.state())
        elif path == "/api/legal":
            self._json(200, app.state()["legal"])
        elif path == "/api/opponents":
            self._json(200, [o.public() for o in app.opponents.values()])
        else:
            self._json(404, {"error": "not found"})

    def _post(self):
        if not self._host_ok():
            return
        app = self.server.app
        path = self.path.split("?", 1)[0]
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            return self._json(415, {"error": "send application/json"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if 0 <= n <= 1 << 16 else None
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return self._json(400, {"error": "bad JSON body"})
        try:
            if path == "/api/new":
                self._json(200, app.new_game(body.get("opponent"), body.get("first")))
            elif path == "/api/play":
                if not all(k in body for k in ("game", "ply", "action")):
                    raise PlayError("expected {\"game\": N, \"ply\": P, \"action\": A}")
                self._json(200, app.play(body["game"], body["ply"], body["action"], body.get("server")))
            else:
                self._json(404, {"error": "not found"})
        except PlayError as e:
            self._json(400, {"error": str(e), "code": e.code})


def make_server(app, port=PORT, host="127.0.0.1"):
    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    server.app = app
    return server


def say(msg):
    print(msg, flush=True)


def build_app(opponent=None, device="cpu", seat="random", seed=None, record=RECORD, log=say, ponder=True):
    """The App for the CLI: the catalogue (plus --opponent if it is a spec not in it), the default opponent loaded."""
    opponents, missing = catalogue()
    notes = list(missing.values())
    for m in notes:
        log(f"note: {m}")
    default = opponents[0].key
    if opponent:
        hit = [o for o in opponents if opponent in (o.key, o.spec)]
        if hit:
            default = hit[0].key
        elif opponent in missing:   # a catalogue key whose checkpoints are not here
            raise ValueError(f"cannot play {opponent}: {missing[opponent]} (available here: "
                             f"{', '.join(o.key for o in opponents)})")
        else:
            kind, body, _ = match.parse_agent(opponent)
            if kind == "nn" and not Path(body).is_file():
                raise ValueError(f"{opponent}: no checkpoint file {body}")
            opponents.insert(0, Opponent("custom", opponent if len(opponent) < 40 else opponent[:37] + "...",
                                         opponent, "Given with --opponent.", short="given with --opponent"))
            default = "custom"
    app = App(opponents, default, device, seat, seed, record, notes, ponder)
    opp = app.opponents[default]
    log(f"loading the AI: {opp.label} [{opp.spec}] on {device} ...")
    t0 = time.time()
    try:
        app.player(opp)
    except (ValueError, OSError, RuntimeError) as e:
        raise ValueError(f"cannot load {opp.spec}: {e}") from e
    log(f"loaded in {time.time() - t0:.1f}s")
    return app


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bbar.play", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opponent", help="AI spec (as bbar.match: pimc:..., nn:PATH, resolve:..., tail:...) or a key: "
                                       "resolve, blueprint, pimc8, pimc4, greedy, random (default: the strongest)")
    ap.add_argument("--seat", default="random", choices=("0", "1", "random"), help="your seat (default random)")
    ap.add_argument("--seed", type=int, help="session seed: deals and the AI's randomness (default: random)")
    ap.add_argument("--port", type=int, default=PORT, help=f"port on 127.0.0.1 (default {PORT})")
    ap.add_argument("--device", default="cpu", help="network device: cpu (default; fastest for this on an M4), mps, "
                                                     "cuda, auto")
    ap.add_argument("--open", action="store_true", help="open the page in the browser")
    ap.add_argument("--record", default=RECORD, help=f"append finished games here (default {RECORD})")
    ap.add_argument("--no-record", action="store_true", help="do not record games")
    ap.add_argument("--no-ponder", action="store_true",
                    help="the re-solving AI does not precompute on your turn (same moves, slower answers)")
    args = ap.parse_args(argv)
    if args.seed is not None and not 0 <= args.seed < 2 ** 63:
        ap.error("--seed must be in 0 .. 2^63 - 1")
    device = str(match.resolve_device(args.device)) if args.device == "auto" else args.device
    try:
        app = build_app(args.opponent, device, args.seat, args.seed, None if args.no_record else args.record,
                        ponder=not args.no_ponder)
    except ValueError as e:
        ap.error(str(e))
    app.new_game()
    try:
        server = make_server(app, args.port)
    except OSError as e:
        ap.error(f"cannot listen on 127.0.0.1:{args.port}: {e} (another bbar.play running? try --port)")
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Beasty Bar: play at {url}  (Ctrl-C to stop)", flush=True)
    if args.open:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("bye")
    finally:
        server.server_close()
        app.close()


if __name__ == "__main__":
    main()
