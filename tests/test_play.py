"""bbar.play: full games through the HTTP API, what the human may see, refused requests, the AI's code path."""
import http.client
import json
import threading
import time

import numpy as np
import pytest

from bbar import match
from bbar._core import Game, VecEnv
from bbar.actions import decode, encode
from bbar import play
from bbar.play import BLUEPRINT, Abandoned, App, Opponent, find, make_server, narrate, plain

CHEAP = "pimc:w=4,d=2"


class Client:
    def __init__(self, port):
        self.port = port

    def request(self, method, path, body=None, headers=None, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        hdrs = {"Content-Type": "application/json"} if body is not None or raw is not None else {}
        hdrs.update(headers or {})
        data = raw if raw is not None else (None if body is None else json.dumps(body))
        conn.request(method, path, body=data, headers=hdrs)
        r = conn.getresponse()
        payload = r.read()
        conn.close()
        ctype = r.getheader("Content-Type") or ""
        return r.status, (json.loads(payload) if "json" in ctype else payload.decode())

    def get(self, path):
        code, data = self.request("GET", path)
        assert code == 200, data
        return data

    def post(self, path, body):
        return self.request("POST", path, body)

    def turn(self, timeout=60):
        """The state once the AI is done: the human's turn, game over or an error."""
        t0 = time.time()
        while True:
            s = self.get("/api/state")
            if s["status"] in ("your_turn", "over", "error"):
                return s
            assert time.time() - t0 < timeout, "the AI took too long"
            time.sleep(0.01)


@pytest.fixture
def serve():
    running = []

    def start(app):
        srv = make_server(app, port=0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        running.append((srv, app))
        return Client(srv.server_address[1])

    yield start
    for srv, app in running:
        srv.shutdown()
        srv.server_close()
        app.close()


def check_state(s):
    """Invariants of every human state."""
    you, ai = s["you"], s["ai"]
    assert len(you["hand"]) <= 4 and ai["hand_count"] <= 4
    assert you["deck"] == sorted(you["deck"], key=ANIMAL_ORDER.index), "the deck as a set, by strength"
    assert you["deck_count"] == len(you["deck"])
    mine = [c for c in s["queue"] + s["bar"] + s["thats_it"] if c["owner"] == "you"]
    assert len(mine) + len(you["hand"]) + you["deck_count"] == 12
    theirs = [c["animal"] for c in s["queue"] + s["bar"] + s["thats_it"] if c["owner"] == "ai"]
    assert sorted(theirs + ai["unplayed"]) == sorted(ANIMAL_ORDER)
    assert ai["hand_count"] + ai["deck_count"] == len(ai["unplayed"])
    for side in ("you", "ai"):
        assert s[side]["score"] == sum(c["points"] for c in s["bar"] if c["owner"] == side)
    assert len(s["log"]) == s["ply"]
    # Each move's log entry names what it sent to the bar and to THAT'S IT (the page marks the last move's).
    assert [c for e in s["log"] for c in e["entered"]] == s["bar"]
    assert [c for e in s["log"] for c in e["bounced"]] == s["thats_it"]


ANIMAL_ORDER = ["skunk", "parrot", "kangaroo", "monkey", "chameleon", "seal", "zebra", "giraffe", "snake", "crocodile",
                "hippo", "lion"]


def random_game(c, rng, first="alternate", new=True):
    s = c.post("/api/new", {"first": first})[1] if new else c.get("/api/state")
    plies = 0
    while True:
        s = c.turn()
        check_state(s)
        assert s["status"] != "error", s["error"]
        if s["status"] == "over":
            return s
        assert s["legal"] and s["to_move"] == "you"
        m = s["legal"][rng.integers(len(s["legal"]))]
        code, s = c.post("/api/play", {"game": s["game"], "ply": s["ply"], "action": m["action"]})
        assert code == 200, s
        assert s["log"][-1]["who"] == "you" and s["log"][-1]["action"] == m["action"]
        plies += 1
        assert plies < 30


@pytest.mark.parametrize("seat", [0, 1])
def test_full_games_against_a_cheap_bot(serve, seat):
    app = App([Opponent("pimc", "PIMC", CHEAP)], seat=str(seat), seed=11 + seat, record=None)
    c = serve(app)
    rng = np.random.default_rng(seat)
    starts = []
    for g in range(3):
        s = random_game(c, rng)
        starts.append(s["start"])
        assert s["over"] and s["status"] == "over" and s["legal"] == []
        r = s["result"]
        assert (r["you"], r["ai"]) == (s["you"]["score"], s["ai"]["score"])
        assert r["winner"] == ("you" if r["you"] > r["ai"] else "ai" if r["ai"] > r["you"] else "draw")
        # The game ends as soon as fewer than 5 cards remain in hands, decks and queue.
        left = len(s["you"]["hand"]) + s["you"]["deck_count"] + s["ai"]["hand_count"] + s["ai"]["deck_count"]
        assert left + len(s["queue"]) < 5
        # Players alternate; every AI move has a think time and a story.
        whos = [e["who"] for e in s["log"]]
        assert whos[0] == s["start"] and all(a != b for a, b in zip(whos, whos[1:]))
        assert all((e["think"] is not None) == (e["who"] == "ai") for e in s["log"])
        assert all(e["text"] and "{{" not in e["text"] for e in s["log"])
        assert s["you"]["seat"] == seat
        # The finished game refuses further moves.
        code, err = c.post("/api/play", {"game": s["game"], "ply": s["ply"], "action": 0})
        assert code == 400 and "over" in err["error"]
    assert starts[:3] in (["you", "ai", "you"], ["ai", "you", "ai"]), "the starting player alternates"


def test_blueprint_network_game(serve):
    path = find(BLUEPRINT)
    if path is None:
        pytest.skip(f"{BLUEPRINT} not available")
    app = App([Opponent("bp", "Blueprint", f"nn:{path}")], seat="1", seed=3, record=None)
    c = serve(app)
    s = random_game(c, np.random.default_rng(3), first="ai")
    assert s["start"] == "ai" and s["status"] == "over"


def test_ai_acts_through_the_env_with_its_seat_rng():
    """The AI's moves are what bbar.match's code path gives: replaying the human's moves in a fresh one-game VecEnv
    with the same deal and seat-RNG key, the built-in agent makes the same decisions."""
    app = App([Opponent("pimc", "PIMC", CHEAP)], seat="0", seed=5, record=None)
    try:
        rng = np.random.default_rng(0)
        for first in ("you", "ai"):
            app.new_game(first=first)
            while True:
                s = app.state()
                if s["status"] == "over":
                    break
                if s["status"] == "your_turn":
                    t = app.table
                    hist = [(p, a) for p, a in t.env.history(0)]
                    assert [a for _, a in hist] == t.history   # the env holds the same game
                    app.play(s["game"], s["ply"], s["legal"][rng.integers(len(s["legal"]))]["action"])
                time.sleep(0.002)
            t = app.table
            env = VecEnv(1, seed=app.seed % 2 ** 63, threads=1)
            env.reset_games([t.deal], np.array([t.key], dtype=np.uint64))
            player = match.make_player(CHEAP)
            for e in s["log"]:
                if e["who"] == "ai":
                    assert int(player.act(env, np.array([0]), None, None)[0]) == e["action"]
                env.step(np.array([e["action"]]))
    finally:
        app.close()


class ScriptAgent:
    """A Python agent (bbar.match.PyPlayer) playing a fixed list of actions, one per ply; `gate`: an Event it waits
    for before answering."""
    name = "script"
    checkpoints = {}
    needs_obs = False

    def __init__(self, script, gate=None):
        self.script, self.gate = script, gate

    def act(self, batch):
        if self.gate is not None:
            assert self.gate.wait(30)
        return np.array([self.script[len(h)] for h in batch.histories], dtype=np.int64)


def lockstep_script(a, b, rng):
    """Actions legal in both games at every ply until the end, or None."""
    a, b = a.copy(), b.copy()
    moves = []
    while not a.is_over():
        common = sorted(set(a.legal_actions()) & set(b.legal_actions()))
        if not common:
            return None
        x = int(rng.choice(common))
        a.play_action(x)
        b.play_action(x)
        moves.append(x)
        assert a.is_over() == b.is_over()
    return moves


def strip(s):
    """A state without timing fields."""
    s = json.loads(json.dumps(s))
    for k in ("server", "version", "thinking_for"):
        s.pop(k, None)
    for e in s["log"]:
        e.pop("think")
    return s


@pytest.mark.parametrize("start", [0, 1])
def test_state_never_shows_the_ai_cards(serve, start):
    """Two games that differ only in the AI's hidden cards (its hand and deck order), played with the same public
    moves, give the human identical states at every turn (and so do the API's legal moves and previews)."""
    rng = np.random.default_rng(100 + start)
    for _ in range(500):
        base = Game.random(int(rng.integers(2 ** 62)), start)
        mine = (base.hand(0), base.deck(0))
        ai = base.hand(1) + base.deck(1)
        ai2 = [ai[i] for i in rng.permutation(12)]
        ga = Game(start, [mine[0], base.hand(1)], [mine[1], base.deck(1)])
        gb = Game(start, [mine[0], ai2[:4]], [mine[1], ai2[4:]])
        if sorted(ai2[:4]) == sorted(base.hand(1)):
            continue
        script = lockstep_script(ga, gb, rng)
        if script is not None:
            break
    else:
        pytest.fail("no scriptable pair of deals found")
    clients = []
    for deal in (ga, gb):
        bot = match.PyPlayer(lambda: ScriptAgent(script))
        app = App([Opponent("script", "Script", "script", player=bot)], seat="0", seed=9, record=None)
        app.new_game(deal=deal)
        clients.append(serve(app))
    ca, cb = clients
    seen = 0
    while True:
        sa, sb = ca.turn(), cb.turn()
        assert sa["status"] != "error", sa["error"]
        assert strip(sa) == strip(sb)
        seen += 1
        if sa["status"] == "over":
            break
        x = script[sa["ply"]]
        assert x in [m["action"] for m in sa["legal"]]
        for c in (ca, cb):
            assert c.post("/api/play", {"game": sa["game"], "ply": sa["ply"], "action": x})[0] == 200
    assert seen >= 8
    # The hidden cards really differed, and the human never saw the AI's hand in either game.
    assert ga.hand(1) != gb.hand(1) or ga.deck(1) != gb.deck(1)


def test_state_hides_the_humans_own_deck_order(serve):
    """Before the human's first draw, two deals differing only in the order of the human's deck look the same."""
    base = Game.random(77, 1)
    deck = base.deck(0)
    other = deck[::-1]
    states = []
    for d in (deck, other):
        g = Game(1, [base.hand(0), base.hand(1)], [d, base.deck(1)])
        script = [base.legal_actions()[0]] + [0] * 30
        bot = match.PyPlayer(lambda: ScriptAgent(script))
        app = App([Opponent("script", "Script", "script", player=bot)], seat="0", seed=9, record=None)
        app.new_game(deal=g)
        states.append(strip(serve(app).turn()))
    assert states[0] == states[1]
    assert states[0]["you"]["deck"] == sorted(deck, key=ANIMAL_ORDER.index)


def test_refused_requests(serve):
    gate = threading.Event()
    script = []
    bot = match.PyPlayer(lambda: ScriptAgent(script, gate))
    app = App([Opponent("script", "Script", "script", player=bot), Opponent("pimc", "PIMC", CHEAP)], seat="0",
              seed=4, record=None)
    deal = Game.random(123, 0)
    app.new_game(deal=deal)
    c = serve(app)
    s = c.turn()
    assert s["status"] == "your_turn" and s["game"] == 0
    legal = [m["action"] for m in s["legal"]]
    assert legal == deal.legal_actions()
    illegal = next(a for a in range(36) if a not in legal)

    def refused(body, code=400):
        got, data = c.post("/api/play", body)
        assert got == code, data
        assert c.get("/api/state")["log"] == [], "a refused move changes nothing"
        return data["error"]

    assert "illegal" in refused({"game": 0, "ply": 0, "action": illegal})
    assert "illegal" in refused({"game": 0, "ply": 0, "action": 99})
    assert "game" in refused({"game": 5, "ply": 0, "action": legal[0]})
    assert "integer" in refused({"game": 0, "ply": 0, "action": "lion"})
    assert "expected" in refused({"game": 0, "action": legal[0]})
    assert "expected" in refused({"action": legal[0]})
    assert c.request("POST", "/api/play", raw="not json")[0] == 400
    assert c.request("POST", "/api/play", raw="[1, 2]")[0] == 400
    assert c.request("POST", "/api/play", raw=json.dumps({"game": 0, "action": legal[0]}),
                     headers={"Content-Type": "text/plain"})[0] == 415
    assert c.request("GET", "/api/state", headers={"Host": "evil.example:8765"})[0] == 403
    assert c.request("GET", "/nope")[0] == 404
    assert c.post("/api/new", {"opponent": "nobody"})[0] == 400
    assert c.post("/api/new", {"first": "sometimes"})[0] == 400

    # The human moves; the AI (blocked on the gate) is thinking: moving now is refused.
    after = deal.copy()
    after.play_action(legal[0])
    script += [legal[0]] + [after.legal_actions()[0]]
    code, s = c.post("/api/play", {"game": 0, "ply": 0, "action": legal[0]})
    assert code == 200 and s["status"] in ("thinking", "loading")
    got, data = c.post("/api/play", {"game": 0, "ply": 1, "action": legal[0]})
    assert got == 400 and "AI's turn" in data["error"]
    gate.set()
    s = c.turn()
    assert s["status"] == "your_turn" and len(s["log"]) == 2

    # A new game abandons the old one: its game number is refused.
    s = c.post("/api/new", {"opponent": "pimc", "first": "you"})[1]
    assert s["game"] == 1 and s["opponent"]["key"] == "pimc"
    s = c.turn()
    got, data = c.post("/api/play", {"game": 0, "ply": 0, "action": s["legal"][0]["action"]})
    assert got == 400 and "abandoned" in data["error"] and data["code"] == "stale_game"
    assert c.post("/api/play", {"game": 1, "ply": 0, "action": s["legal"][0]["action"]})[0] == 200


def test_page_and_apis(serve):
    app = App([Opponent("pimc", "PIMC", CHEAP)], seat="0", seed=1, record=None)
    c = serve(app)
    code, page = c.request("GET", "/")
    assert code == 200 and "Heaven's Gate" in page and "That's it" in page
    assert "http://" not in page.replace("http://www.w3.org", ""), "self-contained: no external resources"
    s = c.get("/api/state")
    assert s["game"] is None and s["status"] == "none"
    s = c.post("/api/new", {"first": "you"})[1]
    assert c.get("/api/legal") == c.turn()["legal"]
    assert [o["key"] for o in c.get("/api/opponents")] == ["pimc"]


def test_narration_matches_the_engine():
    """The move log's step-by-step story (bbar.ref traced on the public queue) agrees with the engine's result on
    every move of random games, and names only cards it may."""
    rng = np.random.default_rng(0)
    seen = set()
    for g in range(400):
        game = Game.random(int(rng.integers(2 ** 62)), g % 2)
        while not game.is_over():
            a = int(rng.choice(game.legal_actions()))
            before = game.copy()
            game.play_action(a)
            story = narrate(before, game, a, 0)
            assert story["checked"], story["text"]
            assert story["text"] == plain(story["rich"]) and "{{" not in story["text"]
            seen.add(decode(a)[0])
            if len(story["entered"]) == 2:
                assert "Five animals" in story["text"]
    assert len(seen) >= 12, seen


def test_narration_example():
    """You: lion (to the front); AI: seal (reverses); you: hippo (stops behind the lion); AI: skunk."""
    hands = [["lion", "hippo", "skunk", "zebra"], ["seal", "skunk", "monkey", "parrot"]]
    decks = [[a for a in ANIMAL_ORDER if a not in h] for h in hands]
    g = Game(0, hands, decks)
    for a in ("lion", "seal", "hippo"):
        g.play(a)
    assert g.queue() == [("seal", 1), ("lion", 0), ("hippo", 0)]
    before = g.copy()
    g.play("skunk")
    story = narrate(before, g, encode("skunk", ()), 0)
    assert story["text"] == "AI played Skunk: bounced your Lion and Hippo."
    assert narrate(before, g, encode("skunk", ()), 1)["text"] == "You played Skunk: bounced AI's Lion and Hippo."


def test_pondering_changes_no_move():
    """The re-solving AI precomputing on the human's turn (App ponder) plays exactly the same moves."""
    from bbar.play import VALUE
    bp, v = find(BLUEPRINT), find(VALUE)
    if bp is None or v is None:
        pytest.skip("checkpoints not available")
    spec = f"resolve:V={v},base=nn:{bp},P0=8,F=10,d=2,iters=20,mode=plain"
    player = match.make_player(spec)
    runs = []
    for ponder in (False, True):
        app = App([Opponent("r", "R", spec, player=player)], seat="0", seed=21, record=None, ponder=ponder)
        try:
            rng = np.random.default_rng(4)
            app.new_game(first="you")
            while True:
                s = app.state()
                if s["status"] == "over":
                    break
                assert s["status"] != "error", s["error"]
                if s["status"] == "your_turn":
                    if s["ply"] in (8, 10):     # the human thinks: let the AI ponder
                        app.table.worker.submit(lambda: None).result(timeout=60)
                        agent = app.table.player.instances[0]
                        hist = tuple(e["action"] for e in s["log"])
                        start = app.table.game.start      # the human (seat 0) started
                        if s["ply"] == 8:
                            assert (("opp", start, hist) in agent._dl) == ponder
                        else:
                            assert ((start, hist, 1) in agent._exact) == ponder
                    app.play(s["game"], s["ply"], s["legal"][rng.integers(len(s["legal"]))]["action"])
                time.sleep(0.002)
            solves = sorted((x["kind"], x["ply"]) for x in app.table.player.instances[0].log)
            runs.append(([e["action"] for e in s["log"]], solves))
        finally:
            app.close()
    assert runs[0] == runs[1]
    assert ("opp", 8) in runs[1][1] and ("exact", 10) in runs[1][1]


def test_moves_from_an_old_board_or_another_server_are_refused(serve):
    """A move carries the ply it was chosen at (and the server id the page saw): a move chosen on an older board (a
    stale poll, another tab) or sent to a restarted server is refused and changes nothing."""
    script = []
    bot = match.PyPlayer(lambda: ScriptAgent(script))
    app = App([Opponent("script", "Script", "script", player=bot)], seat="0", seed=4, record=None)
    deal = Game.random(321, 0)
    app.new_game(deal=deal)
    c = serve(app)
    s = c.turn()
    assert s["server"] == app.server_id and len(s["server"]) >= 8
    a = s["legal"][0]["action"]
    after = deal.copy()
    after.play_action(a)
    script += [a, after.legal_actions()[0]]
    for body, code in (({"game": 0, "ply": 1, "action": a}, "stale_ply"),
                       ({"game": 0, "ply": 0, "action": a, "server": "0123456789ab"}, "restarted"),
                       ({"game": 0, "ply": 0, "action": a, "server": 5}, "restarted")):
        got, data = c.post("/api/play", body)
        assert got == 400 and data["code"] == code, data
        assert c.get("/api/state")["log"] == []
    v0 = c.get("/api/state")["version"]
    got, s = c.post("/api/play", {"game": 0, "ply": 0, "action": a, "server": app.server_id})
    assert got == 200 and s["version"] > v0
    s = c.turn()
    assert s["ply"] == 2
    # The human's next move chosen on the board of ply 0 (e.g. an old poll answer) is refused.
    got, data = c.post("/api/play", {"game": 0, "ply": 0, "action": s["legal"][0]["action"]})
    assert got == 400 and data["code"] == "stale_ply" and c.get("/api/state")["ply"] == 2
    # Another process has another id.
    assert App([Opponent("pimc", "PIMC", CHEAP)], record=None).server_id != app.server_id


def test_malformed_requests_get_a_json_answer(serve):
    """Wrong types are refused with 400 (never converted, never a dropped connection); a handler bug answers 500."""
    app = App([Opponent("pimc", "PIMC", CHEAP)], seat="0", seed=4, record=None)
    c = serve(app)
    c.post("/api/new", {"first": "you"})
    s = c.turn()
    a = s["legal"][0]["action"]
    for raw in ('{"opponent": ["pimc"]}', '{"opponent": {"k": 1}}', '{"opponent": 3}', '{"first": 3}',
                '{"first": ["you"]}'):
        got, data = c.request("POST", "/api/new", raw=raw)
        assert got == 400 and data["error"], (raw, data)
    assert c.get("/api/state")["game"] == 0, "nothing started"
    for raw in ('{"game": 0, "ply": 0, "action": Infinity}', '{"game": 1e999, "ply": 0, "action": 0}',
                f'{{"game": 0.4, "ply": 0, "action": "{a}"}}', f'{{"game": 0, "ply": 0, "action": {a}.0}}',
                '{"game": 0, "ply": 0, "action": true}', f'{{"game": false, "ply": 0, "action": {a}}}',
                f'{{"game": 0, "ply": null, "action": {a}}}', f'{{"game": 0, "ply": [0], "action": {a}}}',
                '[]', '"x"', '{"game": 0'):
        got, data = c.request("POST", "/api/play", raw=raw)
        assert got == 400 and data["error"], (raw, data)
        assert c.get("/api/state")["log"] == [], raw
    got, data = c.request("POST", "/api/play", raw="{}", headers={"Content-Length": "-1"})
    assert got == 400
    # A bug in a handler: JSON 500, and the server keeps serving.
    real = app.state
    app.state = lambda: 1 / 0
    try:
        got, data = c.request("GET", "/api/state")
        assert got == 500 and "ZeroDivisionError" in data["error"]
        got, data = c.post("/api/play", {"game": 0, "ply": 0, "action": a})
        assert got == 500 and data["code"] == "server"
    finally:
        app.state = real
    assert c.turn()["log"][0]["action"] == a, "the move itself was played before state() failed"


class SlowAgent:
    """A Python agent whose search calls its value function until it is stopped (or 30 s pass); `running` is set while
    it searches, `stopped` once it has stopped."""
    name = "slow"
    checkpoints = {}
    needs_obs = False

    def __init__(self, running, stopped):
        self.running, self.stopped = running, stopped
        self.values = lambda: 0.0

    def act(self, batch):
        self.running.set()
        try:
            t0 = time.time()
            while time.time() - t0 < 30:
                self.values()
                time.sleep(0.002)
            return np.array([g.legal_actions()[0] for g in batch.games], dtype=np.int64)
        finally:
            self.stopped.set()


def test_a_new_game_does_not_wait_for_the_abandoned_search(serve):
    """Every game has its own AI thread and agent instance: starting a new game while the AI searches stops that
    search (at its next value-function call) and the new game's AI answers at once, its think time counted."""
    running, stopped = threading.Event(), threading.Event()
    slow = match.PyPlayer(lambda: SlowAgent(running, stopped))
    app = App([Opponent("slow", "Slow", "slow", player=slow), Opponent("pimc", "PIMC", CHEAP)], seat="0", seed=8,
              record=None)
    c = serve(app)
    s = c.post("/api/new", {"opponent": "slow", "first": "you"})[1]
    s = c.turn()
    old = app.table
    assert c.post("/api/play", {"game": 0, "ply": 0, "action": s["legal"][0]["action"]})[0] == 200
    assert running.wait(10) and not stopped.is_set()
    assert c.get("/api/state")["status"] == "thinking"
    t0 = time.time()
    s = c.post("/api/new", {"opponent": "pimc", "first": "ai"})[1]
    assert s["game"] == 1
    statuses = set()
    while True:
        s = c.get("/api/state")
        statuses.add(s["status"])
        if s["status"] == "your_turn":
            break
        assert time.time() - t0 < 10, "the new game waited for the abandoned one"
        time.sleep(0.005)
    assert time.time() - t0 < 5 and statuses <= {"thinking", "loading", "your_turn"}
    assert stopped.wait(5), "the abandoned search was not stopped"
    assert old.abandoned.is_set() and old.error is None and len(old.log) == 1, "the old game is left as it was"
    assert s["log"][0]["who"] == "ai" and s["log"][0]["think"] < 5

    # The same opponent in a new game: a new agent instance (a new PyPlayer over the shared factory).
    running.clear(), stopped.clear()
    s = c.post("/api/new", {"opponent": "slow", "first": "ai"})[1]
    assert running.wait(10)
    first = app.table.player
    assert first is not slow and first.instances[0].stopped is stopped
    s = c.post("/api/new", {"opponent": "slow", "first": "ai"})[1]
    assert stopped.wait(5)
    deadline = time.time() + 10
    while app.table.player is None or 0 not in app.table.player.instances:
        assert time.time() < deadline
        time.sleep(0.005)
    assert app.table.player is not first and app.table.player.instances[0] is not first.instances[0]


def test_cancellable_values():
    stop = threading.Event()

    class V:
        path, needs_games = "v.pt", True

        def __call__(self, x, y=1):
            return x + y

    v = play.Cancellable(V(), stop)
    assert v(1, y=2) == 3 and v.path == "v.pt" and v.needs_games
    stop.set()
    with pytest.raises(Abandoned):
        v(1)


def test_narration_lion():
    """A lion behind monkeys only bounces them, then goes to the front (it passes nobody); into an empty queue it
    stays at the front."""
    hands = [["lion", "hippo", "skunk", "zebra"], ["monkey", "seal", "parrot", "skunk"]]
    decks = [[a for a in ANIMAL_ORDER if a not in h] for h in hands]
    g = Game(1, hands, decks)
    g.play("monkey")
    before = g.copy()
    g.play("lion")
    assert g.queue() == [("lion", 0)] and g.thats_it() == [("monkey", 1)]
    lion = encode("lion", ())
    assert narrate(before, g, lion, 0)["text"] == "You played Lion: bounced AI's Monkey and went to the front."
    g = Game(0, hands, decks)
    before = g.copy()
    g.play("lion")
    assert narrate(before, g, lion, 0)["text"] == "You played Lion: stayed at the front."
    g2 = Game(1, [["hippo", "lion", "skunk", "zebra"], ["seal", "monkey", "parrot", "skunk"]],
              [[a for a in ANIMAL_ORDER if a not in ("hippo", "lion", "skunk", "zebra")],
               [a for a in ANIMAL_ORDER if a not in ("seal", "monkey", "parrot", "skunk")]])
    g2.play("seal")
    before = g2.copy()
    g2.play("lion")
    assert narrate(before, g2, lion, 0)["text"] == "You played Lion: went to the front."


def test_cli_missing_checkpoints_for_a_catalogue_key(monkeypatch, capsys, tmp_path):
    """--opponent resolve (or blueprint) without its checkpoints says which file is missing, not 'unknown agent'."""
    monkeypatch.setattr(play, "VALUE", str(tmp_path / "missing.pt"))
    with pytest.raises(SystemExit) as e:
        play.main(["--opponent", "resolve", "--no-record", "--port", "0"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "re-solving player needs" in err and "missing.pt" in err and "unknown agent" not in err
    monkeypatch.setattr(play, "BLUEPRINT", str(tmp_path / "missing-bp.pt"))
    with pytest.raises(SystemExit) as e:
        play.main(["--opponent", "blueprint", "--no-record", "--port", "0"])
    assert e.value.code == 2 and "blueprint network needs" in capsys.readouterr().err


def test_abandoning_the_resolver_mid_search():
    """The re-solving AI (default settings) abandoned while it re-solves (ply 9): the search stops at once, the new
    game's first AI move (a blueprint move) is not delayed by it, and no 'Loading the AI' is shown."""
    from bbar.play import VALUE
    bp, v = find(BLUEPRINT), find(VALUE)
    if bp is None or v is None:
        pytest.skip("checkpoints not available")
    spec = f"resolve:V={v},base=nn:{bp},P0=8,F=10,d=2,iters=200,mode=plain"
    app = App([Opponent("r", "R", spec, player=match.make_player(spec))], seat="0", seed=2026, record=None)
    try:
        app.new_game(first="you")
        rng = np.random.default_rng(1)
        while True:
            s = app.state()
            assert s["status"] != "error", s["error"]
            if s["status"] == "your_turn":
                app.play(s["game"], s["ply"], s["legal"][rng.integers(len(s["legal"]))]["action"])
                if s["ply"] == 8:
                    break
            time.sleep(0.002)
        time.sleep(0.5)
        s = app.state()
        assert s["status"] == "thinking" and s["ply"] == 9, "the AI is re-solving"
        old = app.table
        t0 = time.time()
        app.new_game(first="ai")
        seen = set()
        while (s := app.state())["status"] != "your_turn":
            seen.add(s["status"])
            assert time.time() - t0 < 5, "the new game waits for the abandoned search"
            time.sleep(0.005)
        assert "loading" not in seen
        old.worker.shutdown(wait=True)
        assert time.time() - t0 < 5, "the abandoned search was not stopped"
        assert len(old.log) == 9 and old.error is None
        assert not any(x.get("ply") == 9 for x in old.player.instances[0].log), "the ply-9 solve was cut short"
    finally:
        app.close()
