"""Pure-Python reference engine, written to be read side by side with RULES.md.

It is the independent cross-check of the fast engine (bbar._core), so it favours clarity over speed.
A card is (animal, owner). Each player owns exactly one of each animal, so a card is unique and is
tracked by value. The queue is a list of cards, front (bar side) first.
"""
from bbar.actions import POINTS, STRENGTH, decode, encode

RECURRING = ("hippo", "crocodile", "giraffe")


class Game:
    def __init__(self, start, hands, decks):
        self.to_move = start
        self._hands = [list(h) for h in hands]
        self._decks = [list(d) for d in decks]  # top card first
        self._queue = []
        self._bar = []
        self._thats_it = []

    def copy(self):
        game = Game(self.to_move, self._hands, self._decks)
        game._queue = list(self._queue)
        game._bar = list(self._bar)
        game._thats_it = list(self._thats_it)
        return game

    # ---- state ------------------------------------------------------------------------------------

    def queue(self):
        return list(self._queue)

    def bar(self):
        return list(self._bar)

    def thats_it(self):
        return list(self._thats_it)

    def hand(self, player):
        return list(self._hands[player])

    def deck(self, player):
        return list(self._decks[player])

    def score(self):
        points = [0, 0]
        for animal, owner in self._bar:
            points[owner] += POINTS[animal]
        return tuple(points)

    def is_over(self):
        """Fewer than 5 cards left in hands, decks and queue: the queue can no longer fill."""
        cards = sum(map(len, self._hands)) + sum(map(len, self._decks)) + len(self._queue)
        return cards < 5

    # ---- moves ------------------------------------------------------------------------------------

    def legal_moves(self):
        """[(animal, params)] for the player to move, ordered by action index."""
        if self.is_over():
            return []
        moves = []
        for animal in self._hands[self.to_move]:
            if animal == "chameleon":
                # Copies any non-chameleon in the queue and takes that animal's parameter.
                for t, (target, _) in enumerate(self._queue):
                    if target != "chameleon":
                        moves += [(animal, (t, *p)) for p in self._params(target)]
            else:
                moves += [(animal, p) for p in self._params(animal)]
        return sorted(moves, key=lambda move: encode(*move))

    def _params(self, animal):
        """Parameter choices for `animal` played onto the current queue (positions before the move)."""
        n = len(self._queue)
        if animal == "parrot":
            return [(t,) for t in range(n)]
        if animal == "kangaroo":
            return [(hop,) for hop in range(1, min(2, n) + 1)] or [()]
        return [()]

    def legal_actions(self):
        return [encode(animal, params) for animal, params in self.legal_moves()]

    def play_action(self, index):
        self.play(*decode(index))

    def play(self, animal, params):
        params = tuple(params)
        if (animal, params) not in self.legal_moves():
            raise ValueError(f"illegal move {animal} {list(params)}")
        player = self.to_move
        card = (animal, player)

        # 1. Play to the back of the queue.  2. It acts.
        self._hands[player].remove(animal)
        self._queue.append(card)
        self._act(animal, card, params)

        # 3. Recurring animals act front to back, each at most once. Re-scanning from the front after
        # every action handles the queue changing underneath (a hippo only passes animals that have
        # already acted). The played giraffe already jumped; a played hippo or crocodile acts again.
        acted = {card} if animal == "giraffe" else set()
        while pending := [c for c in self._queue if c[0] in RECURRING and c not in acted]:
            acted.add(pending[0])
            self._act(pending[0][0], pending[0], ())

        # 4. Five-animal check.
        if len(self._queue) == 5:
            self._bar += self._queue[:2]
            self._thats_it.append(self._queue[4])
            self._queue = self._queue[2:4]

        # 5. Draw.
        if self._decks[player]:
            self._hands[player].append(self._decks[player].pop(0))
        self.to_move = 1 - player

    # ---- actions ----------------------------------------------------------------------------------
    # Each helper performs one animal's action for `actor`, a card in the queue. The actor is usually
    # that animal itself, or a chameleon copying it (which uses the copied animal's strength).

    def _act(self, animal, actor, params):
        getattr(self, "_" + animal)(actor, *params)

    def _bounce(self, card):
        self._queue.remove(card)
        self._thats_it.append(card)

    def _move_to(self, card, position):
        self._queue.remove(card)
        self._queue.insert(position, card)

    def _lion(self, actor):
        if any(c[0] == "lion" and c != actor for c in self._queue):
            self._bounce(actor)
            return
        for monkey in [c for c in self._queue if c[0] == "monkey"]:
            self._bounce(monkey)
        self._move_to(actor, 0)

    def _hippo(self, actor):
        i = self._queue.index(actor)
        while i > 0 and self._passable(self._queue[i - 1], 11):
            i -= 1
        self._move_to(actor, i)

    def _crocodile(self, actor):
        i = self._queue.index(actor)
        while i > 0 and self._passable(self._queue[i - 1], 10):
            self._bounce(self._queue[i - 1])
            i -= 1

    @staticmethod
    def _passable(card, strength):
        """Hippos and crocodiles get past animals weaker than themselves, never a zebra."""
        return card[0] != "zebra" and STRENGTH[card[0]] < strength

    def _snake(self, actor):
        # Stable sort, so equal strengths keep their order. A chameleon sorts itself as strength 9.
        self._queue.sort(key=lambda c: -STRENGTH["snake" if c == actor else c[0]])

    def _giraffe(self, actor):
        i = self._queue.index(actor)
        if i > 0 and STRENGTH[self._queue[i - 1][0]] < STRENGTH["giraffe"]:
            self._move_to(actor, i - 1)

    def _zebra(self, actor):
        pass

    def _seal(self, actor):
        self._queue.reverse()

    def _chameleon(self, actor, target, *params):
        # The played card went to the back, so positions before the move still index the same animals.
        self._act(self._queue[target][0], actor, params)

    def _monkey(self, actor):
        # A chameleon copying a monkey counts as a monkey, so the copied monkey always triggers this.
        others = [c for c in self._queue if c[0] == "monkey" and c != actor]
        if not others:
            return
        for c in [c for c in self._queue if c[0] in ("hippo", "crocodile")]:
            self._bounce(c)
        front = [actor, *reversed(others)]
        self._queue = front + [c for c in self._queue if c not in front]

    def _kangaroo(self, actor, hop=0):
        self._move_to(actor, self._queue.index(actor) - hop)

    def _parrot(self, actor, target):
        self._bounce(self._queue[target])

    def _skunk(self, actor):
        # The actor is never bounced: it is a skunk, or a chameleon acting as one.
        victims = [c for c in self._queue if c[0] != "skunk" and c != actor]
        top_two = sorted({STRENGTH[c[0]] for c in victims}, reverse=True)[:2]
        for c in victims:
            if STRENGTH[c[0]] in top_two:
                self._bounce(c)
