# Golden games

`games.jsonl.gz`: 2,000 games with uniformly random legal moves, one JSON object per line. 45,357 moves; every animal
is played at least 3,700 times and the chameleon copies every other animal at least 180 times.

```json
{
  "start": 0,                          // player who moves first
  "hands": [["lion", ...], [...]],     // opening hands, 4 animals each (hand order carries no meaning)
  "decks": [["zebra", ...], [...]],    // remaining 8 animals each, top card first
  "steps": [                           // one entry per move
    {
      "legal": [["parrot", [0]], ...], // all legal moves for the player to move, before this move
      "move": ["parrot", [0]],         // the move played
      "queue": [["lion", 1], ...],     // after the move: [animal, owner], front (bar side) first
      "bar": [["hippo", 0], ...],      // after the move, in the order animals entered
      "thats_it": [["seal", 1], ...],  // after the move, in the order animals were bounced
      "score": [4, 2]                  // after the move: points in the bar per player
    }
  ],
  "final_score": [13, 9]
}
```

Moves are `[animal, parameters]`. Parameters refer to queue positions before the move (0 = front):

| Animal | Parameters |
|---|---|
| kangaroo | `[hop]`, hop 1 or 2 (at most the queue length); `[]` when the queue is empty |
| parrot | `[target]` |
| chameleon | `[target]` plus the copied animal's parameter, e.g. `[2, 0]` copies the parrot at 2 and bounces the front animal |
| others | `[]` |

Players alternate; the player to move draws the top card of their deck after the move, while it lasts. The game is
over after the last step (fewer than 5 cards left in hands, decks and queue).

Verified: three independently written engines reproduce every move and position of these games exactly, and agreed
on 200,000 further random games.
