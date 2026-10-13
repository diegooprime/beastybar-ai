# Beasty Bar rules (2 players, house scoring)

Sources: the Zoch reference sheet ([PDF](https://cdn.simba-dickie-group.de/downloads/601105155/601105155_BeastyBar-1-Uebersicht_Zoch.pdf))
and the publisher's digital edition manual ([elo-games](https://www.elo-games.com/en/games/966887-beasty-bar/manual)).
One deliberate change: **scoring** (see the end). `golden/` holds games that any engine must reproduce exactly.

## Setup

- Each player owns the same 12 animals. Shuffle them face down; draw 4 as your hand; the other 8 are your deck.
- Nobody knows any deck order, including their own. The opponent's hand is hidden; the set of the opponent's
  not-yet-seen cards is public (same 12 animals each).

| Animal | Strength | Points | Kind |
|---|---|---|---|
| lion | 12 | 2 | |
| hippo | 11 | 2 | recurring |
| crocodile | 10 | 3 | recurring |
| snake | 9 | 2 | |
| giraffe | 8 | 3 | recurring |
| zebra | 7 | 4 | permanent |
| seal | 6 | 2 | |
| chameleon | 5 | 3 | |
| monkey | 4 | 3 | |
| kangaroo | 3 | 4 | |
| parrot | 2 | 4 | |
| skunk | 1 | 4 | |

## A turn

1. **Play** a card from your hand to the back of the queue (the end farthest from the bar).
2. **Act**: the played animal does its action.
3. **Recurring**: every hippo, crocodile and giraffe in the queue acts, front to back. Each animal acts at most once
   per turn (an animal pushed back by a hippo does not act again). A giraffe played this turn does not act here; a
   hippo or crocodile played this turn does.
4. **Five-animal check**: if the queue holds exactly 5 animals, the front 2 enter the bar and the last one is bounced
   (THAT'S IT). Otherwise nothing happens. Cards in the bar or THAT'S IT stay there.
5. **Draw** the top card of your deck, if any.

"Front" is next to the bar. Removing an animal closes the gap.

## Actions

- **Lion**: if another lion is in the queue, the new lion is bounced. Otherwise all monkeys are bounced and the lion
  moves to the front.
- **Hippo** (recurring): moves forward past every animal weaker than 11 that is not a zebra; stops behind a zebra,
  a lion or a hippo.
- **Crocodile** (recurring): bounces the animals directly ahead of it, one at a time, while they are weaker than 10
  and not zebras; stops at a zebra or an animal of strength 10+.
- **Snake**: sorts the queue by strength, strongest at the front; equal strengths keep their order.
- **Giraffe** (recurring): if the animal directly ahead is weaker than 8, swaps with it (one jump per turn).
- **Zebra** (permanent): blocks hippos and crocodiles (see above). No action of its own.
- **Seal**: reverses the queue.
- **Chameleon**: copies the action of an animal in the queue (not another chameleon), using that animal's strength
  for the action only, then is a strength-5 chameleon again before recurring actions. Rulings:
  copying a lion bounces the chameleon (a lion is present); copying a monkey counts the chameleon as a monkey
  (so it triggers the monkey action and goes to the front); copying a skunk, the chameleon is a strength-1 skunk and
  is not bounced; copying a snake, the chameleon sorts as strength 9; copying a kangaroo or parrot takes that
  animal's parameter.
- **Monkey**: if no other monkey is in the queue, nothing. Otherwise all hippos and crocodiles are bounced, the new
  monkey moves to the front, and the other monkeys line up directly behind it in reverse of their previous order.
- **Kangaroo**: jumps over the last 1 or 2 animals (player's choice; at most the number of animals ahead).
- **Parrot**: bounces any one animal in the queue (player's choice).
- **Skunk**: bounces every animal of the two highest strengths present in the queue, never skunks.

## End and scoring (house rule)

- The game ends as soon as fewer than 5 cards remain in hands, decks and queue combined (the queue can no longer
  fill). Animals still in the queue do not score.
- **Score = sum of the points of your animals in the bar. Higher score wins; equal scores are a draw.**
  (Official rules count animals instead, with ties going to the weaker animals. This project uses points.)
- The starting player alternates between games.
