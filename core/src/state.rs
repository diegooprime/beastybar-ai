//! Game state and rules (RULES.md, with the golden games as the final word).

use crate::action::{ActionSet, decode};
use crate::animal::*;
use crate::rng::Rng;

pub const QUEUE_CAP: usize = 5;
pub const DECK_CAP: usize = 8;
/// Each five-animal check sends 2 cards to the bar and removes 3 from play, so at most 16 of 24 reach it.
pub const BAR_CAP: usize = 16;

const RECURRING: u16 = 1 << HIPPO | 1 << CROCODILE | 1 << GIRAFFE;

/// A complete position: 80 bytes of plain `Copy` data for copy-make search.
///
/// Unused array slots are kept zero, so the derived `Eq`/`Hash` compare positions exactly
/// (including the arrival order in the bar and THAT'S IT).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct State {
    /// Card ids, front (bar side) first. At most 4 between turns; the fifth slot is used mid-move.
    pub queue: [u8; QUEUE_CAP],
    pub queue_len: u8,
    /// Animal bitmask per player.
    pub hands: [u16; 2],
    /// Animal ids per player, top card first.
    pub decks: [[u8; DECK_CAP]; 2],
    pub deck_len: [u8; 2],
    /// Card bitmasks.
    pub bar: u32,
    pub thats_it: u32,
    /// Card ids in order of arrival; the lengths are the popcounts of `bar` and `thats_it`.
    pub bar_order: [u8; BAR_CAP],
    pub thats_it_order: [u8; NUM_CARDS],
    /// Points in the bar per player.
    pub scores: [u8; 2],
    pub to_move: u8,
    pub start: u8,
}

impl State {
    /// A game about to start. `hands` are animal bitmasks, `decks` animal ids (top first).
    pub fn new(start: u8, hands: [u16; 2], decks: [&[u8]; 2]) -> State {
        assert!(start < 2, "start must be 0 or 1");
        let mut s = State { hands, start, to_move: start, ..State::default() };
        for (p, deck) in decks.iter().enumerate() {
            assert!(deck.len() <= DECK_CAP, "a deck holds at most {DECK_CAP} cards");
            s.decks[p][..deck.len()].copy_from_slice(deck);
            s.deck_len[p] = deck.len() as u8;
        }
        s
    }

    /// A uniformly random deal: each player's 12 animals shuffled, the first 4 in hand, the rest the deck.
    pub fn deal(rng: &mut Rng, start: u8) -> State {
        let mut hands = [0; 2];
        let mut decks = [[0; DECK_CAP]; 2];
        for p in 0..2 {
            let mut animals: [u8; NUM_ANIMALS] = std::array::from_fn(|a| a as u8);
            rng.shuffle(&mut animals);
            hands[p] = animals[..4].iter().fold(0, |hand, &a| hand | 1 << a);
            decks[p].copy_from_slice(&animals[4..]);
        }
        State::new(start, hands, [&decks[0], &decks[1]])
    }

    #[inline]
    pub fn queue(&self) -> &[u8] {
        &self.queue[..self.queue_len as usize]
    }

    #[inline]
    pub fn deck(&self, player: usize) -> &[u8] {
        &self.decks[player][..self.deck_len[player] as usize]
    }

    pub fn bar_order(&self) -> &[u8] {
        &self.bar_order[..self.bar.count_ones() as usize]
    }

    pub fn thats_it_order(&self) -> &[u8] {
        &self.thats_it_order[..self.thats_it.count_ones() as usize]
    }

    /// The game ends once fewer than 5 cards remain in hands, decks and queue (the queue can no longer fill).
    #[inline]
    pub fn is_over(&self) -> bool {
        let cards = self.hands[0].count_ones()
            + self.hands[1].count_ones()
            + self.deck_len[0] as u32
            + self.deck_len[1] as u32
            + self.queue_len as u32;
        cards < 5
    }

    /// Legal actions for the player to move; empty once the game is over.
    pub fn legal_actions(&self) -> ActionSet {
        if self.is_over() {
            return ActionSet(0);
        }
        let n = self.queue_len as usize;
        let mut set = 0;
        let mut hand = self.hands[self.to_move as usize];
        while hand != 0 {
            let a = hand.trailing_zeros() as u8;
            hand &= hand - 1;
            set |= match a {
                SKUNK => 1,
                PARROT => targets(n) << 1,
                KANGAROO if n == 0 => 1 << 5,
                KANGAROO => hops(n) << 5,
                MONKEY => 1 << 8,
                CHAMELEON => self.chameleon_actions(),
                _ => 1 << (a + 24),
            };
        }
        ActionSet(set)
    }

    /// A chameleon may copy any non-chameleon in the queue, taking a parrot's target or a kangaroo's hop.
    fn chameleon_actions(&self) -> u64 {
        let n = self.queue_len as usize;
        let mut set = 0;
        for (t, &c) in self.queue().iter().enumerate() {
            set |= match animal(c) {
                CHAMELEON => 0,
                PARROT => targets(n) << (13 + 4 * t),
                KANGAROO => hops(n) << (13 + 4 * t),
                _ => 1 << (9 + t),
            };
        }
        set
    }

    /// Plays `action` for the player to move: play, act, recurring actions, five-animal check, draw.
    pub fn apply(&mut self, action: u8) {
        debug_assert!(self.legal_actions().contains(action), "illegal action {action}");
        let player = self.to_move;
        let m = decode(action);
        let [p0, p1] = m.raw_params();
        let played = card(player, m.animal);
        self.hands[player as usize] &= !(1 << m.animal);
        let pos = self.queue_len as usize;
        self.queue[pos] = played;
        self.queue_len += 1;

        if m.animal == CHAMELEON {
            self.act(animal(self.queue[p0 as usize]), pos, p1);
        } else {
            self.act(m.animal, pos, p0);
        }
        // A giraffe played this turn does not act again; a hippo or crocodile played this turn does (which only
        // matters in constructed positions: in play, recurring animals ahead of it have already settled).
        self.recurring(if m.animal == GIRAFFE { 1 << played } else { 0 });
        self.five_animal_check();
        self.draw(player as usize);
        self.to_move ^= 1;
    }

    /// The action of `kind`, performed by the card at `pos`: that animal itself, or a chameleon copying it
    /// (the copy uses `kind`'s strength; the chameleon is a strength-5 chameleon again afterwards).
    fn act(&mut self, kind: u8, pos: usize, param: u8) {
        match kind {
            SKUNK => self.skunk(pos),
            PARROT => self.bounce_at(param as usize),
            KANGAROO => self.queue[pos - param as usize..=pos].rotate_right(1),
            MONKEY => self.monkey(pos),
            SEAL => self.queue[..self.queue_len as usize].reverse(),
            ZEBRA => {}
            GIRAFFE => self.giraffe(pos),
            SNAKE => self.snake(pos),
            CROCODILE => self.crocodile(pos),
            HIPPO => self.hippo(pos),
            LION => self.lion(pos),
            _ => unreachable!("a chameleon never copies a chameleon"),
        }
    }

    fn lion(&mut self, pos: usize) {
        let lion = self.queue[pos];
        if self.queue().iter().any(|&c| c != lion && animal(c) == LION) {
            self.bounce_at(pos);
            return;
        }
        self.bounce_where(|c| animal(c) == MONKEY);
        let i = self.position_of(lion);
        self.queue[..=i].rotate_right(1);
    }

    fn hippo(&mut self, mut pos: usize) {
        while pos > 0 {
            let ahead = animal(self.queue[pos - 1]);
            if ahead >= HIPPO || ahead == ZEBRA {
                break;
            }
            self.queue.swap(pos - 1, pos);
            pos -= 1;
        }
    }

    fn crocodile(&mut self, mut pos: usize) {
        while pos > 0 {
            let ahead = animal(self.queue[pos - 1]);
            if ahead >= CROCODILE || ahead == ZEBRA {
                break;
            }
            self.bounce_at(pos - 1);
            pos -= 1;
        }
    }

    fn giraffe(&mut self, pos: usize) {
        if pos > 0 && animal(self.queue[pos - 1]) < GIRAFFE {
            self.queue.swap(pos - 1, pos);
        }
    }

    /// Stable sort, strongest first. The acting card sorts as a snake (a copying chameleon counts as 9).
    fn snake(&mut self, pos: usize) {
        let n = self.queue_len as usize;
        let mut keys: [u8; QUEUE_CAP] =
            std::array::from_fn(|i| if i == pos { strength(SNAKE) } else { strength(animal(self.queue[i])) });
        for i in 1..n {
            let (key, c) = (keys[i], self.queue[i]);
            let mut j = i;
            while j > 0 && keys[j - 1] < key {
                keys[j] = keys[j - 1];
                self.queue[j] = self.queue[j - 1];
                j -= 1;
            }
            keys[j] = key;
            self.queue[j] = c;
        }
    }

    /// Bounces every animal of the two highest strengths present. Skunks never count, nor does the acting
    /// card (a chameleon copying the skunk is a strength-1 skunk).
    fn skunk(&mut self, pos: usize) {
        let skunk = self.queue[pos];
        let present = self
            .queue()
            .iter()
            .filter(|&&c| c != skunk && animal(c) != SKUNK)
            .fold(0u16, |strengths, &c| strengths | 1 << animal(c));
        let highest = top_bit(present);
        let targets = highest | top_bit(present & !highest);
        self.bounce_where(|c| c != skunk && targets >> animal(c) & 1 != 0);
    }

    /// With another monkey present: bounce all hippos and crocodiles, then the acting monkey goes to the front
    /// and the other monkeys line up directly behind it in reverse of their previous order.
    fn monkey(&mut self, pos: usize) {
        let monkey = self.queue[pos];
        let is_other_monkey = |c: u8| c != monkey && animal(c) == MONKEY;
        if !self.queue().iter().any(|&c| is_other_monkey(c)) {
            return;
        }
        self.bounce_where(|c| matches!(animal(c), HIPPO | CROCODILE));
        let old = self.queue;
        let old = &old[..self.queue_len as usize];
        let others = old.iter().rev().filter(|&&c| is_other_monkey(c));
        let rest = old.iter().filter(|&&c| c != monkey && !is_other_monkey(c));
        for (slot, &c) in self.queue.iter_mut().zip(std::iter::once(&monkey).chain(others).chain(rest)) {
            *slot = c;
        }
    }

    /// Every hippo, crocodile and giraffe acts once, front to back, tracked by card: after each action the scan
    /// restarts at the front and takes the first one that has not acted. (Continuing from the actor's old
    /// index would skip animals that closed ranks behind a crocodile; the golden games rule that out.)
    fn recurring(&mut self, mut acted: u32) {
        'scan: loop {
            for i in 0..self.queue_len as usize {
                let c = self.queue[i];
                if RECURRING >> animal(c) & 1 != 0 && acted >> c & 1 == 0 {
                    acted |= 1 << c;
                    self.act(animal(c), i, 0);
                    continue 'scan;
                }
            }
            return;
        }
    }

    fn five_animal_check(&mut self) {
        if self.queue_len as usize == QUEUE_CAP {
            let [a, b, c, d, e] = self.queue;
            self.enter_bar(a);
            self.enter_bar(b);
            self.enter_thats_it(e);
            self.queue = [c, d, 0, 0, 0];
            self.queue_len = 2;
        }
    }

    fn draw(&mut self, player: usize) {
        let n = self.deck_len[player] as usize;
        if n > 0 {
            let deck = &mut self.decks[player];
            self.hands[player] |= 1 << deck[0];
            deck.copy_within(1..n, 0);
            deck[n - 1] = 0;
            self.deck_len[player] -= 1;
        }
    }

    fn position_of(&self, c: u8) -> usize {
        self.queue().iter().position(|&x| x == c).expect("card in queue")
    }

    fn bounce_at(&mut self, i: usize) {
        let c = self.queue[i];
        let n = self.queue_len as usize;
        self.queue.copy_within(i + 1..n, i);
        self.queue[n - 1] = 0;
        self.queue_len -= 1;
        self.enter_thats_it(c);
    }

    /// Bounces every queue card matching `pred`, front to back; the rest close ranks in order.
    fn bounce_where(&mut self, pred: impl Fn(u8) -> bool) {
        let n = self.queue_len as usize;
        let mut kept = 0;
        for i in 0..n {
            let c = self.queue[i];
            if pred(c) {
                self.enter_thats_it(c);
            } else {
                self.queue[kept] = c;
                kept += 1;
            }
        }
        self.queue[kept..n].fill(0);
        self.queue_len = kept as u8;
    }

    fn enter_bar(&mut self, c: u8) {
        self.bar_order[self.bar.count_ones() as usize] = c;
        self.bar |= 1 << c;
        self.scores[owner(c) as usize] += POINTS[animal(c) as usize];
    }

    fn enter_thats_it(&mut self, c: u8) {
        self.thats_it_order[self.thats_it.count_ones() as usize] = c;
        self.thats_it |= 1 << c;
    }
}

/// Bits 0..n: a target at any of the n queue positions.
#[inline]
fn targets(n: usize) -> u64 {
    (1 << n) - 1
}

/// Bits 1..=min(n, 2): a hop of 1 or 2, at most the queue length.
#[inline]
fn hops(n: usize) -> u64 {
    if n >= 2 { 0b110 } else { 0b010 }
}

#[inline]
fn top_bit(mask: u16) -> u16 {
    if mask == 0 { 0 } else { 1 << (15 - mask.leading_zeros()) }
}
