//! Hand space: the hidden information of one player.
//!
//! Each player owns the same animals and keeps a hand of `k` animals while their deck is non-empty. The only hidden
//! information is how each player's unplayed animals `U_p` (hand ∪ deck, public) split into hand and deck: the hand
//! is a `k`-subset of `U_p` while `|U_p| > k`, and once `|U_p| <= k` the hand is `U_p` itself (public).
//!
//! Hands are animal bitmasks. [`rank`] places every set of `m` animals in a fixed layout of `C(12, m)` slots
//! (combinatorial number system: animals `a_1 < ... < a_m` have rank `sum C(a_i, i)`, which is colex order), so a
//! range over hands of size 4 has 495 slots whatever has been played. [`subsets`] iterates the `m`-subsets of a mask
//! in increasing rank, and [`Transition`] is the sparse play-and-draw operator `H -> {H - c + d}`.
//!
//! [`Variant`] describes the game being played: the animal subset each player owns and the hand size `k`
//! (the real game is all 12 animals with `k = 4`; smaller "mini" variants use the same code).

use crate::animal::{ALL_ANIMALS, CHAMELEON, NUM_ANIMALS, PARROT, POINTS};
use crate::rng::Rng;
use crate::state::{DECK_CAP, State};

/// `BINOM[n][m] = C(n, m)` for `n, m <= 12`.
pub const BINOM: [[u32; NUM_ANIMALS + 1]; NUM_ANIMALS + 1] = {
    let mut t = [[0u32; NUM_ANIMALS + 1]; NUM_ANIMALS + 1];
    let mut n = 0;
    while n <= NUM_ANIMALS {
        t[n][0] = 1;
        let mut m = 1;
        while m <= n {
            t[n][m] = t[n - 1][m - 1] + if m < n { t[n - 1][m] } else { 0 };
            m += 1;
        }
        n += 1;
    }
    t
};

/// Number of slots in the layout of hands of size `m`: `C(12, m)` (495 for `m = 4`).
#[inline]
pub const fn num_hands(m: usize) -> usize {
    BINOM[NUM_ANIMALS][m] as usize
}

const RANKS: [u16; 1 << NUM_ANIMALS] = {
    let mut t = [0u16; 1 << NUM_ANIMALS];
    let mut mask = 0;
    while mask < 1 << NUM_ANIMALS {
        let (mut r, mut i, mut a) = (0u32, 0, 0);
        while a < NUM_ANIMALS {
            if mask >> a & 1 != 0 {
                i += 1;
                r += BINOM[a][i];
            }
            a += 1;
        }
        t[mask] = r as u16;
        mask += 1;
    }
    t
};

/// Points of every animal set (sum of the animals' points).
pub const MASK_POINTS: [u8; 1 << NUM_ANIMALS] = {
    let mut t = [0u8; 1 << NUM_ANIMALS];
    let mut mask = 0;
    while mask < 1 << NUM_ANIMALS {
        let mut a = 0;
        while a < NUM_ANIMALS {
            if mask >> a & 1 != 0 {
                t[mask] += POINTS[a];
            }
            a += 1;
        }
        mask += 1;
    }
    t
};

/// Rank of an animal set among all sets of the same size, in `0..num_hands(hand.count_ones())`.
#[inline]
pub fn rank(hand: u16) -> usize {
    RANKS[hand as usize] as usize
}

/// The set of `m` animals with rank `r` (inverse of [`rank`]).
pub fn unrank(m: usize, mut r: usize) -> u16 {
    assert!(m <= NUM_ANIMALS && r < num_hands(m), "rank {r} out of range for size {m}");
    let mut hand = 0u16;
    let mut top = NUM_ANIMALS;
    for i in (1..=m).rev() {
        // The largest animal a < top with C(a, i) <= r.
        let mut a = top - 1;
        while BINOM[a][i] as usize > r {
            a -= 1;
        }
        r -= BINOM[a][i] as usize;
        hand |= 1 << a;
        top = a;
    }
    hand
}

/// The `m`-subsets of `mask`, in increasing [`rank`] (Gosper's hack on the compressed bits).
#[derive(Clone, Debug)]
pub struct Subsets {
    pos: [u8; NUM_ANIMALS],
    n: u32,
    cur: u32,
    done: bool,
}

/// The `m`-subsets of `mask` in increasing rank (none if `m > |mask|`; the empty set once if `m = 0`).
pub fn subsets(mask: u16, m: usize) -> Subsets {
    let mut pos = [0u8; NUM_ANIMALS];
    let mut n = 0;
    let mut bits = mask & ALL_ANIMALS;
    while bits != 0 {
        pos[n] = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        n += 1;
    }
    Subsets { pos, n: n as u32, cur: (1u32 << m) - 1, done: m > n }
}

impl Iterator for Subsets {
    type Item = u16;

    fn next(&mut self) -> Option<u16> {
        if self.done {
            return None;
        }
        let c = self.cur;
        let mut out = 0u16;
        let mut bits = c;
        while bits != 0 {
            out |= 1 << self.pos[bits.trailing_zeros() as usize];
            bits &= bits - 1;
        }
        if c == 0 {
            self.done = true;
        } else {
            let t = c | (c - 1);
            let next = (t + 1) | (((!t & (t + 1)) - 1) >> (c.trailing_zeros() + 1));
            if next >> self.n != 0 {
                self.done = true;
            } else {
                self.cur = next;
            }
        }
        Some(out)
    }
}

/// Size of the hand of a player with `unplayed` animals and hand size `k`: `min(k, |unplayed|)`.
#[inline]
pub fn hand_size(unplayed: u16, k: usize) -> usize {
    k.min(unplayed.count_ones() as usize)
}

/// Every possible hand of a player with `unplayed` animals: the `k`-subsets while `|unplayed| > k`, else
/// `unplayed` itself. In increasing rank.
pub fn hands(unplayed: u16, k: usize) -> Subsets {
    subsets(unplayed, hand_size(unplayed, k))
}

/// The animals of `player` not yet played: hand ∪ deck.
#[inline]
pub fn unplayed(s: &State, player: usize) -> u16 {
    s.deck(player).iter().fold(s.hands[player], |m, &a| m | 1 << a)
}

/// The hands after a player holding `hand` (of their `unplayed` animals) plays animal `c` and draws:
/// `hand - c + d` for every `d` in the deck `unplayed - hand`, each with probability `1 / |deck|`;
/// just `hand - c` (probability 1) when the deck is empty. In increasing rank.
pub fn play_draw(hand: u16, unplayed: u16, c: u8) -> impl Iterator<Item = u16> {
    debug_assert!(hand >> c & 1 != 0 && hand & !unplayed == 0);
    let rest = hand & !(1 << c);
    let deck = unplayed & !hand;
    // `rest | d` increases in colex order (rank) with `d`.
    subsets(deck, (deck != 0) as usize).map(move |d| rest | d)
}

/// The play-and-draw operator for one player and one played animal, over the global rank layouts.
///
/// For a player with unplayed set `U` and hand size `k` who plays animal `c`: one row per possible hand `H ∋ c`
/// (in increasing rank, in the layout of size `min(k, |U|)`), listing the ranks of the possible next hands
/// `H - c + d` (layout of size `min(k, |U| - 1)`), each with probability [`Transition::prob`].
#[derive(Clone, Debug, PartialEq)]
pub struct Transition {
    /// Rank of each row's hand.
    pub from: Vec<u16>,
    /// Row `i`'s successors are `to[start[i]..start[i + 1]]`.
    pub start: Vec<u32>,
    pub to: Vec<u16>,
    /// The same for every successor: `1 / |deck|`, or 1 with an empty deck.
    pub prob: f64,
    /// Hand sizes before and after (`from` and `to` layouts).
    pub from_size: u8,
    pub to_size: u8,
}

impl Transition {
    pub fn new(unplayed: u16, k: usize, c: u8) -> Transition {
        assert!(unplayed >> c & 1 != 0, "animal {c} already played");
        let from_size = hand_size(unplayed, k);
        let deck = unplayed.count_ones() as usize - from_size;
        let mut t = Transition {
            from: Vec::new(),
            start: vec![0],
            to: Vec::new(),
            prob: if deck > 0 { 1.0 / deck as f64 } else { 1.0 },
            from_size: from_size as u8,
            to_size: hand_size(unplayed & !(1 << c), k) as u8,
        };
        for hand in hands(unplayed, k).filter(|h| h >> c & 1 != 0) {
            t.from.push(rank(hand) as u16);
            t.to.extend(play_draw(hand, unplayed, c).map(|h| rank(h) as u16));
            t.start.push(t.to.len() as u32);
        }
        t
    }

    pub fn rows(&self) -> impl Iterator<Item = (usize, &[u16])> + '_ {
        self.from
            .iter()
            .enumerate()
            .map(|(i, &r)| (r as usize, &self.to[self.start[i] as usize..self.start[i + 1] as usize]))
    }

    /// Range propagation: `out[to] += prob * weight[from]` (a reach vector pushed through the play and the draw).
    pub fn push(&self, weight: &[f64], out: &mut [f64]) {
        for (from, to) in self.rows() {
            let w = weight[from] * self.prob;
            for &j in to {
                out[j as usize] += w;
            }
        }
    }

    /// Expectation over the draw: `out[from] = prob * sum value[to]` for every row (other entries untouched).
    pub fn pull(&self, value: &[f64], out: &mut [f64]) {
        for (from, to) in self.rows() {
            out[from] = self.prob * to.iter().map(|&j| value[j as usize]).sum::<f64>();
        }
    }
}

/// A game variant: every player owns the animals of `animals` and holds `k` of them while their deck lasts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Variant {
    pub animals: u16,
    pub k: u8,
}

impl Variant {
    /// The real game: all 12 animals, hands of 4.
    pub const REAL: Variant = Variant { animals: ALL_ANIMALS, k: 4 };

    pub fn new(animals: u16, k: usize) -> Variant {
        let n = animals.count_ones() as usize;
        assert!(animals & !ALL_ANIMALS == 0, "unknown animals in {animals:#x}");
        assert!(k >= 1 && k <= n, "hand size {k} for {n} animals");
        assert!(n - k <= DECK_CAP, "a deck holds at most {DECK_CAP} cards");
        // Only a parrot (empty queue) or a chameleon (nothing to copy) can be unplayable, so a full hand must
        // always hold another animal. (Once hands shrink, an empty queue means fewer than 5 cards: game over.)
        let stuck = (animals & (1 << PARROT | 1 << CHAMELEON)).count_ones() as usize;
        assert!(k > stuck, "hand size {k} can leave only parrot/chameleon in hand");
        Variant { animals, k: k as u8 }
    }

    /// Animals per player.
    pub fn n(&self) -> usize {
        self.animals.count_ones() as usize
    }

    pub fn k(&self) -> usize {
        self.k as usize
    }

    /// A uniformly random deal: each player's animals shuffled, the first `k` in hand, the rest the deck.
    /// For [`Variant::REAL`] this is exactly [`State::deal`] (same random stream).
    pub fn deal(&self, rng: &mut Rng, start: u8) -> State {
        let mut hands = [0u16; 2];
        let mut decks = [[0u8; NUM_ANIMALS]; 2];
        let n = self.n();
        for p in 0..2 {
            let mut animals = [0u8; NUM_ANIMALS];
            for (i, a) in subsets(self.animals, 1).enumerate() {
                animals[i] = a.trailing_zeros() as u8;
            }
            rng.shuffle(&mut animals[..n]);
            hands[p] = animals[..self.k()].iter().fold(0, |h, &a| h | 1 << a);
            decks[p] = animals;
        }
        let k = self.k();
        State::new(start, hands, [&decks[0][k..n], &decks[1][k..n]])
    }

    /// Moves played so far (the ply): the starting player moves at even plies.
    pub fn ply(&self, s: &State) -> usize {
        2 * self.n() - (unplayed(s, 0).count_ones() + unplayed(s, 1).count_ones()) as usize
    }

    /// Ply at which both decks are empty (all hands public) from then on: `2 (n - k)`.
    pub fn public_ply(&self) -> usize {
        2 * (self.n() - self.k())
    }

    /// Every possible hand of `player` given the public information of `s`, in increasing rank.
    pub fn hands(&self, s: &State, player: usize) -> Subsets {
        hands(unplayed(s, player), self.k())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binomials() {
        assert_eq!(num_hands(4), 495);
        assert_eq!(num_hands(0), 1);
        assert_eq!(num_hands(12), 1);
        assert_eq!((0..=12).map(num_hands).sum::<usize>(), 4096);
    }

    #[test]
    fn rank_is_a_bijection_in_colex_order() {
        for m in 0..=NUM_ANIMALS {
            let all: Vec<u16> = subsets(ALL_ANIMALS, m).collect();
            assert_eq!(all.len(), num_hands(m));
            for (r, &h) in all.iter().enumerate() {
                assert_eq!(h.count_ones() as usize, m);
                assert_eq!(rank(h), r, "subsets come in rank order");
                assert_eq!(unrank(m, r), h);
            }
        }
    }

    #[test]
    fn subsets_of_a_mask() {
        let mut rng = Rng::new(7);
        for _ in 0..500 {
            let mask = (rng.next_u64() & ALL_ANIMALS as u64) as u16;
            for m in 0..=13 {
                let got: Vec<u16> = subsets(mask, m).collect();
                let want: Vec<u16> =
                    (0..1u16 << 12).filter(|&h| h & !mask == 0 && h.count_ones() as usize == m).collect();
                assert_eq!(got, want, "mask {mask:#x} m {m}");
            }
        }
    }

    #[test]
    fn possible_hands() {
        let u = 0b1011_0110_1101; // 8 animals
        assert_eq!(hands(u, 4).count(), 70);
        assert!(hands(u, 4).all(|h| h & !u == 0 && h.count_ones() == 4));
        let small = 0b1010_0001;
        assert_eq!(hands(small, 4).collect::<Vec<_>>(), [small]);
        assert_eq!(hands(0, 4).collect::<Vec<_>>(), [0]);
    }

    #[test]
    fn play_and_draw() {
        let u = 0b11_1110_0000; // animals 5..=9
        let h = 0b01_1110_0000; // 5..=8, deck {9}
        assert_eq!(play_draw(h, u, 5).collect::<Vec<_>>(), [0b11_1100_0000]);
        let u = 0b1111_1100_0000;
        let h = 0b0000_1100_0000 | 0b11 << 10; // 6, 7, 10, 11; deck 8, 9
        let next: Vec<u16> = play_draw(h, u, 10).collect();
        assert_eq!(next, [h & !(1 << 10) | 1 << 8, h & !(1 << 10) | 1 << 9]);
        assert!(next.windows(2).all(|w| rank(w[0]) < rank(w[1])));
        // Empty deck: the hand just shrinks.
        assert_eq!(play_draw(0b111, 0b111, 1).collect::<Vec<_>>(), [0b101]);
    }

    #[test]
    fn transition_rows_are_stochastic_and_match_play_draw() {
        let mut rng = Rng::new(3);
        for _ in 0..300 {
            let u = (rng.next_u64() & ALL_ANIMALS as u64) as u16;
            if u == 0 {
                continue;
            }
            let k = 1 + rng.below(6);
            let animals: Vec<u8> = (0..12).filter(|&a| u >> a & 1 != 0).collect();
            let c = animals[rng.below(animals.len())];
            let t = Transition::new(u, k, c);
            assert_eq!(t.from.len(), hands(u, k).filter(|h| h >> c & 1 != 0).count());
            let deck = u.count_ones() as usize - hand_size(u, k);
            for (from, to) in t.rows() {
                let h = unrank(t.from_size as usize, from);
                assert!(h >> c & 1 != 0);
                assert_eq!(to.len(), deck.max(1));
                assert!((to.len() as f64 * t.prob - 1.0).abs() < 1e-12);
                let want: Vec<u16> = play_draw(h, u, c).map(|x| rank(x) as u16).collect();
                assert_eq!(to, &want[..]);
                for &j in to {
                    let next = unrank(t.to_size as usize, j as usize);
                    assert!(next & !(u & !(1 << c)) == 0, "next hand within the new unplayed set");
                }
            }
            // push conserves mass and pull of a constant is that constant on every row.
            let w: Vec<f64> = (0..num_hands(t.from_size as usize)).map(|i| (i % 7) as f64).collect();
            let mut out = vec![0.0; num_hands(t.to_size as usize)];
            t.push(&w, &mut out);
            let mass: f64 = t.from.iter().map(|&r| w[r as usize]).sum();
            assert!((out.iter().sum::<f64>() - mass).abs() < 1e-9);
            let ones = vec![1.0; num_hands(t.to_size as usize)];
            let mut back = vec![0.0; num_hands(t.from_size as usize)];
            t.pull(&ones, &mut back);
            assert!(t.from.iter().all(|&r| (back[r as usize] - 1.0).abs() < 1e-12));
        }
    }

    #[test]
    fn real_variant_deals_like_state_deal() {
        for seed in 0..50 {
            let a = State::deal(&mut Rng::new(seed), (seed % 2) as u8);
            let b = Variant::REAL.deal(&mut Rng::new(seed), (seed % 2) as u8);
            assert_eq!(a, b);
            assert_eq!(Variant::REAL.ply(&a), 0);
        }
        let v = Variant::new(0b0111_1101_0110, 3);
        let s = v.deal(&mut Rng::new(1), 0);
        for p in 0..2 {
            assert_eq!(s.hands[p].count_ones(), 3);
            assert_eq!(unplayed(&s, p), v.animals);
        }
        assert_eq!(v.n(), 8);
        assert_eq!(v.public_ply(), 10);
    }
}
