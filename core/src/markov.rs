//! Markov key: a `u64` that identifies a position for the purpose of the future game.
//!
//! The future of a position depends only on (RULES.md, `state.rs`):
//! - the queue: its cards in order, with owners;
//! - each player's hand and deck *as sets*: nobody knows any deck order, so as a game the deck is a uniformly
//!   shuffled set (the order a [`State`] stores is one sample of it);
//! - the score difference: the bar only ever gains cards, and the outcome compares the two scores;
//! - the side to move.
//!
//! The game ends when fewer than 5 cards remain in hands, decks and queue, which these determine. Everything else
//! (bar and THAT'S IT contents and order, absolute scores, the starting player, history) is irrelevant.
//!
//! Bit layout of [`key`], low to high:
//!
//! | bits   | field                                                                           |
//! |--------|---------------------------------------------------------------------------------|
//! | 0..3   | queue length (0..=4 between moves)                                              |
//! | 3..23  | queue cards (ids 0..23), 5 bits each, front first; empty slots 0                |
//! | 23..35 | `U_0`: player 0's unplayed animals, hand ∪ deck                                 |
//! | 35..47 | `U_1`                                                                           |
//! | 47..51 | `|H_0|`: player 0's hand size (deck size `|U_0| - |H_0|`)                        |
//! | 51..55 | `|H_1|`                                                                         |
//! | 55..62 | `s_0 - s_1 + 64`: exact, since each player owns at most 36 points             |
//! | 62     | side to move                                                                    |
//!
//! Bit 63 is 0 and no key is 0 (the score field is at least 28).
//!
//! What the key covers: when a player's deck is empty (`|H_p| = |U_p|`) their hand is `U_p`, public, and
//! determined by the key. While the deck is non-empty the key does **not** say which `|H_p|`-subset of `U_p` is
//! the hand (the only hidden information): it then identifies the public state, and [`full_key`]
//! `= (key, H_0, H_1)` identifies the position. So:
//! - positions with equal [`full_key`] have the same legal actions, the same multiset of successor full keys for
//!   every action (over the uniformly random draw, [`successors`]) and the same outcome when over;
//! - when both decks are empty (every position from ply `2 (n - k)` on, 16 in the real game, and every [`public`]
//!   position) the key alone is a full key: the same holds for [`key`], with deterministic successors.
//!
//! `tests/markov_fuzz.rs` checks both on positions reached by different histories.

use crate::hands::unplayed;
use crate::state::State;

/// The Markov key of `s` (see the module documentation for what it covers).
#[inline]
pub fn key(s: &State) -> u64 {
    debug_assert!(s.queue_len <= 4, "keys are taken between moves");
    let mut k = s.queue_len as u64;
    for (i, &c) in s.queue().iter().enumerate() {
        k |= (c as u64) << (3 + 5 * i);
    }
    k |= (unplayed(s, 0) as u64) << 23 | (unplayed(s, 1) as u64) << 35;
    k |= (s.hands[0].count_ones() as u64) << 47 | (s.hands[1].count_ones() as u64) << 51;
    let diff = s.scores[0] as i64 - s.scores[1] as i64;
    debug_assert!(diff.abs() <= 63);
    k | ((diff + 64) as u64) << 55 | (s.to_move as u64) << 62
}

/// The key plus both hands (bits 64..76 and 76..88): identifies the position for the future game even while
/// hands are hidden.
#[inline]
pub fn full_key(s: &State) -> u128 {
    key(s) as u128 | (s.hands[0] as u128) << 64 | (s.hands[1] as u128) << 76
}

/// The public position: every unplayed animal in hand, decks empty. `apply` then never draws and `is_over` counts
/// the same cards; its legal actions are the union over all possible hands, and its key is the public state's.
pub fn public(s: &State) -> State {
    let mut p = *s;
    for player in 0..2 {
        p.hands[player] = unplayed(s, player);
        p.decks[player] = [0; crate::state::DECK_CAP];
        p.deck_len[player] = 0;
    }
    p
}

/// Key of the public position: the same for every possible split of the hands.
#[inline]
pub fn public_key(s: &State) -> u64 {
    key(&public(s))
}

/// The fields of a key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyFields {
    pub queue: [u8; 4],
    pub queue_len: u8,
    pub unplayed: [u16; 2],
    pub hand_size: [u8; 2],
    /// `s_0 - s_1`.
    pub score_diff: i8,
    pub to_move: u8,
}

pub fn decode(key: u64) -> KeyFields {
    KeyFields {
        queue: std::array::from_fn(|i| (key >> (3 + 5 * i) & 31) as u8),
        queue_len: (key & 7) as u8,
        unplayed: [(key >> 23 & 0xfff) as u16, (key >> 35 & 0xfff) as u16],
        hand_size: [(key >> 47 & 15) as u8, (key >> 51 & 15) as u8],
        score_diff: ((key >> 55 & 127) as i64 - 64) as i8,
        to_move: (key >> 62 & 1) as u8,
    }
}

/// The positions after `action`, one per possible draw (each with probability `1 / deck size`): the mover's deck
/// is reordered so that each of its animals in turn is on top. Just one position when the deck is empty.
pub fn successors(s: &State, action: u8) -> impl Iterator<Item = State> + '_ {
    let p = s.to_move as usize;
    let n = s.deck_len[p] as usize;
    (0..n.max(1)).map(move |i| {
        let mut t = *s;
        if n > 0 {
            t.decks[p][..=i].rotate_right(1);
        }
        t.apply(action);
        t
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animal::ALL_ANIMALS;
    use crate::hands::{Variant, hands};
    use crate::rng::Rng;

    fn random_state(rng: &mut Rng, variant: Variant, plies: usize) -> State {
        let start = rng.below(2) as u8;
        let mut s = variant.deal(rng, start);
        for _ in 0..plies {
            if s.is_over() {
                break;
            }
            let legal = s.legal_actions();
            s.apply(legal.nth_action(rng.below(legal.len())));
        }
        s
    }

    #[test]
    fn decode_inverts_key() {
        let mut rng = Rng::new(11);
        for _ in 0..2000 {
            let plies = rng.below(24);
            let s = random_state(&mut rng, Variant::REAL, plies);
            let f = decode(key(&s));
            assert_eq!(&f.queue[..s.queue_len as usize], s.queue());
            assert_eq!(f.queue_len, s.queue_len);
            assert_eq!(f.unplayed, [unplayed(&s, 0), unplayed(&s, 1)]);
            assert_eq!(f.hand_size, [s.hands[0].count_ones() as u8, s.hands[1].count_ones() as u8]);
            assert_eq!(f.score_diff as i32, s.scores[0] as i32 - s.scores[1] as i32);
            assert_eq!(f.to_move, s.to_move);
            assert!(key(&s) != 0 && key(&s) >> 63 == 0);
        }
    }

    /// Changing what the key ignores changes nothing about the future.
    #[test]
    fn ignored_fields_do_not_matter() {
        let mut rng = Rng::new(5);
        for _ in 0..3000 {
            let plies = rng.below(24);
            let s = random_state(&mut rng, Variant::REAL, plies);
            let mut t = s;
            t.start ^= 1;
            let shift = rng.below(20) as u8;
            t.scores = [s.scores[0] + shift, s.scores[1] + shift];
            let nb = s.bar.count_ones() as usize;
            rng.shuffle(&mut t.bar_order[..nb]);
            let nt = s.thats_it.count_ones() as usize;
            rng.shuffle(&mut t.thats_it_order[..nt]);
            for p in 0..2 {
                let n = s.deck_len[p] as usize;
                rng.shuffle(&mut t.decks[p][..n]);
            }
            assert_eq!(full_key(&s), full_key(&t));
            assert_eq!(s.legal_actions(), t.legal_actions());
            assert_eq!(s.is_over(), t.is_over());
            for a in s.legal_actions() {
                let mut x: Vec<u128> = successors(&s, a).map(|c| full_key(&c)).collect();
                let mut y: Vec<u128> = successors(&t, a).map(|c| full_key(&c)).collect();
                x.sort_unstable();
                y.sort_unstable();
                assert_eq!(x, y);
            }
        }
    }

    /// The public position stands for every possible split: same end, union of the legal actions, and its
    /// successors are the public successors.
    #[test]
    fn public_position_represents_the_public_state() {
        let mut rng = Rng::new(9);
        for variant in [Variant::REAL, Variant::new(0b1111_1011_0110, 3)] {
            for _ in 0..1000 {
                let plies = rng.below(2 * variant.n());
                let s = random_state(&mut rng, variant, plies);
                let p = public(&s);
                assert_eq!(s.is_over(), p.is_over());
                assert_eq!(public_key(&s), key(&p));
                // The hand sizes tell the two representations apart unless they are the same position.
                assert_eq!(key(&s) == key(&p), s.deck_len == [0, 0]);
                if s.is_over() {
                    continue;
                }
                let mover = s.to_move as usize;
                let mut union = 0;
                for h in hands(unplayed(&s, mover), variant.k()) {
                    let mut t = s;
                    t.hands[mover] = h;
                    union |= t.legal_actions().0;
                }
                assert_eq!(p.legal_actions().0, union);
                for a in s.legal_actions() {
                    let mut c = s;
                    c.apply(a);
                    let mut q = p;
                    q.apply(a);
                    assert_eq!(public_key(&c), key(&q));
                    assert_eq!(c.is_over(), q.is_over());
                }
            }
        }
        // Public positions of the full game have every animal in hand.
        let s = Variant::REAL.deal(&mut rng, 0);
        assert_eq!(public(&s).hands, [ALL_ANIMALS; 2]);
    }

    #[test]
    fn successors_enumerate_the_draws() {
        let mut rng = Rng::new(2);
        let s = Variant::REAL.deal(&mut rng, 0);
        let a = s.legal_actions().nth_action(0);
        let kids: Vec<State> = successors(&s, a).collect();
        assert_eq!(kids.len(), 8);
        let drawn: u16 = kids.iter().map(|c| c.hands[0] & !s.hands[0]).fold(0, |m, d| {
            assert_eq!(d.count_ones(), 1);
            assert_eq!(m & d, 0);
            m | d
        });
        assert_eq!(drawn, unplayed(&s, 0) & !s.hands[0]);
    }
}
