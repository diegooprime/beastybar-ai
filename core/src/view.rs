//! Information views. Agents receive positions only through [`view`]: hidden information is never visible.
//!
//! Player p knows the public information (queue, bar, THAT'S IT, scores, whose turn, who started), every deck's
//! length and the opponent's hand size, their own hand, and therefore the set of animals in their own deck and the
//! set of the opponent's unplayed animals (hand ∪ deck: every animal not yet seen). They do not know any deck
//! order, nor how the opponent's unplayed animals split between hand and deck.

use crate::animal::NUM_ANIMALS;
use crate::rng::Rng;
use crate::state::{DECK_CAP, State};

/// Animal bitmask of player `p`'s deck.
#[inline]
pub fn deck_mask(state: &State, p: usize) -> u16 {
    state.deck(p).iter().fold(0, |mask, &a| mask | 1 << a)
}

/// A position consistent with everything player `p` knows, with all hidden information freshly randomized:
/// p's deck order shuffled, and the opponent's unplayed animals re-split uniformly at random into a hand of the
/// same size (min(4, unplayed) in every reachable position) and a shuffled deck. Public information and p's hand
/// are unchanged.
///
/// The result is a function of p's information and `rng` only: hidden cards are put in a canonical order before
/// shuffling, so `view(view(s, p, r1), p, r2) == view(s, p, r2)` for any `r1`.
pub fn view(state: &State, p: u8, rng: &mut Rng) -> State {
    let mut v = *state;
    let me = p as usize;
    let opp = 1 - me;
    let (own, n) = shuffled(deck_mask(state, me), rng);
    v.decks[me] = deck(&own[..n]);

    let hand_size = state.hands[opp].count_ones() as usize;
    let (animals, n) = shuffled(state.hands[opp] | deck_mask(state, opp), rng);
    v.hands[opp] = animals[..hand_size].iter().fold(0, |hand, &a| hand | 1 << a);
    v.decks[opp] = deck(&animals[hand_size..n]);
    v
}

/// The true position with both deck orders reshuffled; hands unchanged. For oracle agents that measure the value
/// of knowing the opponent's hand.
pub fn shuffle_decks(state: &State, rng: &mut Rng) -> State {
    let mut v = *state;
    for p in 0..2 {
        let (animals, n) = shuffled(deck_mask(state, p), rng);
        v.decks[p] = deck(&animals[..n]);
    }
    v
}

/// The animals of `mask` in a uniformly random order (zero-padded), and how many there are.
fn shuffled(mask: u16, rng: &mut Rng) -> ([u8; NUM_ANIMALS], usize) {
    let mut animals = [0u8; NUM_ANIMALS];
    let mut n = 0;
    let mut bits = mask;
    while bits != 0 {
        animals[n] = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        n += 1;
    }
    rng.shuffle(&mut animals[..n]);
    (animals, n)
}

/// A deck array holding `animals` (top first), zero-padded.
fn deck(animals: &[u8]) -> [u8; DECK_CAP] {
    let mut deck = [0; DECK_CAP];
    deck[..animals.len()].copy_from_slice(animals);
    deck
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Positions from random games: every position of `games` games.
    fn positions(games: u32, seed: u64) -> Vec<State> {
        let mut rng = Rng::new(seed);
        let mut out = Vec::new();
        for i in 0..games {
            let mut s = State::deal(&mut rng, (i % 2) as u8);
            while !s.is_over() {
                out.push(s);
                let legal = s.legal_actions();
                s.apply(legal.nth_action(rng.below(legal.len())));
            }
            out.push(s);
        }
        out
    }

    #[test]
    fn views_preserve_what_p_knows() {
        let mut rng = Rng::new(7);
        for s in positions(300, 1) {
            for p in 0..2u8 {
                let (me, opp) = (p as usize, 1 - p as usize);
                let v = view(&s, p, &mut rng);
                // Public information.
                assert_eq!(v.queue, s.queue);
                assert_eq!(v.queue_len, s.queue_len);
                assert_eq!((v.bar, v.thats_it), (s.bar, s.thats_it));
                assert_eq!(v.bar_order, s.bar_order);
                assert_eq!(v.thats_it_order, s.thats_it_order);
                assert_eq!(v.scores, s.scores);
                assert_eq!((v.to_move, v.start), (s.to_move, s.start));
                assert_eq!(v.deck_len, s.deck_len);
                assert_eq!(v.is_over(), s.is_over());
                // p's own cards: same hand, same deck set.
                assert_eq!(v.hands[me], s.hands[me]);
                assert_eq!(deck_mask(&v, me), deck_mask(&s, me));
                // The opponent: same unplayed set and hand size, min(4, unplayed).
                let unplayed = s.hands[opp] | deck_mask(&s, opp);
                assert_eq!(v.hands[opp] | deck_mask(&v, opp), unplayed);
                assert_eq!(v.hands[opp] & deck_mask(&v, opp), 0);
                assert_eq!(v.hands[opp].count_ones(), s.hands[opp].count_ones());
                assert_eq!(s.hands[opp].count_ones(), unplayed.count_ones().min(4));
                // Unused deck slots stay zero (State's derived Eq/Hash rely on it).
                for q in 0..2 {
                    assert!(v.decks[q][v.deck_len[q] as usize..].iter().all(|&a| a == 0));
                }
                if s.to_move == p {
                    assert_eq!(v.legal_actions(), s.legal_actions());
                }
                // A view depends only on p's information: viewing a view with the same rng gives the same result.
                let seed = rng.next_u64();
                assert_eq!(view(&v, p, &mut Rng::new(seed)), view(&s, p, &mut Rng::new(seed)));
            }
        }
    }

    /// Pearson chi-square statistic of `counts` against equal expected counts.
    fn chi_square(counts: &[u64]) -> f64 {
        let total: u64 = counts.iter().sum();
        let expected = total as f64 / counts.len() as f64;
        counts.iter().map(|&c| (c as f64 - expected).powi(2) / expected).sum()
    }

    /// Fails if `counts` is far from uniform: chi-square more than 6 standard deviations above its mean.
    fn assert_uniform(counts: &[u64], what: &str) {
        let df = (counts.len() - 1) as f64;
        let chi2 = chi_square(counts);
        assert!(
            chi2 < df + 6.0 * (2.0 * df).sqrt(),
            "{what}: chi-square {chi2:.1} with {df} degrees of freedom"
        );
    }

    #[test]
    fn opponent_hands_and_deck_orders_are_uniform() {
        // Mid-game: after 4 plies the opponent has 10 unplayed animals, 4 in hand (C(10, 4) = 210 hands).
        let mut rng = Rng::new(3);
        let mut s = State::deal(&mut rng, 0);
        for _ in 0..4 {
            let legal = s.legal_actions();
            s.apply(legal.nth_action(rng.below(legal.len())));
        }
        assert!(!s.is_over());
        let p = s.to_move;
        let (me, opp) = (p as usize, 1 - p as usize);
        let unplayed = s.hands[opp] | deck_mask(&s, opp);
        let u = unplayed.count_ones() as usize;
        assert_eq!((u, s.hands[opp].count_ones()), (10, 4));

        // Index every 4-subset of the unplayed animals.
        let animals: Vec<u8> = (0..NUM_ANIMALS as u8).filter(|&a| unplayed >> a & 1 != 0).collect();
        let mut subsets = std::collections::HashMap::new();
        for m in 0u16..1 << u {
            if m.count_ones() == 4 {
                let hand = (0..u).filter(|&i| m >> i & 1 != 0).fold(0u16, |h, i| h | 1 << animals[i]);
                let k = subsets.len();
                subsets.insert(hand, k);
            }
        }
        assert_eq!(subsets.len(), 210);

        let mut hands = vec![0u64; 210];
        let mut opp_top = [0u64; NUM_ANIMALS];
        let mut own_top = [0u64; NUM_ANIMALS];
        let n = 210_000;
        for _ in 0..n {
            let v = view(&s, p, &mut rng);
            hands[subsets[&v.hands[opp]]] += 1;
            opp_top[v.decks[opp][0] as usize] += 1;
            own_top[v.decks[me][0] as usize] += 1;
        }
        assert_uniform(&hands, "opponent hand");
        let opp_top: Vec<u64> = animals.iter().map(|&a| opp_top[a as usize]).collect();
        assert_uniform(&opp_top, "opponent deck top card");
        let own: Vec<u64> = s.deck(me).iter().map(|&a| own_top[a as usize]).collect();
        assert_eq!(own.iter().sum::<u64>(), n);
        assert_uniform(&own, "own deck top card");

        // The check has power: a sampler that never puts the lowest unplayed animal in the hand fails it.
        let mut biased = hands.clone();
        for (&hand, &k) in &subsets {
            if hand & 1 << animals[0] != 0 {
                biased[k] = 0;
            }
        }
        let df: f64 = 209.0;
        assert!(chi_square(&biased) > df + 6.0 * (2.0 * df).sqrt());
    }

    #[test]
    fn shuffle_decks_keeps_hands() {
        let mut rng = Rng::new(5);
        for s in positions(50, 2) {
            let v = shuffle_decks(&s, &mut rng);
            assert_eq!(v.hands, s.hands);
            for p in 0..2 {
                assert_eq!(deck_mask(&v, p), deck_mask(&s, p));
            }
            assert_eq!(State { decks: s.decks, ..v }, s);
        }
    }
}
