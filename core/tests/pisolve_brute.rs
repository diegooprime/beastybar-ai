//! The perfect-information solver equals plain minimax (no pruning, no table).
//!
//! Positions come from random play from random deals, played on until both decks are empty (all hands public)
//! and then 0-3 plies more: plies 16-19 of the real game and the corresponding plies of smaller variants.
//! Frontier nodes (the ply before, where the mover's deck holds one card and the opponent's hand is public) are
//! checked hand by hand against minimax on the real position with that hand in it.
//!
//!     cargo test --release --test pisolve_brute                 # 12k positions, 2.4k frontier nodes
//!     cargo test --release --test pisolve_brute -- --ignored    # 100k positions, 20k frontier nodes (~1 min)

use bbcore::hands::{Variant, hands, unplayed};
use bbcore::markov::public;
use bbcore::pisolve::{PiSolver, minimax};
use bbcore::{Rng, State};

/// Variants and their share of the positions.
const VARIANTS: [(Variant, usize); 5] = [
    (Variant::REAL, 6),
    (Variant { animals: 0b1111_1111_1100, k: 4 }, 3), // 10 animals, public from ply 12
    (Variant { animals: 0b1011_1011_1011, k: 3 }, 2), // 9 animals, public from ply 12
    (Variant { animals: 0b0001_1111_1000, k: 2 }, 1), // 6 animals, public from ply 8
    (Variant { animals: 0b1111_1111_1111, k: 5 }, 0), // hands of 5, public from ply 14: bigger trees
];

fn play(s: &mut State, rng: &mut Rng) {
    let legal = s.legal_actions();
    s.apply(legal.nth_action(rng.below(legal.len())));
}

/// A random unfinished position with both decks empty, up to `extra` plies after they emptied (fewer if the
/// game would end).
fn public_position(rng: &mut Rng, v: Variant, extra: usize) -> State {
    loop {
        let start = rng.below(2) as u8;
        let mut s = v.deal(rng, start);
        while s.deck_len != [0, 0] && !s.is_over() {
            play(&mut s, rng);
        }
        if s.is_over() {
            continue;
        }
        for _ in 0..extra {
            let mut t = s;
            play(&mut t, rng);
            if t.is_over() {
                break;
            }
            s = t;
        }
        return s;
    }
}

/// A random real position at the frontier ply (one card left in the mover's deck).
fn frontier_position(rng: &mut Rng, v: Variant) -> State {
    loop {
        let start = rng.below(2) as u8;
        let mut s = v.deal(rng, start);
        while v.ply(&s) + 1 < v.public_ply() && !s.is_over() {
            play(&mut s, rng);
        }
        if !s.is_over() {
            return s;
        }
    }
}

/// `n` positions (plus 2% more from the hand-of-5 variant).
fn check_values(n: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let mut warm = PiSolver::new(16);
    let mut tiny = PiSolver::new(2);
    for (v, share) in VARIANTS {
        let n_per_variant = if share == 0 { n / 50 } else { n * share / 12 };
        let mut by_ply = std::collections::BTreeMap::new();
        let mut brute_nodes = 0;
        for i in 0..n_per_variant {
            let s = public_position(&mut rng, v, i % 4);
            let want = minimax(&s, &mut brute_nodes);
            assert_eq!(warm.value(&s), want, "warm table, {s:?}");
            if i % 8 == 0 {
                assert_eq!(tiny.value(&s), want, "tiny table, {s:?}");
            }
            let mover = warm.value_for_mover(&s);
            assert_eq!(if s.to_move == 0 { mover } else { 2 - mover }, want);
            *by_ply.entry(v.ply(&s)).or_insert(0) += 1;
        }
        println!(
            "{v:?}: {n_per_variant} positions by ply {by_ply:?}, minimax nodes mean {:.0}, solver nodes mean {:.0}",
            brute_nodes as f64 / n_per_variant as f64,
            warm.stats.nodes as f64 / n_per_variant as f64
        );
        warm.reset_stats();
    }
}

/// `n` frontier nodes (plus 0.5% more from the hand-of-5 variant, whose minimax takes ~4M nodes per node).
fn check_frontiers(n: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let mut solver = PiSolver::new(16);
    for (v, share) in VARIANTS {
        let n_per_variant = if share == 0 { n / 200 } else { n * share / 12 };
        let k = v.k();
        let mut frontier_hands = 0;
        for _ in 0..n_per_variant {
            let s = frontier_position(&mut rng, v);
            let p = s.to_move as usize;
            assert_eq!((s.deck_len[p], s.deck_len[1 - p]), (1, 0));
            let u = unplayed(&s, p);
            let values = solver.frontier_values(&s, k);
            assert_eq!(values, solver.frontier_values(&public(&s), k), "real and public inputs agree");
            let expected_hands: Vec<u16> = hands(u, k).collect();
            assert_eq!(values.iter().map(|&(h, _)| h).collect::<Vec<_>>(), expected_hands);
            assert_eq!(values.len(), k + 1);
            frontier_hands += values.len();
            for &(h, value) in &values {
                // The real position with hand h: the deck holds the one remaining card, so minimax is exact.
                let mut t = s;
                t.hands[p] = h;
                t.decks[p][0] = (u & !h).trailing_zeros() as u8;
                let want = minimax(&t, &mut 0);
                assert_eq!(value, want, "hand {h:#x} of {s:?}");
                if h == s.hands[p] {
                    assert_eq!(value, minimax(&s, &mut 0));
                }
            }
        }
        println!("{v:?}: {n_per_variant} frontier nodes, {frontier_hands} hand values");
        // With an empty deck the only hand is the unplayed set, valued like the position.
        for _ in 0..n_per_variant / 4 {
            let s = public_position(&mut rng, v, 0);
            let p = s.to_move as usize;
            if unplayed(&s, 1 - p).count_ones() as usize > k {
                continue;
            }
            assert_eq!(solver.frontier_values(&s, k), [(s.hands[p], solver.value(&s))]);
        }
    }
}

#[test]
fn values_equal_minimax() {
    check_values(12_000, 1);
}

#[test]
fn frontier_values_equal_minimax() {
    check_frontiers(2_400, 2);
}

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn values_equal_minimax_100k() {
    check_values(100_000, 3);
}

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn frontier_values_equal_minimax_20k() {
    check_frontiers(20_000, 4);
}
