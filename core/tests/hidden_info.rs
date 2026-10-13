//! The hidden-information facts the solvers rely on, checked on the engine itself.
//!
//! - The only hidden information is each player's split of their unplayed animals `U_p` into hand and deck: the
//!   hand is a `k`-subset of `U_p` while `|U_p| > k`, and `U_p` itself (public) once `|U_p| <= k`.
//! - Ply convention: ply = moves played; the starting player moves at even plies. In the real game both decks are
//!   empty from ply 16, and at ply 15 the second mover's deck holds exactly one card.
//! - Exact Bayes: for any public history and any behaviour strategies that see only public information and the
//!   player's own hand, the posterior over (H_0, H_1), computed by enumerating every deal, is the product of two
//!   independent ranges, each obtained by the filter `r'(H') = sum_H r(H) sigma(a | H) T_c(H -> H')` with the
//!   play-and-draw operator of `hands::Transition` (new hand `H - c + d`, `d` uniform over the deck `U - H`).
//!   So merging a player's information sets by (public history, current hand) loses nothing.

use std::collections::HashMap;

use bbcore::hands::{Transition, Variant, hand_size, hands, num_hands, rank, unplayed};
use bbcore::markov::public_key;
use bbcore::{Rng, State};

fn play_random(s: &mut State, rng: &mut Rng) -> u8 {
    let legal = s.legal_actions();
    let a = legal.nth_action(rng.below(legal.len()));
    s.apply(a);
    a
}

#[test]
fn hidden_information_is_the_hand_deck_split() {
    let variants = [
        Variant::REAL,
        Variant::new(0b1111_1111_1100, 4),
        Variant::new(0b1011_1011_1011, 3),
        Variant::new(0b0001_1111_1000, 2),
        Variant::new(0b1111_1111_1111, 5),
    ];
    let mut rng = Rng::new(17);
    for (i, &v) in variants.iter().cycle().take(5000).enumerate() {
        let start = (i % 2) as u8;
        let mut s = v.deal(&mut rng, start);
        let mut ply = 0;
        loop {
            assert_eq!(v.ply(&s), ply);
            assert_eq!(s.to_move == start, ply % 2 == 0, "the starting player moves at even plies");
            for p in 0..2 {
                let u = unplayed(&s, p);
                assert_eq!(u & !v.animals, 0);
                assert_eq!(s.hands[p] & !u, 0);
                assert_eq!(s.hands[p].count_ones() as usize, hand_size(u, v.k()), "hand size min(k, |U|)");
                assert_eq!(s.deck_len[p] == 0, u.count_ones() as usize <= v.k(), "deck empty iff |U| <= k");
                assert!(hands(u, v.k()).any(|h| h == s.hands[p]), "the hand is one of the possible hands");
            }
            let mover = s.to_move as usize;
            if ply + 1 == v.public_ply() {
                assert_eq!((s.deck_len[mover], s.deck_len[1 - mover]), (1, 0), "frontier ply: one card left");
                assert_ne!(s.to_move, start, "the second mover");
            }
            if ply >= v.public_ply() {
                assert_eq!(s.deck_len, [0, 0], "all hands public");
            }
            if s.is_over() {
                break;
            }
            play_random(&mut s, &mut rng);
            ply += 1;
        }
    }
    assert_eq!(Variant::REAL.public_ply(), 16);
}

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A behaviour strategy that depends on public information and the mover's own hand only:
/// `sigma(a | public state, hand)` over the actions legal with that hand.
fn sigma(s: &State, hand: u16, action: u8, salt: u64) -> f64 {
    let mut t = *s;
    t.hands[s.to_move as usize] = hand;
    let legal = t.legal_actions();
    if !legal.contains(action) {
        return 0.0;
    }
    let pk = public_key(s);
    let w = |a: u8| 1.0 + (mix(pk ^ salt ^ (hand as u64) << 1 ^ (a as u64) << 20) % 4) as f64;
    w(action) / legal.map(w).sum::<f64>()
}

/// Every deal of `v`: the orders of each player's animals (hand = first k).
fn orders(v: Variant) -> Vec<Vec<u8>> {
    fn perms(xs: &mut Vec<u8>, k: usize, out: &mut Vec<Vec<u8>>) {
        if k == xs.len() {
            out.push(xs.clone());
            return;
        }
        for i in k..xs.len() {
            xs.swap(k, i);
            perms(xs, k + 1, out);
            xs.swap(k, i);
        }
    }
    let mut xs: Vec<u8> = (0..12).filter(|&a| v.animals >> a & 1 != 0).collect();
    let mut out = Vec::new();
    perms(&mut xs, 0, &mut out);
    out
}

fn deal_from(v: Variant, start: u8, o0: &[u8], o1: &[u8]) -> State {
    let k = v.k();
    let hand = |o: &[u8]| o[..k].iter().fold(0u16, |h, &a| h | 1 << a);
    State::new(start, [hand(o0), hand(o1)], [&o0[k..], &o1[k..]])
}

#[test]
fn exact_bayes_filter_and_independent_hands() {
    let variants = [
        Variant::new(0b10_1010_0110, 2),   // parrot, kangaroo, seal, giraffe, crocodile
        Variant::new(0b1001_0101_0010, 3), // parrot, chameleon, zebra, snake, lion (deck 2)
    ];
    let mut rng = Rng::new(23);
    let mut checked = 0;
    for &v in &variants {
        let all = orders(v);
        for history in 0..12 {
            let salt = rng.next_u64();
            let start = (history % 2) as u8;
            // A public history: one game played with sigma from a random deal.
            let (i0, i1) = (rng.below(all.len()), rng.below(all.len()));
            let mut s = deal_from(v, start, &all[i0], &all[i1]);
            let mut actions = Vec::new();
            let mut publics = vec![s];
            while !s.is_over() {
                let p = s.to_move as usize;
                let legal: Vec<u8> = s.legal_actions().collect();
                let weights: Vec<f64> = legal.iter().map(|&a| sigma(&s, s.hands[p], a, salt)).collect();
                let mut x = rng.next_u64() as f64 / u64::MAX as f64 * weights.iter().sum::<f64>();
                let mut a = *legal.last().unwrap();
                for (&b, &w) in legal.iter().zip(&weights) {
                    if x < w {
                        a = b;
                        break;
                    }
                    x -= w;
                }
                s.apply(a);
                actions.push(a);
                publics.push(s);
            }

            // Posterior over (H_0, H_1) after each prefix, by enumerating every deal.
            let mut joint: Vec<HashMap<(u16, u16), f64>> = vec![HashMap::new(); actions.len() + 1];
            for o0 in &all {
                for o1 in &all {
                    let mut t = deal_from(v, start, o0, o1);
                    let mut w = 1.0;
                    for (step, &a) in actions.iter().enumerate() {
                        *joint[step].entry((t.hands[0], t.hands[1])).or_default() += w;
                        assert_eq!(public_key(&t), public_key(&publics[step]));
                        w *= sigma(&t, t.hands[t.to_move as usize], a, salt);
                        if w == 0.0 {
                            break;
                        }
                        t.apply(a);
                    }
                    if w > 0.0 {
                        *joint[actions.len()].entry((t.hands[0], t.hands[1])).or_default() += w;
                    }
                }
            }

            // The filter: one independent range per player, starting uniform over the k-subsets.
            let mut ranges: [Vec<f64>; 2] = std::array::from_fn(|_| {
                let mut r = vec![0.0; num_hands(v.k())];
                for h in hands(v.animals, v.k()) {
                    r[rank(h)] = 1.0;
                }
                r
            });
            for (step, s) in publics.iter().enumerate() {
                let u = [unplayed(s, 0), unplayed(s, 1)];
                let total: f64 = joint[step].values().sum();
                let mass: [f64; 2] = std::array::from_fn(|p| ranges[p].iter().sum());
                for h0 in hands(u[0], v.k()) {
                    for h1 in hands(u[1], v.k()) {
                        let exact = joint[step].get(&(h0, h1)).copied().unwrap_or(0.0) / total;
                        let filtered = ranges[0][rank(h0)] / mass[0] * ranges[1][rank(h1)] / mass[1];
                        assert!((exact - filtered).abs() < 1e-12, "step {step}: {exact} vs {filtered}");
                        checked += 1;
                    }
                }
                let Some(&a) = actions.get(step) else { break };
                let p = s.to_move as usize;
                let c = bbcore::decode(a).animal;
                let mut weighted = vec![0.0; ranges[p].len()];
                for h in hands(u[p], v.k()) {
                    weighted[rank(h)] = ranges[p][rank(h)] * sigma(s, h, a, salt);
                }
                let t = Transition::new(u[p], v.k(), c);
                let mut next = vec![0.0; num_hands(t.to_size as usize)];
                t.push(&weighted, &mut next);
                ranges[p] = next;
            }
        }
    }
    assert!(checked > 1000);
}
