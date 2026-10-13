//! Adversarial review tests for search, PIMC and the perfect-information shortcut.
//! Independent re-implementations; they use only the public API.

use bbcore::agents::{Spec, parse};
use bbcore::eval::evaluate;
use bbcore::search::Search;
use bbcore::view::{shuffle_decks, view};
use bbcore::{ActionSet, NUM_ACTIONS, Rng, State};

/// Plain minimax, no pruning, no table, no depth normalization.
fn minimax(s: &State, rem: u32) -> f32 {
    if rem == 0 || s.is_over() {
        return evaluate(s, s.to_move);
    }
    let mut best = f32::NEG_INFINITY;
    for a in s.legal_actions() {
        let mut c = *s;
        c.apply(a);
        best = best.max(-minimax(&c, rem - 1));
    }
    best
}

fn brute(s: &State, depth: u32) -> Vec<(u8, f32)> {
    let rem = if depth == 0 { 64 } else { depth };
    s.legal_actions()
        .map(|a| {
            let mut c = *s;
            c.apply(a);
            (a, -minimax(&c, rem - 1))
        })
        .collect()
}

fn unplayed(s: &State) -> u32 {
    s.hands[0].count_ones() + s.hands[1].count_ones() + s.deck_len[0] as u32 + s.deck_len[1] as u32
}

fn position_at(rng: &mut Rng, plies: u32) -> Option<State> {
    let start = rng.below(2) as u8;
    let mut s = State::deal(rng, start);
    for _ in 0..plies {
        if s.is_over() {
            return None;
        }
        let l = s.legal_actions();
        s.apply(l.nth_action(rng.below(l.len())));
    }
    (!s.is_over()).then_some(s)
}

fn check(search: &mut Search, s: &State, depth: u32, what: &str) {
    let got = search.root_values(s, depth);
    let want = brute(s, depth);
    assert_eq!(got.actions.len(), want.len(), "{what}");
    for (a, v) in want {
        let g = got.values[a as usize];
        assert!(g == v, "{what}: action {a} depth {depth}: search {g} vs minimax {v} in {s:?}");
    }
}

/// Every ply of the game, several depths in shuffled order, tables of 0, 1, 3 and 14 bits shared across all
/// positions (heavy replacement and stale entries), and full solves wherever brute force is affordable.
#[test]
fn search_equals_minimax_at_every_ply_and_depth() {
    let mut rng = Rng::new(0xBEEF);
    let mut searches: Vec<(u32, Search)> = [0, 1, 3, 14].iter().map(|&b| (b, Search::new(b))).collect();
    let mut checked = 0;
    for round in 0..40 {
        for plies in 0..22u32 {
            let Some(s) = position_at(&mut rng, plies) else { continue };
            let mut depths: Vec<u32> = (1..=5).collect();
            if round % 4 == 0 {
                depths.push(6);
            }
            if round % 10 == 0 {
                depths.push(7);
            }
            if unplayed(&s) <= 10 {
                depths.push(0);
            }
            rng.shuffle(&mut depths);
            for &d in &depths {
                for (bits, search) in searches.iter_mut() {
                    check(search, &s, d, &format!("tt_bits={bits} ply={plies}"));
                }
            }
            checked += 1;
        }
    }
    assert!(checked > 600, "{checked}");
}

/// The same positions searched in many PIMC worlds (views), which is how agents use the shared table.
#[test]
fn search_equals_minimax_on_views_with_shared_table() {
    let mut rng = Rng::new(0xCAFE);
    let mut search = Search::new(6);
    for _ in 0..150 {
        let plies = rng.below(16) as u32;
        let Some(s) = position_at(&mut rng, plies) else { continue };
        for _ in 0..4 {
            let w = view(&s, s.to_move, &mut rng);
            check(&mut search, &w, 4, "view d4");
            check(&mut search, &w, 3, "view d3");
        }
    }
}

/// PIMC must play argmax of the summed root values of its sampled worlds. Re-implemented from the spec: draw
/// `w` worlds with `view(state, to_move, rng)`, search each to depth `d`, sum, argmax with uniform tie-break
/// `rng.below(ties)` over the tied actions in increasing order.
#[test]
fn pimc_plays_argmax_of_summed_world_values() {
    let mut rng = Rng::new(77);
    let mut checked = 0;
    while checked < 300 {
        let plies = rng.below(14) as u32;
        let Some(truth) = position_at(&mut rng, plies) else { continue };
        let s = view(&truth, truth.to_move, &mut rng);
        let legal = s.legal_actions();
        let (me, opp) = (s.to_move as usize, 1 - s.to_move as usize);
        if legal.len() < 2 || (s.deck_len[me] <= 1 && s.deck_len[opp] == 0) {
            continue;
        }
        let (w, d) = (1 + rng.below(6) as u32, 1 + rng.below(4) as u32);
        let seed = rng.next_u64();

        let mut r = Rng::new(seed);
        let mut sum = [0f32; NUM_ACTIONS];
        let mut search = Search::new(0);
        for _ in 0..w {
            let world = view(&s, s.to_move, &mut r);
            let v = search.root_values(&world, d);
            for a in legal {
                sum[a as usize] += v.values[a as usize];
            }
        }
        let best = legal.map(|a| sum[a as usize]).fold(f32::NEG_INFINITY, f32::max);
        let ties = ActionSet(legal.filter(|&a| sum[a as usize] == best).fold(0, |m, a| m | 1 << a));
        let want = ties.nth_action(r.below(ties.len()));

        let got = parse(&format!("pimc:w={w},d={d}")).unwrap().act(&s, &[], &mut Rng::new(seed));
        assert_eq!(got, want, "w={w} d={d} in {s:?}");
        checked += 1;
    }
}

/// Whenever an agent takes the perfect-information shortcut, every world it could sample must be the position
/// itself; and once the shortcut applies, the view really is the true state.
#[test]
fn shortcut_fires_only_when_worlds_are_identical() {
    let mut rng = Rng::new(99);
    let mut fired = [0u32; 2];
    for g in 0..3000 {
        let mut s = State::deal(&mut rng, (g % 2) as u8);
        while !s.is_over() {
            let (me, opp) = (s.to_move as usize, 1 - s.to_move as usize);
            let pimc_det = s.deck_len[me] <= 1 && s.deck_len[opp] == 0;
            let hand_det = s.deck_len[me] <= 1 && s.deck_len[opp] <= 1;
            if pimc_det {
                fired[0] += 1;
                for _ in 0..8 {
                    assert_eq!(view(&s, s.to_move, &mut rng), s);
                }
            }
            if hand_det {
                fired[1] += 1;
                for _ in 0..8 {
                    assert_eq!(shuffle_decks(&s, &mut rng), s);
                }
            }
            let l = s.legal_actions();
            s.apply(l.nth_action(rng.below(l.len())));
        }
    }
    assert!(fired[0] > 1000 && fired[1] > fired[0], "{fired:?}");
}

/// When the shortcut applies, every flavour's move must be exactly optimal (full-depth minimax), for any depth.
#[test]
fn shortcut_moves_are_exactly_optimal() {
    let mut rng = Rng::new(1234);
    let mut checked = 0;
    while checked < 200 {
        let start = rng.below(2) as u8;
        let mut s = State::deal(&mut rng, start);
        let stop = 14 + rng.below(4) as u32;
        let mut ply = 0;
        while !s.is_over() && ply < stop {
            let l = s.legal_actions();
            s.apply(l.nth_action(rng.below(l.len())));
            ply += 1;
        }
        if s.is_over() || s.legal_actions().len() < 2 {
            continue;
        }
        let exact = brute(&s, 0);
        let best = exact.iter().map(|&(_, v)| v).fold(f32::NEG_INFINITY, f32::max);
        let (me, opp) = (s.to_move as usize, 1 - s.to_move as usize);
        for (spec, det) in [
            ("pimc:w=2,d=1", s.deck_len[me] <= 1 && s.deck_len[opp] == 0),
            ("pimc-hand:w=2,d=1", s.deck_len[me] <= 1 && s.deck_len[opp] <= 1),
            ("cheat:d=1", s.deck_len[me] <= 1 && s.deck_len[opp] <= 1),
        ] {
            if !det {
                continue;
            }
            let oracle = Spec::parse(spec).unwrap().is_oracle();
            let input = if oracle { s } else { view(&s, s.to_move, &mut rng) };
            let a = parse(spec).unwrap().act(&input, &[], &mut rng);
            let v = exact.iter().find(|&&(x, _)| x == a).unwrap().1;
            assert_eq!(v, best, "{spec} ply {ply}");
        }
        checked += 1;
    }
}

/// Heavier version of the minimax check (run with --ignored): 20k positions, depths 1..=6, full solves to 12 plies.
#[test]
#[ignore]
fn stress_search_equals_minimax() {
    let mut rng = Rng::new(0x5EED);
    let mut searches: Vec<(u32, Search)> = [0, 2, 10, 14].iter().map(|&b| (b, Search::new(b))).collect();
    for round in 0..1000 {
        for plies in 0..22u32 {
            let Some(s) = position_at(&mut rng, plies) else { continue };
            let mut depths: Vec<u32> = (1..=5).collect();
            if round % 3 == 0 {
                depths.push(6);
            }
            if unplayed(&s) <= 11 || (unplayed(&s) == 12 && round % 5 == 0) {
                depths.push(0);
            }
            rng.shuffle(&mut depths);
            for &d in &depths {
                for (bits, search) in searches.iter_mut() {
                    check(search, &s, d, &format!("tt_bits={bits} ply={plies}"));
                }
            }
        }
    }
}
