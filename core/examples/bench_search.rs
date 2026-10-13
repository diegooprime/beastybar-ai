//! Search speed: nodes/sec at fixed depths on mid-game positions, and full-depth solves with and without the
//! transposition table.
//!
//!     cargo run --release --example bench_search [positions]

use std::time::Instant;

use bbcore::search::{DEFAULT_TT_BITS, Search};
use bbcore::{Rng, State};

/// Random positions `lo..hi` plies into random games (true, fully determinized states).
fn positions(n: usize, lo: usize, hi: usize, seed: u64) -> Vec<State> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    while out.len() < n {
        let mut s = State::deal(&mut rng, 0);
        let plies = lo + rng.below(hi - lo);
        for _ in 0..plies {
            if s.is_over() {
                break;
            }
            let legal = s.legal_actions();
            s.apply(legal.nth_action(rng.below(legal.len())));
        }
        if !s.is_over() && s.legal_actions().len() > 1 {
            out.push(s);
        }
    }
    out
}

/// Positions where both decks have just run out (perfect information, up to 8 plies left).
fn endgames(n: usize, seed: u64) -> Vec<State> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    while out.len() < n {
        let mut s = State::deal(&mut rng, 0);
        while !s.is_over() && s.deck_len != [0, 0] {
            let legal = s.legal_actions();
            s.apply(legal.nth_action(rng.below(legal.len())));
        }
        if !s.is_over() && s.legal_actions().len() > 1 {
            out.push(s);
        }
    }
    out
}

/// Searches every state (best of 5 timed passes, table allocated outside the timing; a fresh table each pass).
fn run(label: &str, states: &[State], depth: u32, tt_bits: u32) -> f64 {
    let mut best = f64::INFINITY;
    let mut nodes = 0;
    let mut check = 0.0f64;
    for _ in 0..5 {
        let mut search = Search::new(tt_bits);
        check = 0.0;
        let t = Instant::now();
        for s in states {
            let r = search.root_values(s, depth);
            check += r.actions.map(|a| r.values[a as usize] as f64).sum::<f64>();
        }
        best = best.min(t.elapsed().as_secs_f64());
        nodes = search.nodes;
    }
    println!(
        "{label:<34} tt={tt:<3} {n:>5} searches {nodes:>10} nodes {us:>9.1} us/search {mnps:>6.2} M nodes/s (checksum {check:.1})",
        tt = if tt_bits > 0 { format!("2^{tt_bits}") } else { "off".into() },
        n = states.len(),
        us = 1e6 * best / states.len() as f64,
        mnps = nodes as f64 / best / 1e6,
    );
    best
}

fn main() {
    let n: usize = std::env::args().nth(1).map_or(400, |s| s.parse().expect("positions"));
    let mid = positions(n, 0, 12, 1);
    // Table sizes in bits (0 = none); agents use DEFAULT_TT_BITS.
    let tts = [0, 12, DEFAULT_TT_BITS, 16, 18];
    for depth in [4, 6, 8] {
        for tt in tts {
            run(&format!("mid-game, depth {depth}"), &mid, depth, tt);
        }
    }
    let end = endgames(n, 2);
    for tt in tts {
        run("endgame (decks empty), to the end", &end, 0, tt);
    }
    let late = positions(n / 4, 13, 15, 3);
    for tt in tts {
        run("13-14 plies in, to the end", &late, 0, tt);
    }
}
