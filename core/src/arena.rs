//! Duplicate-deal matches between two agents, reproducible regardless of thread count.
//!
//! Deal i is `State::deal(Rng(seed, i), start = i % 2)`. Game 2i seats A at 0 and B at 1; game 2i+1 replays the same
//! deal with the seats swapped (A plays seat 1's cards), so card luck cancels within a pair. Each seat of each game
//! has its own RNG derived from (seed, i, game, seat), used both to build that seat's views and by its agent.
//!
//! Every RNG seed comes from [`derive_seed`], SipHash-2-4 keyed by the match seed: a seat's RNG reveals nothing
//! about the seed, so it cannot be traced back to the deal or to the other seat's RNG. That protects the hidden
//! information only while the seed itself cannot be guessed: anyone who knows the seed can recompute every deal.
//! bbar.arena therefore draws a fresh random 64-bit seed by default (and logs it). An explicit seed such as `--seed 1`
//! makes the deals reproducible but enumerable, which is fine for agents that follow the trust model in agents.rs
//! (as every agent in this crate does) and not for untrusted agent code.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use crate::agents::{Agent, Spec};
use crate::rng::Rng;
use crate::state::State;
use crate::view::view;

/// One finished game, from A's point of view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameRecord {
    pub deal: u32,
    /// A's seat: 0 in game 2i, 1 in game 2i+1.
    pub a_seat: u8,
    /// The seat that moved first.
    pub start: u8,
    pub a_points: u8,
    pub b_points: u8,
    /// Every action, in order.
    pub moves: Vec<u8>,
}

/// Shared progress and cancellation for a running match.
#[derive(Default)]
pub struct Control {
    /// Set to stop early: agents abandon their current search, the deal in progress is discarded, and only deals
    /// finished before the stop are returned.
    pub stop: Arc<AtomicBool>,
    /// Deals finished so far.
    pub done: AtomicU32,
}

/// The seed of the RNG at `path` (e.g. deal i, or game g of deal i, seat s) in a match with `seed`: SipHash-2-4 of the
/// path keyed by the seed. A keyed pseudorandom function, so seeds for different paths are independent and no seed
/// (nor the RNG built from it) reveals the match seed or any other path's seed.
pub fn derive_seed(seed: u64, path: &[u64]) -> u64 {
    siphash24(seed, 0x6A09_E667_F3BC_C909, path)
}

/// SipHash-2-4 (Aumasson and Bernstein, 2012) under the key (k0, k1) of the message made of `words`, each as 8
/// little-endian bytes.
pub fn siphash24(k0: u64, k1: u64, words: &[u64]) -> u64 {
    fn round(v: &mut [u64; 4]) {
        v[0] = v[0].wrapping_add(v[1]);
        v[1] = v[1].rotate_left(13) ^ v[0];
        v[0] = v[0].rotate_left(32);
        v[2] = v[2].wrapping_add(v[3]);
        v[3] = v[3].rotate_left(16) ^ v[2];
        v[0] = v[0].wrapping_add(v[3]);
        v[3] = v[3].rotate_left(21) ^ v[0];
        v[2] = v[2].wrapping_add(v[1]);
        v[1] = v[1].rotate_left(17) ^ v[2];
        v[2] = v[2].rotate_left(32);
    }
    fn compress(v: &mut [u64; 4], m: u64) {
        v[3] ^= m;
        round(v);
        round(v);
        v[0] ^= m;
    }
    let mut v = [
        k0 ^ 0x736f_6d65_7073_6575,
        k1 ^ 0x646f_7261_6e64_6f6d,
        k0 ^ 0x6c79_6765_6e65_7261,
        k1 ^ 0x7465_6462_7974_6573,
    ];
    for &m in words {
        compress(&mut v, m);
    }
    // Last block: no leftover bytes, the message length in bytes (mod 256) in the top byte.
    compress(&mut v, ((words.len() as u64 * 8) & 0xff) << 56);
    v[2] ^= 0xff;
    for _ in 0..4 {
        round(&mut v);
    }
    v[0] ^ v[1] ^ v[2] ^ v[3]
}

const DEAL: u64 = 1;
const GAME: u64 = 2;

/// Deal `i` of a match with `seed`.
pub fn deal(seed: u64, i: u32) -> State {
    State::deal(&mut Rng::new(derive_seed(seed, &[DEAL, i as u64])), (i % 2) as u8)
}

/// Plays one game from `state`. `agents[s]` and `rngs[s]` belong to seat s; `oracle[s]` seats get the true state,
/// the others only `view(state, s, rng)`. Returns the final state and every action.
pub fn play_game(
    mut state: State,
    agents: [&mut dyn Agent; 2],
    oracle: [bool; 2],
    rngs: &mut [Rng; 2],
) -> (State, Vec<u8>) {
    let mut history = Vec::with_capacity(24);
    while !state.is_over() {
        let p = state.to_move as usize;
        let input = if oracle[p] { state } else { view(&state, p as u8, &mut rngs[p]) };
        let a = agents[p].act(&input, &history, &mut rngs[p]);
        assert!(state.legal_actions().contains(a), "agent in seat {p} played illegal action {a}");
        state.apply(a);
        history.push(a);
    }
    (state, history)
}

/// The number of threads `play_match` uses: `threads` (0 = available cores), at most one per deal.
pub fn thread_count(threads: usize, deals: u32) -> usize {
    let threads =
        if threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { threads };
    threads.min(deals as usize).max(1)
}

/// Plays `deals` duplicate pairs (2 * deals games) of `a` against `b` on `threads` threads (0 = all cores).
/// Records come back ordered by game (deal, then A's seat). Once `control.stop` is set, agents abandon their searches
/// and the match ends promptly, returning the deals finished before the stop only.
pub fn play_match(
    a: &Spec,
    b: &Spec,
    deals: u32,
    seed: u64,
    threads: usize,
    control: &Control,
) -> Vec<GameRecord> {
    let threads = thread_count(threads, deals);
    let next = AtomicU32::new(0);
    let results = Mutex::new(Vec::with_capacity(2 * deals as usize));
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let (mut agent_a, mut agent_b) =
                    (a.build_stoppable(&control.stop), b.build_stoppable(&control.stop));
                let mut local = Vec::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= deals || control.stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let start = deal(seed, i);
                    let mut pair = Vec::with_capacity(2);
                    for g in 0..2u8 {
                        let mut rngs: [Rng; 2] = std::array::from_fn(|s| {
                            Rng::new(derive_seed(seed, &[GAME, i as u64, g as u64, s as u64]))
                        });
                        let (agents, oracle): ([&mut dyn Agent; 2], _) = if g == 0 {
                            ([agent_a.as_mut(), agent_b.as_mut()], [a.is_oracle(), b.is_oracle()])
                        } else {
                            ([agent_b.as_mut(), agent_a.as_mut()], [b.is_oracle(), a.is_oracle()])
                        };
                        let (end, moves) = play_game(start, agents, oracle, &mut rngs);
                        pair.push(GameRecord {
                            deal: i,
                            a_seat: g,
                            start: start.start,
                            a_points: end.scores[g as usize],
                            b_points: end.scores[1 - g as usize],
                            moves,
                        });
                    }
                    // A stop during the deal may have cut searches short: its games are not results.
                    if control.stop.load(Ordering::Relaxed) {
                        break;
                    }
                    local.extend(pair);
                    control.done.fetch_add(1, Ordering::Relaxed);
                }
                results.lock().unwrap().extend(local);
            });
        }
    });
    let mut records = results.into_inner().unwrap();
    records.sort_by_key(|r| (r.deal, r.a_seat));
    records
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(a: &str, b: &str, deals: u32, seed: u64, threads: usize) -> Vec<GameRecord> {
        play_match(
            &Spec::parse(a).unwrap(),
            &Spec::parse(b).unwrap(),
            deals,
            seed,
            threads,
            &Control::default(),
        )
    }

    #[test]
    fn duplicate_pairs_replay_the_deal_with_seats_swapped() {
        let records = run("random", "greedy", 10, 3, 2);
        assert_eq!(records.len(), 20);
        for (k, r) in records.iter().enumerate() {
            assert_eq!((r.deal, r.a_seat), ((k / 2) as u32, (k % 2) as u8));
            assert_eq!(r.start, (r.deal % 2) as u8);
            // Replaying the moves on the deal reproduces the recorded score.
            let mut s = deal(3, r.deal);
            for &m in &r.moves {
                assert!(s.legal_actions().contains(m));
                s.apply(m);
            }
            assert!(s.is_over());
            assert_eq!(
                (s.scores[r.a_seat as usize], s.scores[1 - r.a_seat as usize]),
                (r.a_points, r.b_points)
            );
        }
        // Deals depend only on (seed, i).
        assert_eq!(deal(3, 4), deal(3, 4));
        assert_ne!(deal(3, 4), deal(4, 4));
    }

    #[test]
    fn results_do_not_depend_on_thread_count() {
        for (a, b) in [("random", "greedy"), ("pimc:w=2,d=4", "cheat:d=5"), ("pimc-hand:w=2,d=3", "greedy")] {
            let one = run(a, b, 12, 9, 1);
            assert_eq!(one, run(a, b, 12, 9, 3));
            assert_eq!(one, run(a, b, 12, 9, 8));
            assert_ne!(one, run(a, b, 12, 10, 1));
        }
    }

    #[test]
    fn siphash_matches_the_reference() {
        // The SipHash-2-4 paper's test key 00 01 .. 0f; messages 00 01 .. (n - 1).
        let (k0, k1) = (0x0706_0504_0302_0100, 0x0f0e_0d0c_0b0a_0908);
        assert_eq!(siphash24(k0, k1, &[]), 0x726f_db47_dd0e_0e31);
        assert_eq!(siphash24(k0, k1, &[0x0706_0504_0302_0100]), 0x93f5_f579_9a93_2462);
        // And std's (deprecated, but specified as SipHash-2-4) implementation on random keys and messages.
        #[allow(deprecated)]
        fn std_siphash(k0: u64, k1: u64, words: &[u64]) -> u64 {
            use std::hash::Hasher;
            let mut h = std::hash::SipHasher::new_with_keys(k0, k1);
            for w in words {
                h.write(&w.to_le_bytes());
            }
            h.finish()
        }
        let mut rng = Rng::new(1);
        for n in 0..40 {
            let (k0, k1) = (rng.next_u64(), rng.next_u64());
            let words: Vec<u64> = (0..n % 7).map(|_| rng.next_u64()).collect();
            assert_eq!(siphash24(k0, k1, &words), std_siphash(k0, k1, &words));
        }
    }

    #[test]
    fn seat_seeds_do_not_give_the_match_seed_back() {
        // The previous derivation chained invertible SplitMix64 steps: from a seat's seed and its (deal, game, seat),
        // a chain of inverse steps gave the match seed back, however large and random it was. SipHash has no such
        // inverse.
        const C: u64 = 0x9E37_79B9_7F4A_7C15;
        fn mix(mut z: u64) -> u64 {
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn inverse(m: u64) -> u64 {
            let mut x = m; // Newton's iteration for the inverse of an odd number mod 2^64.
            for _ in 0..6 {
                x = x.wrapping_mul(2u64.wrapping_sub(m.wrapping_mul(x)));
            }
            x
        }
        fn unmix(mut z: u64) -> u64 {
            z ^= z >> 31 ^ z >> 62;
            z = z.wrapping_mul(inverse(0x94D0_49BB_1331_11EB));
            z ^= z >> 27 ^ z >> 54;
            z = z.wrapping_mul(inverse(0xBF58_476D_1CE4_E5B9));
            z ^ z >> 30 ^ z >> 60
        }
        fn old_derive(seed: u64, path: &[u64]) -> u64 {
            let mut h = seed ^ 0x6A09_E667_F3BC_C909;
            for &x in path {
                h = mix(h.wrapping_add(C).wrapping_add(mix(x)));
            }
            mix(h)
        }
        fn invert(out: u64, path: &[u64]) -> u64 {
            let mut h = unmix(out);
            for &x in path.iter().rev() {
                h = unmix(h).wrapping_sub(C).wrapping_sub(mix(x));
            }
            h ^ 0x6A09_E667_F3BC_C909
        }
        let mut rng = Rng::new(2);
        for _ in 0..100 {
            let seed = rng.next_u64();
            let path = [GAME, rng.below(100_000) as u64, rng.below(2) as u64, rng.below(2) as u64];
            assert_eq!(invert(old_derive(seed, &path), &path), seed);
            assert_ne!(invert(derive_seed(seed, &path), &path), seed);
        }
    }

    #[test]
    fn a_stop_ends_an_endless_match_promptly_with_finished_deals_only() {
        use std::time::{Duration, Instant};
        // pimc to depth 24 from the opening: hours per move. Ctrl-C in Python sets the same flag.
        let control = Control::default();
        let (a, b) = (Spec::parse("pimc:w=1,d=24").unwrap(), Spec::parse("random").unwrap());
        let t = Instant::now();
        let records = std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(200));
                control.stop.store(true, Ordering::Relaxed);
            });
            play_match(&a, &b, 4, 1, 2, &control)
        });
        assert!(t.elapsed() < Duration::from_secs(10), "took {:?}", t.elapsed());
        assert!(records.is_empty());
        assert_eq!(control.done.load(Ordering::Relaxed), 0);
        // Fast agents: whatever finished before the stop comes back in whole, valid pairs.
        let control = Control::default();
        let (a, b) = (Spec::parse("pimc:w=2,d=3").unwrap(), Spec::parse("greedy").unwrap());
        let records = std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(50));
                control.stop.store(true, Ordering::Relaxed);
            });
            play_match(&a, &b, 1_000_000, 1, 2, &control)
        });
        assert_eq!(records.len(), 2 * control.done.load(Ordering::Relaxed) as usize);
        assert!(records.len() < 2_000_000);
        let full = run("pimc:w=2,d=3", "greedy", records.len() as u32 / 2 + 3, 1, 1);
        for r in &records {
            assert_eq!(*r, full[(2 * r.deal + r.a_seat as u32) as usize]);
        }
    }

    #[test]
    fn only_oracle_seats_see_the_true_state() {
        struct Recorder(Vec<State>);
        impl Agent for Recorder {
            fn act(&mut self, view: &State, _: &[u8], _: &mut Rng) -> u8 {
                self.0.push(*view);
                view.legal_actions().nth_action(0)
            }
        }
        let s = deal(5, 0);
        assert_eq!(s.start, 0);
        // Swap a card between player 1's hand and deck: player 0 cannot tell s and t apart.
        let mut t = s;
        let (in_hand, in_deck) = (t.hands[1].trailing_zeros() as u8, t.decks[1][3]);
        t.hands[1] = t.hands[1] & !(1 << in_hand) | 1 << in_deck;
        t.decks[1][3] = in_hand;
        let first_input = |state: State, oracle: bool| {
            let (mut r0, mut r1) = (Recorder(Vec::new()), Recorder(Vec::new()));
            play_game(state, [&mut r0, &mut r1], [oracle, false], &mut [Rng::new(1), Rng::new(2)]);
            r0.0[0]
        };
        assert_eq!(first_input(s, false), first_input(t, false));
        assert_ne!(first_input(s, false), s);
        assert_eq!(first_input(s, true), s);
        assert_eq!(first_input(t, true), t);
    }
}
