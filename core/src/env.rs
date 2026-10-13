//! Batched environment for self-play training: `n` independent games, each auto-resetting to a fresh random deal
//! when it ends, encoded for the networks ([`crate::features`]).
//!
//! Every env always has a player to move. Env `i` draws its deals from its own RNG stream, derived from
//! `(seed, i)`, and its `k`-th game (k = 0, 1, ...) starts with player `(i + k) % 2`, so a run is reproducible from
//! the seed and the starting seat is balanced. [`VecEnv::reset_deals`] installs given deals (duplicate-deal
//! evaluation) without touching the RNG streams.
//!
//! Every game also carries one agent RNG per seat, for built-in agents acting inside the env (`crate::env_act`) and
//! for the sampling draws of network agents: an agent in seat s builds its views and acts with seat s's RNG, exactly
//! as in the arena. Seeds come from the keyed, non-invertible [`derive_seed`] (SipHash): for env `i`'s `k`-th
//! own-stream game, `derive_seed(seed, [SEAT_STREAM, i, k, s])`; [`VecEnv::reset_arena`] uses the arena's own seat
//! RNGs instead ([`crate::arena::seat_rng`]), so a match played in the env replays the arena move for move. As in
//! the arena, whoever knows the seed can recompute the deals: built-in agents never look (trust model, agents.rs).
//!
//! Episodes end where [`episode_end`] says so. Today that is game over, with the house-rule result; the planned
//! exact-solver integration ends them earlier, at the perfect-information frontier, with the solver's exact value.

use std::thread;

use crate::action::NUM_ACTIONS;
use crate::arena::{self, derive_seed};
use crate::features::{History, OBS_DIM, PRIV_DIM, encode_mask, encode_obs_zeroed, encode_privileged, ply};
use crate::rng::Rng;
use crate::state::State;

/// `derive_seed` path tags of the seat RNGs (distinct from the arena's DEAL = 1 and GAME = 2).
const SEAT_STREAM: u64 = 3;
const SEAT_RESET: u64 = 4;

/// How an episode ended, from player 0's side.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Outcome {
    /// Player 0's game score: 1 win, 0.5 draw, 0 loss (an expected score once episodes can end at a frontier).
    pub p0_score: f32,
    /// Player 0's points minus player 1's.
    pub margin: i32,
    /// Points in the bar per player.
    pub scores: [u8; 2],
}

impl Outcome {
    /// The house-rule result of a finished game: higher points win, equal points draw.
    pub fn game_over(s: &State) -> Outcome {
        let margin = s.scores[0] as i32 - s.scores[1] as i32;
        let p0_score = match margin.signum() {
            1 => 1.0,
            0 => 0.5,
            _ => 0.0,
        };
        Outcome { p0_score, margin, scores: s.scores }
    }
}

/// `Some(outcome)` if the episode ends at `s`.
///
/// This is the single place that decides episode ends, so the exact-solver integration only changes this
/// function: for `ply(s) >= frontier` (both decks known to the player to move) it will return the solver's exact
/// value instead of letting the game be played out.
#[inline]
pub fn episode_end(s: &State) -> Option<Outcome> {
    s.is_over().then(|| Outcome::game_over(s))
}

#[derive(Clone, Debug)]
struct Slot {
    state: State,
    hist: History,
    rng: Rng,
    /// Games started from this slot's own RNG stream.
    games: u64,
    /// Final scores of the last episode that ended in this slot.
    last_scores: [u8; 2],
    /// The agent RNG of each seat in the current game.
    seat_rngs: [Rng; 2],
}

/// What an agent seated in an env acts on: the position, its public history, and the seats' agent RNGs.
pub(crate) struct AgentSlot<'a> {
    pub state: &'a State,
    pub hist: &'a History,
    pub seat_rngs: &'a mut [Rng; 2],
}

/// Independent per-env seed: a SplitMix64 finalizer over `(seed, i)`.
fn stream_seed(seed: u64, i: u64) -> u64 {
    let mut z = seed ^ i.wrapping_add(1).wrapping_mul(0xD1B5_4A32_D192_ED03);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Seat RNGs keyed by `key` at `path` + seat.
fn seat_rngs(key: u64, path: [u64; 3]) -> [Rng; 2] {
    std::array::from_fn(|s| Rng::new(derive_seed(key, &[path[0], path[1], path[2], s as u64])))
}

impl Slot {
    fn new(seed: u64, i: usize) -> Slot {
        let mut slot = Slot {
            state: State::default(),
            hist: History::default(),
            rng: Rng::new(stream_seed(seed, i as u64)),
            games: 0,
            last_scores: [0; 2],
            seat_rngs: [Rng::new(0), Rng::new(0)],
        };
        slot.next_game(seed, i);
        slot
    }

    fn next_game(&mut self, seed: u64, i: usize) {
        let start = ((i as u64 + self.games) % 2) as u8;
        self.state = State::deal(&mut self.rng, start);
        self.hist.clear();
        self.seat_rngs = seat_rngs(seed, [SEAT_STREAM, i as u64, self.games]);
        self.games += 1;
    }
}

/// Envs per thread below which encoding stays on one thread.
const MIN_CHUNK: usize = 128;

pub struct VecEnv {
    slots: Vec<Slot>,
    threads: usize,
    /// The seed: the key of the seat RNGs.
    seed: u64,
}

impl VecEnv {
    /// `n` games from `seed`; `threads` = 0 uses every available core for encoding.
    pub fn new(n: usize, seed: u64, threads: usize) -> VecEnv {
        assert!(n > 0, "need at least one env");
        let threads = if threads == 0 { thread::available_parallelism().map_or(1, |t| t.get()) } else { threads };
        VecEnv { slots: (0..n).map(|i| Slot::new(seed, i)).collect(), threads, seed }
    }

    /// Worker threads for encoding and for built-in agents.
    #[inline]
    pub fn threads(&self) -> usize {
        self.threads
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn state(&self, i: usize) -> &State {
        &self.slots[i].state
    }

    pub fn history(&self, i: usize) -> &History {
        &self.slots[i].hist
    }

    /// Final scores of the last episode that ended in each env (zeros before the first).
    pub fn final_scores(&self) -> Vec<[u8; 2]> {
        self.slots.iter().map(|s| s.last_scores).collect()
    }

    /// Envs per encoding chunk: all of them on one thread when small.
    fn chunk(&self) -> usize {
        let n = self.len();
        if n <= MIN_CHUNK || self.threads <= 1 { n } else { n.div_ceil(self.threads).max(MIN_CHUNK) }
    }

    /// Writes, for every env, the mover's observation into `obs` (`n * OBS_DIM`), the legal mask
    /// (`n * NUM_ACTIONS`), the player to move and the ply. Encodes in parallel.
    pub fn observe(&self, obs: &mut [f32], mask: &mut [u8], mover: &mut [i64], plies: &mut [i64]) {
        self.observe_impl(obs, mask, mover, plies, false);
    }

    /// [`VecEnv::observe`] for an `obs` buffer that is already all zeros (skips clearing it).
    pub fn observe_zeroed(&self, obs: &mut [f32], mask: &mut [u8], mover: &mut [i64], plies: &mut [i64]) {
        self.observe_impl(obs, mask, mover, plies, true);
    }

    fn observe_impl(&self, obs: &mut [f32], mask: &mut [u8], mover: &mut [i64], plies: &mut [i64], zeroed: bool) {
        let n = self.len();
        assert!(obs.len() == n * OBS_DIM && mask.len() == n * NUM_ACTIONS && mover.len() == n && plies.len() == n);
        let chunk = self.chunk();
        let work = |slots: &[Slot], obs: &mut [f32], mask: &mut [u8], mover: &mut [i64], plies: &mut [i64]| {
            if !zeroed {
                obs.fill(0.0);
            }
            for (j, slot) in slots.iter().enumerate() {
                let s = &slot.state;
                encode_obs_zeroed(s, &slot.hist, &mut obs[j * OBS_DIM..(j + 1) * OBS_DIM]);
                encode_mask(s, &mut mask[j * NUM_ACTIONS..(j + 1) * NUM_ACTIONS]);
                mover[j] = s.to_move as i64;
                plies[j] = ply(s) as i64;
            }
        };
        thread::scope(|scope| {
            let mut parts = self
                .slots
                .chunks(chunk)
                .zip(obs.chunks_mut(chunk * OBS_DIM))
                .zip(mask.chunks_mut(chunk * NUM_ACTIONS))
                .zip(mover.chunks_mut(chunk).zip(plies.chunks_mut(chunk)));
            let first = parts.next();
            for (((slots, obs), mask), (mover, plies)) in parts {
                scope.spawn(move || work(slots, obs, mask, mover, plies));
            }
            if let Some((((slots, obs), mask), (mover, plies))) = first {
                work(slots, obs, mask, mover, plies);
            }
        });
    }

    /// Privileged (hidden-information) features of every env's mover, `n * PRIV_DIM`. For critics only.
    pub fn privileged(&self, out: &mut [f32]) {
        assert_eq!(out.len(), self.len() * PRIV_DIM);
        for (slot, out) in self.slots.iter().zip(out.chunks_mut(PRIV_DIM)) {
            encode_privileged(&slot.state, out);
        }
    }

    /// Plays `actions[i]` in env `i` (a negative action leaves env `i` untouched). For every env whose episode
    /// ends, sets `done[i]`, `result[i]` = player 0's game score (1/0.5/0) and `margin[i]` = player 0's points
    /// minus player 1's, and starts its next game; other envs get `false, 0, 0`. All actions are checked before
    /// any is played: an illegal one is an error and changes nothing.
    pub fn step(&mut self, actions: &[i64], done: &mut [bool], result: &mut [f32], margin: &mut [i64]) -> Result<(), String> {
        let n = self.len();
        assert!(actions.len() == n && done.len() == n && result.len() == n && margin.len() == n);
        for (i, (&a, slot)) in actions.iter().zip(&self.slots).enumerate() {
            if a >= 0 && !(a < NUM_ACTIONS as i64 && slot.state.legal_actions().contains(a as u8)) {
                return Err(format!("env {i}: illegal action {a}"));
            }
        }
        for (i, (&a, slot)) in actions.iter().zip(self.slots.iter_mut()).enumerate() {
            done[i] = false;
            result[i] = 0.0;
            margin[i] = 0;
            if a < 0 {
                continue;
            }
            slot.hist.push(&slot.state, a as u8);
            slot.state.apply(a as u8);
            if let Some(end) = episode_end(&slot.state) {
                done[i] = true;
                result[i] = end.p0_score;
                margin[i] = end.margin as i64;
                slot.last_scores = end.scores;
                slot.next_game(self.seed, i);
            }
        }
        Ok(())
    }

    /// Env `i` restarts from `State::deal(&mut Rng::new(seeds[i]), starts[i])` (the deal of `Game.random(seed,
    /// start)`). The envs' own deal streams are untouched. The seat RNGs restart from
    /// `derive_seed(env seed, [SEAT_RESET, i, seeds[i], s])` (the same deal in the same env plays the same way).
    pub fn reset_deals(&mut self, seeds: &[u64], starts: &[u8]) {
        assert!(seeds.len() == self.len() && starts.len() == self.len());
        for (i, ((slot, &seed), &start)) in self.slots.iter_mut().zip(seeds).zip(starts).enumerate() {
            assert!(start < 2, "start must be 0 or 1");
            slot.state = State::deal(&mut Rng::new(seed), start);
            slot.hist.clear();
            slot.seat_rngs = seat_rngs(self.seed, [SEAT_RESET, i as u64, seed]);
        }
    }

    /// Env `j` restarts from game `games[j]` of deal `deals[j]` of an arena match with `seed`
    /// (`core/src/arena.rs`): the deal `arena::deal(seed, deals[j])`, and the seat RNGs the arena gives that game
    /// ([`arena::seat_rng`]). In game 0, agent A sits in seat 0; in game 1, in seat 1. The envs' own deal streams
    /// are untouched.
    pub fn reset_arena(&mut self, seed: u64, deals: &[u32], games: &[u8]) {
        assert!(deals.len() == self.len() && games.len() == self.len());
        for ((slot, &i), &g) in self.slots.iter_mut().zip(deals).zip(games) {
            assert!(g < 2, "game must be 0 or 1");
            slot.state = arena::deal(seed, i);
            slot.hist.clear();
            slot.seat_rngs = std::array::from_fn(|s| arena::seat_rng(seed, i, g, s as u8));
        }
    }

    /// Every env's position, history and seat RNGs, for agents acting in the env (`crate::env_act`).
    pub(crate) fn agent_slots(&mut self) -> impl Iterator<Item = AgentSlot<'_>> {
        self.slots.iter_mut().map(|s| AgentSlot { state: &s.state, hist: &s.hist, seat_rngs: &mut s.seat_rngs })
    }

    /// Puts env `i` at a given position with its public history (`hist.len()` must equal its ply). The seat RNGs
    /// continue their streams.
    pub fn set(&mut self, i: usize, state: State, hist: History) {
        assert_eq!(ply(&state), hist.len(), "history length must equal the ply");
        assert!(episode_end(&state).is_none(), "the position must have a player to move");
        self.slots[i].state = state;
        self.slots[i].hist = hist;
    }
}

/// Seconds spent in each part of `steps` env steps at `n` envs, with uniformly random legal actions.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bench {
    pub observe: f64,
    pub privileged: f64,
    pub step: f64,
    pub decisions: u64,
    pub games: u64,
}

pub fn bench(n: usize, steps: usize, seed: u64, threads: usize) -> Bench {
    use std::time::Instant;
    let mut env = VecEnv::new(n, seed, threads);
    let mut rng = Rng::new(seed ^ 0xABCD);
    let (mut obs, mut mask) = (vec![0f32; n * OBS_DIM], vec![0u8; n * NUM_ACTIONS]);
    let (mut mover, mut plies, mut privileged) = (vec![0i64; n], vec![0i64; n], vec![0f32; n * PRIV_DIM]);
    let (mut actions, mut done, mut result, mut margin) = (vec![0i64; n], vec![false; n], vec![0f32; n], vec![0i64; n]);
    let mut b = Bench::default();
    for _ in 0..steps {
        let t = Instant::now();
        env.observe(&mut obs, &mut mask, &mut mover, &mut plies);
        b.observe += t.elapsed().as_secs_f64();
        let t = Instant::now();
        env.privileged(&mut privileged);
        b.privileged += t.elapsed().as_secs_f64();
        for (i, a) in actions.iter_mut().enumerate() {
            let legal: Vec<usize> = (0..NUM_ACTIONS).filter(|&k| mask[i * NUM_ACTIONS + k] != 0).collect();
            *a = legal[rng.below(legal.len())] as i64;
        }
        let t = Instant::now();
        env.step(&actions, &mut done, &mut result, &mut margin).expect("legal actions");
        b.step += t.elapsed().as_secs_f64();
        b.decisions += n as u64;
        b.games += done.iter().filter(|&&d| d).count() as u64;
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_actions(env: &VecEnv, rng: &mut Rng) -> Vec<i64> {
        (0..env.len())
            .map(|i| {
                let legal = env.state(i).legal_actions();
                legal.nth_action(rng.below(legal.len())) as i64
            })
            .collect()
    }

    /// Runs `steps` steps of random play; returns every (done, result, margin) and the final observations.
    fn run(env: &mut VecEnv, steps: usize, seed: u64) -> (Vec<(bool, f32, i64)>, Vec<f32>) {
        let n = env.len();
        let mut rng = Rng::new(seed);
        let (mut done, mut result, mut margin) = (vec![false; n], vec![0.0; n], vec![0; n]);
        let mut log = Vec::new();
        for _ in 0..steps {
            let actions = random_actions(env, &mut rng);
            env.step(&actions, &mut done, &mut result, &mut margin).unwrap();
            log.extend((0..n).map(|i| (done[i], result[i], margin[i])));
        }
        let mut obs = vec![1.0; n * OBS_DIM];
        let (mut mask, mut mover, mut plies) = (vec![0; n * NUM_ACTIONS], vec![0; n], vec![0; n]);
        env.observe(&mut obs, &mut mask, &mut mover, &mut plies);
        (log, obs)
    }

    #[test]
    fn reproducible_from_seed() {
        // Different thread counts must not change anything.
        let a = run(&mut VecEnv::new(300, 5, 1), 60, 9);
        let b = run(&mut VecEnv::new(300, 5, 4), 60, 9);
        assert_eq!(a, b);
        let c = run(&mut VecEnv::new(300, 6, 4), 60, 9);
        assert_ne!(a.0, c.0);
    }

    #[test]
    fn episodes_end_with_the_house_result_and_alternate_starts() {
        let mut env = VecEnv::new(64, 1, 2);
        let n = env.len();
        let mut rng = Rng::new(2);
        let (mut done, mut result, mut margin) = (vec![false; n], vec![0.0; n], vec![0; n]);
        let mut starts: Vec<Vec<u8>> = (0..n).map(|i| vec![env.state(i).start]).collect();
        let mut finished = 0;
        for _ in 0..200 {
            let before: Vec<State> = (0..n).map(|i| *env.state(i)).collect();
            let actions = random_actions(&env, &mut rng);
            env.step(&actions, &mut done, &mut result, &mut margin).unwrap();
            for i in 0..n {
                let mut s = before[i];
                s.apply(actions[i] as u8);
                assert_eq!(done[i], s.is_over());
                if done[i] {
                    finished += 1;
                    let o = Outcome::game_over(&s);
                    assert_eq!((result[i], margin[i]), (o.p0_score, o.margin as i64));
                    assert_eq!(env.final_scores()[i], s.scores);
                    assert_eq!(ply(env.state(i)), 0, "a finished env starts a fresh game");
                    starts[i].push(env.state(i).start);
                } else {
                    assert_eq!(*env.state(i), s);
                    assert_eq!((result[i], margin[i]), (0.0, 0));
                }
            }
        }
        assert!(finished > 300);
        for (i, st) in starts.iter().enumerate() {
            for (k, &s) in st.iter().enumerate() {
                assert_eq!(s as usize, (i + k) % 2, "starting player alternates per env");
            }
        }
    }

    #[test]
    fn reset_deals_and_skipped_envs() {
        let mut env = VecEnv::new(4, 0, 1);
        env.reset_deals(&[11, 11, 12, 13], &[0, 0, 1, 0]);
        for i in 0..4 {
            let seed = [11, 11, 12, 13][i];
            let start = [0, 0, 1, 0][i];
            assert_eq!(*env.state(i), State::deal(&mut Rng::new(seed), start));
        }
        let before = *env.state(1);
        let a0 = env.state(0).legal_actions().nth_action(0) as i64;
        let (mut done, mut result, mut margin) = (vec![false; 4], vec![0.0; 4], vec![0; 4]);
        let a2 = env.state(2).legal_actions().nth_action(0) as i64;
        let a3 = env.state(3).legal_actions().nth_action(0) as i64;
        env.step(&[a0, -1, a2, a3], &mut done, &mut result, &mut margin).unwrap();
        assert_eq!(*env.state(1), before, "a negative action skips the env");
        assert_eq!(env.history(0).len(), 1);
        assert_eq!(env.history(1).len(), 0);
        // Illegal actions are rejected before anything is played.
        let snapshot: Vec<State> = (0..4).map(|i| *env.state(i)).collect();
        assert!(env.step(&[a0, -1, 99, -1], &mut done, &mut result, &mut margin).is_err());
        assert_eq!((0..4).map(|i| *env.state(i)).collect::<Vec<_>>(), snapshot);
    }

    #[test]
    fn observe_matches_single_encoder() {
        let mut env = VecEnv::new(400, 3, 3);
        let _ = run(&mut env, 7, 1);
        let n = env.len();
        let mut obs = vec![0.0; n * OBS_DIM];
        let (mut mask, mut mover, mut plies) = (vec![0; n * NUM_ACTIONS], vec![0; n], vec![0; n]);
        env.observe(&mut obs, &mut mask, &mut mover, &mut plies);
        for i in 0..n {
            let mut one = vec![0.0; OBS_DIM];
            crate::features::encode_obs(env.state(i), env.history(i), &mut one);
            assert_eq!(&obs[i * OBS_DIM..(i + 1) * OBS_DIM], &one[..]);
            let legal = env.state(i).legal_actions();
            for a in 0..NUM_ACTIONS {
                assert_eq!(mask[i * NUM_ACTIONS + a] != 0, legal.contains(a as u8));
            }
            assert_eq!(mover[i], env.state(i).to_move as i64);
            assert_eq!(plies[i] as usize, env.history(i).len());
            // Ply convention: the starting player moves at even plies.
            assert_eq!(mover[i] as u8, env.state(i).start ^ (plies[i] % 2) as u8);
        }
    }
}
