//! Built-in agents (agents.rs specs) acting inside a [`VecEnv`], so that every kind of agent can play every other in
//! one batched environment: networks act from observations, built-in agents through [`VecEnv::act`].
//!
//! An agent acts exactly as in the arena (`arena::play_game`): the player to move gets `view(state, p, rng)` (oracle
//! specs: the true state) and the history of every action played so far, and acts with `rng`, where `rng` is that
//! seat's agent RNG in that env (`crate::env`): after [`VecEnv::reset_arena`] these are the arena's own seat RNGs, so
//! an arena match replayed in a VecEnv makes the same decisions move for move (tested below and in
//! tests/test_match.py). Decisions depend only on the env's position, history and seat RNG, never on the thread
//! count or on which pooled agent instance handles the env (search tables do not change decisions: info_leaks.rs).
//!
//! [`VecEnv::uniforms`] gives network agents their sampling draws from the same seat RNGs: one draw per decision.
//! [`VecEnv::views`] gives agents written in Python the same draw (so a Python agent that samples a network with it
//! replays that network move for move), plus the mover's view and a private seed from a stream derived from that
//! draw (the seat RNG is not advanced further).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::agents::{Agent, Spec};
use crate::arena::derive_seed;
use crate::env::{AgentSlot, VecEnv};
use crate::features::MAX_PLIES;
use crate::rng::Rng;
use crate::state::State;
use crate::view::view;

/// `derive_seed` path tag of the private stream of a Python agent's decision ([`VecEnv::views`]).
const PRIVATE_STREAM: u64 = 5;

/// A uniform in [0, 1) from the top 53 bits of a draw.
fn unit(x: u64) -> f64 {
    (x >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

/// What an agent outside this crate gets for one decision ([`VecEnv::views`]).
#[derive(Clone, Debug, PartialEq)]
pub struct OutsideDecision {
    /// The mover's view: own hand and public state true, hidden cards freshly randomized.
    pub view: State,
    /// Every action played so far.
    pub history: Vec<u8>,
    /// A seed for the agent's own private random stream.
    pub seed: u64,
    /// The seat RNG's draw for this decision, as [`VecEnv::uniforms`] would give it.
    pub uniform: f64,
}

/// One agent per worker thread for one spec, reusable across calls (searchers keep their allocations).
pub struct AgentPool {
    spec: Option<Spec>,
    oracle: bool,
    agents: Vec<Box<dyn Agent>>,
    stop: Arc<AtomicBool>,
}

impl AgentPool {
    /// `threads` agents for `spec` (0 = one per available core), abandoning their searches once
    /// [`AgentPool::stop_flag`] is set.
    pub fn new(spec: Spec, threads: usize) -> AgentPool {
        let threads =
            if threads == 0 { thread::available_parallelism().map_or(1, |t| t.get()) } else { threads };
        let stop = Arc::new(AtomicBool::new(false));
        let agents = (0..threads.max(1)).map(|_| spec.build_stoppable(&stop)).collect();
        AgentPool { spec: Some(spec), oracle: spec.is_oracle(), agents, stop }
    }

    /// A pool of given agents (one per thread); `oracle` agents get the true state.
    pub fn from_agents(agents: Vec<Box<dyn Agent>>, oracle: bool) -> AgentPool {
        assert!(!agents.is_empty(), "need at least one agent");
        AgentPool { spec: None, oracle, agents, stop: Arc::new(AtomicBool::new(false)) }
    }

    pub fn spec(&self) -> Option<Spec> {
        self.spec
    }

    pub fn threads(&self) -> usize {
        self.agents.len()
    }

    /// Set it to abandon the current [`VecEnv::act`] call promptly (its actions are then meaningless, and the seat
    /// RNGs of the envs involved have advanced by an unspecified amount). Each call clears it when it starts.
    pub fn stop_flag(&self) -> &Arc<AtomicBool> {
        &self.stop
    }
}

/// Envs handed to a worker at a time: small, because one PIMC decision can cost much more than another.
const CHUNK: usize = 4;

/// The agent's action in one env, as `arena::play_game` would get it.
fn decide(agent: &mut dyn Agent, oracle: bool, slot: &mut AgentSlot<'_>) -> u8 {
    let s = slot.state;
    let p = s.to_move as usize;
    let rng = &mut slot.seat_rngs[p];
    let input = if oracle { *s } else { view(s, p as u8, rng) };
    let mut history = [0u8; MAX_PLIES];
    let plies = slot.hist.plies();
    for (h, ply) in history.iter_mut().zip(plies) {
        *h = ply.action;
    }
    let a = agent.act(&input, &history[..plies.len()], rng);
    assert!(s.legal_actions().contains(a), "agent played illegal action {a}");
    a
}

impl VecEnv {
    /// The action of `pool`'s agent for the player to move in each env of `envs` (distinct indices), written to
    /// `out[k]` for `envs[k]`; nothing is played. Runs on the pool's threads, each taking a few envs at a time.
    /// Errors on a bad or repeated index (before any agent acts).
    pub fn act(&mut self, pool: &mut AgentPool, envs: &[usize], out: &mut [u8]) -> Result<(), String> {
        let n = self.len();
        assert_eq!(out.len(), envs.len(), "one output per env");
        let mut pos = vec![usize::MAX; n];
        for (k, &i) in envs.iter().enumerate() {
            if i >= n {
                return Err(format!("env index {i} out of range ({n} envs)"));
            }
            if pos[i] != usize::MAX {
                return Err(format!("env {i} given twice"));
            }
            pos[i] = k;
        }
        let mut work: Vec<(usize, AgentSlot<'_>)> = self
            .agent_slots()
            .enumerate()
            .filter(|(i, _)| pos[*i] != usize::MAX)
            .map(|(i, s)| (pos[i], s))
            .collect();
        pool.stop.store(false, Ordering::Relaxed);
        let oracle = pool.oracle;
        let threads = pool.agents.len().min(work.len().div_ceil(CHUNK)).max(1);
        let queue = Mutex::new(work.chunks_mut(CHUNK));
        let worker = |agent: &mut Box<dyn Agent>| {
            let mut local = Vec::new();
            loop {
                let Some(chunk) = queue.lock().unwrap().next() else { break };
                for (k, slot) in chunk.iter_mut() {
                    local.push((*k, decide(agent.as_mut(), oracle, slot)));
                }
            }
            local
        };
        let worker = &worker;
        let decided: Vec<Vec<(usize, u8)>> = thread::scope(|scope| {
            let (first, rest) = pool.agents[..threads].split_first_mut().expect("at least one agent");
            let handles: Vec<_> = rest.iter_mut().map(|agent| scope.spawn(move || worker(agent))).collect();
            let mut all = vec![worker(first)];
            all.extend(handles.into_iter().map(|h| h.join().expect("agent thread panicked")));
            all
        });
        for (k, a) in decided.into_iter().flatten() {
            out[k] = a;
        }
        Ok(())
    }

    /// For agents written outside this crate (Python agents, bbar/match.py), one decision per env of `envs` (distinct
    /// indices), for the player to move there. Takes exactly one draw `x` from that seat's agent RNG, the draw a
    /// network in that seat takes ([`VecEnv::uniforms`]), and returns it as `uniform`; the view (`view(state, p, rng)`,
    /// as a built-in agent gets it: hidden cards freshly randomized) and the private `seed` come from a stream keyed by
    /// `x` (`derive_seed(x, [PRIVATE_STREAM])`), so the seat RNG advances by one draw per decision whatever the agent
    /// is: a Python agent sampling a network with `uniform` makes that network's decisions move for move. Nothing is
    /// played.
    pub fn views(&mut self, envs: &[usize]) -> Result<Vec<OutsideDecision>, String> {
        let n = self.len();
        let mut seen = vec![false; n];
        for &i in envs {
            if i >= n {
                return Err(format!("env index {i} out of range ({n} envs)"));
            }
            if std::mem::replace(&mut seen[i], true) {
                return Err(format!("env {i} given twice"));
            }
        }
        let mut slots: Vec<AgentSlot<'_>> = self.agent_slots().collect();
        Ok(envs
            .iter()
            .map(|&i| {
                let slot = &mut slots[i];
                let s = slot.state;
                let p = s.to_move;
                let x = slot.seat_rngs[p as usize].next_u64();
                let mut private = Rng::new(derive_seed(x, &[PRIVATE_STREAM]));
                let view = view(s, p, &mut private);
                let seed = private.next_u64();
                let history = slot.hist.plies().iter().map(|ply| ply.action).collect();
                OutsideDecision { view, history, seed, uniform: unit(x) }
            })
            .collect())
    }

    /// One uniform draw in [0, 1) (53 bits) from the agent RNG of the player to move in each env of `envs`: the
    /// sampling randomness of a network agent in that seat (inverse-CDF sampling, bbar/match.py).
    pub fn uniforms(&mut self, envs: &[usize], out: &mut [f64]) -> Result<(), String> {
        let n = self.len();
        assert_eq!(out.len(), envs.len(), "one output per env");
        if let Some(&i) = envs.iter().find(|&&i| i >= n) {
            return Err(format!("env index {i} out of range ({n} envs)"));
        }
        let mut slots: Vec<AgentSlot<'_>> = self.agent_slots().collect();
        for (o, &i) in out.iter_mut().zip(envs) {
            let slot = &mut slots[i];
            let p = slot.state.to_move as usize;
            *o = unit(slot.seat_rngs[p].next_u64());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{self, Control, GameRecord};
    use crate::rng::Rng;
    use crate::state::State;

    /// Plays `deals` duplicate pairs of `a` against `b` the way bbar/match.py does: env 2i+g holds game g of deal i
    /// (`reset_arena`), every step each spec acts in the envs where its seat is to move, then all envs step.
    fn env_match(a: &str, b: &str, deals: u32, seed: u64, threads: usize) -> Vec<GameRecord> {
        let n = 2 * deals as usize;
        let mut env = VecEnv::new(n, 0, threads);
        let deal_idx: Vec<u32> = (0..n as u32).map(|j| j / 2).collect();
        let games: Vec<u8> = (0..n).map(|j| (j % 2) as u8).collect();
        env.reset_arena(seed, &deal_idx, &games);
        let starts: Vec<u8> = (0..n).map(|j| env.state(j).start).collect();
        let mut pools = [
            AgentPool::new(Spec::parse(a).unwrap(), threads),
            AgentPool::new(Spec::parse(b).unwrap(), threads),
        ];
        let mut moves: Vec<Vec<u8>> = vec![Vec::new(); n];
        let mut finished = vec![false; n];
        let mut scores = vec![[0u8; 2]; n];
        let (mut done, mut result, mut margin) = (vec![false; n], vec![0.0; n], vec![0i64; n]);
        while !finished.iter().all(|&f| f) {
            let mut actions = vec![-1i64; n];
            for (who, pool) in pools.iter_mut().enumerate() {
                // A (who = 0) sits in seat g of game g.
                let envs: Vec<usize> = (0..n)
                    .filter(|&j| !finished[j] && (env.state(j).to_move as usize == j % 2) == (who == 0))
                    .collect();
                let mut out = vec![0u8; envs.len()];
                env.act(pool, &envs, &mut out).unwrap();
                for (&j, &a) in envs.iter().zip(&out) {
                    actions[j] = a as i64;
                    moves[j].push(a);
                }
            }
            env.step(&actions, &mut done, &mut result, &mut margin).unwrap();
            for j in 0..n {
                if done[j] && !finished[j] {
                    finished[j] = true;
                    scores[j] = env.final_scores()[j];
                }
            }
        }
        (0..n)
            .map(|j| GameRecord {
                deal: (j / 2) as u32,
                a_seat: (j % 2) as u8,
                start: starts[j],
                a_points: scores[j][j % 2],
                b_points: scores[j][1 - j % 2],
                moves: moves[j].clone(),
            })
            .collect()
    }

    #[test]
    fn agents_in_the_env_replay_the_arena_move_for_move() {
        let pairs =
            [("random", "greedy"), ("pimc:w=3,d=2", "cheat:d=3"), ("pimc-hand:w=2,d=2", "pimc:w=2,d=3")];
        for (a, b) in pairs {
            let (sa, sb) = (Spec::parse(a).unwrap(), Spec::parse(b).unwrap());
            let deals = 10;
            let arena = arena::play_match(&sa, &sb, deals, 7, 2, &Control::default());
            for threads in [1, 3] {
                assert_eq!(env_match(a, b, deals, 7, threads), arena, "{a} vs {b}, {threads} threads");
            }
        }
    }

    /// Every (input, history) an agent was given.
    type Seen = Arc<Mutex<Vec<(State, Vec<u8>)>>>;

    /// Records what it is given; plays the first legal action.
    struct Recorder(Seen);

    impl Agent for Recorder {
        fn act(&mut self, input: &State, history: &[u8], _: &mut Rng) -> u8 {
            self.0.lock().unwrap().push((*input, history.to_vec()));
            input.legal_actions().nth_action(0)
        }
    }

    #[test]
    fn agents_get_views_and_the_move_history() {
        for oracle in [false, true] {
            let seen = Arc::new(Mutex::new(Vec::new()));
            let mut pool = AgentPool::from_agents(
                (0..3).map(|_| Box::new(Recorder(seen.clone())) as Box<dyn Agent>).collect(),
                oracle,
            );
            let n = 40;
            let mut env = VecEnv::new(n, 3, 2);
            let mut rng = Rng::new(1);
            let (mut done, mut result, mut margin) = (vec![false; n], vec![0.0; n], vec![0i64; n]);
            let mut hidden_differs = 0;
            for step in 0..30 {
                let envs: Vec<usize> = (0..n).filter(|&i| (i + step) % 3 != 0).collect();
                let mut out = vec![0u8; envs.len()];
                seen.lock().unwrap().clear();
                env.act(&mut pool, &envs, &mut out).unwrap();
                let mut seen = seen.lock().unwrap().clone();
                assert_eq!(seen.len(), envs.len());
                // Every env's agent got its move history and, oracles, the true state; others a position in the
                // mover's information set (a view of it equals a view of the true state under the same RNG).
                for &i in &envs {
                    let s = env.state(i);
                    let p = s.to_move;
                    let hist: Vec<u8> = env.history(i).plies().iter().map(|p| p.action).collect();
                    let k = seen
                        .iter()
                        .position(|(v, h)| {
                            *h == hist
                                && if oracle {
                                    v == s
                                } else {
                                    view(v, p, &mut Rng::new(9)) == view(s, p, &mut Rng::new(9))
                                }
                        })
                        .unwrap_or_else(|| {
                            panic!("env {i}: no agent got its history and information (oracle {oracle})")
                        });
                    let (v, _) = seen.swap_remove(k);
                    hidden_differs += (v != *s) as usize;
                }
                let actions: Vec<i64> = (0..n)
                    .map(|i| {
                        let legal = env.state(i).legal_actions();
                        legal.nth_action(rng.below(legal.len())) as i64
                    })
                    .collect();
                env.step(&actions, &mut done, &mut result, &mut margin).unwrap();
            }
            if !oracle {
                assert!(hidden_differs > 100, "views hide the hidden cards");
            }
        }
    }

    #[test]
    fn decisions_are_reproducible_from_the_env_seed_for_any_thread_count() {
        let run = |seed: u64, threads: usize| {
            let n = 64;
            let mut env = VecEnv::new(n, seed, threads);
            let mut pool = AgentPool::new(Spec::parse("pimc:w=2,d=2").unwrap(), threads);
            let (mut done, mut result, mut margin) = (vec![false; n], vec![0.0; n], vec![0i64; n]);
            let mut log = Vec::new();
            for _ in 0..60 {
                // Unordered, partial index lists: the output follows the given order.
                let envs: Vec<usize> = (0..n).rev().filter(|i| i % 5 != 1).collect();
                let mut out = vec![0u8; envs.len()];
                env.act(&mut pool, &envs, &mut out).unwrap();
                let mut u = vec![0.0; envs.len()];
                env.uniforms(&envs, &mut u).unwrap();
                assert!(u.iter().all(|&x| (0.0..1.0).contains(&x)));
                let mut actions: Vec<i64> =
                    (0..n).map(|i| env.state(i).legal_actions().nth_action(0) as i64).collect();
                for (&i, &a) in envs.iter().zip(&out) {
                    assert!(env.state(i).legal_actions().contains(a));
                    actions[i] = a as i64;
                }
                env.step(&actions, &mut done, &mut result, &mut margin).unwrap();
                log.push((out, u, result.clone()));
            }
            log
        };
        let one = run(5, 1);
        assert_eq!(one, run(5, 4));
        assert_ne!(one, run(6, 4));
    }

    #[test]
    fn seat_rngs_restart_with_every_game() {
        // The same deal installed twice in the same env plays the same way; another env (other seat RNGs) need not.
        let spec = Spec::parse("random").unwrap();
        let first_moves = |env: &mut VecEnv, pool: &mut AgentPool| {
            env.reset_deals(&[11, 11, 11, 11], &[0, 0, 1, 1]);
            let mut out = vec![0u8; 4];
            let mut log = Vec::new();
            let (mut d, mut r, mut m) = (vec![false; 4], vec![0.0; 4], vec![0i64; 4]);
            for _ in 0..10 {
                env.act(pool, &[0, 1, 2, 3], &mut out).unwrap();
                log.push(out.clone());
                env.step(&out.iter().map(|&a| a as i64).collect::<Vec<_>>(), &mut d, &mut r, &mut m).unwrap();
            }
            log
        };
        let mut env = VecEnv::new(4, 1, 1);
        let mut pool = AgentPool::new(spec, 1);
        let a = first_moves(&mut env, &mut pool);
        assert_eq!(a, first_moves(&mut env, &mut pool));
        let col = |j: usize| a.iter().map(|o| o[j]).collect::<Vec<_>>();
        assert_ne!(col(0), col(1), "seat RNGs differ between envs");
        // Arena resets: the arena's seat RNGs, whatever the env's seed.
        let mut other = VecEnv::new(2, 99, 1);
        let mut env2 = VecEnv::new(2, 5, 2);
        for e in [&mut other, &mut env2] {
            e.reset_arena(3, &[4, 4], &[0, 1]);
        }
        let mut o1 = vec![0u8; 2];
        let mut o2 = vec![0u8; 2];
        other.act(&mut pool, &[0, 1], &mut o1).unwrap();
        env2.act(&mut pool, &[0, 1], &mut o2).unwrap();
        assert_eq!(o1, o2);
    }

    #[test]
    fn views_for_outside_agents_are_the_movers_information_from_the_seat_rng() {
        // A twin env asks for the network draws (`uniforms`) where the first asks for views: the draws are the
        // same, so the seat RNGs advance by exactly one draw per decision in both.
        let run = |threads: usize| {
            let n = 30;
            let mut env = VecEnv::new(n, 4, threads);
            let mut twin = VecEnv::new(n, 4, threads);
            let mut rng = Rng::new(3);
            let (mut done, mut result, mut margin) = (vec![false; n], vec![0.0; n], vec![0i64; n]);
            let mut log = Vec::new();
            let mut hidden_differs = 0;
            for _ in 0..25 {
                let envs: Vec<usize> = (0..n).rev().filter(|i| i % 4 != 2).collect();
                let got = env.views(&envs).unwrap();
                let mut want_u = vec![0.0; envs.len()];
                twin.uniforms(&envs, &mut want_u).unwrap();
                for ((&i, d), &u) in envs.iter().zip(&got).zip(&want_u) {
                    let s = env.state(i);
                    let p = s.to_move;
                    // In the mover's information set: its view of the view is its view of the truth.
                    assert_eq!(view(&d.view, p, &mut Rng::new(1)), view(s, p, &mut Rng::new(1)));
                    let want: Vec<u8> = env.history(i).plies().iter().map(|x| x.action).collect();
                    assert_eq!(d.history, want);
                    assert_eq!(d.uniform, u);
                    assert!((0.0..1.0).contains(&d.uniform));
                    hidden_differs += (d.view != *s) as usize;
                }
                log.push(got);
                let actions: Vec<i64> = (0..n)
                    .map(|i| {
                        let legal = env.state(i).legal_actions();
                        legal.nth_action(rng.below(legal.len())) as i64
                    })
                    .collect();
                env.step(&actions, &mut done, &mut result, &mut margin).unwrap();
                twin.step(&actions, &mut done, &mut result, &mut margin).unwrap();
            }
            assert!(hidden_differs > 100);
            let seeds: std::collections::HashSet<u64> = log.iter().flatten().map(|d| d.seed).collect();
            assert_eq!(seeds.len(), log.iter().map(Vec::len).sum::<usize>());
            log
        };
        assert_eq!(run(1), run(3));
        let mut env = VecEnv::new(4, 0, 1);
        assert!(env.views(&[1, 1]).is_err());
        assert!(env.views(&[4]).is_err());
    }

    #[test]
    fn bad_indices_are_errors_before_anything_is_played() {
        let mut env = VecEnv::new(4, 0, 1);
        let mut pool = AgentPool::new(Spec::parse("random").unwrap(), 2);
        let mut out = vec![0u8; 2];
        assert!(env.act(&mut pool, &[1, 1], &mut out).is_err());
        assert!(env.act(&mut pool, &[0, 4], &mut out).is_err());
        let mut u = vec![0.0; 1];
        assert!(env.uniforms(&[4], &mut u).is_err());
        env.act(&mut pool, &[], &mut []).unwrap();
    }

    #[test]
    fn a_stop_abandons_a_long_search_promptly() {
        use std::time::{Duration, Instant};
        let mut env = VecEnv::new(4, 0, 2);
        let mut pool = AgentPool::new(Spec::parse("pimc:w=1,d=24").unwrap(), 2);
        let stop = pool.stop_flag().clone();
        let t = Instant::now();
        thread::scope(|scope| {
            scope.spawn(|| {
                thread::sleep(Duration::from_millis(100));
                stop.store(true, Ordering::Relaxed);
            });
            let mut out = vec![0u8; 4];
            env.act(&mut pool, &[0, 1, 2, 3], &mut out).unwrap();
            for (i, &a) in out.iter().enumerate() {
                assert!(env.state(i).legal_actions().contains(a));
            }
        });
        assert!(t.elapsed() < Duration::from_secs(10), "took {:?}", t.elapsed());
        // A stop left over from an interrupted call does not cut the next call short.
        let spec = Spec::parse("pimc:w=3,d=3").unwrap();
        let mut stale = AgentPool::new(spec, 2);
        stale.stop_flag().store(true, Ordering::Relaxed);
        let (mut a, mut b) = (vec![0u8; 4], vec![0u8; 4]);
        for (pool, out) in [(&mut stale, &mut a), (&mut AgentPool::new(spec, 2), &mut b)] {
            let mut env = VecEnv::new(4, 3, 2);
            env.act(pool, &[0, 1, 2, 3], out).unwrap();
        }
        assert_eq!(a, b);
    }
}
