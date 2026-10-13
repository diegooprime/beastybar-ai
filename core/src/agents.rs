//! Agents, built from spec strings such as `"random"`, `"greedy"` or `"pimc:w=32,d=6"`.
//!
//! An agent sees a position only through [`crate::view::view`] (hidden information freshly randomized), except the
//! oracle specs, which exist to measure the value of information and are never real players:
//! `pimc-hand` sees the opponent's true hand, `cheat` the true full state. The arena passes the true state to these
//! and only these ([`Spec::is_oracle`]).
//!
//! Trust model: an agent learns about the game only through `act` (its view, the history and its own RNG). Agent code
//! must not reach into the arena (`arena::deal`, `derive_seed`) or use `unsafe`; `core/tests/info_leaks.rs` lints this
//! file, search.rs, eval.rs and view.rs for it.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::action::{ActionSet, NUM_ACTIONS};
use crate::eval::evaluate;
use crate::rng::Rng;
use crate::search::Search;
use crate::state::State;
use crate::view::{shuffle_decks, view};

pub trait Agent: Send {
    /// The action to play at `view` (for the player to move). `history` holds every action played so far; movers
    /// alternate starting from `view.start`. Deterministic given `rng`.
    fn act(&mut self, view: &State, history: &[u8], rng: &mut Rng) -> u8;
}

/// A parsed agent spec. `Display` gives the canonical spec string (every parameter explicit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spec {
    /// Uniform over legal actions.
    Random,
    /// One ply: maximize the evaluation after its own move.
    Greedy,
    /// Perfect-information Monte Carlo: average every root action's value over `worlds` fresh views searched
    /// `depth` plies (at least 1; 24 or more reaches the end of the game, affordable only late), play the argmax.
    /// Once the position is perfect information for it (ply 15 on), one world searched to the end, whatever `depth`.
    Pimc { worlds: u32, depth: u32 },
    /// Oracle: PIMC knowing the opponent's true hand (only deck orders are sampled).
    PimcHand { worlds: u32, depth: u32 },
    /// Oracle: search of the true full state.
    Cheat { depth: u32 },
}

const DEFAULT_WORLDS: u32 = 32;
const DEFAULT_DEPTH: u32 = 6;

impl Spec {
    /// Parses `name` or `name:key=value,...`. Unknown names, unknown or repeated keys and bad values are errors.
    pub fn parse(spec: &str) -> Result<Spec, String> {
        let spec = spec.trim();
        let (name, args) = spec.split_once(':').unwrap_or((spec, ""));
        let mut params: Vec<(&str, u32)> = Vec::new();
        for part in args.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            let (k, v) =
                part.split_once('=').ok_or_else(|| format!("{spec:?}: expected key=value, got {part:?}"))?;
            let (k, v) = (k.trim(), v.trim());
            let v = v
                .parse::<u32>()
                .map_err(|_| format!("{spec:?}: {k} must be a non-negative integer, got {v:?}"))?;
            if params.iter().any(|&(seen, _)| seen == k) {
                return Err(format!("{spec:?}: repeated key {k:?}"));
            }
            params.push((k, v));
        }
        let allowed: &[&str] = match name {
            "random" | "greedy" => &[],
            "pimc" | "pimc-hand" => &["w", "d"],
            "cheat" => &["d"],
            _ => {
                return Err(format!(
                    "unknown agent {name:?} (expected random, greedy, pimc, pimc-hand or cheat)"
                ));
            }
        };
        if let Some(&(k, _)) = params.iter().find(|(k, _)| !allowed.contains(k)) {
            return Err(format!("{spec:?}: unknown key {k:?} for {name} (allowed: {})", allowed.join(", ")));
        }
        let get =
            |key: &str, default: u32| params.iter().find(|&&(k, _)| k == key).map_or(default, |&(_, v)| v);
        let (worlds, depth) = (get("w", DEFAULT_WORLDS), get("d", DEFAULT_DEPTH));
        if worlds == 0 {
            return Err(format!("{spec:?}: w must be at least 1"));
        }
        // d=0 used to mean "to the end": a search of hours from the opening, one typo away from d=6.
        if depth == 0 {
            return Err(format!(
                "{spec:?}: d must be at least 1 (d=24 searches to the end of the game; the exact endgame solve from \
                 ply 15 is automatic)"
            ));
        }
        Ok(match name {
            "random" => Spec::Random,
            "greedy" => Spec::Greedy,
            "pimc" => Spec::Pimc { worlds, depth },
            "pimc-hand" => Spec::PimcHand { worlds, depth },
            _ => Spec::Cheat { depth },
        })
    }

    /// Oracle specs receive the true state instead of a view.
    pub fn is_oracle(&self) -> bool {
        matches!(self, Spec::PimcHand { .. } | Spec::Cheat { .. })
    }

    pub fn build(&self) -> Box<dyn Agent> {
        self.build_with(None)
    }

    /// The agent, abandoning its current search as soon as `stop` is set (its move is then meaningless). The arena
    /// builds agents this way so that Ctrl-C interrupts even a search that would take hours.
    pub fn build_stoppable(&self, stop: &Arc<AtomicBool>) -> Box<dyn Agent> {
        self.build_with(Some(stop.clone()))
    }

    fn build_with(&self, stop: Option<Arc<AtomicBool>>) -> Box<dyn Agent> {
        match *self {
            Spec::Random => Box::new(RandomAgent),
            Spec::Greedy => Box::new(GreedyAgent),
            Spec::Pimc { worlds, depth } => Box::new(Pimc::new(Sampling::View, worlds, depth, stop)),
            Spec::PimcHand { worlds, depth } => Box::new(Pimc::new(Sampling::Decks, worlds, depth, stop)),
            Spec::Cheat { depth } => Box::new(Pimc::new(Sampling::None, 1, depth, stop)),
        }
    }
}

impl fmt::Display for Spec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Spec::Random => write!(f, "random"),
            Spec::Greedy => write!(f, "greedy"),
            Spec::Pimc { worlds, depth } => write!(f, "pimc:w={worlds},d={depth}"),
            Spec::PimcHand { worlds, depth } => write!(f, "pimc-hand:w={worlds},d={depth}"),
            Spec::Cheat { depth } => write!(f, "cheat:d={depth}"),
        }
    }
}

/// Builds the agent for a spec string.
pub fn parse(spec: &str) -> Result<Box<dyn Agent>, String> {
    Spec::parse(spec).map(|s| s.build())
}

/// The best actions by `values` (exact ties broken uniformly at random).
fn argmax(actions: ActionSet, values: &[f32; NUM_ACTIONS], rng: &mut Rng) -> u8 {
    let best = actions.map(|a| values[a as usize]).fold(f32::NEG_INFINITY, f32::max);
    let ties = ActionSet(actions.filter(|&a| values[a as usize] == best).fold(0, |set, a| set | 1 << a));
    ties.nth_action(rng.below(ties.len()))
}

pub struct RandomAgent;

impl Agent for RandomAgent {
    fn act(&mut self, view: &State, _history: &[u8], rng: &mut Rng) -> u8 {
        let legal = view.legal_actions();
        legal.nth_action(rng.below(legal.len()))
    }
}

pub struct GreedyAgent;

impl Agent for GreedyAgent {
    fn act(&mut self, view: &State, _history: &[u8], rng: &mut Rng) -> u8 {
        let legal = view.legal_actions();
        let mut values = [0.0; NUM_ACTIONS];
        for a in legal {
            let mut child = *view;
            child.apply(a);
            values[a as usize] = evaluate(&child, view.to_move);
        }
        argmax(legal, &values, rng)
    }
}

/// How a PIMC agent samples worlds from the position it receives.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Sampling {
    /// Fresh views (it receives a view).
    View,
    /// Deck orders only (it receives the true state).
    Decks,
    /// The position itself (it receives the true state).
    None,
}

pub struct Pimc {
    sampling: Sampling,
    worlds: u32,
    depth: u32,
    pub search: Search,
}

impl Pimc {
    fn new(sampling: Sampling, worlds: u32, depth: u32, stop: Option<Arc<AtomicBool>>) -> Pimc {
        let search = match stop {
            Some(stop) => Search::default().with_stop(stop),
            None => Search::default(),
        };
        Pimc { sampling, worlds, depth, search }
    }
}

impl Pimc {
    /// Whether every world this agent could sample is the position it received: its own deck holds at most one
    /// card (known by elimination) and the opponent's split between hand and deck is known (their deck is empty,
    /// or, for the oracles, holds at most one card). From then on the game is perfect information for this agent
    /// (both decks are empty from ply 16; at ply 15 the second mover already knows everything).
    fn determined(&self, s: &State) -> bool {
        let (me, opp) = (s.to_move as usize, 1 - s.to_move as usize);
        let opp_max = if self.sampling == Sampling::View { 0 } else { 1 };
        s.deck_len[me] <= 1 && s.deck_len[opp] <= opp_max
    }
}

impl Agent for Pimc {
    /// Once the stop flag is set, searches abort: the move is still legal but meaningless (the arena discards the
    /// deal).
    fn act(&mut self, state: &State, _history: &[u8], rng: &mut Rng) -> u8 {
        let legal = state.legal_actions();
        if legal.len() == 1 {
            return legal.nth_action(0);
        }
        // Perfect information: one world, searched to the end (exact).
        if self.determined(state) {
            let r = self.search.root_values(state, 0);
            return argmax(legal, &r.values, rng);
        }
        if self.sampling == Sampling::None {
            let r = self.search.root_values(state, self.depth);
            return argmax(legal, &r.values, rng);
        }
        let mut sum = [0.0f32; NUM_ACTIONS];
        for _ in 0..self.worlds {
            let world = match self.sampling {
                Sampling::View => view(state, state.to_move, rng),
                _ => shuffle_decks(state, rng),
            };
            let r = self.search.root_values(&world, self.depth);
            if self.search.aborted() {
                break;
            }
            for a in legal {
                sum[a as usize] += r.values[a as usize];
            }
        }
        argmax(legal, &sum, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_parse_and_print_canonically() {
        let cases = [
            ("random", "random"),
            ("greedy", "greedy"),
            ("pimc:w=16,d=4", "pimc:w=16,d=4"),
            ("pimc: d=4 , w=16", "pimc:w=16,d=4"),
            ("pimc", "pimc:w=32,d=6"),
            ("pimc:d=24", "pimc:w=32,d=24"),
            ("pimc-hand:w=8,d=2", "pimc-hand:w=8,d=2"),
            ("cheat:d=5", "cheat:d=5"),
        ];
        for (input, canonical) in cases {
            let spec = Spec::parse(input).unwrap();
            assert_eq!(spec.to_string(), canonical);
            assert_eq!(Spec::parse(canonical).unwrap(), spec);
        }
        for bad in [
            "",
            "minimax",
            "random:w=1",
            "greedy:d=2",
            "pimc:x=1",
            "pimc:w=1,w=2",
            "pimc:w=0",
            "pimc:w=-1",
            "pimc:w",
            "cheat:w=4",
            "pimc:w=1.5",
            "pimc:d=0",
            "pimc:w=8,d=0",
            "pimc-hand:d=0",
            "cheat:d=0",
        ] {
            assert!(Spec::parse(bad).is_err(), "{bad:?} should not parse");
        }
        assert!(!Spec::parse("pimc").unwrap().is_oracle());
        assert!(Spec::parse("pimc-hand").unwrap().is_oracle());
        assert!(Spec::parse("cheat").unwrap().is_oracle());
    }

    #[test]
    fn agents_play_legal_deterministic_moves() {
        let specs = ["random", "greedy", "pimc:w=3,d=2", "pimc-hand:w=3,d=2", "cheat:d=3", "pimc:w=2,d=24"];
        let mut rng = Rng::new(21);
        for spec in specs {
            for g in 0..20u64 {
                let deal = State::deal(&mut rng, (g % 2) as u8);
                let play = |seed: u64| {
                    let mut agent = parse(spec).unwrap();
                    let mut rng = Rng::new(seed);
                    let mut s = deal;
                    let mut history = Vec::new();
                    // d=24 from the first move is a full-game search: start it late.
                    let skip = if spec.ends_with("d=24") { 12 } else { 0 };
                    while !s.is_over() {
                        let legal = s.legal_actions();
                        let a = if history.len() < skip {
                            legal.nth_action(0)
                        } else {
                            let input = if Spec::parse(spec).unwrap().is_oracle() {
                                s
                            } else {
                                view(&s, s.to_move, &mut rng)
                            };
                            agent.act(&input, &history, &mut rng)
                        };
                        assert!(legal.contains(a), "{spec} played illegal {a}");
                        s.apply(a);
                        history.push(a);
                    }
                    history
                };
                assert_eq!(play(g), play(g), "{spec} not deterministic");
            }
        }
    }

    #[test]
    fn greedy_takes_the_immediate_points() {
        use crate::animal::*;
        // Four animals queued; player 0 completes the queue with a zebra so the front two (its own) enter the bar,
        // or plays a parrot to bounce one of them instead.
        let mut s = State::new(0, [1 << ZEBRA | 1 << PARROT, crate::animal::ALL_ANIMALS], [&[], &[]]);
        for (i, c) in
            [card(0, KANGAROO), card(0, SKUNK), card(1, SEAL), card(1, LION)].into_iter().enumerate()
        {
            s.queue[i] = c;
            s.hands[1] &= !(1 << animal(c));
        }
        s.queue_len = 4;
        let a = GreedyAgent.act(&s, &[], &mut Rng::new(0));
        assert_eq!(a, crate::action::encode(ZEBRA, &[]).unwrap());
    }

    #[test]
    fn perfect_information_endgame_is_solved_exactly() {
        // From the moment the mover knows every card's location (ply 15 for the second mover, then both decks
        // empty), every PIMC flavour plays an exactly best move, whatever its depth.
        let mut rng = Rng::new(22);
        let mut solved = 0;
        while solved < 60 {
            let mut s = State::deal(&mut rng, (solved % 2) as u8);
            let mut ply = 0;
            while !s.is_over() && ply < 15 {
                let legal = s.legal_actions();
                s.apply(legal.nth_action(rng.below(legal.len())));
                ply += 1;
            }
            if s.is_over() || s.legal_actions().len() < 2 {
                continue;
            }
            assert_eq!((s.deck_len[s.to_move as usize], s.deck_len[1 - s.to_move as usize]), (1, 0));
            // At ply 15 the mover's view is the true position.
            assert_eq!(view(&s, s.to_move, &mut rng), s);
            let exact = Search::new(0).root_values(&s, 0);
            let best = exact.actions.map(|a| exact.values[a as usize]).fold(f32::NEG_INFINITY, f32::max);
            for spec in ["pimc:w=4,d=1", "pimc-hand:w=4,d=1", "cheat:d=1"] {
                let a = parse(spec).unwrap().act(&s, &[], &mut rng);
                assert_eq!(exact.values[a as usize], best, "{spec}");
            }
            solved += 1;
            // One ply later both decks are empty.
            let mut t = s;
            t.apply(exact.actions.nth_action(0));
            if !t.is_over() && t.legal_actions().len() > 1 {
                assert_eq!(t.deck_len, [0, 0]);
                let exact = Search::new(0).root_values(&t, 0);
                let best = exact.actions.map(|a| exact.values[a as usize]).fold(f32::NEG_INFINITY, f32::max);
                let a = parse("pimc:w=2,d=1").unwrap().act(&t, &[], &mut rng);
                assert_eq!(exact.values[a as usize], best);
            }
        }
    }

    #[test]
    fn a_stoppable_agent_returns_a_legal_move_promptly_once_stopped() {
        use std::sync::atomic::Ordering;
        use std::time::{Duration, Instant};
        let s = State::deal(&mut Rng::new(24), 1);
        for spec in ["pimc:w=1,d=24", "pimc-hand:w=1,d=24", "cheat:d=24"] {
            let stop = Arc::new(AtomicBool::new(false));
            let mut agent = Spec::parse(spec).unwrap().build_stoppable(&stop);
            let setter = {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(Duration::from_millis(100));
                    stop.store(true, Ordering::Relaxed);
                })
            };
            let t = Instant::now();
            let input =
                if Spec::parse(spec).unwrap().is_oracle() { s } else { view(&s, 1, &mut Rng::new(1)) };
            let a = agent.act(&input, &[], &mut Rng::new(2));
            assert!(t.elapsed() < Duration::from_secs(5), "{spec}: took {:?} to stop", t.elapsed());
            assert!(s.legal_actions().contains(a));
            setter.join().unwrap();
        }
    }

    #[test]
    fn determined_only_with_perfect_information() {
        let mut rng = Rng::new(23);
        let mut s = State::deal(&mut rng, 0);
        let pimc = Pimc::new(Sampling::View, 4, 2, None);
        let hand = Pimc::new(Sampling::Decks, 4, 2, None);
        for ply in 0..16 {
            // Ply 14: both decks hold one card; the opponent's split between hand and deck is still hidden.
            assert_eq!(pimc.determined(&s), ply >= 15, "ply {ply}");
            assert_eq!(hand.determined(&s), ply >= 14, "ply {ply}");
            if s.is_over() {
                break;
            }
            let legal = s.legal_actions();
            s.apply(legal.nth_action(rng.below(legal.len())));
        }
    }
}
