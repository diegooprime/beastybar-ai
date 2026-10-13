//! Python bindings: `bbar._core.Game`, the engine API checked by `bbar/golden.py`, `bench`, the arena
//! (`play_match`, `parse_spec`, `arena_deal`) and `build_id`.

use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::agents::Spec;
use crate::arena::{self, Control};

use crate::animal::{NAMES, animal, animal_id, owner};
use crate::state::DECK_CAP;
use crate::{BUILD_ID, Rng, State, decode, encode, random_playout};

fn parse_animal(name: &str) -> PyResult<u8> {
    animal_id(name).ok_or_else(|| PyValueError::new_err(format!("unknown animal {name:?}")))
}

fn player_index(player: i64) -> PyResult<usize> {
    match player {
        0 | 1 => Ok(player as usize),
        _ => Err(PyValueError::new_err(format!("player must be 0 or 1, got {player}"))),
    }
}

fn named(cards: &[u8]) -> Vec<(&'static str, u8)> {
    cards.iter().map(|&c| (NAMES[animal(c) as usize], owner(c))).collect()
}

/// A Beasty Bar game. `Game(start, hands, decks)`: animal names, decks top card first.
#[pyclass(module = "bbar._core", skip_from_py_object)]
#[derive(Clone)]
pub struct Game {
    pub(crate) state: State,
}

#[pymethods]
impl Game {
    #[new]
    fn new(start: i64, hands: Vec<Vec<String>>, decks: Vec<Vec<String>>) -> PyResult<Game> {
        let start = player_index(start)? as u8;
        if hands.len() != 2 || decks.len() != 2 {
            return Err(PyValueError::new_err("expected two hands and two decks"));
        }
        let mut hand_masks = [0u16; 2];
        let mut deck_ids: [Vec<u8>; 2] = Default::default();
        for p in 0..2 {
            let mut seen = 0u16;
            let mut take = |name: &str| {
                let a = parse_animal(name)?;
                if seen >> a & 1 != 0 {
                    return Err(PyValueError::new_err(format!("player {p} has two {name}s")));
                }
                seen |= 1 << a;
                Ok(a)
            };
            for name in &hands[p] {
                hand_masks[p] |= 1 << take(name)?;
            }
            for name in &decks[p] {
                deck_ids[p].push(take(name)?);
            }
            if deck_ids[p].len() > DECK_CAP {
                return Err(PyValueError::new_err(format!("a deck holds at most {DECK_CAP} cards")));
            }
        }
        Ok(Game { state: State::new(start, hand_masks, [&deck_ids[0], &deck_ids[1]]) })
    }

    /// A uniformly random deal from `seed`.
    #[staticmethod]
    fn random(seed: u64, start: i64) -> PyResult<Game> {
        let start = player_index(start)? as u8;
        Ok(Game { state: State::deal(&mut Rng::new(seed), start) })
    }

    #[getter]
    fn to_move(&self) -> u8 {
        self.state.to_move
    }

    #[getter]
    fn start(&self) -> u8 {
        self.state.start
    }

    /// `[(animal, (params...)), ...]` in action-index order.
    fn legal_moves<'py>(&self, py: Python<'py>) -> PyResult<Vec<(&'static str, Bound<'py, PyTuple>)>> {
        self.state
            .legal_actions()
            .map(|a| {
                let m = decode(a);
                Ok((NAMES[m.animal as usize], PyTuple::new(py, m.params())?))
            })
            .collect()
    }

    /// Sorted action indices (bbar/actions.py layout).
    fn legal_actions(&self) -> Vec<usize> {
        self.state.legal_actions().map(usize::from).collect()
    }

    #[pyo3(signature = (animal, params=Vec::new()))]
    fn play(&mut self, animal: &str, params: Vec<i64>) -> PyResult<()> {
        let id = parse_animal(animal)?;
        let action = params
            .iter()
            .map(|&x| u8::try_from(x).ok())
            .collect::<Option<Vec<u8>>>()
            .and_then(|p| encode(id, &p))
            .filter(|&a| self.state.legal_actions().contains(a));
        match action {
            Some(a) => {
                self.state.apply(a);
                Ok(())
            }
            None => Err(PyValueError::new_err(format!("illegal move {animal} {params:?}"))),
        }
    }

    fn play_action(&mut self, action: i64) -> PyResult<()> {
        match u8::try_from(action) {
            Ok(a) if self.state.legal_actions().contains(a) => {
                self.state.apply(a);
                Ok(())
            }
            _ => Err(PyValueError::new_err(format!("illegal action {action}"))),
        }
    }

    /// `[(animal, owner), ...]`, front (bar side) first.
    fn queue(&self) -> Vec<(&'static str, u8)> {
        named(self.state.queue())
    }

    /// `[(animal, owner), ...]` in order of arrival.
    fn bar(&self) -> Vec<(&'static str, u8)> {
        named(self.state.bar_order())
    }

    /// `[(animal, owner), ...]` in order of arrival.
    fn thats_it(&self) -> Vec<(&'static str, u8)> {
        named(self.state.thats_it_order())
    }

    fn score(&self) -> (u8, u8) {
        (self.state.scores[0], self.state.scores[1])
    }

    fn is_over(&self) -> bool {
        self.state.is_over()
    }

    /// Animals in hand, weakest first.
    fn hand(&self, player: i64) -> PyResult<Vec<&'static str>> {
        let hand = self.state.hands[player_index(player)?];
        Ok((0..NAMES.len()).filter(|&a| hand >> a & 1 != 0).map(|a| NAMES[a]).collect())
    }

    /// Animals in the deck, top first.
    fn deck(&self, player: i64) -> PyResult<Vec<&'static str>> {
        let deck = self.state.deck(player_index(player)?);
        Ok(deck.iter().map(|&a| NAMES[a as usize]).collect())
    }

    fn copy(&self) -> Game {
        self.clone()
    }
}

/// Plays `n_games` uniformly random games in Rust; returns `(n_games, n_moves, seconds)`.
#[pyfunction]
fn bench(py: Python<'_>, n_games: u64, seed: u64) -> (u64, u64, f64) {
    py.detach(|| {
        let mut rng = Rng::new(seed);
        let t = Instant::now();
        let mut moves = 0;
        for i in 0..n_games {
            let mut state = State::deal(&mut rng, (i % 2) as u8);
            moves += random_playout(&mut state, &mut rng) as u64;
        }
        (n_games, moves, t.elapsed().as_secs_f64())
    })
}

/// Fingerprint of the Rust sources this extension was compiled from (core/build.rs); bbar.provenance compares it
/// with the current sources to detect a stale build.
#[pyfunction]
fn build_id() -> &'static str {
    BUILD_ID
}

/// Deal `i` of every arena match played with `seed`, as a Game (for replaying recorded moves).
#[pyfunction]
fn arena_deal(seed: u64, i: u32) -> Game {
    Game { state: arena::deal(seed, i) }
}

fn parse(spec: &str) -> PyResult<Spec> {
    Spec::parse(spec).map_err(PyValueError::new_err)
}

/// The canonical form of an agent spec (every parameter explicit); raises ValueError if it is invalid.
#[pyfunction]
fn parse_spec(spec: &str) -> PyResult<String> {
    Ok(parse(spec)?.to_string())
}

/// Plays `deals` duplicate pairs of `spec_a` against `spec_b` (see core/src/arena.rs) on `threads` threads
/// (0 = all cores), without holding the GIL. `seed` has no default: a guessable seed lets agent code that breaks the
/// trust model recompute the deals (bbar.arena draws a random one). Ctrl-C stops it within milliseconds, even
/// mid-search (KeyboardInterrupt). With `progress`, prints the deals done to stderr every 10 s.
///
/// Returns `{"a", "b"` (canonical specs), `"seconds"`, `"threads"` (used), `"build"` (`build_id()`), and per game
/// (ordered by deal, then A's seat): `"deal", "a_seat", "start", "a_points", "b_points"` (lists of ints) and
/// `"moves"` (list of action lists)}.
#[pyfunction]
#[pyo3(signature = (spec_a, spec_b, deals, seed, threads=0, progress=false))]
fn play_match<'py>(
    py: Python<'py>,
    spec_a: &str,
    spec_b: &str,
    deals: u32,
    seed: u64,
    threads: usize,
    progress: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let (a, b) = (parse(spec_a)?, parse(spec_b)?);
    let control = Control::default();
    let t = Instant::now();
    let (records, interrupted) = py.detach(|| {
        std::thread::scope(|scope| {
            let (a, b, control) = (&a, &b, &control);
            let (done_tx, done_rx) = mpsc::channel::<()>();
            let worker = scope.spawn(move || {
                let records = arena::play_match(a, b, deals, seed, threads, control);
                let _ = done_tx.send(());
                records
            });
            let mut interrupted = false;
            let mut last_report = Instant::now();
            while let Err(RecvTimeoutError::Timeout) = done_rx.recv_timeout(Duration::from_millis(50)) {
                if !interrupted && Python::attach(|py| py.check_signals()).is_err() {
                    interrupted = true;
                    control.stop.store(true, Ordering::Relaxed);
                }
                if progress && last_report.elapsed() >= Duration::from_secs(10) {
                    last_report = Instant::now();
                    let done = control.done.load(Ordering::Relaxed);
                    let secs = t.elapsed().as_secs_f64();
                    let eta = if done > 0 { secs * (deals - done) as f64 / done as f64 } else { f64::NAN };
                    eprintln!("  {a} vs {b}: {done}/{deals} deals, {secs:.0}s elapsed, ~{eta:.0}s left");
                }
            }
            (worker.join().expect("arena thread panicked"), interrupted)
        })
    });
    if interrupted {
        return Err(pyo3::exceptions::PyKeyboardInterrupt::new_err("match interrupted"));
    }
    let seconds = t.elapsed().as_secs_f64();
    let d = PyDict::new(py);
    d.set_item("a", a.to_string())?;
    d.set_item("b", b.to_string())?;
    d.set_item("seconds", seconds)?;
    d.set_item("threads", arena::thread_count(threads, deals))?;
    d.set_item("build", BUILD_ID)?;
    // Lists of ints (u32: a Vec<u8> would convert to bytes).
    let column = |f: fn(&arena::GameRecord) -> u32| records.iter().map(f).collect::<Vec<u32>>();
    d.set_item("deal", column(|r| r.deal))?;
    d.set_item("a_seat", column(|r| r.a_seat as u32))?;
    d.set_item("start", column(|r| r.start as u32))?;
    d.set_item("a_points", column(|r| r.a_points as u32))?;
    d.set_item("b_points", column(|r| r.b_points as u32))?;
    let moves: Vec<Vec<u32>> = records.iter().map(|r| r.moves.iter().map(|&m| m as u32).collect()).collect();
    d.set_item("moves", moves)?;
    Ok(d)
}

#[pymodule]
mod _core {
    #[pymodule_export]
    use super::{Game, arena_deal, bench, build_id, parse_spec, play_match};

    /// Training bindings (VecEnv, observation layout) in `python_env.rs`; solvers (Window etc.) in `python_solve.rs`.
    #[pymodule_init]
    fn init(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
        crate::python_env::register(m)?;
        crate::python_tail::register(m)?;
        crate::python_solve::register(m)
    }
}
