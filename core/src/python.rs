//! Python bindings: `bbar._core.Game`, the engine API checked by `bbar/golden.py`, and `bench`.

use std::time::Instant;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::animal::{NAMES, animal, animal_id, owner};
use crate::state::DECK_CAP;
use crate::{Rng, State, decode, encode, random_playout};

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
    state: State,
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

#[pymodule]
mod _core {
    #[pymodule_export]
    use super::{Game, bench};

    /// Training bindings (VecEnv, observation layout): `python_env.rs`.
    #[pymodule_init]
    fn init(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
        crate::python_env::register(m)
    }
}
