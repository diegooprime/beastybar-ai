//! Beasty Bar engine: 2 players, house points scoring (RULES.md), reproducing every golden game.
//!
//! [`State`] is a small `Copy` position. [`State::legal_actions`] returns an [`ActionSet`] over the
//! 36-action layout of `bbar/actions.py` and [`State::apply`] plays one; neither allocates.
//! With the `pyo3` feature (enabled by maturin) the crate is the Python module `bbar._core`.

pub mod action;
pub mod agents;
pub mod animal;
pub mod arena;
pub mod env;
pub mod env_act;
pub mod eval;
pub mod features;
pub mod rng;
pub mod search;
mod state;
pub mod view;

#[cfg(feature = "pyo3")]
mod python;
#[cfg(feature = "pyo3")]
mod python_env;

/// Fingerprint of the sources this crate was compiled from (core/build.rs; bbar/provenance.py checks it).
pub const BUILD_ID: &str = env!("BBAR_BUILD_ID");

pub use action::{ActionSet, Move, NUM_ACTIONS, decode, encode};
pub use rng::Rng;
pub use state::{BAR_CAP, DECK_CAP, QUEUE_CAP, State};

/// Plays uniformly random legal moves until the game is over; returns the number of moves played.
pub fn random_playout(state: &mut State, rng: &mut Rng) -> u32 {
    let mut moves = 0;
    while !state.is_over() {
        let legal = state.legal_actions();
        state.apply(legal.nth_action(rng.below(legal.len())));
        moves += 1;
    }
    moves
}
