//! Python bindings for per-hand policy inputs (`policy_obs.rs`): beliefs over a player's hands and policy tables on
//! windows. Registered into `bbar._core` by the module init in `python.rs`.
//!
//! - `belief_rows(game, actions, player, k=4)`: every decision of `player` in `actions` (from the start position
//!   `game`) with every hand they could hold there: `(ply, hand_rank, hand_mask)` int64 arrays, one row per (decision,
//!   hand), decisions in order, hands in increasing rank.
//! - `belief_obs(game, actions, player)`: the same rows plus the observation (layout v1) and legal mask of each: what
//!   a policy network needs for `sigma(a_t | h)` for every candidate hand `h` (real game only, from the deal).
//! - `bayes_filter(game, actions, player, likelihood, k=4, eps=1e-9)`: the exact Bayes range from per-row
//!   likelihoods of the observed actions; `(range, steps)`, steps `[(ply, prob, impossible), ...]`.
//! - `agent_likelihoods(spec, game, actions, player, samples=8, alpha=0.5, seed=0, threads=0, k=4)`: per-row
//!   likelihoods of a built-in agent's observed actions, estimated from `samples` runs per row (smoothed by `alpha`).
//! - `WindowTree(game, actions, k=4, leaves="frontier")`: the public tree of the window below the position reached by
//!   `actions` from `game`, in `Window`'s layout, for filling strategy tables (rows = (node, hand of the mover)).
//!
//! Long computations release the GIL; agent runs stop on Ctrl-C (KeyboardInterrupt).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

use pyo3::exceptions::{PyKeyboardInterrupt, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict};

use crate::action::NUM_ACTIONS;
use crate::agents::Spec;
use crate::animal::ALL_ANIMALS;
use crate::features::OBS_DIM;
use crate::hands::{hands, rank, unplayed};
use crate::markov::public;
use crate::policy_obs::{
    self, WindowTree, agent_likelihoods as agent_lik, bayes_filter as filter, decisions, encode_hand,
    replay_public,
};
use crate::python::Game;
use crate::window::Leaves;

fn err(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn player_index(player: i64) -> PyResult<usize> {
    match player {
        0 | 1 => Ok(player as usize),
        _ => Err(err(format!("player must be 0 or 1, got {player}"))),
    }
}

fn action_list(actions: &[i64]) -> PyResult<Vec<u8>> {
    actions
        .iter()
        .map(|&a| {
            u8::try_from(a)
                .ok()
                .filter(|&a| (a as usize) < NUM_ACTIONS)
                .ok_or_else(|| err(format!("bad action {a}")))
        })
        .collect()
}

fn parse_leaves(leaves: &str) -> PyResult<Leaves> {
    match leaves {
        "frontier" => Ok(Leaves::Frontier),
        "public" => Ok(Leaves::Public),
        "terminal" => Ok(Leaves::Terminal),
        _ => Err(err(format!("leaves must be frontier, public or terminal, not {leaves:?}"))),
    }
}

fn parse_spec(spec: &str) -> PyResult<Spec> {
    Spec::parse(spec).map_err(err)
}

/// A zeroed bytearray of `len` elements of `T`.
fn zeroed<'py, T>(py: Python<'py>, len: usize) -> PyResult<Bound<'py, PyByteArray>> {
    PyByteArray::new_with(py, len * size_of::<T>(), |_| Ok(()))
}

/// The buffer of a bytearray made by [`zeroed`] that Python has not seen yet, as `&mut [T]`.
///
/// Safety: as `python_env::slice_mut`: nothing else can read, write or resize it while the slice lives.
#[allow(clippy::mut_from_ref)]
unsafe fn slice_mut<'a, T>(buf: &'a Bound<'_, PyByteArray>) -> PyResult<&'a mut [T]> {
    let bytes = unsafe { buf.as_bytes_mut() };
    let (head, body, tail) = unsafe { bytes.align_to_mut::<T>() };
    if !head.is_empty() || !tail.is_empty() {
        return Err(err("misaligned buffer"));
    }
    Ok(body)
}

/// `numpy.frombuffer(buf, dtype).reshape(shape)`.
fn ndarray<'py>(buf: &Bound<'py, PyByteArray>, dtype: &str, shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    let np = buf.py().import("numpy")?;
    np.getattr("frombuffer")?.call1((buf, dtype))?.call_method1("reshape", (shape.to_vec(),))
}

fn i64_array<'py>(py: Python<'py>, data: &[i64]) -> PyResult<Bound<'py, PyAny>> {
    let buf = zeroed::<i64>(py, data.len())?;
    unsafe { slice_mut::<i64>(&buf)? }.copy_from_slice(data);
    ndarray(&buf, "int64", &[data.len()])
}

fn f64_array<'py>(py: Python<'py>, data: &[f64]) -> PyResult<Bound<'py, PyAny>> {
    let buf = zeroed::<f64>(py, data.len())?;
    unsafe { slice_mut::<f64>(&buf)? }.copy_from_slice(data);
    ndarray(&buf, "float64", &[data.len()])
}

fn to_f64s(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let np = obj.py().import("numpy")?;
    let arr = np.getattr("ascontiguousarray")?.call1((obj, np.getattr("float64")?))?;
    let bytes = arr.call_method0("tobytes")?;
    let b = bytes.cast::<PyBytes>()?.as_bytes();
    Ok(b.as_chunks::<8>().0.iter().map(|c| f64::from_ne_bytes(*c)).collect())
}

/// Runs `f` without the GIL on a worker thread, setting `stop` on Ctrl-C (then KeyboardInterrupt).
fn interruptible<T: Send>(
    py: Python<'_>,
    stop: &Arc<AtomicBool>,
    f: impl FnOnce() -> T + Send,
) -> PyResult<T> {
    let (out, interrupted) = py.detach(|| {
        std::thread::scope(|scope| {
            let (tx, rx) = mpsc::channel::<()>();
            let worker = scope.spawn(move || {
                let r = f();
                let _ = tx.send(());
                r
            });
            let mut interrupted = false;
            while let Err(RecvTimeoutError::Timeout) = rx.recv_timeout(Duration::from_millis(50)) {
                if !interrupted && Python::attach(|py| py.check_signals()).is_err() {
                    interrupted = true;
                    stop.store(true, Ordering::Relaxed);
                }
            }
            (worker.join().expect("worker thread panicked"), interrupted)
        })
    });
    if interrupted {
        return Err(PyKeyboardInterrupt::new_err("interrupted"));
    }
    Ok(out)
}

/// Every decision of `player` in `actions` (played from the start position `game`) with every hand they could hold:
/// `(ply, hand_rank, hand_mask)` int64 arrays, one row per (decision, hand), hands in increasing rank (`hand_rank`).
/// `game` may be any start position (mini variants included); its hidden split is ignored.
#[pyfunction]
#[pyo3(signature = (game, actions, player, k=4))]
fn belief_rows<'py>(
    py: Python<'py>,
    game: PyRef<'_, Game>,
    actions: Vec<i64>,
    player: i64,
    k: usize,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (p, actions) = (player_index(player)?, action_list(&actions)?);
    let ds = decisions(&game.state, &actions, p, k).map_err(err)?;
    let (mut plies, mut ranks, mut masks) = (Vec::new(), Vec::new(), Vec::new());
    for d in &ds {
        for &h in &d.hands {
            plies.push(d.ply as i64);
            ranks.push(rank(h) as i64);
            masks.push(h as i64);
        }
    }
    Ok((i64_array(py, &plies)?, i64_array(py, &ranks)?, i64_array(py, &masks)?))
}

/// `belief_rows` (real game, from the deal) plus each row's observation (float32 [R, OBS_DIM]) and legal mask
/// (uint8 [R, 36]): the input of a policy network for the mover holding that hand at that point of the history.
#[pyfunction]
fn belief_obs<'py>(
    py: Python<'py>,
    game: PyRef<'_, Game>,
    actions: Vec<i64>,
    player: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let (p, actions) = (player_index(player)?, action_list(&actions)?);
    let start = game.state;
    if public(&start).hands != [ALL_ANIMALS; 2] {
        return Err(err("belief_obs needs the deal of the real game as the start position"));
    }
    let k = 4;
    let publics = replay_public(&start, &actions).map_err(err)?;
    let ds = decisions(&start, &actions, p, k).map_err(err)?;
    let n: usize = ds.iter().map(|d| d.hands.len()).sum();
    let (obs, legal) = (zeroed::<f32>(py, n * OBS_DIM)?, zeroed::<u8>(py, n * NUM_ACTIONS)?);
    let (mut plies, mut ranks, mut masks) = (Vec::new(), Vec::new(), Vec::new());
    {
        let (o, l) = unsafe { (slice_mut::<f32>(&obs)?, slice_mut::<u8>(&legal)?) };
        let mut r = 0;
        for d in &ds {
            let hist = policy_obs::history_of(&publics[..d.ply], &actions[..d.ply]);
            for &h in &d.hands {
                encode_hand(
                    &d.public,
                    h,
                    k,
                    &hist,
                    &mut o[r * OBS_DIM..(r + 1) * OBS_DIM],
                    &mut l[r * NUM_ACTIONS..(r + 1) * NUM_ACTIONS],
                );
                plies.push(d.ply as i64);
                ranks.push(rank(h) as i64);
                masks.push(h as i64);
                r += 1;
            }
        }
    }
    let out = (
        i64_array(py, &plies)?,
        i64_array(py, &ranks)?,
        i64_array(py, &masks)?,
        ndarray(&obs, "float32", &[n, OBS_DIM])?,
        ndarray(&legal, "uint8", &[n, NUM_ACTIONS])?,
    );
    Ok(out.into_pyobject(py)?.into_any())
}

/// Exact Bayes range of `player`'s hand after `actions` (from the start position `game`, whose feasible hands get
/// the uniform deal prior), given `likelihood[r]` = the model's probability of the observed action for the hand of
/// row `r` of `belief_rows`. Hands that cannot play the observed action are eliminated; the others' likelihoods are
/// floored at `eps`, and a decision where the model gives the observed action probability <= eps is flagged
/// impossible (core/src/policy_obs.rs). Returns `(range float64 [C(12, m)] in the hand_rank layout, normalised;
/// steps [(ply, prob of the observed action given the public history, impossible), ...])`.
#[pyfunction]
#[pyo3(signature = (game, actions, player, likelihood, k=4, eps=1e-9))]
#[allow(clippy::type_complexity)]
fn bayes_filter<'py>(
    py: Python<'py>,
    game: PyRef<'_, Game>,
    actions: Vec<i64>,
    player: i64,
    likelihood: &Bound<'py, PyAny>,
    k: usize,
    eps: f64,
) -> PyResult<(Bound<'py, PyAny>, Vec<(usize, f64, bool)>)> {
    let (p, actions) = (player_index(player)?, action_list(&actions)?);
    let lik = to_f64s(likelihood)?;
    let (range, steps) = filter(&game.state, &actions, p, k, &lik, eps).map_err(err)?;
    Ok((f64_array(py, &range)?, steps.iter().map(|s| (s.ply, s.prob, s.impossible)).collect()))
}

/// Per-row likelihoods (rows of `belief_rows`) of `player`'s observed actions under the built-in agent `spec`: each
/// row's position (the mover holding that hand, with the history so far) is played `samples` times from fresh views,
/// and the likelihood is `(count + alpha) / (samples + alpha * legal actions of the hand)` (0 for hands that cannot
/// play it). Deterministic given `seed`, whatever `threads` (0: all cores).
#[pyfunction]
#[pyo3(signature = (spec, game, actions, player, samples=8, alpha=0.5, seed=0, threads=0, k=4))]
#[allow(clippy::too_many_arguments)]
fn agent_likelihoods<'py>(
    py: Python<'py>,
    spec: &str,
    game: PyRef<'_, Game>,
    actions: Vec<i64>,
    player: i64,
    samples: u32,
    alpha: f64,
    seed: u64,
    threads: usize,
    k: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let spec = parse_spec(spec)?;
    let (p, actions) = (player_index(player)?, action_list(&actions)?);
    if !(alpha >= 0.0 && alpha.is_finite()) {
        return Err(err("alpha must be finite and non-negative"));
    }
    let start = game.state;
    let stop = Arc::new(AtomicBool::new(false));
    let out = interruptible(py, &stop, || {
        agent_lik(spec, &start, &actions, p, k, samples, alpha, seed, threads, &stop)
    })?
    .map_err(err)?;
    f64_array(py, &out)
}

/// The public tree of a window, laid out exactly like `Window` (same nodes in preorder, same strategy-entry offsets:
/// a table filled here is a profile for `Window.exploitability` etc. of the window at the same position), with the
/// real move history up to its root.
///
/// `WindowTree(game, actions, k=4, leaves="frontier")`: `game` is the start position (the deal), `actions` every
/// action played up to the window root. Rows of player p: one per (node where p moves, possible hand of p there), in
/// preorder, hands in increasing rank; `encode` gives each row's observation (the real history plus the path inside
/// the window) and `scatter` maps the rows' action probabilities to strategy entries.
#[pyclass(name = "WindowTree", module = "bbar._core", skip_from_py_object)]
pub struct PyWindowTree {
    t: WindowTree,
}

impl PyWindowTree {
    fn node_index(&self, n: i64) -> PyResult<usize> {
        usize::try_from(n)
            .ok()
            .filter(|&n| n < self.t.nodes.len())
            .ok_or_else(|| err(format!("node {n} out of range ({} nodes)", self.t.nodes.len())))
    }
}

#[pymethods]
impl PyWindowTree {
    #[new]
    #[pyo3(signature = (game, actions, k=4, leaves="frontier"))]
    fn new(
        py: Python<'_>,
        game: PyRef<'_, Game>,
        actions: Vec<i64>,
        k: usize,
        leaves: &str,
    ) -> PyResult<Self> {
        let actions = action_list(&actions)?;
        let leaves = parse_leaves(leaves)?;
        let start = game.state;
        let t = py.detach(|| WindowTree::new(&start, &actions, k, leaves)).map_err(err)?;
        Ok(PyWindowTree { t })
    }

    #[getter]
    fn num_entries(&self) -> usize {
        self.t.n_entries
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.t.nodes.len()
    }

    /// Moves played before the window root.
    #[getter]
    fn ply(&self) -> usize {
        self.t.root_actions.len()
    }

    /// The public position at the window root (every unplayed animal in hand, decks empty).
    fn root_game(&self) -> Game {
        Game { state: *self.t.root() }
    }

    /// Rows of `player`: one per (node where they move, possible hand).
    fn num_rows(&self, player: i64) -> PyResult<usize> {
        Ok(self.t.n_rows(player_index(player)?))
    }

    /// `(obs float32 [r, OBS_DIM], legal uint8 [r, 36])` for rows `start .. stop` of `player` (default: all), encoded
    /// on `threads` threads (0: all cores).
    #[pyo3(signature = (player, start=0, stop=None, threads=0))]
    fn encode<'py>(
        &self,
        py: Python<'py>,
        player: i64,
        start: usize,
        stop: Option<usize>,
        threads: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let p = player_index(player)?;
        let stop = stop.unwrap_or(self.t.n_rows(p));
        if start > stop || stop > self.t.n_rows(p) {
            return Err(err(format!("rows {start}..{stop} of {}", self.t.n_rows(p))));
        }
        let n = stop - start;
        let (obs, legal) = (zeroed::<f32>(py, n * OBS_DIM)?, zeroed::<u8>(py, n * NUM_ACTIONS)?);
        {
            let (o, l) = unsafe { (slice_mut::<f32>(&obs)?, slice_mut::<u8>(&legal)?) };
            let t = &self.t;
            py.detach(|| t.encode(p, start, stop, o, l, threads));
        }
        let out = (ndarray(&obs, "float32", &[n, OBS_DIM])?, ndarray(&legal, "uint8", &[n, NUM_ACTIONS])?);
        Ok(out.into_pyobject(py)?.into_any())
    }

    /// `(row, action, entry)` int64 arrays over every strategy entry of `player`: `table[entry] = probs[row, action]`
    /// fills `player`'s part of a strategy table from per-row action probabilities.
    fn scatter<'py>(&self, py: Python<'py>, player: i64) -> PyResult<Bound<'py, PyAny>> {
        let p = player_index(player)?;
        let (r, a, e) = self.t.scatter(p);
        let out = (i64_array(py, &r)?, i64_array(py, &a)?, i64_array(py, &e)?);
        Ok(out.into_pyobject(py)?.into_any())
    }

    /// int64 [num_entries]: the mover of every strategy entry.
    fn entry_movers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let m: Vec<i64> = self.t.entry_movers().into_iter().map(i64::from).collect();
        i64_array(py, &m)
    }

    /// float64 [num_entries]: `player`'s entries from the built-in agent `spec`, each (node, hand) row played
    /// `samples` times from fresh views of a position where the mover holds that hand, with the real history plus the
    /// path (frequencies of the actions chosen; other entries 0). Deterministic given `seed`, whatever `threads`
    /// (0: all cores). Ctrl-C stops it (KeyboardInterrupt).
    #[pyo3(signature = (spec, player, samples=1, seed=0, threads=0))]
    fn agent_table<'py>(
        &self,
        py: Python<'py>,
        spec: &str,
        player: i64,
        samples: u32,
        seed: u64,
        threads: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let spec = parse_spec(spec)?;
        let p = player_index(player)?;
        let stop = Arc::new(AtomicBool::new(false));
        let t = &self.t;
        let table = interruptible(py, &stop, || t.agent_table(spec, p, samples, seed, threads, &stop))?
            .map_err(err)?;
        f64_array(py, &table)
    }

    /// The node reached by `path` (public actions from the window root), or None if it is a leaf or not in the window.
    fn find(&self, path: Vec<i64>) -> PyResult<Option<usize>> {
        Ok(self.t.find(&action_list(&path)?))
    }

    /// Node `n`: `{depth, ply, mover, entry, entry_end (end of its subtree's entries), n_hands, actions, path}`.
    fn node<'py>(&self, py: Python<'py>, n: i64) -> PyResult<Bound<'py, PyDict>> {
        let n = self.node_index(n)?;
        let node = &self.t.nodes[n];
        let d = PyDict::new(py);
        d.set_item("depth", node.depth)?;
        d.set_item("ply", self.t.root_actions.len() + node.depth as usize)?;
        d.set_item("mover", node.mover)?;
        d.set_item("entry", node.entry)?;
        d.set_item("entry_end", self.t.entry_end(n))?;
        d.set_item("n_hands", node.n_hands)?;
        d.set_item("actions", node.legal.map(u32::from).collect::<Vec<u32>>())?;
        d.set_item("path", self.t.path(n).into_iter().map(u32::from).collect::<Vec<u32>>())?;
        Ok(d)
    }

    /// Nodes `depth` plies below the root (where `mover` moves, if given), in preorder.
    #[pyo3(signature = (depth, mover=None))]
    fn nodes_at(&self, depth: usize, mover: Option<i64>) -> PyResult<Vec<usize>> {
        let mover = mover.map(player_index).transpose()?;
        Ok((0..self.t.nodes.len())
            .filter(|&n| {
                let node = &self.t.nodes[n];
                node.depth as usize == depth && mover.is_none_or(|m| node.mover as usize == m)
            })
            .collect())
    }

    /// Possible hands of node `n`'s mover as `hand_rank` slots, increasing (the row order of its entries).
    fn hands(&self, n: i64) -> PyResult<Vec<usize>> {
        let n = self.node_index(n)?;
        Ok(self.t.hands(n).into_iter().map(rank).collect())
    }
}

/// The possible hands of `player` at the public position of `game` (`hand_rank` slots, increasing).
#[pyfunction]
#[pyo3(signature = (game, player, k=4))]
fn possible_hands(game: PyRef<'_, Game>, player: i64, k: usize) -> PyResult<Vec<usize>> {
    let p = player_index(player)?;
    Ok(hands(unplayed(&game.state, p), k).map(rank).collect())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(belief_rows, m)?)?;
    m.add_function(wrap_pyfunction!(belief_obs, m)?)?;
    m.add_function(wrap_pyfunction!(bayes_filter, m)?)?;
    m.add_function(wrap_pyfunction!(agent_likelihoods, m)?)?;
    m.add_function(wrap_pyfunction!(possible_hands, m)?)?;
    m.add_class::<PyWindowTree>()?;
    Ok(())
}
