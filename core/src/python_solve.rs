//! Python bindings for the Markov key, the hand layout and the perfect-information solver, registered into
//! `bbar._core` by the module init in `python.rs`.
//!
//! - `markov_key(game)`, `public_key(game)`: the `u64` keys of `markov.rs`.
//! - `pi_value(game)`: exact value for player 0 (0, .5 or 1) of a game whose decks are both empty.
//! - `frontier_values(game, k=4)`: `[(hand, value), ...]` for player 0, one per possible hand of the mover in
//!   increasing rank, at a node where the opponent's hand is public and the mover's deck holds at most one card
//!   (ply 15 of the real game). `game` may be the real position or its public form.
//! - `hand_rank(animals)`, `hand_unrank(size, rank)`: the fixed hand layout (`C(12, size)` slots).
//! - `Window`: the exact hidden-information window solver (`window.rs`): build, solve (DCFR / PCFR+ / CFR+),
//!   exact exploitability of any profile, best responses, counterfactual values, policies and the re-solving gadget;
//!   depth-limited windows (`leaf_ply`) whose value leaves are valued from Python every half-iteration
//!   (`solve_external`, `resolve_external` with the gadget, `ext_begin` / `ext_end`), counterfactual values at
//!   internal nodes (`node_values`), and profile assembly (`compose`, `embed`).
//!
//! The perfect-information functions share one transposition table (2^18 buckets, 8 MiB); everything releases the
//! GIL while computing.

use std::sync::Mutex;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::NUM_ACTIONS;
use crate::animal::{NAMES, animal_id};
use crate::hands::{num_hands, rank, unplayed, unrank};
use crate::markov;
use crate::pisolve::PiSolver;
use crate::python::Game;
use crate::window::{Algo, Cfr, LeafQuery, Leaves, Window};

static SOLVER: Mutex<Option<PiSolver>> = Mutex::new(None);

fn with_solver<R>(f: impl FnOnce(&mut PiSolver) -> R) -> R {
    let mut guard = SOLVER.lock().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(|| PiSolver::new(18)))
}

fn names(mask: u16) -> Vec<&'static str> {
    (0..NAMES.len()).filter(|&a| mask >> a & 1 != 0).map(|a| NAMES[a]).collect()
}

/// Markov key of the position (`markov.rs`): identifies it for the future game when both decks are empty.
#[pyfunction]
fn markov_key(game: PyRef<'_, Game>) -> u64 {
    markov::key(&game.state)
}

/// Key of the public position (every unplayed animal in hand, decks empty): the same for every hidden split.
#[pyfunction]
fn public_key(game: PyRef<'_, Game>) -> u64 {
    markov::public_key(&game.state)
}

/// Exact value for player 0 (1 win, .5 draw, 0 loss) of a game whose decks are both empty (all hands public).
#[pyfunction]
fn pi_value(py: Python<'_>, game: PyRef<'_, Game>) -> PyResult<f64> {
    let s = game.state;
    if s.deck_len != [0, 0] {
        return Err(PyValueError::new_err("pi_value needs both decks empty (all hands public)"));
    }
    Ok(py.detach(|| with_solver(|solver| solver.value(&s))) as f64 / 2.0)
}

/// Per-hand values for player 0 at a frontier node: `[(hand, value), ...]`, hands (animals weakest first) in
/// increasing rank. The opponent's deck must be empty with at most `k` unplayed animals, and the mover must have
/// at most `k + 1` unplayed animals (hand plus at most one deck card).
#[pyfunction]
#[pyo3(signature = (game, k=4))]
fn frontier_values(
    py: Python<'_>,
    game: PyRef<'_, Game>,
    k: usize,
) -> PyResult<Vec<(Vec<&'static str>, f64)>> {
    let s = game.state;
    let p = s.to_move as usize;
    if s.deck_len[1 - p] != 0 || unplayed(&s, 1 - p).count_ones() as usize > k {
        return Err(PyValueError::new_err("frontier_values: the opponent's hand must be public"));
    }
    if unplayed(&s, p).count_ones() as usize > k + 1 {
        return Err(PyValueError::new_err("frontier_values: the mover's deck must hold at most one card"));
    }
    let values = py.detach(|| with_solver(|solver| solver.frontier_values(&s, k)));
    Ok(values.into_iter().map(|(h, v)| (names(h), v as f64 / 2.0)).collect())
}

/// Rank of a set of animals among all sets of the same size (`C(12, size)` slots; 495 for 4).
#[pyfunction]
fn hand_rank(animals: Vec<String>) -> PyResult<usize> {
    let mut mask = 0u16;
    for name in &animals {
        let a = animal_id(name).ok_or_else(|| PyValueError::new_err(format!("unknown animal {name:?}")))?;
        if mask >> a & 1 != 0 {
            return Err(PyValueError::new_err(format!("{name} twice")));
        }
        mask |= 1 << a;
    }
    Ok(rank(mask))
}

/// The set of `size` animals (weakest first) with rank `rank`.
#[pyfunction]
fn hand_unrank(size: usize, rank: usize) -> PyResult<Vec<&'static str>> {
    if size > NAMES.len() || rank >= num_hands(size) {
        return Err(PyValueError::new_err(format!("no hand of size {size} has rank {rank}")));
    }
    Ok(names(unrank(size, rank)))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(markov_key, m)?)?;
    m.add_function(wrap_pyfunction!(public_key, m)?)?;
    m.add_function(wrap_pyfunction!(pi_value, m)?)?;
    m.add_function(wrap_pyfunction!(frontier_values, m)?)?;
    m.add_function(wrap_pyfunction!(hand_rank, m)?)?;
    m.add_function(wrap_pyfunction!(hand_unrank, m)?)?;
    m.add_class::<PyWindow>()?;
    Ok(())
}

// ---- Window: the exact hidden-information window solver (window.rs) -------------------------------------------

/// A numpy array (or any array-like) as a flat `f64` vector, through `numpy.ascontiguousarray(x, float64)`.
fn to_f64s(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    let np = obj.py().import("numpy")?;
    let arr = np.getattr("ascontiguousarray")?.call1((obj, np.getattr("float64")?))?;
    let bytes = arr.call_method0("tobytes")?;
    let b = bytes.cast::<PyBytes>()?.as_bytes();
    Ok(b.as_chunks::<8>().0.iter().map(|c| f64::from_ne_bytes(*c)).collect())
}

/// A new float64 numpy array of `shape` holding `data`.
fn to_numpy<'py>(py: Python<'py>, data: &[f64], shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_ne_bytes()).collect();
    let arr = np.getattr("frombuffer")?.call1((PyBytes::new(py, &bytes), np.getattr("float64")?))?;
    arr.call_method1("reshape", (shape.to_vec(),))?.call_method0("copy")
}

/// Exact solver of a hidden-information window (core/src/window.rs).
///
/// `Window(game, r0=None, r1=None, k=4, leaves="frontier", threads=1)`: the public subtree below the public state
/// of `game` (its hands and decks are ignored; only the unplayed sets matter), down to terminal positions and
/// `leaves`: "frontier" (opponent hand public, mover deck at most one card: ply 15 of the real game; exact values per
/// mover hand), "public" (both decks empty: ply 16) or "terminal" (no cut; small variants only).
/// Root ranges `r0`, `r1`: arrays over the `C(12, m)` hand slots (`hand_rank` layout, `m` the root hand size),
/// zero outside the feasible hands; uniform over the feasible hands when `None`. They are normalised.
///
/// Values are win probabilities (win 1, draw .5, loss 0). Hand vectors returned use the same slot layout (0 for
/// infeasible hands); policies are `(slots, 36)` arrays over the action layout of `bbar/actions.py`.
///
/// `leaf_ply=L` (greater than the root's ply; ply = moves played) makes a *depth-limited* window: every node at
/// ply `L` that is neither over nor a `leaves` leaf is a value leaf, valued from outside (window.rs, "Depth-limited
/// solving"). Leaf arrays are `(num_value_leaves, 2, C(12, k))`: `[j, p, slot]` is about player `p` (0/1, not
/// mover-relative) at value leaf `j` (in `value_leaves()` order); leaf values are the owner's win probability per
/// hand against the other player's belief. Solve with `solve_external(fn)`; evaluate with `leaf_values=`.
///
/// Strategy *profiles* are flat arrays of `num_entries` behaviour probabilities: node `n` (see `nodes()`) owns
/// entries `offset .. offset + len(hands) * len(actions)`, row-major by hand, for its mover. Wherever a profile is
/// accepted: `None` (the average strategy of the last `solve`/`resolve`), `"uniform"`, `"average"`, `"current"`, or
/// an array (rows normalised over the legal actions; rows without positive mass become uniform).
#[pyclass(name = "Window", module = "bbar._core", skip_from_py_object)]
pub struct PyWindow {
    w: Window,
    /// Root ranges, local layout, normalised.
    ranges: [Vec<f64>; 2],
    cfr: Option<Cfr>,
    threads: usize,
    /// Ply of the root and of the value leaves (if any).
    ply: usize,
    leaf_ply: Option<usize>,
    /// `cfr` was started by `ext_start` (ext_begin / ext_end may drive it).
    external: bool,
}

impl PyWindow {
    /// Slots of a leaf array's last axis: `C(12, k)`.
    fn leaf_slots(&self) -> usize {
        num_hands(self.w.k)
    }

    /// A `(num_value_leaves, 2, slots)` array -> leaf-values layout (only feasible slots are read; they must be
    /// finite).
    fn leaf_local(&self, obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<f64>> {
        let v = to_f64s(obj)?;
        let (n, slots) = (self.w.value_leaves.len(), self.leaf_slots());
        if v.len() != n * 2 * slots {
            return Err(PyValueError::new_err(format!(
                "{what}: {} entries, expected {n} x 2 x {slots} (value leaves x players x hand slots)",
                v.len()
            )));
        }
        let mut out = vec![0.0; self.w.n_leaf_values];
        for (j, leaf) in self.w.value_leaves.iter().enumerate() {
            for p in 0..2 {
                for (h, &mask) in self.w.leaf_hands(j, p).iter().enumerate() {
                    let x = v[(j * 2 + p) * slots + rank(mask)];
                    if !x.is_finite() {
                        return Err(PyValueError::new_err(format!(
                            "{what}[{j}, {p}, {}] is {x}",
                            rank(mask)
                        )));
                    }
                    out[leaf.off[p] as usize + h] = x;
                }
            }
        }
        Ok(out)
    }

    /// Leaf-values layout -> `(num_value_leaves, 2, slots)` array (0 at infeasible slots).
    fn leaf_global<'py>(&self, py: Python<'py>, local: &[f64]) -> PyResult<Bound<'py, PyAny>> {
        let (n, slots) = (self.w.value_leaves.len(), self.leaf_slots());
        let mut out = vec![0.0; n * 2 * slots];
        for (j, leaf) in self.w.value_leaves.iter().enumerate() {
            for p in 0..2 {
                for (h, &mask) in self.w.leaf_hands(j, p).iter().enumerate() {
                    out[(j * 2 + p) * slots + rank(mask)] = local[leaf.off[p] as usize + h];
                }
            }
        }
        to_numpy(py, &out, &[n, 2, slots])
    }

    fn query_numpy<'py>(
        &self,
        py: Python<'py>,
        q: &LeafQuery,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let mass: Vec<f64> = q.mass.iter().flatten().copied().collect();
        Ok((self.leaf_global(py, &q.beliefs)?, to_numpy(py, &mass, &[q.mass.len(), 2])?))
    }

    /// The leaf values to evaluate with: required (and only accepted) for a window with value leaves.
    fn ext_values(&self, leaf_values: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f64>> {
        match (leaf_values, self.w.value_leaves.is_empty()) {
            (None, true) => Ok(Vec::new()),
            (None, false) => Err(PyValueError::new_err(
                "this window has value leaves: pass leaf_values (num_value_leaves, 2, slots)",
            )),
            (Some(v), _) => self.leaf_local(v, "leaf_values"),
        }
    }

    fn no_value_leaves(&self, what: &str) -> PyResult<()> {
        if self.w.value_leaves.is_empty() {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!("{what}: this window has value leaves (use solve_external)")))
        }
    }

    fn slots(&self, n: usize, player: usize) -> (usize, &[u16]) {
        let masks = self.w.node_hands(n, player);
        (num_hands(masks[0].count_ones() as usize), masks)
    }

    fn root_slots(&self, player: usize) -> (usize, &[u16]) {
        let masks = self.w.root_hands(player);
        (num_hands(masks[0].count_ones() as usize), masks)
    }

    /// A hand vector in the slot layout -> local layout (checks zeros outside the feasible hands).
    fn local(&self, player: usize, obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<f64>> {
        let v = to_f64s(obj)?;
        let (slots, masks) = self.root_slots(player);
        if v.len() != slots {
            return Err(PyValueError::new_err(format!(
                "{what} of player {player}: {} entries, expected {slots}",
                v.len()
            )));
        }
        if v.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err(format!("{what} of player {player} is not finite")));
        }
        let mut feasible = vec![false; slots];
        let out: Vec<f64> = masks
            .iter()
            .map(|&h| {
                feasible[rank(h)] = true;
                v[rank(h)]
            })
            .collect();
        if let Some(r) = (0..slots).find(|&r| !feasible[r] && v[r] != 0.0) {
            return Err(PyValueError::new_err(format!(
                "{what} of player {player} is non-zero at slot {r}, which is not a feasible hand"
            )));
        }
        Ok(out)
    }

    fn range(&self, player: usize, obj: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f64>> {
        let r = match obj {
            None => vec![1.0; self.w.root_hands(player).len()],
            Some(o) => self.local(player, o, "range")?,
        };
        if r.iter().any(|&x| x < 0.0) || r.iter().sum::<f64>() <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "range of player {player} must be non-negative with positive mass"
            )));
        }
        let s: f64 = r.iter().sum();
        Ok(r.iter().map(|x| x / s).collect())
    }

    fn global(&self, player: usize, local: &[f64]) -> Vec<f64> {
        let (slots, masks) = self.root_slots(player);
        let mut out = vec![0.0; slots];
        for (&h, &x) in masks.iter().zip(local) {
            out[rank(h)] = x;
        }
        out
    }

    fn profile(&self, s: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f64>> {
        fn need(cfr: &Option<Cfr>) -> PyResult<&Cfr> {
            cfr.as_ref()
                .ok_or_else(|| PyValueError::new_err("no strategy yet: call solve() or resolve() first"))
        }
        match s {
            None => Ok(need(&self.cfr)?.average(&self.w)),
            Some(o) => {
                if let Ok(name) = o.extract::<String>() {
                    return match name.as_str() {
                        "uniform" => Ok(self.w.uniform_profile()),
                        "average" => Ok(need(&self.cfr)?.average(&self.w)),
                        "current" => Ok(need(&self.cfr)?.current(&self.w)),
                        _ => Err(PyValueError::new_err(format!("unknown profile {name:?}"))),
                    };
                }
                self.w.sanitize(&to_f64s(o)?).map_err(PyValueError::new_err)
            }
        }
    }

    /// Player 0's entries from `s0`, player 1's from `s1`.
    fn compose_profiles(
        &self,
        s0: Option<&Bound<'_, PyAny>>,
        s1: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<f64>> {
        let p = [self.profile(s0)?, self.profile(s1)?];
        let mut out = vec![0.0; self.w.n_entries];
        for (n, node) in self.w.nodes.iter().enumerate() {
            let len = self.w.node_hands(n, node.mover as usize).len() * node.n_edges as usize;
            let r = node.entry..node.entry + len;
            out[r.clone()].copy_from_slice(&p[node.mover as usize][r]);
        }
        Ok(out)
    }

    /// `(slots, 36)` policy of node `n` from `profile`.
    fn node_policy<'py>(&self, py: Python<'py>, n: usize, profile: &[f64]) -> PyResult<Bound<'py, PyAny>> {
        let node = &self.w.nodes[n];
        let (slots, masks) = self.slots(n, node.mover as usize);
        let ne = node.n_edges as usize;
        let mut out = vec![0.0; slots * NUM_ACTIONS];
        for (h, &mask) in masks.iter().enumerate() {
            for (e, edge) in self.w.node_edges(n).iter().enumerate() {
                out[rank(mask) * NUM_ACTIONS + edge.action as usize] = profile[node.entry + h * ne + e];
            }
        }
        to_numpy(py, &out, &[slots, NUM_ACTIONS])
    }

    /// The half-iteration loop of `solve_external` / `resolve_external` (after `ext_start`).
    fn run_external<'py>(
        &mut self,
        py: Python<'py>,
        f: Bound<'py, PyAny>,
        iters: u32,
        algo: &str,
        t0: std::time::Instant,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut cb = 0.0;
        for _ in 0..2 * iters as usize {
            let (player, t, b, m) = self.ext_begin(py)?;
            let t1 = std::time::Instant::now();
            let values = f.call1((player, t, b, m))?;
            cb += t1.elapsed().as_secs_f64();
            self.ext_end(py, values)?;
        }
        let d = PyDict::new(py);
        d.set_item("iterations", self.cfr.as_ref().map_or(0, |c| c.t))?;
        d.set_item("seconds", t0.elapsed().as_secs_f64())?;
        d.set_item("callback_seconds", cb)?;
        d.set_item("algo", Self::algo(algo)?.name())?;
        Ok(d)
    }

    fn player(player: i64) -> PyResult<usize> {
        match player {
            0 | 1 => Ok(player as usize),
            _ => Err(PyValueError::new_err(format!("player must be 0 or 1, got {player}"))),
        }
    }

    fn algo(name: &str) -> PyResult<Algo> {
        Algo::parse(name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown algorithm {name:?} (dcfr, pcfr+, cfr+)")))
    }
}

#[pymethods]
impl PyWindow {
    #[new]
    #[pyo3(signature = (game, r0=None, r1=None, k=4, leaves="frontier", threads=1, leaf_ply=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        game: PyRef<'_, Game>,
        r0: Option<Bound<'_, PyAny>>,
        r1: Option<Bound<'_, PyAny>>,
        k: usize,
        leaves: &str,
        threads: usize,
        leaf_ply: Option<usize>,
    ) -> PyResult<PyWindow> {
        let leaves = match leaves {
            "frontier" => Leaves::Frontier,
            "public" => Leaves::Public,
            "terminal" => Leaves::Terminal,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "leaves must be frontier, public or terminal, not {leaves:?}"
                )));
            }
        };
        let s = game.state;
        let threads = threads.max(1);
        let ply = crate::features::ply(&s);
        let cut = match leaf_ply {
            None => None,
            Some(l) if l > ply => Some(l - ply),
            Some(l) => {
                return Err(PyValueError::new_err(format!("leaf_ply {l} must exceed the root's ply {ply}")));
            }
        };
        let w =
            py.detach(|| Window::build_cut(&s, k, leaves, cut, threads)).map_err(PyValueError::new_err)?;
        if let Some(j) = (0..w.value_leaves.len())
            .find(|&j| (0..2).any(|p| w.leaf_hands(j, p)[0].count_ones() as usize != k))
        {
            return Err(PyValueError::new_err(format!(
                "value leaf {j} has a public hand (fewer than k unplayed animals): choose another leaf_ply"
            )));
        }
        let mut win =
            PyWindow { w, ranges: [Vec::new(), Vec::new()], cfr: None, threads, ply, leaf_ply, external: false };
        win.ranges = [win.range(0, r0.as_ref())?, win.range(1, r1.as_ref())?];
        Ok(win)
    }

    /// Ply of the root (moves played).
    #[getter]
    fn ply(&self) -> usize {
        self.ply
    }

    /// Ply of the value leaves (None: not a depth-limited window).
    #[getter]
    fn leaf_ply(&self) -> Option<usize> {
        self.leaf_ply
    }

    #[getter]
    fn num_value_leaves(&self) -> usize {
        self.w.value_leaves.len()
    }

    /// Every value leaf, in the order of leaf arrays: `(history, mover)`, history = public action indices from the
    /// root.
    fn value_leaves(&self) -> Vec<(Vec<usize>, u8)> {
        (0..self.w.value_leaves.len())
            .map(|j| {
                (self.w.leaf_history(j).into_iter().map(usize::from).collect(), self.w.value_leaves[j].mover)
            })
            .collect()
    }

    /// The public position (every unplayed animal in hand, decks empty) at value leaf `j`.
    fn leaf_game(&self, j: usize) -> PyResult<Game> {
        let leaf = self.w.value_leaves.get(j).ok_or_else(|| {
            PyValueError::new_err(format!("no value leaf {j} ({} leaves)", self.w.value_leaves.len()))
        })?;
        Ok(Game { state: leaf.state })
    }

    /// float64 `(num_value_leaves, 2, slots)`: 1 at each player's feasible hands at each value leaf.
    fn leaf_feasible<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.leaf_global(py, &vec![1.0; self.w.n_leaf_values])
    }

    /// Replaces the root ranges (slot layout; None = uniform over the feasible hands). A solver state is kept: the
    /// next `solve(warm=True)` restarts from its regrets.
    #[pyo3(signature = (r0=None, r1=None))]
    fn set_ranges(&mut self, r0: Option<Bound<'_, PyAny>>, r1: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        self.ranges = [self.range(0, r0.as_ref())?, self.range(1, r1.as_ref())?];
        Ok(())
    }

    /// Window size: internal nodes, leaves (terminal and frontier), distinct frontier positions, value leaves
    /// (depth-limited windows), root hands per player, strategy entries, depth, build times.
    fn sizes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let st = &self.w.stats;
        let d = PyDict::new(py);
        d.set_item("internal", st.internal)?;
        d.set_item("leaves", st.terminal_leaves + st.frontier_leaves)?;
        d.set_item("terminal_leaves", st.terminal_leaves)?;
        d.set_item("frontier_leaves", st.frontier_leaves)?;
        d.set_item("frontier_positions", st.frontier_positions)?;
        d.set_item("value_leaves", st.value_leaves)?;
        d.set_item("hands", (self.w.root_hands(0).len(), self.w.root_hands(1).len()))?;
        d.set_item("entries", self.w.n_entries)?;
        d.set_item("max_depth", st.max_depth)?;
        d.set_item("pi_nodes", st.pi_nodes)?;
        d.set_item("tree_seconds", st.tree_seconds)?;
        d.set_item("frontier_seconds", st.frontier_seconds)?;
        Ok(d)
    }

    #[getter]
    fn num_entries(&self) -> usize {
        self.w.n_entries
    }

    #[getter]
    fn iterations(&self) -> u32 {
        self.cfr.as_ref().map_or(0, |c| c.t)
    }

    #[getter]
    fn threads(&self) -> usize {
        self.threads
    }

    #[setter]
    fn set_threads(&mut self, threads: usize) {
        self.threads = threads.max(1);
    }

    /// Feasible root hands of `player`, as slot indices (`hand_rank`), increasing.
    fn hands(&self, player: i64) -> PyResult<Vec<usize>> {
        Ok(self.w.root_hands(Self::player(player)?).iter().map(|&h| rank(h)).collect())
    }

    /// The (normalised) root ranges in the slot layout.
    fn ranges<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let r0 = self.global(0, &self.ranges[0]);
        let r1 = self.global(1, &self.ranges[1]);
        Ok((to_numpy(py, &r0, &[r0.len()])?, to_numpy(py, &r1, &[r1.len()])?))
    }

    /// Solves the window from scratch until the exploitability of the average strategy is at most `target`
    /// (checked every `check_every` iterations) or `max_iters`. Returns iterations, exploitability, value (player
    /// 0's expected value), nashconv, br (both best-response values) and seconds.
    ///
    /// `warm=True` restarts from the regrets of the previous `solve` (same algorithm) with the current ranges (see
    /// `set_ranges`): a re-solve whose ranges moved a little converges in fewer iterations. Its average strategy
    /// starts over.
    #[pyo3(signature = (target=1e-4, max_iters=100_000, check_every=10, algo="dcfr", warm=false))]
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        target: f64,
        max_iters: u32,
        check_every: u32,
        algo: &str,
        warm: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.no_value_leaves("solve")?;
        let algo = Self::algo(algo)?;
        let mut cfr = match self.cfr.take() {
            Some(mut c) if warm && c.algo == algo && c.gadget.is_none() && c.ext.is_none() => {
                c.warm_restart(&self.w, self.ranges.clone());
                c
            }
            _ => Cfr::new(&self.w, self.ranges.clone(), algo),
        };
        cfr.threads = self.threads;
        let w = &self.w;
        let r = py.detach(|| cfr.solve(w, target, max_iters, check_every));
        self.cfr = Some(cfr);
        self.external = false;
        let d = PyDict::new(py);
        d.set_item("iterations", r.iterations)?;
        d.set_item("exploitability", r.exploitability)?;
        d.set_item("value", r.expl.value)?;
        d.set_item("nashconv", r.expl.nashconv)?;
        d.set_item("br", (r.expl.br[0], r.expl.br[1]))?;
        d.set_item("seconds", r.seconds)?;
        d.set_item("algo", algo.name())?;
        Ok(d)
    }

    /// Re-solving gadget (CFR-D / DeepStack): `player`'s range is the window's; the opponent may terminate with
    /// each hand at `opp_cfvs` (slot layout, opponent win probability with our range normalised) or follow into the
    /// window. `prior`: the opponent's gadget prior (slot layout; uniform over the feasible hands when None).
    /// Solves until the gadget game's exploitability is at most `target`. Afterwards the average strategy is the
    /// re-solved one. Returns iterations, exploitability (of the gadget game), max_violation (largest excess of the
    /// opponent's best-response cfv over `opp_cfvs`), mean_violation and seconds.
    #[pyo3(signature = (player, opp_cfvs, target=1e-4, max_iters=100_000, check_every=10, algo="dcfr", prior=None))]
    #[allow(clippy::too_many_arguments)]
    fn resolve<'py>(
        &mut self,
        py: Python<'py>,
        player: i64,
        opp_cfvs: Bound<'py, PyAny>,
        target: f64,
        max_iters: u32,
        check_every: u32,
        algo: &str,
        prior: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.no_value_leaves("resolve")?;
        let p = Self::player(player)?;
        let algo = Self::algo(algo)?;
        let cfvs = self.local(1 - p, &opp_cfvs, "opp_cfvs")?;
        let prior = match prior {
            None => None,
            Some(o) => Some(self.range(1 - p, Some(&o))?),
        };
        let mut cfr = Cfr::new_gadget(&self.w, p, self.ranges[p].clone(), cfvs, prior, algo);
        cfr.threads = self.threads;
        let w = &self.w;
        let r = py.detach(|| cfr.solve(w, target, max_iters, check_every));
        self.cfr = Some(cfr);
        self.external = false;
        let g = r.gadget.unwrap_or_default();
        let d = PyDict::new(py);
        d.set_item("iterations", r.iterations)?;
        d.set_item("exploitability", g.exploitability)?;
        d.set_item("max_violation", g.max_violation)?;
        d.set_item("mean_violation", g.mean_violation)?;
        d.set_item("seconds", r.seconds)?;
        d.set_item("algo", algo.name())?;
        Ok(d)
    }

    /// Exact exploitability of a profile on the window with its root ranges: player 0 plays `s0`, player 1 `s1`.
    /// Returns nashconv (BR_0 + BR_1 - 1), exploitability (nashconv / 2), br (both best-response values, each in
    /// that player's win probability) and value (player 0's expected value).
    ///
    /// With value leaves, `leaf_values` (`(num_value_leaves, 2, slots)`) fixes their payoffs: the numbers are those
    /// of the depth-limited game with these leaf values.
    #[pyo3(signature = (s0=None, s1=None, leaf_values=None))]
    fn exploitability<'py>(
        &self,
        py: Python<'py>,
        s0: Option<Bound<'py, PyAny>>,
        s1: Option<Bound<'py, PyAny>>,
        leaf_values: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ext = self.ext_values(leaf_values.as_ref())?;
        let prof = self.compose_profiles(s0.as_ref(), s1.as_ref())?;
        let (w, ranges, threads) = (&self.w, &self.ranges, self.threads);
        let e = py.detach(|| w.exploitability_ext(&prof, ranges, true, &ext, threads));
        let d = PyDict::new(py);
        d.set_item("nashconv", e.nashconv)?;
        d.set_item("exploitability", e.exploitability)?;
        d.set_item("br", (e.br[0], e.br[1]))?;
        d.set_item("value", e.value)?;
        Ok(d)
    }

    /// Best-response counterfactual values of `player` per root hand (slot layout) against the other player's
    /// strategy in profile `s`, the other player's root reach being its window range. `sum(range * values)` is
    /// `player`'s best-response value.
    #[pyo3(signature = (player, s=None, leaf_values=None))]
    fn br_values<'py>(
        &self,
        py: Python<'py>,
        player: i64,
        s: Option<Bound<'py, PyAny>>,
        leaf_values: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let p = Self::player(player)?;
        let ext = self.ext_values(leaf_values.as_ref())?;
        let prof = self.profile(s.as_ref())?;
        let (w, ro, threads) = (&self.w, &self.ranges[1 - p], self.threads);
        let (best, _) = py.detach(|| w.root_cfvs_ext(p, &prof, ro, false, &ext, threads));
        let g = self.global(p, &best);
        to_numpy(py, &g, &[g.len()])
    }

    /// Counterfactual values of `player` per root hand (slot layout) when both follow profile `s`: the opponent's
    /// reach included (its window range), `player`'s excluded. For continual re-solving these are the opponent
    /// cfvs to carry.
    #[pyo3(signature = (player, s=None, leaf_values=None))]
    fn cfvs<'py>(
        &self,
        py: Python<'py>,
        player: i64,
        s: Option<Bound<'py, PyAny>>,
        leaf_values: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let p = Self::player(player)?;
        let ext = self.ext_values(leaf_values.as_ref())?;
        let prof = self.profile(s.as_ref())?;
        let (w, ro, threads) = (&self.w, &self.ranges[1 - p], self.threads);
        let (_, fol) = py.detach(|| w.root_cfvs_ext(p, &prof, ro, true, &ext, threads));
        let g = self.global(p, &fol);
        to_numpy(py, &g, &[g.len()])
    }

    /// A pure best response of `player` to profile `s` (a full profile array: `player`'s entries replaced).
    #[pyo3(signature = (player, s=None, leaf_values=None))]
    fn best_response<'py>(
        &self,
        py: Python<'py>,
        player: i64,
        s: Option<Bound<'py, PyAny>>,
        leaf_values: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let p = Self::player(player)?;
        let ext = self.ext_values(leaf_values.as_ref())?;
        let prof = self.profile(s.as_ref())?;
        let (w, ro) = (&self.w, &self.ranges[1 - p]);
        let br = py.detach(|| w.best_response_ext(p, &prof, ro, &ext));
        to_numpy(py, &br, &[br.len()])
    }

    /// Starts a fresh depth-limited solve (`algo`: dcfr, pcfr+, cfr+), replacing any previous solver state. Then
    /// alternate `ext_begin()` and `ext_end(values)`; `solve_external` does it for you.
    ///
    /// With `player` and `opp_cfvs` it is a re-solve with the gadget (see `resolve`): `player`'s range is the
    /// window's, the opponent may terminate with each hand at `opp_cfvs` (slot layout, our range normalised) or
    /// follow into the window, with gadget prior `prior` (uniform over the feasible hands when None); the opponent's
    /// root reach at the value leaves is its prior times its current follow probability (`resolve_external`).
    #[pyo3(signature = (algo="dcfr", player=None, opp_cfvs=None, prior=None))]
    fn ext_start(
        &mut self,
        algo: &str,
        player: Option<i64>,
        opp_cfvs: Option<Bound<'_, PyAny>>,
        prior: Option<Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let algo = Self::algo(algo)?;
        let mut cfr = match (player, opp_cfvs) {
            (None, None) => {
                if prior.is_some() {
                    return Err(PyValueError::new_err("prior is the gadget's: pass player and opp_cfvs too"));
                }
                Cfr::new(&self.w, self.ranges.clone(), algo)
            }
            (Some(p), Some(c)) => {
                let p = Self::player(p)?;
                let cfvs = self.local(1 - p, &c, "opp_cfvs")?;
                let prior = match prior {
                    None => None,
                    Some(o) => Some(self.range(1 - p, Some(&o))?),
                };
                Cfr::new_gadget(&self.w, p, self.ranges[p].clone(), cfvs, prior, algo)
            }
            _ => return Err(PyValueError::new_err("a gadget re-solve needs both player and opp_cfvs")),
        };
        cfr.threads = self.threads;
        self.cfr = Some(cfr);
        self.external = true;
        Ok(())
    }

    /// Starts the next half-iteration (player 0 then player 1 each iteration): computes the current strategies and
    /// returns `(player, iteration, beliefs, mass)`, beliefs float64 `(num_value_leaves, 2, slots)` (each player's
    /// normalised reach at each leaf; uniform over the feasible hands where its reach mass is 0) and mass float64
    /// `(num_value_leaves, 2)` (reach masses). Answer with `ext_end(values)`.
    #[allow(clippy::type_complexity)]
    fn ext_begin<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(usize, u32, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let cfr = self.cfr.as_mut().filter(|_| self.external).ok_or_else(|| {
            PyValueError::new_err("no depth-limited solve in progress: call ext_start() first")
        })?;
        if cfr.ext.as_ref().is_some_and(|e| e.query.is_some()) {
            return Err(PyValueError::new_err("ext_begin called twice without ext_end"));
        }
        let w = &self.w;
        let q = py.detach(|| cfr.ext_begin(w).clone());
        let (player, t) = (cfr.ext.as_ref().map_or(0, |e| e.phase), cfr.t);
        let (b, m) = self.query_numpy(py, &q)?;
        Ok((player, t, b, m))
    }

    /// Ends the half-iteration: `values` float64 `(num_value_leaves, 2, slots)`, `[j, p, slot]` = player `p`'s win
    /// probability at leaf `j` holding that hand, against the other player's belief of `ext_begin` (only feasible
    /// slots are read).
    fn ext_end(&mut self, py: Python<'_>, values: Bound<'_, PyAny>) -> PyResult<()> {
        let vals = self.leaf_local(&values, "values")?;
        let external = self.external;
        let cfr = self
            .cfr
            .as_mut()
            .filter(|c| external && c.ext.as_ref().is_some_and(|e| e.query.is_some()))
            .ok_or_else(|| PyValueError::new_err("ext_end without ext_begin"))?;
        let w = &self.w;
        py.detach(|| cfr.ext_end(w, vals)).map_err(PyValueError::new_err)
    }

    /// Depth-limited solve: `iters` iterations of alternating CFR (`algo`) whose value leaves are valued by
    /// `fn(player, iteration, beliefs, mass) -> values` once per half-iteration (see `ext_begin` / `ext_end` for the
    /// arrays). The leaf beliefs are those of the current iterate (CFR-D; window.rs). Afterwards the average
    /// strategy is the solution (`profile_array()`), `leaf_query("average")` its leaf beliefs and
    /// `leaf_values_avg()` the averaged leaf values (the opponent cfvs for re-solving below a leaf). Returns
    /// iterations, seconds (total) and callback_seconds.
    #[pyo3(signature = (f, iters=200, algo="dcfr"))]
    fn solve_external<'py>(
        &mut self,
        py: Python<'py>,
        f: Bound<'py, PyAny>,
        iters: u32,
        algo: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let t0 = std::time::Instant::now();
        self.ext_start(algo, None, None, None)?;
        self.run_external(py, f, iters, algo, t0)
    }

    /// Depth-limited re-solve with the gadget (`ext_start` with `player`, `opp_cfvs`, `prior`; continual re-solving):
    /// `iters` iterations whose value leaves are valued by `fn` as in `solve_external`. Afterwards the average strategy
    /// is the re-solved one, `gadget_stats(leaf_values)` its evaluation. Returns iterations, seconds and
    /// callback_seconds.
    #[pyo3(signature = (player, opp_cfvs, f, iters=200, algo="dcfr", prior=None))]
    #[allow(clippy::too_many_arguments)]
    fn resolve_external<'py>(
        &mut self,
        py: Python<'py>,
        player: i64,
        opp_cfvs: Bound<'py, PyAny>,
        f: Bound<'py, PyAny>,
        iters: u32,
        algo: &str,
        prior: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let t0 = std::time::Instant::now();
        self.ext_start(algo, Some(player), Some(opp_cfvs), prior)?;
        self.run_external(py, f, iters, algo, t0)
    }

    /// The gadget game's evaluation of the last re-solve (`resolve`, `resolve_external`): exploitability, max_violation
    /// and mean_violation (as `resolve` returns them), with fixed value-leaf values `leaf_values` (required for a
    /// window with value leaves, e.g. `leaf_values_avg()`: then the numbers are those of the depth-limited gadget game
    /// with these leaf payoffs).
    #[pyo3(signature = (leaf_values=None))]
    fn gadget_stats<'py>(
        &self,
        py: Python<'py>,
        leaf_values: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ext = self.ext_values(leaf_values.as_ref())?;
        let cfr = self
            .cfr
            .as_ref()
            .filter(|c| c.gadget.is_some())
            .ok_or_else(|| PyValueError::new_err("no gadget re-solve yet"))?;
        let w = &self.w;
        let g = py.detach(|| cfr.gadget_exploitability_ext(w, &ext));
        let d = PyDict::new(py);
        d.set_item("exploitability", g.exploitability)?;
        d.set_item("max_violation", g.max_violation)?;
        d.set_item("mean_violation", g.mean_violation)?;
        Ok(d)
    }

    /// Counterfactual values of `player` per hand (slot layout of `player`'s hands there) at the internal node
    /// reached by `history`, against the other player's strategy in profile `s` (default: the average), the other
    /// player's root reach being its window range (or, after a gadget re-solve, the opponent's prior times its average
    /// follow probability), through its own actions on the path. Returns `(best, follow, mass)`: best-response values,
    /// the values of following `s`, and the other player's reach mass at the node; `best / mass` are in units where
    /// the other's reach there sums to 1 (the opponent cfvs of a re-solve rooted there). `leaf_values`: as in
    /// `exploitability` (required with value leaves).
    #[pyo3(signature = (player, history, s=None, leaf_values=None))]
    #[allow(clippy::type_complexity)]
    fn node_values<'py>(
        &self,
        py: Python<'py>,
        player: i64,
        history: Vec<u8>,
        s: Option<Bound<'py, PyAny>>,
        leaf_values: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, f64)> {
        let p = Self::player(player)?;
        let n = self
            .w
            .find(&history)
            .ok_or_else(|| PyValueError::new_err(format!("no internal node at history {history:?}")))?;
        let ext = self.ext_values(leaf_values.as_ref())?;
        let prof = self.profile(s.as_ref())?;
        let ro = match &self.cfr {
            Some(c) if c.gadget.is_some() => c.root_reaches(true)[1 - p].clone(),
            _ => self.ranges[1 - p].clone(),
        };
        let (w, threads) = (&self.w, self.threads);
        let (best, fol, mass) = py.detach(|| w.node_cfvs_ext(p, &prof, &ro, n, true, &ext, threads));
        let (slots, masks) = self.slots(n, p);
        let (mut gb, mut gf) = (vec![0.0; slots], vec![0.0; slots]);
        for (h, &mask) in masks.iter().enumerate() {
            gb[rank(mask)] = best[h];
            gf[rank(mask)] = fol[h];
        }
        Ok((to_numpy(py, &gb, &[slots])?, to_numpy(py, &gf, &[slots])?, mass))
    }

    /// `(beliefs, mass)` at the value leaves under profile `s` (default: the average strategy), as `ext_begin`
    /// returns them (after a gadget re-solve, the opponent's root reach is its prior times its average follow
    /// probability).
    #[pyo3(signature = (s=None))]
    fn leaf_query<'py>(
        &self,
        py: Python<'py>,
        s: Option<Bound<'py, PyAny>>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let prof = self.profile(s.as_ref())?;
        let ranges = match &self.cfr {
            Some(c) if c.gadget.is_some() => c.root_reaches(true),
            _ => self.ranges.clone(),
        };
        let w = &self.w;
        let q = py.detach(|| LeafQuery::from_reaches(w, &w.leaf_reaches(&prof, &ranges)));
        self.query_numpy(py, &q)
    }

    /// float64 `(num_value_leaves, 2, slots)`: each player's leaf values averaged over the other player's
    /// half-iterations with the weights of the other's average strategy: `[j, p]` = `p`'s values against the other's
    /// average strategy at leaf `j`, in units where the other's reach there sums to 1. For re-solving player `q`'s
    /// strategy below leaf `j`, `[j, 1 - q]` are the opponent cfvs (window.rs).
    fn leaf_values_avg<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let cfr = self
            .cfr
            .as_ref()
            .filter(|c| c.ext.is_some())
            .ok_or_else(|| PyValueError::new_err("no depth-limited solve yet"))?;
        let v = cfr.avg_leaf_values(&self.w);
        self.leaf_global(py, &v)
    }

    /// One profile array with player 0's entries from `s0` and player 1's from `s1` (profiles as anywhere).
    #[pyo3(signature = (s0=None, s1=None))]
    fn compose<'py>(
        &self,
        py: Python<'py>,
        s0: Option<Bound<'py, PyAny>>,
        s1: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let prof = self.compose_profiles(s0.as_ref(), s1.as_ref())?;
        to_numpy(py, &prof, &[prof.len()])
    }

    /// Copies window `src`'s profile `src_profile` (default its average) into `profile` (one of this window's, not
    /// modified): `src`'s node `src_history` sits at this window's node `history`, and each node below it overwrites
    /// the node of this window with the same history from there (same public position, hands and actions, else
    /// ValueError). A value leaf of either window may be an internal node of the other (nothing below it is copied):
    /// a depth-limited trunk and the re-solved subtrees below its leaves assemble into a full-window profile, and a
    /// trunk or a leaf subtree can be cut out of a full profile. Returns `(new profile, entries copied)`.
    #[pyo3(signature = (profile, src, src_profile=None, history=Vec::new(), src_history=Vec::new()))]
    fn embed<'py>(
        &self,
        py: Python<'py>,
        profile: Bound<'py, PyAny>,
        src: PyRef<'py, PyWindow>,
        src_profile: Option<Bound<'py, PyAny>>,
        history: Vec<u8>,
        src_history: Vec<u8>,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let mut dest = to_f64s(&profile)?;
        if dest.len() != self.w.n_entries {
            return Err(PyValueError::new_err(format!(
                "profile has {} entries, the window {}",
                dest.len(),
                self.w.n_entries
            )));
        }
        let sp = src.profile(src_profile.as_ref())?;
        let at = self
            .w
            .find(&history)
            .ok_or_else(|| PyValueError::new_err(format!("no internal node at history {history:?}")))?;
        let src_at = src.w.find(&src_history).ok_or_else(|| {
            PyValueError::new_err(format!("no internal node of src at history {src_history:?}"))
        })?;
        let n = self.w.embed(&mut dest, &src.w, &sp, at, src_at).map_err(PyValueError::new_err)?;
        Ok((to_numpy(py, &dest, &[dest.len()])?, n))
    }

    /// `embed` for many sources at once (one copy of the profile): `parts` is a list of `(src, src_profile,
    /// history)` (src_profile None = src's average), each source's root at this window's node `history`. Returns
    /// `(new profile, entries copied)`.
    #[allow(clippy::type_complexity)]
    fn embed_all<'py>(
        &self,
        py: Python<'py>,
        profile: Bound<'py, PyAny>,
        parts: Vec<(PyRef<'py, PyWindow>, Option<Bound<'py, PyAny>>, Vec<u8>)>,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let mut dest = to_f64s(&profile)?;
        if dest.len() != self.w.n_entries {
            return Err(PyValueError::new_err(format!(
                "profile has {} entries, the window {}",
                dest.len(),
                self.w.n_entries
            )));
        }
        let mut n = 0;
        for (src, sp, history) in &parts {
            let sp = src.profile(sp.as_ref())?;
            let at = self
                .w
                .find(history)
                .ok_or_else(|| PyValueError::new_err(format!("no internal node at history {history:?}")))?;
            let Some(src_root) = src.w.find(&[]) else { continue };
            n += self.w.embed(&mut dest, &src.w, &sp, at, src_root).map_err(PyValueError::new_err)?;
        }
        Ok((to_numpy(py, &dest, &[dest.len()])?, n))
    }

    /// The profile `s` as an array (`"uniform"`, `"average"`, `"current"`, or None for the average).
    #[pyo3(signature = (s=None))]
    fn profile_array<'py>(
        &self,
        py: Python<'py>,
        s: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let prof = self.profile(s.as_ref())?;
        to_numpy(py, &prof, &[prof.len()])
    }

    /// `(slots, 36)` policy of the root mover from profile `s` (rows of infeasible hands are 0).
    #[pyo3(signature = (s=None))]
    fn root_policy<'py>(&self, py: Python<'py>, s: Option<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
        if self.w.nodes.is_empty() {
            return Err(PyValueError::new_err("the root is a leaf: no decision"));
        }
        let prof = self.profile(s.as_ref())?;
        self.node_policy(py, 0, &prof)
    }

    /// `(slots, 36)` policy of the mover at the node reached by `history` (public action indices from the root).
    #[pyo3(signature = (history, s=None))]
    fn policy<'py>(
        &self,
        py: Python<'py>,
        history: Vec<u8>,
        s: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let n = self
            .w
            .find(&history)
            .ok_or_else(|| PyValueError::new_err(format!("no internal node at history {history:?}")))?;
        let prof = self.profile(s.as_ref())?;
        self.node_policy(py, n, &prof)
    }

    /// Every internal node in preorder: `(history, mover, offset, hands, actions)` with `hands` the mover's possible
    /// hands (tuples of animal names, weakest first; in increasing `hand_rank`) and `actions` the public action
    /// indices. The node's profile entries are `offset + i * len(actions) + j` for hand `hands[i]` and action
    /// `actions[j]` (0 when the hand lacks the action's animal).
    #[allow(clippy::type_complexity)]
    fn nodes(&self) -> Vec<(Vec<usize>, u8, usize, Vec<Vec<&'static str>>, Vec<usize>)> {
        (0..self.w.nodes.len())
            .map(|n| {
                let node = &self.w.nodes[n];
                let hands = self.w.node_hands(n, node.mover as usize).iter().map(|&h| names(h)).collect();
                let actions = self.w.node_edges(n).iter().map(|e| e.action as usize).collect();
                let history = self.w.history(n).into_iter().map(usize::from).collect();
                (history, node.mover, node.entry, hands, actions)
            })
            .collect()
    }

    /// The public position (every unplayed animal in hand, decks empty) at the node reached by `history`.
    fn node_game(&self, history: Vec<u8>) -> PyResult<Game> {
        let n = self
            .w
            .find(&history)
            .ok_or_else(|| PyValueError::new_err(format!("no internal node at history {history:?}")))?;
        Ok(Game { state: self.w.nodes[n].state })
    }
}
