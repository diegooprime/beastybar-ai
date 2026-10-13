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
//!   exact exploitability of any profile, best responses, counterfactual values, policies and the re-solving gadget.
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
use crate::window::{Algo, Cfr, Leaves, Window};

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
}

impl PyWindow {
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
    fn compose(&self, s0: Option<&Bound<'_, PyAny>>, s1: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<f64>> {
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
    #[pyo3(signature = (game, r0=None, r1=None, k=4, leaves="frontier", threads=1))]
    fn new(
        py: Python<'_>,
        game: PyRef<'_, Game>,
        r0: Option<Bound<'_, PyAny>>,
        r1: Option<Bound<'_, PyAny>>,
        k: usize,
        leaves: &str,
        threads: usize,
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
        let w = py.detach(|| Window::build(&s, k, leaves, threads)).map_err(PyValueError::new_err)?;
        let mut win = PyWindow { w, ranges: [Vec::new(), Vec::new()], cfr: None, threads };
        win.ranges = [win.range(0, r0.as_ref())?, win.range(1, r1.as_ref())?];
        Ok(win)
    }

    /// Window size: internal nodes, leaves (terminal and frontier), distinct frontier positions, root hands per
    /// player, strategy entries, depth, build times.
    fn sizes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let st = &self.w.stats;
        let d = PyDict::new(py);
        d.set_item("internal", st.internal)?;
        d.set_item("leaves", st.terminal_leaves + st.frontier_leaves)?;
        d.set_item("terminal_leaves", st.terminal_leaves)?;
        d.set_item("frontier_leaves", st.frontier_leaves)?;
        d.set_item("frontier_positions", st.frontier_positions)?;
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
    #[pyo3(signature = (target=1e-4, max_iters=100_000, check_every=10, algo="dcfr"))]
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        target: f64,
        max_iters: u32,
        check_every: u32,
        algo: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let algo = Self::algo(algo)?;
        let mut cfr = Cfr::new(&self.w, self.ranges.clone(), algo);
        cfr.threads = self.threads;
        let w = &self.w;
        let r = py.detach(|| cfr.solve(w, target, max_iters, check_every));
        self.cfr = Some(cfr);
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
    #[pyo3(signature = (s0=None, s1=None))]
    fn exploitability<'py>(
        &self,
        py: Python<'py>,
        s0: Option<Bound<'py, PyAny>>,
        s1: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let prof = self.compose(s0.as_ref(), s1.as_ref())?;
        let (w, ranges, threads) = (&self.w, &self.ranges, self.threads);
        let e = py.detach(|| w.exploitability(&prof, ranges, true, threads));
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
    #[pyo3(signature = (player, s=None))]
    fn br_values<'py>(
        &self,
        py: Python<'py>,
        player: i64,
        s: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let p = Self::player(player)?;
        let prof = self.profile(s.as_ref())?;
        let (w, ro, threads) = (&self.w, &self.ranges[1 - p], self.threads);
        let (best, _) = py.detach(|| w.root_cfvs(p, &prof, ro, false, threads));
        let g = self.global(p, &best);
        to_numpy(py, &g, &[g.len()])
    }

    /// Counterfactual values of `player` per root hand (slot layout) when both follow profile `s`: the opponent's
    /// reach included (its window range), `player`'s excluded. For continual re-solving these are the opponent
    /// cfvs to carry.
    #[pyo3(signature = (player, s=None))]
    fn cfvs<'py>(
        &self,
        py: Python<'py>,
        player: i64,
        s: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let p = Self::player(player)?;
        let prof = self.profile(s.as_ref())?;
        let (w, ro, threads) = (&self.w, &self.ranges[1 - p], self.threads);
        let (_, fol) = py.detach(|| w.root_cfvs(p, &prof, ro, true, threads));
        let g = self.global(p, &fol);
        to_numpy(py, &g, &[g.len()])
    }

    /// A pure best response of `player` to profile `s` (a full profile array: `player`'s entries replaced).
    #[pyo3(signature = (player, s=None))]
    fn best_response<'py>(
        &self,
        py: Python<'py>,
        player: i64,
        s: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let p = Self::player(player)?;
        let prof = self.profile(s.as_ref())?;
        let (w, ro) = (&self.w, &self.ranges[1 - p]);
        let br = py.detach(|| w.best_response(p, &prof, ro));
        to_numpy(py, &br, &[br.len()])
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
