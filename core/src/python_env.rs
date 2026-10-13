//! Python bindings for training: `bbar._core.VecEnv`, the observation layout, `encode_moves` and `env_bench`.
//! Registered from the module init in `python.rs`.
//!
//! Arrays come back as numpy arrays over fresh `bytearray`s that Rust fills in place with the GIL released (no
//! copy after encoding). Array arguments are anything `numpy.ascontiguousarray` accepts.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict};

use crate::action::NUM_ACTIONS;
use crate::animal::{NUM_ANIMALS, animal_id};
use crate::env::{self, VecEnv};
use crate::features::{
    self, History, MAX_PLIES, OBS_DIM, OBS_LAYOUT, OBS_VERSION, PRIV_DIM, encode_mask, encode_obs, encode_privileged,
};
use crate::state::{DECK_CAP, State};

/// A zeroed bytearray of `len` elements of `T`.
fn zeroed<'py, T>(py: Python<'py>, len: usize) -> PyResult<Bound<'py, PyByteArray>> {
    PyByteArray::new_with(py, len * size_of::<T>(), |_| Ok(()))
}

/// The bytearray's buffer as `&mut [T]` (the memory belongs to the Python object, not to the `&` borrow).
///
/// Safety: `buf` must be a bytearray created by [`zeroed`] that no Python code has seen yet (nothing else can
/// read, write or resize it while the slice lives), and `T` a plain number type.
#[allow(clippy::mut_from_ref)]
unsafe fn slice_mut<'a, T>(buf: &'a Bound<'_, PyByteArray>) -> PyResult<&'a mut [T]> {
    let bytes = unsafe { buf.as_bytes_mut() };
    let (head, body, tail) = unsafe { bytes.align_to_mut::<T>() };
    if !head.is_empty() || !tail.is_empty() {
        return Err(PyValueError::new_err("misaligned buffer"));
    }
    Ok(body)
}

/// The data of `arr`, a writable C-contiguous numpy array of `len` elements of numpy type `typestr` (e.g. "<f4"),
/// read from its `__array_interface__`. Returns (address, byte length).
fn array_data(arr: &Bound<'_, PyAny>, typestr: &str, len: usize, align: usize) -> PyResult<(usize, usize)> {
    let err = |what: &str| PyValueError::new_err(format!("output array: {what}"));
    let ai = arr.getattr("__array_interface__").map_err(|_| err("not a numpy array"))?;
    let ai = ai.cast::<PyDict>()?;
    let get = |k: &str| ai.get_item(k)?.ok_or_else(|| err(&format!("no {k} in __array_interface__")));
    let (addr, readonly): (usize, bool) = get("data")?.extract()?;
    let ts: String = get("typestr")?.extract()?;
    let shape: Vec<usize> = get("shape")?.extract()?;
    let contiguous = ai.get_item("strides")?.is_none_or(|s| s.is_none());
    if ts != typestr {
        return Err(err(&format!("dtype {ts}, expected {typestr}")));
    }
    if readonly || !contiguous || shape.iter().product::<usize>() != len || addr % align != 0 {
        return Err(err(&format!("need a writable C-contiguous array of {len} elements")));
    }
    Ok((addr, len * typestr[2..].parse::<usize>().unwrap()))
}

/// `spans` (address, byte length) unchanged if no two overlap (so mutable slices over them do not alias).
fn disjoint<const N: usize>(spans: [(usize, usize); N]) -> PyResult<[(usize, usize); N]> {
    for i in 0..N {
        for j in 0..i {
            let (a, b) = (spans[i], spans[j]);
            if a.0 < b.0 + b.1 && b.0 < a.0 + a.1 {
                return Err(PyValueError::new_err("output arrays overlap"));
            }
        }
    }
    Ok(spans)
}

/// `numpy.frombuffer(buf, dtype).reshape(shape)`: a writable array sharing the bytearray's memory.
fn ndarray<'py>(buf: &Bound<'py, PyByteArray>, dtype: &str, shape: &[usize]) -> PyResult<Bound<'py, PyAny>> {
    let np = buf.py().import("numpy")?;
    np.getattr("frombuffer")?.call1((buf, dtype))?.call_method1("reshape", (shape.to_vec(),))
}

/// A contiguous copy of `obj` as `dtype` (an 8-byte integer type), as raw native-endian words.
fn words(obj: &Bound<'_, PyAny>, dtype: &str) -> PyResult<Vec<[u8; 8]>> {
    let arr = obj.py().import("numpy")?.call_method1("ascontiguousarray", (obj, dtype))?;
    let bytes = arr.call_method0("tobytes")?;
    let bytes = bytes.cast::<PyBytes>()?.as_bytes();
    let (words, rest) = bytes.as_chunks::<8>();
    debug_assert!(rest.is_empty());
    Ok(words.to_vec())
}

fn i64s(obj: &Bound<'_, PyAny>) -> PyResult<Vec<i64>> {
    Ok(words(obj, "int64")?.into_iter().map(i64::from_ne_bytes).collect())
}

fn u64s(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u64>> {
    Ok(words(obj, "uint64")?.into_iter().map(u64::from_ne_bytes).collect())
}

/// `n_envs` Beasty Bar games for batched self-play; see `core/src/env.rs` and `core/src/features.rs`.
///
/// `VecEnv(n_envs, seed=0, threads=0)`; `threads` = 0 encodes on every core. Env `i`'s `k`-th game is a fresh
/// random deal from its own stream of `seed`, started by player `(i + k) % 2`.
#[pyclass(module = "bbar._core", name = "VecEnv")]
pub struct PyVecEnv {
    env: VecEnv,
}

#[pymethods]
impl PyVecEnv {
    #[new]
    #[pyo3(signature = (n_envs, seed = 0, threads = 0))]
    fn new(n_envs: usize, seed: u64, threads: usize) -> PyResult<Self> {
        if n_envs == 0 {
            return Err(PyValueError::new_err("n_envs must be positive"));
        }
        Ok(PyVecEnv { env: VecEnv::new(n_envs, seed, threads) })
    }

    #[getter]
    fn n_envs(&self) -> usize {
        self.env.len()
    }

    fn __len__(&self) -> usize {
        self.env.len()
    }

    /// `(obs float32 [n, OBS_DIM], legal uint8 [n, 36], mover int64 [n], ply int64 [n])` for each env's player to
    /// move. Only information that player has.
    fn observe<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.env.len();
        let (obs, mask) = (zeroed::<f32>(py, n * OBS_DIM)?, zeroed::<u8>(py, n * NUM_ACTIONS)?);
        let (mover, plies) = (zeroed::<i64>(py, n)?, zeroed::<i64>(py, n)?);
        {
            // Safety: fresh bytearrays, not yet visible to Python.
            let (o, m) = unsafe { (slice_mut::<f32>(&obs)?, slice_mut::<u8>(&mask)?) };
            let (mv, pl) = unsafe { (slice_mut::<i64>(&mover)?, slice_mut::<i64>(&plies)?) };
            let env = &self.env;
            py.detach(|| env.observe_zeroed(o, m, mv, pl));
        }
        let out = (
            ndarray(&obs, "float32", &[n, OBS_DIM])?,
            ndarray(&mask, "uint8", &[n, NUM_ACTIONS])?,
            ndarray(&mover, "int64", &[n])?,
            ndarray(&plies, "int64", &[n])?,
        );
        Ok(out.into_pyobject(py)?.into_any())
    }

    /// `observe()` into caller-owned arrays (e.g. numpy views of pinned torch tensors for fast host-to-device
    /// copies): obs float32 [n, OBS_DIM], legal uint8 [n, 36], mover int64 [n], ply int64 [n], all C-contiguous.
    /// The arrays are overwritten in place; nothing is allocated.
    fn observe_into(
        &self,
        py: Python<'_>,
        obs: &Bound<'_, PyAny>,
        legal: &Bound<'_, PyAny>,
        mover: &Bound<'_, PyAny>,
        ply: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let n = self.env.len();
        let spans = disjoint([
            array_data(obs, "<f4", n * OBS_DIM, 4)?,
            array_data(legal, "|u1", n * NUM_ACTIONS, 1)?,
            array_data(mover, "<i8", n, 8)?,
            array_data(ply, "<i8", n, 8)?,
        ])?;
        // Safety: validated above (type, length, alignment, writable, disjoint); the arrays are kept alive by the
        // borrowed references for the duration of the call.
        let (o, m, mv, pl) = unsafe {
            (
                std::slice::from_raw_parts_mut(spans[0].0 as *mut f32, n * OBS_DIM),
                std::slice::from_raw_parts_mut(spans[1].0 as *mut u8, n * NUM_ACTIONS),
                std::slice::from_raw_parts_mut(spans[2].0 as *mut i64, n),
                std::slice::from_raw_parts_mut(spans[3].0 as *mut i64, n),
            )
        };
        let env = &self.env;
        py.detach(|| env.observe(o, m, mv, pl));
        Ok(())
    }

    /// `privileged()` into a caller-owned float32 [n, PRIV_DIM] C-contiguous array.
    fn privileged_into(&self, py: Python<'_>, out: &Bound<'_, PyAny>) -> PyResult<()> {
        let n = self.env.len();
        let (addr, _) = array_data(out, "<f4", n * PRIV_DIM, 4)?;
        let out = unsafe { std::slice::from_raw_parts_mut(addr as *mut f32, n * PRIV_DIM) };
        let env = &self.env;
        py.detach(|| env.privileged(out));
        Ok(())
    }

    /// float32 [n, PRIV_DIM]: hidden information about each env's position (opponent hand, both next draws), for
    /// critics only.
    fn privileged<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.env.len();
        let buf = zeroed::<f32>(py, n * PRIV_DIM)?;
        {
            let out = unsafe { slice_mut::<f32>(&buf)? };
            let env = &self.env;
            py.detach(|| env.privileged(out));
        }
        ndarray(&buf, "float32", &[n, PRIV_DIM])
    }

    /// Plays `actions[i]` in env `i` (negative = leave env `i` as it is). Returns `(done bool [n], result float32
    /// [n], margin int64 [n])`: for envs whose game just ended, player 0's game score (1/0.5/0) and points margin;
    /// those envs have already moved on to a fresh deal. Raises ValueError (playing nothing) on an illegal action.
    fn step<'py>(&mut self, py: Python<'py>, actions: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.env.len();
        let actions = i64s(actions)?;
        if actions.len() != n {
            return Err(PyValueError::new_err(format!("expected {n} actions, got {}", actions.len())));
        }
        let (done, result, margin) = (zeroed::<u8>(py, n)?, zeroed::<f32>(py, n)?, zeroed::<i64>(py, n)?);
        {
            let d = unsafe { slice_mut::<u8>(&done)? };
            let (r, m) = unsafe { (slice_mut::<f32>(&result)?, slice_mut::<i64>(&margin)?) };
            let mut flags = vec![false; n];
            let env = &mut self.env;
            py.detach(|| env.step(&actions, &mut flags, r, m)).map_err(PyValueError::new_err)?;
            for (d, f) in d.iter_mut().zip(flags) {
                *d = f as u8;
            }
        }
        let out = (ndarray(&done, "bool", &[n])?, ndarray(&result, "float32", &[n])?, ndarray(&margin, "int64", &[n])?);
        Ok(out.into_pyobject(py)?.into_any())
    }

    /// Restarts env `i` from the deal `Game.random(seeds[i], starts[i])`. Auto-reset deals are unaffected.
    fn reset_deals(&mut self, seeds: &Bound<'_, PyAny>, starts: &Bound<'_, PyAny>) -> PyResult<()> {
        let (seeds, starts) = (u64s(seeds)?, i64s(starts)?);
        let n = self.env.len();
        if seeds.len() != n || starts.len() != n {
            return Err(PyValueError::new_err(format!("expected {n} seeds and {n} starts")));
        }
        if starts.iter().any(|&s| s != 0 && s != 1) {
            return Err(PyValueError::new_err("starts must be 0 or 1"));
        }
        let starts: Vec<u8> = starts.into_iter().map(|s| s as u8).collect();
        self.env.reset_deals(&seeds, &starts);
        Ok(())
    }

    /// int64 [n, 2]: final points of both players in the last game that ended in each env.
    fn final_scores<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.env.len();
        let buf = zeroed::<i64>(py, 2 * n)?;
        let out = unsafe { slice_mut::<i64>(&buf)? };
        for (o, s) in out.chunks_mut(2).zip(self.env.final_scores()) {
            o[0] = s[0] as i64;
            o[1] = s[1] as i64;
        }
        ndarray(&buf, "int64", &[n, 2])
    }

    /// Public move history of env `i`'s current game: `[(player, action), ...]`, oldest first.
    fn history(&self, i: usize) -> PyResult<Vec<(u8, u8)>> {
        if i >= self.env.len() {
            return Err(PyValueError::new_err("env index out of range"));
        }
        Ok(self.env.history(i).plies().iter().map(|p| (p.player, p.action)).collect())
    }
}

fn parse_animal(name: &str) -> PyResult<u8> {
    animal_id(name).ok_or_else(|| PyValueError::new_err(format!("unknown animal {name:?}")))
}

/// Player `p`'s hand mask and deck (top first) from animal names, checked to be a dealt player: 4 animals in hand
/// and 8 in the deck, all 12 distinct (the duplicate check is the one `Game` makes, plus the deal sizes).
fn parse_dealt_player(p: usize, hand: &[String], deck: &[String]) -> PyResult<(u16, Vec<u8>)> {
    let hand_size = NUM_ANIMALS - DECK_CAP;
    if hand.len() != hand_size || deck.len() != DECK_CAP {
        return Err(PyValueError::new_err(format!(
            "player {p}: a deal has {hand_size} animals in hand and {DECK_CAP} in the deck, got {} and {}",
            hand.len(),
            deck.len()
        )));
    }
    let (mut seen, mut hand_mask, mut deck_ids) = (0u16, 0u16, Vec::with_capacity(DECK_CAP));
    for (i, name) in hand.iter().chain(deck).enumerate() {
        let a = parse_animal(name)?;
        if seen >> a & 1 != 0 {
            return Err(PyValueError::new_err(format!("player {p} has two {name}s")));
        }
        seen |= 1 << a;
        if i < hand_size {
            hand_mask |= 1 << a;
        } else {
            deck_ids.push(a);
        }
    }
    Ok((hand_mask, deck_ids))
}

/// `(obs float32 [OBS_DIM], legal uint8 [36], privileged float32 [PRIV_DIM])` for the player to move after
/// `actions` (action indices) from the deal `start, hands, decks` (animal names, decks top first, as in `Game`).
/// The deal must be a full one: each player has 4 animals in hand and the other 8 in the deck. Raises ValueError
/// otherwise, on an illegal action, or if the game is over.
#[pyfunction]
fn encode_moves<'py>(
    py: Python<'py>,
    start: u8,
    hands: Vec<Vec<String>>,
    decks: Vec<Vec<String>>,
    actions: Vec<i64>,
) -> PyResult<Bound<'py, PyAny>> {
    if start > 1 || hands.len() != 2 || decks.len() != 2 {
        return Err(PyValueError::new_err("expected start 0/1, two hands and two decks"));
    }
    let mut hand_masks = [0u16; 2];
    let mut deck_ids: [Vec<u8>; 2] = Default::default();
    for p in 0..2 {
        (hand_masks[p], deck_ids[p]) = parse_dealt_player(p, &hands[p], &decks[p])?;
    }
    let mut s = State::new(start, hand_masks, [&deck_ids[0], &deck_ids[1]]);
    let mut h = History::default();
    for &a in &actions {
        if !(0..NUM_ACTIONS as i64).contains(&a) || !s.legal_actions().contains(a as u8) {
            return Err(PyValueError::new_err(format!("illegal action {a} at ply {}", h.len())));
        }
        h.push(&s, a as u8);
        s.apply(a as u8);
    }
    if s.is_over() {
        return Err(PyValueError::new_err("the game is over"));
    }
    let (obs, mask, privileged) = (zeroed::<f32>(py, OBS_DIM)?, zeroed::<u8>(py, NUM_ACTIONS)?, zeroed::<f32>(py, PRIV_DIM)?);
    unsafe {
        encode_obs(&s, &h, slice_mut::<f32>(&obs)?);
        encode_mask(&s, slice_mut::<u8>(&mask)?);
        encode_privileged(&s, slice_mut::<f32>(&privileged)?);
    }
    let out = (
        ndarray(&obs, "float32", &[OBS_DIM])?,
        ndarray(&mask, "uint8", &[NUM_ACTIONS])?,
        ndarray(&privileged, "float32", &[PRIV_DIM])?,
    );
    Ok(out.into_pyobject(py)?.into_any())
}

/// `{name: (offset, size)}` for every block of the observation (and of the privileged vector, `priv_*`).
#[pyfunction]
fn obs_layout<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (name, offset, size) in OBS_LAYOUT {
        d.set_item(name, (offset, size))?;
    }
    for (name, offset) in [
        ("history_by_me", features::HIST_BY_ME),
        ("history_by_opp", features::HIST_BY_OPP),
        ("history_action", features::HIST_ACTION),
        ("history_queue", features::HIST_QUEUE),
    ] {
        d.set_item(name, (offset, 1))?;
    }
    d.set_item("history_slot", (0, features::HISTORY_SLOT))?;
    d.set_item("queue_slot", (0, features::QUEUE_SLOT))?;
    Ok(d)
}

/// Times `steps` VecEnv steps at `n_envs` envs with uniformly random legal actions, all in Rust. Returns seconds
/// spent in observe / privileged / step, and the decisions and games played.
#[pyfunction]
#[pyo3(signature = (n_envs, steps, seed = 0, threads = 0))]
fn env_bench<'py>(py: Python<'py>, n_envs: usize, steps: usize, seed: u64, threads: usize) -> PyResult<Bound<'py, PyDict>> {
    let b = py.detach(|| env::bench(n_envs, steps, seed, threads));
    let d = PyDict::new(py);
    d.set_item("observe", b.observe)?;
    d.set_item("privileged", b.privileged)?;
    d.set_item("step", b.step)?;
    d.set_item("decisions", b.decisions)?;
    d.set_item("games", b.games)?;
    Ok(d)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVecEnv>()?;
    m.add_function(wrap_pyfunction!(encode_moves, m)?)?;
    m.add_function(wrap_pyfunction!(obs_layout, m)?)?;
    m.add_function(wrap_pyfunction!(env_bench, m)?)?;
    m.add("OBS_DIM", OBS_DIM)?;
    m.add("PRIV_DIM", PRIV_DIM)?;
    m.add("OBS_VERSION", OBS_VERSION)?;
    m.add("MAX_PLIES", MAX_PLIES)?;
    m.add("NUM_ACTIONS", NUM_ACTIONS)?;
    Ok(())
}
