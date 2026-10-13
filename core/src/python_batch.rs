//! `bbar._core.WindowBatch`: the half-iterations of many depth-limited windows (`Window` with `leaf_ply`, after
//! `ext_start`) in one call each, for `bbar/value/dl.solve_many`. Registered from `python_solve::register`.
//!
//! `Window.ext_begin` / `ext_end` move every window's leaf beliefs and values through fresh float64
//! `(num_value_leaves, 2, 495)` numpy arrays with the GIL held; with 128 windows of ~70 value leaves that is ~70 MB
//! each way per half-iteration, plus Python's concatenation and splitting, all serial. A batch instead runs
//! `Cfr::ext_begin` / `Cfr::ext_end` of all its windows on `threads` threads with the GIL released, reading and
//! writing ONE caller-owned array for all value leaves: rows in window order (each window's value leaves in
//! `value_leaves()` order, as `np.concatenate` of the per-window arrays), `[row, r, x]` with `r` the player (absolute,
//! or with `relative=True` the leaf's mover first, as the value network wants it) and `x` the entry:
//! - layout "slots": `x` is the hand slot (`hand_rank`, `C(12, k)` of them, 0 at infeasible slots): exactly
//!   `Window.ext_begin`'s arrays, concatenated;
//! - layout "compact": `x` runs over that player's feasible hands at the leaf in increasing slot order
//!   (`slots()[row, r, x]`), padded with 0 to the batch's `width` (most feasible hands at any leaf): ~7x smaller at
//!   ply 10.
//!
//! Arrays are numpy arrays (numpy.ndarray, nothing else: `python_env::array_info`), float32 or float64, C-contiguous
//! (outputs writable). Beliefs are converted from the solver's float64 by rounding to nearest (as numpy's
//! `astype(float32)`), values widened exactly: float32 transport is what the value network sees and returns anyway
//! (bbar/value/net.NetValues), so a solve through a float32 batch follows the same trajectory bit for bit. Each
//! window's computation is the same as through `ext_begin` / `ext_end` whatever the number of threads.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::hands::{num_hands, rank};
use crate::python_env::{array_info, ndarray, slice_mut, zeroed};
use crate::python_solve::PyWindow;
use crate::window::Window;

/// Element type of a leaf array.
trait Elem: Copy + Send + Sync + 'static {
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl Elem for f32 {
    #[inline]
    fn from_f64(x: f64) -> f32 {
        x as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl Elem for f64 {
    #[inline]
    fn from_f64(x: f64) -> f64 {
        x
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

/// Address and element type (true: float32, false: float64) of `arr`, a C-contiguous float32 or float64 numpy array
/// of `len` elements (writable if `write`). Only real numpy arrays pass (`python_env::array_info`: numpy's own report
/// of the array's memory, which the caller's reference keeps alive): anything else, e.g. an object with a made-up
/// `__array_interface__`, is a ValueError, never an address.
fn leaf_array(arr: &Bound<'_, PyAny>, len: usize, write: bool, what: &str) -> PyResult<(usize, bool)> {
    let err = |m: String| PyValueError::new_err(format!("{what}: {m}"));
    let a = array_info(arr, what)?;
    let f32_ = match a.typestr.as_str() {
        "<f4" => true,
        "<f8" => false,
        ts => return Err(err(format!("dtype {ts}, expected float32 or float64"))),
    };
    if a.len != len {
        return Err(err(format!("{} elements, expected {len} (rows x 2 x width)", a.len)));
    }
    if !a.c_contiguous {
        return Err(err("not C-contiguous".into()));
    }
    if write && a.readonly {
        return Err(err("read-only".into()));
    }
    if len > 0 && a.addr % if f32_ { 4 } else { 8 } != 0 {
        return Err(err("misaligned".into()));
    }
    Ok((a.addr, f32_))
}

/// The `len` elements at `addr` (an array validated by [`leaf_array`]); an empty slice when `len` is 0, whatever the
/// address.
///
/// Safety: `addr` must hold `len` aligned elements of `T` that stay alive and unaliased for `'a`.
unsafe fn slice_at<'a, T>(addr: usize, len: usize) -> &'a [T] {
    if len == 0 { &[] } else { unsafe { std::slice::from_raw_parts(addr as *const T, len) } }
}

/// [`slice_at`], mutable.
///
/// Safety: as [`slice_at`], and nothing else may read or write the elements for `'a`.
#[allow(clippy::mut_from_ref)]
unsafe fn slice_at_mut<'a, T>(addr: usize, len: usize) -> &'a mut [T] {
    if len == 0 { &mut [] } else { unsafe { std::slice::from_raw_parts_mut(addr as *mut T, len) } }
}

/// Static layout of one window's value leaves.
struct Layout {
    /// Per leaf: the mover, and both players' offsets and counts in the leaf-values buffer.
    mover: Vec<u8>,
    off: Vec<[u32; 2]>,
    n: Vec<[u32; 2]>,
    /// Hand slot of every leaf-values entry.
    slot: Vec<u16>,
    n_leaf_values: usize,
}

impl Layout {
    fn new(w: &Window) -> Layout {
        let mut slot = vec![0u16; w.n_leaf_values];
        for (j, leaf) in w.value_leaves.iter().enumerate() {
            for p in 0..2 {
                for (h, &mask) in w.leaf_hands(j, p).iter().enumerate() {
                    slot[leaf.off[p] as usize + h] = rank(mask) as u16;
                }
            }
        }
        Layout {
            mover: w.value_leaves.iter().map(|l| l.mover).collect(),
            off: w.value_leaves.iter().map(|l| l.off).collect(),
            n: w.value_leaves.iter().map(|l| l.n).collect(),
            slot,
            n_leaf_values: w.n_leaf_values,
        }
    }

    fn leaves(&self) -> usize {
        self.mover.len()
    }

    /// The absolute player of row `r` of leaf `j`.
    #[inline]
    fn player(&self, j: usize, r: usize, relative: bool) -> usize {
        if relative { r ^ self.mover[j] as usize } else { r }
    }
}

/// A non-finite value read: (value leaf of the window, row player, entry, value).
type BadValue = (usize, usize, usize, f64);

/// Shape of the batch's arrays and how rows map to players.
#[derive(Clone, Copy)]
struct Shape {
    width: usize,
    compact: bool,
    relative: bool,
}

impl Shape {
    /// Writes a window's leaf-values buffer `src` (beliefs) into its rows `dst`.
    fn write<T: Elem>(self, lay: &Layout, src: &[f64], dst: &mut [T]) {
        let wd = self.width;
        for j in 0..lay.leaves() {
            for r in 0..2 {
                let p = lay.player(j, r, self.relative);
                let (off, n) = (lay.off[j][p] as usize, lay.n[j][p] as usize);
                let row = &mut dst[(2 * j + r) * wd..][..wd];
                let s = &src[off..off + n];
                if self.compact {
                    for (d, &x) in row.iter_mut().zip(s) {
                        *d = T::from_f64(x);
                    }
                    row[n..].fill(T::from_f64(0.0));
                } else {
                    row.fill(T::from_f64(0.0));
                    for (&x, &sl) in s.iter().zip(&lay.slot[off..off + n]) {
                        row[sl as usize] = T::from_f64(x);
                    }
                }
            }
        }
    }

    /// Reads a window's rows `src` (values) into a leaf-values buffer; Err((value leaf, row player, entry, value)) at
    /// the first non-finite entry read (only feasible entries are read).
    fn read<T: Elem>(self, lay: &Layout, src: &[T]) -> Result<Vec<f64>, BadValue> {
        let wd = self.width;
        let mut out = vec![0.0; lay.n_leaf_values];
        for j in 0..lay.leaves() {
            for r in 0..2 {
                let p = lay.player(j, r, self.relative);
                let (off, n) = (lay.off[j][p] as usize, lay.n[j][p] as usize);
                let row = &src[(2 * j + r) * wd..][..wd];
                for h in 0..n {
                    let x = if self.compact { row[h] } else { row[lay.slot[off + h] as usize] }.to_f64();
                    if !x.is_finite() {
                        let col = if self.compact { h } else { lay.slot[off + h] as usize };
                        return Err((j, r, col, x));
                    }
                    out[off + h] = x;
                }
            }
        }
        Ok(out)
    }
}

/// Runs `f` on every item, on up to `threads` scoped threads taking items in order (dynamic scheduling).
fn par_each<I: Send>(items: Vec<I>, threads: usize, f: impl Fn(I) + Sync) {
    let threads = threads.max(1).min(items.len());
    if threads <= 1 {
        items.into_iter().for_each(f);
        return;
    }
    let cells: Vec<Mutex<Option<I>>> = items.into_iter().map(|x| Mutex::new(Some(x))).collect();
    let next = AtomicUsize::new(0);
    let work = || {
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(cell) = cells.get(i) else { break };
            let item = cell.lock().unwrap_or_else(|e| e.into_inner()).take().expect("item taken twice");
            f(item);
        }
    };
    thread::scope(|sc| {
        for _ in 1..threads {
            sc.spawn(work);
        }
        work();
    });
}

/// Splits `buf` into consecutive chunks of `lens`.
fn chunks<T>(mut buf: &mut [T], lens: impl Iterator<Item = usize>) -> Vec<&mut [T]> {
    let mut out = Vec::new();
    for len in lens {
        let (a, b) = std::mem::take(&mut buf).split_at_mut(len);
        out.push(a);
        buf = b;
    }
    out
}

/// Many depth-limited windows' half-iterations in one call each (module docs of `core/src/python_batch.rs`).
///
/// `WindowBatch(windows, layout="compact", relative=True, threads=1)`: `windows` are `Window`s with the same hand size,
/// each started with `ext_start` before `begin` (a window may hold no value leaf: it still iterates). `begin(out)` runs
/// `ext_begin` on every window and writes all leaf beliefs to `out` (float32 or float64, `(num_leaves, 2, width)`);
/// `end(values, out=None)` takes all leaf values in the same layout and runs `ext_end` on every window (values are
/// checked finite before any window is updated), then, given `out`, the next `begin(out)` in the same pass. Each
/// window ends exactly as through its own `ext_begin` / `ext_end` with the same values.
#[pyclass(name = "WindowBatch", module = "bbar._core", skip_from_py_object)]
pub struct PyWindowBatch {
    windows: Vec<Py<PyWindow>>,
    layouts: Vec<Layout>,
    /// First row of each window, then the total.
    rows: Vec<usize>,
    shape: Shape,
    slots: usize,
    threads: usize,
    /// Windows in the order threads take them: most leaf values first.
    order: Vec<usize>,
    /// Seconds inside begin / end (GIL released), and calls.
    seconds: [f64; 2],
    calls: [u64; 2],
}

impl PyWindowBatch {
    fn len(&self) -> usize {
        self.rows[self.windows.len()] * 2 * self.shape.width
    }

    /// Exclusive borrows of every window, checked: an external solve started and (`pending`) a query in progress
    /// or not.
    fn borrow<'py>(&self, py: Python<'py>, pending: bool) -> PyResult<Vec<PyRefMut<'py, PyWindow>>> {
        let mut out = Vec::with_capacity(self.windows.len());
        for (i, w) in self.windows.iter().enumerate() {
            let g = w
                .bind(py)
                .try_borrow_mut()
                .map_err(|_| PyValueError::new_err(format!("window {i} of the batch is in use elsewhere")))?;
            let state = g
                .cfr
                .as_ref()
                .filter(|_| g.external)
                .map(|c| c.ext.as_ref().is_some_and(|e| e.query.is_some()));
            match (state, pending) {
                (None, _) => {
                    return Err(PyValueError::new_err(format!(
                        "window {i}: no depth-limited solve in progress: call ext_start() first"
                    )));
                }
                (Some(true), false) => {
                    return Err(PyValueError::new_err(format!("window {i}: begin called twice without end")));
                }
                (Some(false), true) => {
                    return Err(PyValueError::new_err(format!("window {i}: end without begin")));
                }
                _ => {}
            }
            out.push(g);
        }
        Ok(out)
    }

    fn begin_t<T: Elem>(&self, wins: Vec<&mut PyWindow>, out: &mut [T]) {
        let (shape, wd) = (self.shape, self.shape.width);
        let parts = chunks(out, (0..wins.len()).map(|i| (self.rows[i + 1] - self.rows[i]) * 2 * wd));
        let mut items: Vec<_> = wins.into_iter().zip(parts).zip(&self.layouts).map(Some).collect();
        let items: Vec<_> = self.order.iter().map(|&i| items[i].take().expect("window order")).collect();
        par_each(items, self.threads, |((pw, dst), lay)| {
            let PyWindow { w, cfr, .. } = pw;
            let q = cfr.as_mut().expect("checked").ext_begin(w);
            shape.write(lay, &q.beliefs, dst);
        });
    }

    /// Values -> every window's leaf-values buffer (Err: the first non-finite value, in batch order).
    fn read_t<T: Elem>(&self, values: &[T]) -> PyResult<Vec<Vec<f64>>> {
        let (shape, wd) = (self.shape, self.shape.width);
        let n = self.windows.len();
        let mut parts = Vec::with_capacity(n);
        let mut rest = values;
        for i in 0..n {
            let (a, b) = rest.split_at((self.rows[i + 1] - self.rows[i]) * 2 * wd);
            parts.push(a);
            rest = b;
        }
        // Each window's buffer, in batch order (threads take the windows in `order`).
        let results = Mutex::new((0..n).map(|_| None).collect::<Vec<Option<Result<Vec<f64>, BadValue>>>>());
        par_each(self.order.clone(), self.threads, |i| {
            let r = shape.read(&self.layouts[i], parts[i]);
            results.lock().unwrap_or_else(|e| e.into_inner())[i] = Some(r);
        });
        let mut out = Vec::with_capacity(n);
        for (i, r) in results.into_inner().unwrap_or_else(|e| e.into_inner()).into_iter().enumerate() {
            match r.expect("read") {
                Ok(v) => out.push(v),
                Err((j, p, x, v)) => {
                    return Err(PyValueError::new_err(format!(
                        "values[{}, {p}, {x}] is {v} (window {i}, value leaf {j})",
                        self.rows[i] + j
                    )));
                }
            }
        }
        Ok(out)
    }

    fn end_t<U: Elem>(
        &self,
        wins: Vec<&mut PyWindow>,
        vals: Vec<Vec<f64>>,
        out: Option<&mut [U]>,
    ) -> Result<(), String> {
        let (shape, wd) = (self.shape, self.shape.width);
        let n = wins.len();
        let mut parts: Vec<Option<&mut [U]>> = match out {
            Some(o) => chunks(o, (0..n).map(|i| (self.rows[i + 1] - self.rows[i]) * 2 * wd))
                .into_iter()
                .map(Some)
                .collect(),
            None => (0..n).map(|_| None).collect(),
        };
        let mut items: Vec<_> = wins
            .into_iter()
            .zip(vals)
            .zip(parts.iter_mut().map(|p| p.take()))
            .zip(&self.layouts)
            .enumerate()
            .map(Some)
            .collect();
        let items: Vec<_> = self.order.iter().map(|&i| items[i].take().expect("window order")).collect();
        let err: Mutex<Option<(usize, String)>> = Mutex::new(None);
        par_each(items, self.threads, |(i, (((pw, v), dst), lay))| {
            let PyWindow { w, cfr, .. } = pw;
            let cfr = cfr.as_mut().expect("checked");
            if let Err(e) = cfr.ext_end(w, v) {
                let mut g = err.lock().unwrap_or_else(|e| e.into_inner());
                if g.as_ref().is_none_or(|(k, _)| i < *k) {
                    *g = Some((i, e));
                }
                return;
            }
            if let Some(dst) = dst {
                let q = cfr.ext_begin(w);
                shape.write(lay, &q.beliefs, dst);
            }
        });
        match err.into_inner().unwrap_or_else(|e| e.into_inner()) {
            Some((i, e)) => Err(format!("window {i}: {e}")),
            None => Ok(()),
        }
    }
}

#[pymethods]
impl PyWindowBatch {
    #[new]
    #[pyo3(signature = (windows, layout="compact", relative=true, threads=1))]
    fn new(
        py: Python<'_>,
        windows: Vec<Py<PyWindow>>,
        layout: &str,
        relative: bool,
        threads: usize,
    ) -> PyResult<PyWindowBatch> {
        let compact = match layout {
            "compact" => true,
            "slots" => false,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "layout must be compact or slots, not {layout:?}"
                )));
            }
        };
        let mut layouts = Vec::with_capacity(windows.len());
        let mut rows = vec![0usize];
        let mut k = None;
        let mut seen = std::collections::HashSet::new();
        for (i, w) in windows.iter().enumerate() {
            if !seen.insert(w.as_ptr() as usize) {
                return Err(PyValueError::new_err(format!("window {i} is in the batch twice")));
            }
            let g = w
                .bind(py)
                .try_borrow()
                .map_err(|_| PyValueError::new_err(format!("window {i} is in use")))?;
            if *k.get_or_insert(g.w.k) != g.w.k {
                return Err(PyValueError::new_err("the windows of a batch must have the same hand size k"));
            }
            let lay = Layout::new(&g.w);
            rows.push(rows[i] + lay.leaves());
            layouts.push(lay);
        }
        let slots = num_hands(k.unwrap_or(4));
        let width = if compact {
            layouts.iter().flat_map(|l| l.n.iter().flatten()).map(|&n| n as usize).max().unwrap_or(0)
        } else {
            slots
        };
        let mut order: Vec<usize> = (0..windows.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(layouts[i].n_leaf_values));
        Ok(PyWindowBatch {
            windows,
            layouts,
            rows,
            shape: Shape { width, compact, relative },
            slots,
            threads: threads.max(1),
            order,
            seconds: [0.0; 2],
            calls: [0; 2],
        })
    }

    /// Value leaves of all windows (rows of the arrays).
    #[getter]
    fn num_leaves(&self) -> usize {
        self.rows[self.windows.len()]
    }

    /// Last axis of the arrays: hand slots (layout "slots") or the most feasible hands at any leaf ("compact").
    #[getter]
    fn width(&self) -> usize {
        self.shape.width
    }

    #[getter]
    fn layout(&self) -> &'static str {
        if self.shape.compact { "compact" } else { "slots" }
    }

    #[getter]
    fn relative(&self) -> bool {
        self.shape.relative
    }

    #[getter]
    fn threads(&self) -> usize {
        self.threads
    }

    #[setter]
    fn set_threads(&mut self, threads: usize) {
        self.threads = threads.max(1);
    }

    /// First row of each window, then the number of rows.
    fn rows(&self) -> Vec<usize> {
        self.rows.clone()
    }

    /// int64 `(num_leaves,)`: the player to move at each row's value leaf.
    fn movers<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let n = self.num_leaves();
        let buf = zeroed::<i64>(py, n)?;
        // Safety: a fresh bytearray, not yet visible to Python.
        let out = unsafe { slice_mut::<i64>(&buf)? };
        for (x, m) in out.iter_mut().zip(self.layouts.iter().flat_map(|l| l.mover.iter())) {
            *x = *m as i64;
        }
        ndarray(&buf, "int64", &[n])
    }

    /// int64 `(num_leaves, 2, width)`: the hand slot of each entry of the compact layout (padding: a slot that is not
    /// that player's feasible hand at the leaf, where beliefs are 0) and the number of feasible hands `(num_leaves, 2)`.
    fn slots<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        if !self.shape.compact {
            return Err(PyValueError::new_err("slots(): the layout is already by slot"));
        }
        let (n, wd) = (self.num_leaves(), self.shape.width);
        let buf = zeroed::<i64>(py, n * 2 * wd)?;
        let cnt = zeroed::<i64>(py, n * 2)?;
        {
            // Safety: fresh bytearrays, not yet visible to Python.
            let (out, counts) = unsafe { (slice_mut::<i64>(&buf)?, slice_mut::<i64>(&cnt)?) };
            let mut row = 0;
            for lay in &self.layouts {
                for j in 0..lay.leaves() {
                    for r in 0..2 {
                        let p = lay.player(j, r, self.shape.relative);
                        let (off, k) = (lay.off[j][p] as usize, lay.n[j][p] as usize);
                        let dst = &mut out[(2 * row + r) * wd..][..wd];
                        let feas = &lay.slot[off..off + k];
                        for (d, &s) in dst.iter_mut().zip(feas) {
                            *d = s as i64;
                        }
                        if k < wd {
                            // Feasible slots are increasing: the first gap is not feasible.
                            let pad = (0..self.slots)
                                .find(|&s| feas.binary_search(&(s as u16)).is_err())
                                .expect("gap");
                            dst[k..].fill(pad as i64);
                        }
                        counts[2 * row + r] = k as i64;
                    }
                    row += 1;
                }
            }
        }
        Ok((ndarray(&buf, "int64", &[n, 2, wd])?, ndarray(&cnt, "int64", &[n, 2])?))
    }

    /// Starts the next half-iteration of every window (`Window.ext_begin`) and writes all leaf beliefs to `out`.
    fn begin(&mut self, py: Python<'_>, out: &Bound<'_, PyAny>) -> PyResult<()> {
        let len = self.len();
        let (addr, f32_) = leaf_array(out, len, true, "out")?;
        let mut guards = self.borrow(py, false)?;
        let wins: Vec<&mut PyWindow> = guards.iter_mut().map(|g| &mut **g).collect();
        let t0 = Instant::now();
        let this = &*self;
        // Safety: validated above (type, length, alignment, writable); `out` is kept alive by the borrowed reference
        // for the duration of the call.
        if f32_ {
            let o = unsafe { slice_at_mut::<f32>(addr, len) };
            py.detach(|| this.begin_t(wins, o));
        } else {
            let o = unsafe { slice_at_mut::<f64>(addr, len) };
            py.detach(|| this.begin_t(wins, o));
        }
        self.seconds[0] += t0.elapsed().as_secs_f64();
        self.calls[0] += 1;
        Ok(())
    }

    /// Ends the half-iteration of every window (`Window.ext_end`) with all leaf values `values` (float32 or float64,
    /// the batch's layout; only feasible entries are read and must be finite: checked before any window changes).
    /// Given `out`, then starts the next half-iteration and writes the beliefs there (as `begin(out)`).
    #[pyo3(signature = (values, out=None))]
    fn end(
        &mut self,
        py: Python<'_>,
        values: &Bound<'_, PyAny>,
        out: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let len = self.len();
        let (va, vf32) = leaf_array(values, len, false, "values")?;
        let o = match out {
            Some(o) => {
                let (oa, of32) = leaf_array(o, len, true, "out")?;
                let bytes = |f: bool| len * if f { 4 } else { 8 };
                if len > 0 && oa < va + bytes(vf32) && va < oa + bytes(of32) {
                    return Err(PyValueError::new_err("values and out overlap"));
                }
                Some((oa, of32))
            }
            None => None,
        };
        let mut guards = self.borrow(py, true)?;
        let wins: Vec<&mut PyWindow> = guards.iter_mut().map(|g| &mut **g).collect();
        let t0 = Instant::now();
        let this = &*self;
        // Safety: validated above (type, length, alignment, writable out, no overlap); the arrays are kept alive by
        // the borrowed references for the duration of the call.
        let vals = if vf32 {
            let v = unsafe { slice_at::<f32>(va, len) };
            py.detach(|| this.read_t(v))
        } else {
            let v = unsafe { slice_at::<f64>(va, len) };
            py.detach(|| this.read_t(v))
        }?;
        let res = match o {
            None => py.detach(|| this.end_t::<f64>(wins, vals, None)),
            Some((oa, true)) => {
                let o = unsafe { slice_at_mut::<f32>(oa, len) };
                py.detach(|| this.end_t::<f32>(wins, vals, Some(o)))
            }
            Some((oa, false)) => {
                let o = unsafe { slice_at_mut::<f64>(oa, len) };
                py.detach(|| this.end_t::<f64>(wins, vals, Some(o)))
            }
        };
        self.seconds[1] += t0.elapsed().as_secs_f64();
        self.calls[1] += 1;
        if o.is_some() {
            self.calls[0] += 1;
        }
        res.map_err(PyValueError::new_err)
    }

    /// Frees every window's between-half-iteration caches (`Cfr::ext_release`), e.g. once the solves are done;
    /// windows with a half-iteration in progress keep theirs.
    fn release(&self, py: Python<'_>) -> PyResult<()> {
        for (i, w) in self.windows.iter().enumerate() {
            let mut g = w
                .bind(py)
                .try_borrow_mut()
                .map_err(|_| PyValueError::new_err(format!("window {i} of the batch is in use elsewhere")))?;
            if let Some(c) = g.cfr.as_mut() {
                c.ext_release();
            }
        }
        Ok(())
    }

    /// Seconds spent inside begin / end (the GIL released) and their calls (an `end` with `out` counts as a begin
    /// too).
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("begin_seconds", self.seconds[0])?;
        d.set_item("end_seconds", self.seconds[1])?;
        d.set_item("begins", self.calls[0])?;
        d.set_item("ends", self.calls[1])?;
        Ok(d)
    }
}
