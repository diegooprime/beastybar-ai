//! Per-hand policy inputs: everything needed to ask a policy `sigma(a | that player's information)` for EVERY hand a
//! player could hold at a public history, in batches.
//!
//! - [`with_hand`]: a concrete position consistent with a public position in which one player holds a given hand
//!   (the other player's split and every deck order are arbitrary: views and observations ignore them).
//! - [`replay_public`] / [`decisions`]: the public positions of a move history (from the public start position, every
//!   unplayed animal in hand) and, for one player, each of their decisions with every hand they could hold there.
//! - [`bayes_filter`]: exact Bayes over a player's hands given per-hand likelihoods of their observed actions, pushed
//!   through the play-and-draw operator ([`Transition`]): `r'(H') = sum_H r(H) L(a | H) P(H -> H')`, starting from
//!   the deal prior (uniform over the feasible hands). Zero-probability observations: see the function.
//! - [`WindowTree`]: the public tree of a window (`window.rs`) walked the same way [`crate::window::Window::build`]
//!   walks it (same preorder, same strategy-entry offsets, no leaf values), with the real game history up to its root,
//!   so a policy can fill a window strategy table: one row per (node, hand of the mover), encoded as the observation
//!   (layout v1, `features.rs`) of the real game history plus the path inside the window.
//! - [`agent_counts`]: a built-in agent (agents.rs spec) called on many (position, history) tasks, each with a view
//!   consistent with the hand (the agent's own world sampling), deterministically from a seed, multithreaded.
//!
//! Ply convention: ply k = k moves played; the starting player moves at even plies.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;

use crate::action::{ActionSet, NUM_ACTIONS, decode};
use crate::agents::Spec;
use crate::arena::derive_seed;
use crate::features::{History, MAX_PLIES, OBS_DIM, encode_mask, encode_obs_zeroed};
use crate::hands::{Transition, hand_size, hands, num_hands, rank, unplayed};
use crate::markov::public;
use crate::rng::Rng;
use crate::state::{DECK_CAP, State};
use crate::view::view;
use crate::window::Leaves;

/// `derive_seed` path tags of the agent tasks.
pub const TAG_TABLE: u64 = 0x7461_626c; // "tabl"
pub const TAG_BELIEF: u64 = 0x6265_6c66; // "belf"

/// A position consistent with the public information of `ps` (a real or public position) in which `player` holds
/// `hand` (a possible hand: `hand ⊆ U_player` of size `min(k, |U_player|)`) and the other player holds the canonical
/// split of their unplayed animals (weakest first in hand). Decks are in increasing animal order. Observations and
/// views depend only on public information and the hand of the player they are for.
pub fn with_hand(ps: &State, player: usize, hand: u16, k: usize) -> State {
    let mut s = *ps;
    for q in 0..2 {
        let u = unplayed(ps, q);
        let m = hand_size(u, k);
        let h = if q == player {
            assert!(
                hand & !u == 0 && hand.count_ones() as usize == m,
                "hand {hand:#x} is not possible for {u:#x}"
            );
            hand
        } else {
            let mut h = 0u16;
            let mut bits = u;
            for _ in 0..m {
                h |= 1 << bits.trailing_zeros();
                bits &= bits - 1;
            }
            h
        };
        s.hands[q] = h;
        let mut deck = [0u8; DECK_CAP];
        let mut n = 0;
        let mut rest = u & !h;
        while rest != 0 {
            deck[n] = rest.trailing_zeros() as u8;
            rest &= rest - 1;
            n += 1;
        }
        s.decks[q] = deck;
        s.deck_len[q] = n as u8;
    }
    s
}

/// The public positions of a game: before each action and after the last (`actions.len() + 1` states), from the
/// public form of `start` (the position before `actions[0]`). Every action must be legal on the public position.
pub fn replay_public(start: &State, actions: &[u8]) -> Result<Vec<State>, String> {
    let mut s = public(start);
    let mut out = Vec::with_capacity(actions.len() + 1);
    for (t, &a) in actions.iter().enumerate() {
        if !s.legal_actions().contains(a) {
            return Err(format!("action {a} at ply {t} is not legal in the public position"));
        }
        out.push(s);
        s.apply(a);
    }
    out.push(s);
    Ok(out)
}

/// The public history (queues, movers, actions) of `actions` played from `start`.
pub fn history_of(publics: &[State], actions: &[u8]) -> History {
    let mut h = History::default();
    for (s, &a) in publics.iter().zip(actions) {
        h.push(s, a);
    }
    h
}

/// One decision of a player in a move history: the ply, the public position before it, and every hand the player
/// could hold there (increasing rank).
#[derive(Clone, Debug)]
pub struct Decision {
    pub ply: usize,
    pub public: State,
    pub hands: Vec<u16>,
}

/// Every decision of `player` in `actions` (from `start`), with the player's possible hands.
pub fn decisions(start: &State, actions: &[u8], player: usize, k: usize) -> Result<Vec<Decision>, String> {
    let publics = replay_public(start, actions)?;
    Ok((0..actions.len())
        .filter(|&t| publics[t].to_move as usize == player)
        .map(|t| Decision {
            ply: t,
            public: publics[t],
            hands: hands(unplayed(&publics[t], player), k).collect(),
        })
        .collect())
}

/// What the filter saw at one decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterStep {
    pub ply: usize,
    /// Probability of the observed action under the model given the public history:
    /// `sum_H r(H) L(a | H) / sum_H r(H)`.
    pub prob: f64,
    /// `prob <= eps`: the model (all but) rules the observed play out.
    pub impossible: bool,
}

/// Exact Bayes range of `player`'s hand after `actions` (from `start`), given the likelihood of each observed action
/// of theirs for every hand they could hold: `likelihood` has one entry per row of [`decisions`] (decision by
/// decision, hands in increasing rank). Starts from the deal prior: uniform over the feasible hands at `start`.
///
/// Zero probabilities. An action whose animal is not in a hand is impossible for that hand whatever the model (exact
/// elimination: the true hand always holds the played animal). For hands that hold it, the likelihood is floored at
/// `eps` (`max(L, eps)`), so a model that gives the observed play (almost) zero probability never wipes out the
/// true hand; the step is recorded as `impossible` when the model's probability of the observed action given the
/// public history is at most `eps`. With `eps = 0` and a zero-probability observation, the step is uninformative
/// (every hand that holds the animal keeps its weight) and recorded as impossible.
///
/// Returns the range in the slot layout of size `num_hands(min(k, |U|))` (`hands::rank`), normalised, and the steps.
pub fn bayes_filter(
    start: &State,
    actions: &[u8],
    player: usize,
    k: usize,
    likelihood: &[f64],
    eps: f64,
) -> Result<(Vec<f64>, Vec<FilterStep>), String> {
    if !(eps >= 0.0 && eps.is_finite()) {
        return Err(format!("eps must be finite and non-negative, got {eps}"));
    }
    let publics = replay_public(start, actions)?;
    let u0 = unplayed(&publics[0], player);
    let mut range = vec![0.0; num_hands(hand_size(u0, k))];
    for h in hands(u0, k) {
        range[rank(h)] = 1.0;
    }
    let mut steps = Vec::new();
    let mut j = 0;
    for (t, &a) in actions.iter().enumerate() {
        let s = &publics[t];
        if s.to_move as usize != player {
            continue;
        }
        let u = unplayed(s, player);
        let c = decode(a).animal;
        let hs: Vec<u16> = hands(u, k).collect();
        if j + hs.len() > likelihood.len() {
            return Err(format!("{} likelihoods given, decision at ply {t} needs more", likelihood.len()));
        }
        let lik = &likelihood[j..j + hs.len()];
        j += hs.len();
        if let Some(x) = lik.iter().find(|x| !x.is_finite() || **x < 0.0) {
            return Err(format!("likelihood {x} at ply {t} (must be finite and non-negative)"));
        }
        let (mut total, mut model, mut legal) = (0.0, 0.0, 0.0);
        for (&h, &l) in hs.iter().zip(lik) {
            let r = range[rank(h)];
            total += r;
            if h >> c & 1 != 0 {
                legal += r;
                model += r * l;
            }
        }
        if legal <= 0.0 {
            return Err(format!("ply {t}: no hand in the range holds the played animal"));
        }
        let prob = model / total;
        let impossible = prob <= eps;
        let mut weighted = vec![0.0; range.len()];
        for (&h, &l) in hs.iter().zip(lik) {
            if h >> c & 1 != 0 {
                let l = if eps == 0.0 && model <= 0.0 { 1.0 } else { l.max(eps) };
                weighted[rank(h)] = range[rank(h)] * l;
            }
        }
        let mass: f64 = weighted.iter().sum();
        for w in &mut weighted {
            *w /= mass;
        }
        let tr = Transition::new(u, k, c);
        let mut next = vec![0.0; num_hands(tr.to_size as usize)];
        tr.push(&weighted, &mut next);
        range = next;
        steps.push(FilterStep { ply: t, prob, impossible });
    }
    if j != likelihood.len() {
        return Err(format!("{} likelihoods given, the decisions have {j} hands", likelihood.len()));
    }
    let mass: f64 = range.iter().sum();
    for r in &mut range {
        *r /= mass;
    }
    Ok((range, steps))
}

/// The observation (layout v1) of the mover of `ps` (a real or public position) holding `hand`, with the public
/// history `hist`, into `obs` (zeroed, length OBS_DIM), and the legal mask of that hand into `legal`.
pub fn encode_hand(ps: &State, hand: u16, k: usize, hist: &History, obs: &mut [f32], legal: &mut [u8]) {
    let s = with_hand(ps, ps.to_move as usize, hand, k);
    encode_obs_zeroed(&s, hist, obs);
    encode_mask(&s, legal);
}

// ---- windows ------------------------------------------------------------------------------------------------------

/// An internal node of a [`WindowTree`] (a public position where a player decides).
#[derive(Clone, Copy, Debug)]
pub struct TreeNode {
    /// The public position (every unplayed animal in hand, decks empty).
    pub state: State,
    /// Parent node (`u32::MAX` at the root) and the action that led here.
    pub parent: u32,
    pub action: u8,
    pub mover: u8,
    /// Plies below the window root.
    pub depth: u8,
    /// The mover's possible hands.
    pub n_hands: u16,
    /// Public actions, in increasing index (the window's edges).
    pub legal: ActionSet,
    /// First strategy entry: `n_hands * legal.len()` entries, row-major by hand.
    pub entry: usize,
    /// Preorder: the subtree is `nodes[id..end]`.
    pub end: u32,
}

/// The public tree of a window, laid out like [`crate::window::Window`]: nodes in preorder, strategy entries
/// `entry .. entry + n_hands * n_actions` per node. Built from a start position and the real move history up to the
/// window root, which observations need.
#[derive(Clone, Debug)]
pub struct WindowTree {
    pub k: usize,
    pub leaves: Leaves,
    /// Public positions of the history up to the root (the last one is the root).
    pub publics: Vec<State>,
    pub root_actions: Vec<u8>,
    pub root_hist: History,
    pub nodes: Vec<TreeNode>,
    pub n_entries: usize,
    /// Per player: `(node, first row)` of every node they move at, in preorder.
    rows: [Vec<(u32, u32)>; 2],
    n_rows: [usize; 2],
}

/// Whether the public position `s` (not over) is a leaf of a window with these leaves (the rule of `window.rs`).
pub fn is_leaf(s: &State, k: usize, leaves: Leaves) -> bool {
    let p = s.to_move as usize;
    let n = |q: usize| unplayed(s, q).count_ones() as usize;
    match leaves {
        Leaves::Frontier => n(1 - p) <= k && n(p) <= k + 1,
        Leaves::Public => n(0) <= k && n(1) <= k,
        Leaves::Terminal => false,
    }
}

impl WindowTree {
    /// The window below the position reached by `actions` from `start`.
    pub fn new(start: &State, actions: &[u8], k: usize, leaves: Leaves) -> Result<WindowTree, String> {
        if actions.len() > MAX_PLIES {
            return Err(format!("{} actions: a game has at most {MAX_PLIES}", actions.len()));
        }
        let publics = replay_public(start, actions)?;
        let root_hist = history_of(&publics, actions);
        let mut t = WindowTree {
            k,
            leaves,
            publics,
            root_actions: actions.to_vec(),
            root_hist,
            nodes: Vec::new(),
            n_entries: 0,
            rows: [Vec::new(), Vec::new()],
            n_rows: [0, 0],
        };
        let root = *t.publics.last().unwrap();
        t.walk(root, u32::MAX, 0, 0);
        for (n, node) in t.nodes.iter().enumerate() {
            let p = node.mover as usize;
            t.rows[p].push((n as u32, t.n_rows[p] as u32));
            t.n_rows[p] += node.n_hands as usize;
        }
        Ok(t)
    }

    fn walk(&mut self, s: State, parent: u32, action: u8, depth: usize) {
        if s.is_over() || is_leaf(&s, self.k, self.leaves) {
            return;
        }
        let id = self.nodes.len();
        let p = s.to_move as usize;
        let legal = s.legal_actions();
        let n_hands = hands(unplayed(&s, p), self.k).count();
        self.nodes.push(TreeNode {
            state: s,
            parent,
            action,
            mover: p as u8,
            depth: depth as u8,
            n_hands: n_hands as u16,
            legal,
            entry: self.n_entries,
            end: 0,
        });
        self.n_entries += n_hands * legal.len();
        for a in legal {
            let mut c = s;
            c.apply(a);
            self.walk(c, id as u32, a, depth + 1);
        }
        self.nodes[id].end = self.nodes.len() as u32;
    }

    /// The public position at the window root.
    pub fn root(&self) -> &State {
        self.publics.last().unwrap()
    }

    /// Rows of `player`: one per (node where they move, possible hand).
    pub fn n_rows(&self, player: usize) -> usize {
        self.n_rows[player]
    }

    /// End (exclusive) of the strategy entries of node `n`'s subtree.
    pub fn entry_end(&self, n: usize) -> usize {
        let end = self.nodes[n].end as usize;
        if end < self.nodes.len() { self.nodes[end].entry } else { self.n_entries }
    }

    /// The node and hand index of `player`'s row `r`.
    pub fn row(&self, player: usize, r: usize) -> (usize, usize) {
        let rows = &self.rows[player];
        let i = rows.partition_point(|&(_, first)| first as usize <= r) - 1;
        let (n, first) = rows[i];
        (n as usize, r - first as usize)
    }

    /// Actions from the window root to node `n`.
    pub fn path(&self, mut n: usize) -> Vec<u8> {
        let mut out = Vec::new();
        while self.nodes[n].parent != u32::MAX {
            out.push(self.nodes[n].action);
            n = self.nodes[n].parent as usize;
        }
        out.reverse();
        out
    }

    /// Every action from the deal to node `n` (the root history plus the path).
    pub fn full_actions(&self, n: usize) -> Vec<u8> {
        let mut a = self.root_actions.clone();
        a.extend(self.path(n));
        a
    }

    /// The public history from the deal to node `n`.
    pub fn history(&self, n: usize) -> History {
        let mut chain = Vec::new();
        let mut m = n;
        while self.nodes[m].parent != u32::MAX {
            let parent = self.nodes[m].parent as usize;
            chain.push((parent, self.nodes[m].action));
            m = parent;
        }
        let mut h = self.root_hist;
        for &(parent, a) in chain.iter().rev() {
            h.push(&self.nodes[parent].state, a);
        }
        h
    }

    /// The internal node reached by `path` (public actions from the root), if any.
    pub fn find(&self, path: &[u8]) -> Option<usize> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut n = 0;
        for &a in path {
            let mut c = n + 1;
            let end = self.nodes[n].end as usize;
            let mut found = None;
            while c < end {
                if self.nodes[c].action == a {
                    found = Some(c);
                    break;
                }
                c = self.nodes[c].end as usize;
            }
            n = found?;
        }
        Some(n)
    }

    /// The possible hands of node `n`'s mover, increasing rank (the row order of the node's entries).
    pub fn hands(&self, n: usize) -> Vec<u16> {
        let node = &self.nodes[n];
        hands(unplayed(&node.state, node.mover as usize), self.k).collect()
    }

    /// Observations and legal masks of `player`'s rows `r0..r1` into `obs` (`(r1 - r0) * OBS_DIM`, zeroed) and
    /// `legal` (`(r1 - r0) * NUM_ACTIONS`), on `threads` threads (0: all cores).
    pub fn encode(
        &self,
        player: usize,
        r0: usize,
        r1: usize,
        obs: &mut [f32],
        legal: &mut [u8],
        threads: usize,
    ) {
        assert!(r0 <= r1 && r1 <= self.n_rows[player], "rows {r0}..{r1} of {}", self.n_rows[player]);
        let n = r1 - r0;
        assert!(obs.len() == n * OBS_DIM && legal.len() == n * NUM_ACTIONS);
        if n == 0 {
            return;
        }
        let threads = resolve_threads(threads).min(n.div_ceil(256)).max(1);
        let chunk = n.div_ceil(threads);
        let work = |a: usize, obs: &mut [f32], legal: &mut [u8]| {
            let b = (a + chunk).min(r1);
            let (mut node, mut i) = self.row(player, a);
            let mut cur = (usize::MAX, History::default(), Vec::new());
            for (j, r) in (a..b).enumerate() {
                if i >= self.nodes[node].n_hands as usize {
                    node = self.rows[player]
                        [self.rows[player].partition_point(|&(m, _)| (m as usize) <= node)]
                    .0 as usize;
                    i = 0;
                }
                debug_assert_eq!(self.row(player, r), (node, i));
                if cur.0 != node {
                    cur = (node, self.history(node), self.hands(node));
                }
                encode_hand(
                    &self.nodes[node].state,
                    cur.2[i],
                    self.k,
                    &cur.1,
                    &mut obs[j * OBS_DIM..(j + 1) * OBS_DIM],
                    &mut legal[j * NUM_ACTIONS..(j + 1) * NUM_ACTIONS],
                );
                i += 1;
            }
        };
        thread::scope(|sc| {
            let mut rest_obs = obs;
            let mut rest_legal = legal;
            let mut a = r0;
            let mut handles = Vec::new();
            while a < r1 {
                let len = chunk.min(r1 - a);
                let (o, ro) = std::mem::take(&mut rest_obs).split_at_mut(len * OBS_DIM);
                let (l, rl) = std::mem::take(&mut rest_legal).split_at_mut(len * NUM_ACTIONS);
                rest_obs = ro;
                rest_legal = rl;
                let work = &work;
                handles.push(sc.spawn(move || work(a, o, l)));
                a += len;
            }
            for h in handles {
                h.join().expect("encoder thread panicked");
            }
        });
    }

    /// `(row, action, entry)` for every strategy entry of `player`: the entry of the row's hand and that action.
    pub fn scatter(&self, player: usize) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        let (mut rows, mut acts, mut entries) = (Vec::new(), Vec::new(), Vec::new());
        for &(n, first) in &self.rows[player] {
            let node = &self.nodes[n as usize];
            let ne = node.legal.len();
            for i in 0..node.n_hands as usize {
                for (j, a) in node.legal.enumerate() {
                    rows.push(first as i64 + i as i64);
                    acts.push(a as i64);
                    entries.push((node.entry + i * ne + j) as i64);
                }
            }
        }
        (rows, acts, entries)
    }

    /// The mover of every strategy entry.
    pub fn entry_movers(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.n_entries];
        for node in &self.nodes {
            let len = node.n_hands as usize * node.legal.len();
            out[node.entry..node.entry + len].fill(node.mover);
        }
        out
    }

    /// A strategy table with `player`'s entries from `spec`'s agent: each (node, hand) row is played `samples` times,
    /// each time from a fresh view of a position where the mover holds that hand (the agent's own world sampling)
    /// with the history up to the node, and gets the frequencies of the actions chosen. Other entries are 0. The RNG
    /// of sample `j` of row `r` is `derive_seed(seed, [TAG_TABLE, player, r, j])`: results do not depend on `threads`.
    pub fn agent_table(
        &self,
        spec: Spec,
        player: usize,
        samples: u32,
        seed: u64,
        threads: usize,
        stop: &Arc<AtomicBool>,
    ) -> Result<Vec<f64>, String> {
        let task = |r: usize| {
            let (n, i) = self.row(player, r);
            let node = &self.nodes[n];
            let h = hands(unplayed(&node.state, player), self.k).nth(i).unwrap();
            (with_hand(&node.state, player, h, self.k), self.full_actions(n))
        };
        let counts = agent_counts(
            spec,
            self.n_rows[player],
            task,
            samples,
            seed,
            &[TAG_TABLE, player as u64],
            threads,
            stop,
        )?;
        let mut table = vec![0.0; self.n_entries];
        for &(n, first) in &self.rows[player] {
            let node = &self.nodes[n as usize];
            let ne = node.legal.len();
            for i in 0..node.n_hands as usize {
                let c = &counts[first as usize + i];
                for (j, a) in node.legal.enumerate() {
                    table[node.entry + i * ne + j] = c[a as usize] as f64 / samples as f64;
                }
            }
        }
        Ok(table)
    }
}

pub fn resolve_threads(threads: usize) -> usize {
    if threads == 0 { thread::available_parallelism().map_or(1, |t| t.get()) } else { threads }
}

/// Tasks handed to a worker at a time.
const CHUNK: usize = 4;

/// Runs `spec`'s agent `samples` times on each of `n` tasks (`task(i)` = a position whose mover holds the hand in
/// question, and every action played before it) and counts the actions chosen. Each run builds a view of the
/// position with its own RNG, `derive_seed(seed, path + [i, j])` for sample `j` of task `i`, and acts with the same
/// RNG, as the arena does. Agents are built once per thread; results do not depend on `threads` (search tables never
/// change decisions: tests/info_leaks.rs). Oracle specs are refused (they would see an arbitrary hidden split).
/// Setting `stop` abandons the work (Err).
#[allow(clippy::too_many_arguments)]
pub fn agent_counts<F>(
    spec: Spec,
    n: usize,
    task: F,
    samples: u32,
    seed: u64,
    path: &[u64],
    threads: usize,
    stop: &Arc<AtomicBool>,
) -> Result<Vec<[u32; NUM_ACTIONS]>, String>
where
    F: Fn(usize) -> (State, Vec<u8>) + Sync,
{
    if spec.is_oracle() {
        return Err(format!("{spec} is an oracle: it would see an arbitrary hidden split, not a real one"));
    }
    if samples == 0 {
        return Err("samples must be at least 1".into());
    }
    let mut out = vec![[0u32; NUM_ACTIONS]; n];
    if n == 0 {
        return Ok(out);
    }
    let threads = resolve_threads(threads).min(n.div_ceil(CHUNK)).max(1);
    let next = AtomicUsize::new(0);
    type Counts = Vec<(usize, [u32; NUM_ACTIONS])>;
    let worker = || -> Result<Counts, String> {
        let mut agent = spec.build_stoppable(stop);
        let mut local = Vec::new();
        let mut key: Vec<u64> = path.to_vec();
        key.extend([0, 0]);
        loop {
            let i0 = next.fetch_add(CHUNK, Ordering::Relaxed);
            if i0 >= n {
                break;
            }
            for i in i0..(i0 + CHUNK).min(n) {
                if stop.load(Ordering::Relaxed) {
                    return Err("stopped".into());
                }
                let (s, history) = task(i);
                let legal = s.legal_actions();
                let mut c = [0u32; NUM_ACTIONS];
                for j in 0..samples {
                    let len = key.len();
                    key[len - 2] = i as u64;
                    key[len - 1] = j as u64;
                    let mut rng = Rng::new(derive_seed(seed, &key));
                    let v = view(&s, s.to_move, &mut rng);
                    let a = agent.act(&v, &history, &mut rng);
                    if !legal.contains(a) {
                        return Err(format!("{spec} played illegal action {a}"));
                    }
                    c[a as usize] += 1;
                }
                local.push((i, c));
            }
        }
        Ok(local)
    };
    let results: Vec<Result<Counts, String>> = thread::scope(|sc| {
        let handles: Vec<_> = (0..threads).map(|_| sc.spawn(worker)).collect();
        handles.into_iter().map(|h| h.join().expect("agent thread panicked")).collect()
    });
    for r in results {
        for (i, c) in r? {
            out[i] = c;
        }
    }
    if stop.load(Ordering::Relaxed) {
        return Err("stopped".into());
    }
    Ok(out)
}

/// Per-row likelihoods of `player`'s observed actions under `spec`'s agent (rows of [`decisions`]): the frequency of
/// the observed action over `samples` runs, smoothed: `(count + alpha) / (samples + alpha * legal actions of the
/// hand)` (alpha = 0: raw frequencies). Rows of hands that cannot play the observed action get 0.
#[allow(clippy::too_many_arguments)]
pub fn agent_likelihoods(
    spec: Spec,
    start: &State,
    actions: &[u8],
    player: usize,
    k: usize,
    samples: u32,
    alpha: f64,
    seed: u64,
    threads: usize,
    stop: &Arc<AtomicBool>,
) -> Result<Vec<f64>, String> {
    let ds = decisions(start, actions, player, k)?;
    let mut rows: Vec<(usize, u16)> = Vec::new();
    for (d, dec) in ds.iter().enumerate() {
        for &h in &dec.hands {
            rows.push((d, h));
        }
    }
    let task = |i: usize| {
        let (d, h) = rows[i];
        let dec = &ds[d];
        (with_hand(&dec.public, player, h, k), actions[..dec.ply].to_vec())
    };
    let counts =
        agent_counts(spec, rows.len(), task, samples, seed, &[TAG_BELIEF, player as u64], threads, stop)?;
    Ok(rows
        .iter()
        .zip(&counts)
        .map(|(&(d, h), c)| {
            let a = actions[ds[d].ply];
            let s = with_hand(&ds[d].public, player, h, k);
            let legal = s.legal_actions();
            if !legal.contains(a) {
                return 0.0;
            }
            (c[a as usize] as f64 + alpha) / (samples as f64 + alpha * legal.len() as f64)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::encode_obs;
    use crate::hands::Variant;
    use crate::window::Window;

    fn random_game(rng: &mut Rng, plies: usize) -> Option<(State, State, Vec<u8>)> {
        let start = rng.below(2) as u8;
        let deal = State::deal(rng, start);
        let mut s = deal;
        let mut actions = Vec::new();
        for _ in 0..plies {
            if s.is_over() {
                return None;
            }
            let legal = s.legal_actions();
            let a = legal.nth_action(rng.below(legal.len()));
            actions.push(a);
            s.apply(a);
        }
        (!s.is_over()).then_some((deal, s, actions))
    }

    #[test]
    fn tree_has_the_window_layout() {
        let mut rng = Rng::new(3);
        let mut checked = 0;
        while checked < 12 {
            let ply = 11 + rng.below(4);
            let Some((deal, s, actions)) = random_game(&mut rng, ply) else { continue };
            for leaves in [Leaves::Frontier, Leaves::Public] {
                let t = WindowTree::new(&deal, &actions, 4, leaves).unwrap();
                let w = Window::build(&s, 4, leaves, 1).unwrap();
                assert_eq!(t.nodes.len(), w.nodes.len());
                assert_eq!(t.n_entries, w.n_entries);
                assert_eq!(*t.root(), w.root_state);
                for (n, (a, b)) in t.nodes.iter().zip(&w.nodes).enumerate() {
                    assert_eq!(
                        (a.state, a.parent, a.action, a.mover, a.entry, a.end),
                        (b.state, b.parent, b.action, b.mover, b.entry, b.end)
                    );
                    assert_eq!(a.legal.len(), b.n_edges as usize);
                    assert_eq!(t.hands(n), w.node_hands(n, a.mover as usize));
                    assert_eq!(t.path(n), w.history(n));
                    assert_eq!(t.find(&t.path(n)), Some(n));
                }
            }
            checked += 1;
        }
    }

    /// Rows encode the observation of the real position: real games continued inside the window.
    #[test]
    fn rows_are_the_real_observations() {
        let mut rng = Rng::new(5);
        let mut checked = 0;
        while checked < 300 {
            let ply = 10 + rng.below(4);
            let Some((deal, root, actions)) = random_game(&mut rng, ply) else { continue };
            let t = WindowTree::new(&deal, &actions, 4, Leaves::Frontier).unwrap();
            // A real continuation.
            let mut s = root;
            let mut h = history_of(&replay_public(&deal, &actions).unwrap(), &actions);
            let mut path = Vec::new();
            for _ in 0..rng.below(4) {
                let legal = s.legal_actions();
                let a = legal.nth_action(rng.below(legal.len()));
                h.push(&s, a);
                s.apply(a);
                path.push(a);
            }
            let Some(n) = t.find(&path) else { continue };
            let p = s.to_move as usize;
            let i = t.hands(n).iter().position(|&x| x == s.hands[p]).unwrap();
            let first = t.rows[p].iter().find(|&&(m, _)| m as usize == n).unwrap().1 as usize;
            let (mut obs, mut legal) = (vec![0f32; OBS_DIM], vec![0u8; NUM_ACTIONS]);
            t.encode(p, first + i, first + i + 1, &mut obs, &mut legal, 1);
            let mut want = vec![0f32; OBS_DIM];
            encode_obs(&s, &h, &mut want);
            assert_eq!(obs, want);
            let mut want_legal = vec![0u8; NUM_ACTIONS];
            encode_mask(&s, &mut want_legal);
            assert_eq!(legal, want_legal);
            checked += 1;
        }
    }

    #[test]
    fn encoding_in_chunks_and_threads_is_the_same() {
        let mut rng = Rng::new(9);
        let (deal, _, actions) = loop {
            if let Some(g) = random_game(&mut rng, 11) {
                break g;
            }
        };
        let t = WindowTree::new(&deal, &actions, 4, Leaves::Frontier).unwrap();
        for p in 0..2 {
            let n = t.n_rows(p);
            let (mut a, mut la) = (vec![0f32; n * OBS_DIM], vec![0u8; n * NUM_ACTIONS]);
            t.encode(p, 0, n, &mut a, &mut la, 1);
            let (mut b, mut lb) = (vec![0f32; n * OBS_DIM], vec![0u8; n * NUM_ACTIONS]);
            let mid = n / 3;
            t.encode(p, 0, mid, &mut b[..mid * OBS_DIM], &mut lb[..mid * NUM_ACTIONS], 3);
            t.encode(p, mid, n, &mut b[mid * OBS_DIM..], &mut lb[mid * NUM_ACTIONS..], 4);
            assert!(a == b && la == lb);
            for r in 0..n {
                let (node, i) = t.row(p, r);
                assert_eq!(t.nodes[node].mover as usize, p);
                assert!(i < t.nodes[node].n_hands as usize);
            }
        }
    }

    #[test]
    fn agent_tables_are_distributions_and_thread_independent() {
        let mut rng = Rng::new(4);
        let (deal, _, actions) = loop {
            if let Some(g) = random_game(&mut rng, 12) {
                break g;
            }
        };
        let t = WindowTree::new(&deal, &actions, 4, Leaves::Frontier).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let spec = Spec::parse("pimc:w=2,d=2").unwrap();
        for p in 0..2 {
            let a = t.agent_table(spec, p, 3, 7, 1, &stop).unwrap();
            let b = t.agent_table(spec, p, 3, 7, 4, &stop).unwrap();
            assert_eq!(a, b);
            for (n, node) in t.nodes.iter().enumerate() {
                let ne = node.legal.len();
                for (i, h) in t.hands(n).into_iter().enumerate() {
                    let row = &a[node.entry + i * ne..node.entry + (i + 1) * ne];
                    if node.mover as usize != p {
                        assert!(row.iter().all(|&x| x == 0.0));
                        continue;
                    }
                    assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
                    for (j, act) in node.legal.enumerate() {
                        if h >> decode(act).animal & 1 == 0 {
                            assert_eq!(row[j], 0.0);
                        }
                    }
                }
            }
        }
        assert!(t.agent_table(Spec::parse("cheat:d=2").unwrap(), 0, 1, 0, 1, &stop).is_err());
    }

    /// The filter against brute-force enumeration of every deal, on a mini variant with a hand-dependent policy.
    #[test]
    fn filter_matches_enumeration() {
        let v = Variant::new(0b10_1010_0110, 2); // parrot, kangaroo, seal, giraffe, crocodile
        let animals: Vec<u8> = (0..12).filter(|&a| v.animals >> a & 1 != 0).collect();
        fn perms(xs: &[u8]) -> Vec<Vec<u8>> {
            if xs.len() <= 1 {
                return vec![xs.to_vec()];
            }
            let mut out = Vec::new();
            for i in 0..xs.len() {
                let mut rest = xs.to_vec();
                let x = rest.remove(i);
                for mut p in perms(&rest) {
                    p.insert(0, x);
                    out.push(p);
                }
            }
            out
        }
        let orders = perms(&animals);
        let sigma = |s: &State, a: u8| -> f64 {
            let legal = s.legal_actions();
            if !legal.contains(a) {
                return 0.0;
            }
            let p = s.to_move as usize;
            let w = |b: u8| {
                ((crate::markov::public_key(s) ^ (s.hands[p] as u64) << 7 ^ (b as u64) << 23) % 5) as f64
            };
            let tot: f64 = legal.map(w).sum();
            if tot == 0.0 { 1.0 / legal.len() as f64 } else { w(a) / tot }
        };
        let deal_of = |start: u8, o0: &[u8], o1: &[u8]| {
            let hand = |o: &[u8]| o[..2].iter().fold(0u16, |h, &a| h | 1 << a);
            State::new(start, [hand(o0), hand(o1)], [&o0[2..], &o1[2..]])
        };
        let mut rng = Rng::new(31);
        for g in 0..6 {
            let start = (g % 2) as u8;
            let deal = deal_of(start, &orders[rng.below(orders.len())], &orders[rng.below(orders.len())]);
            // A history played with sigma.
            let mut s = deal;
            let mut actions = Vec::new();
            while !s.is_over() && actions.len() < 6 {
                let legal: Vec<u8> = s.legal_actions().collect();
                let ws: Vec<f64> = legal.iter().map(|&a| sigma(&s, a)).collect();
                let mut x = rng.next_u64() as f64 / u64::MAX as f64 * ws.iter().sum::<f64>();
                let mut a = *legal.last().unwrap();
                for (&b, &w) in legal.iter().zip(&ws) {
                    if x < w {
                        a = b;
                        break;
                    }
                    x -= w;
                }
                actions.push(a);
                s.apply(a);
            }
            for player in 0..2 {
                // Brute force: posterior over the player's final hand.
                let mut post = vec![0.0; num_hands(2)];
                for o0 in &orders {
                    for o1 in &orders {
                        let mut t = deal_of(start, o0, o1);
                        let mut w = 1.0;
                        for &a in &actions {
                            w *= sigma(&t, a);
                            if w == 0.0 {
                                break;
                            }
                            t.apply(a);
                        }
                        if w > 0.0 {
                            post[rank(t.hands[player])] += w;
                        }
                    }
                }
                let tot: f64 = post.iter().sum();
                let ds = decisions(&deal, &actions, player, 2).unwrap();
                let lik: Vec<f64> = ds
                    .iter()
                    .flat_map(|d| {
                        let a = actions[d.ply];
                        d.hands
                            .iter()
                            .map(move |&h| sigma(&with_hand(&d.public, player, h, 2), a))
                            .collect::<Vec<_>>()
                    })
                    .collect();
                let (range, steps) = bayes_filter(&deal, &actions, player, 2, &lik, 0.0).unwrap();
                assert!(steps.iter().all(|s| !s.impossible));
                let m = s.hands[player].count_ones() as usize;
                assert_eq!(range.len(), num_hands(m));
                for (r, &x) in range.iter().enumerate() {
                    assert!((x - post[r] / tot).abs() < 1e-12, "game {g} player {player} slot {r}");
                }
            }
        }
    }

    #[test]
    fn zero_probability_observations() {
        let mut rng = Rng::new(8);
        let (deal, _, actions) = loop {
            if let Some(g) = random_game(&mut rng, 9) {
                break g;
            }
        };
        let p = deal.start as usize;
        let ds = decisions(&deal, &actions, p, 4).unwrap();
        let n: usize = ds.iter().map(|d| d.hands.len()).sum();
        // A model that gives the observed actions zero probability everywhere.
        let zeros = vec![0.0; n];
        let (r, steps) = bayes_filter(&deal, &actions, p, 4, &zeros, 1e-9).unwrap();
        assert!(steps.iter().all(|s| s.impossible && s.prob == 0.0));
        // Flooring makes the zero model uninformative: only the exact eliminations remain, as with a constant model.
        let ones = vec![1.0; n];
        let (u, steps1) = bayes_filter(&deal, &actions, p, 4, &ones, 1e-9).unwrap();
        assert!(steps1.iter().all(|s| !s.impossible && s.prob > 0.0));
        for (a, b) in r.iter().zip(&u) {
            assert!((a - b).abs() < 1e-12);
        }
        let (z, _) = bayes_filter(&deal, &actions, p, 4, &zeros, 0.0).unwrap();
        assert_eq!(z, u);
        // The true hand keeps positive mass.
        let mut s = deal;
        for &a in &actions {
            s.apply(a);
        }
        assert!(r[rank(s.hands[p])] > 0.0);
        assert!(bayes_filter(&deal, &actions, p, 4, &ones[1..], 1e-9).is_err());
    }
}
