//! Exact hidden-information window solver.
//!
//! A *window* is the public subtree below a root public state (any ply) down to leaves, with given root ranges
//! (distributions over each player's possible hands). Nothing is abstracted:
//! - **Nodes** are public states. The public actions at a node are every action legal for at least one of the
//!   mover's possible hands: the legal actions of the [`public`] position (every unplayed animal in hand). An action
//!   is legal for hand `H` iff its animal is in `H` (and it is legal on the queue, which every public action is).
//! - **Hands.** A player's possible hands at a node are `hands(U_p, k)` (the `k`-subsets of the unplayed animals
//!   while the deck is non-empty, else `U_p`). Vectors over hands use the *local* layout: the possible hands in
//!   increasing global rank ([`local_rank`]).
//! - **Draws.** Playing animal `c` from hand `H` with a non-empty deck leads to `H - c + d`, `d` uniform over the deck
//!   `U - H` ([`Trans`], with local indices). The two players' hands are independent and only the mover's changes.
//! - **Leaves** ([`Leaves`]): terminal positions (payoff from the public scores), and either frontier nodes (the
//!   opponent's hand is public and the mover's deck holds at most one card: ply 15 of the real game; valued per
//!   mover hand by the perfect-information solver), or public nodes (both decks empty; one exact value), or
//!   nothing else (the whole game to the end, for small variants).
//!
//! - **Value leaves** (optional, [`Window::build_cut`]): every other node at a given depth below the root (ply `L`
//!   of the game) ends the window too. Their values come from outside, every half-iteration (see *Depth-limited
//!   solving* below).
//!
//! Information sets are merged by (public node, current hand), which is lossless: draws depend only on the current
//! hand, payoffs only on public information, and the hands are independent (see `tests/hidden_info.rs` and the
//! explicit-infoset cross-check `bbar/xcfr.py`). Values are win probabilities (win 1, draw .5, loss 0); `u` below is
//! player 0's.
//!
//! # Vector CFR
//! Strategies are behaviour strategies per (node, mover hand) over the node's public actions, zero on actions
//! illegal for the hand. Counterfactual values (cfv) of player `i` at a node, per hand of `i`, exclude `i`'s own
//! reach and include the opponent's reach: at a terminal leaf `cfv_i(h) = u_i * sum(r_o)`; at a frontier leaf whose
//! mover is `i`, `cfv_i(h) = u_i(h) * sum(r_o)`, else `cfv_i = sum_h' r_o(h') u_i(h')`. The mover's reach goes down
//! through the play and the draw, `r'(h') = sum_h r(h) sigma(a | h) P(h -> h')`, and action values come back up
//! through the transpose, `q(h, a) = sum_h' P(h -> h') cfv(child, h')`. Updates alternate between the players.
//! Algorithms ([`Algo`]): DCFR (alpha 1.5, beta 0, gamma 2), PCFR+ (predictive regret matching+, quadratic
//! averaging) and CFR+ (linear averaging).
//!
//! # Exact evaluation
//! [`Window::exploitability`] computes, for any profile (the solver's average or an external table), both exact
//! best responses (one pass per player): `NashConv = BR_0 + BR_1 - 1` (each in the best responder's win
//! probability), exploitability `= NashConv / 2`.
//!
//! # Re-solving gadget
//! [`Cfr::new_gadget`] (Burch et al. CFR-D, DeepStack): our range is exact; for each opponent hand the opponent
//! may terminate at a given counterfactual value or follow into the window. Solving the gadget gives a strategy whose
//! opponent best-response cfvs do not exceed the given ones (up to the gadget's exploitability). At a leaf root
//! (no decisions of ours) the gadget is solved exactly without iterations: each opponent hand takes the better of
//! its fixed follow value and its terminate value.
//!
//! # Depth-limited solving (external leaf values)
//! With value leaves, the window is the trunk of a depth-limited solve (CFR-D, Burch et al. 2014; DeepStack;
//! ReBeL). A value leaf's payoff is given per hand by a *value function* of the public belief state there: for each
//! player `p` and hand `h`, `v_p(h)` = `p`'s win probability holding `h` (the owner's value, not player 0's), against
//! the opponent's belief. The leaf cfv of player `i` is `v_i(h) * m_o`, `m_o` the opponent's reach mass at the leaf
//! (`cfv_i(h) = sum_h' r_o(h') u_i(h, h') = m_o sum_h' b_o(h') u_i(h, h')`).
//!
//! [`Cfr::ext_begin`] / [`Cfr::ext_end`] run one *half-iteration* (the updates alternate between the players, as in
//! [`Cfr::iterate`]): `ext_begin` computes the current strategies (regret matching), both players' reaches at every
//! value leaf and returns the beliefs (normalised reaches; uniform over the feasible hands when a player's mass is 0)
//! and masses; the caller evaluates the value function at those beliefs; `ext_end` takes the values of both players
//! and updates the regrets and the average strategy of the player whose half-iteration it is.
//!
//! **Which policy defines the leaf beliefs: the current iterate.** This is CFR-D (Burch et al. 2014) and DeepStack's
//! depth-limited lookahead: on every iteration the leaf PBS is the one the iterate induces, and the leaf values are
//! the (approximate) subgame-equilibrium values for it. With an exact oracle, each iterate is then a full-game
//! strategy (trunk iterate + subgame play) and the trunk regrets are regrets in the full game, so the average trunk
//! strategy converges to the trunk of an equilibrium; that is the guarantee behind re-solving. The leaf values react
//! to the trunk strategy through the beliefs, so this converges more slowly than CFR on a fixed tree (about
//! `1/sqrt(T)` on hard positions). ReBeL's CFR-AVG takes the leaf PBS of the *average* policy instead; tried on a slow
//! position (golden game 1, ply 11, leaves at 12, oracle values, 100 iterations) it gave a better trunk but a worse
//! re-solved composite (3.3e-4 against 2.2e-4), so the iterate it is.
//!
//! **What a re-solve below a leaf needs.** In `p`'s half-iteration the values the caller returns for the *other*
//! player `o` are `o`'s values against `p`'s implicit subgame strategy for that iterate. [`Cfr::avg_leaf_values`]
//! averages them with the weights of `p`'s average strategy (`w_t * m_p^t`): `o`'s values against `p`'s average
//! (trunk plus implicit subgame) strategy, in units where `p`'s reach at the leaf sums to 1 — upper bounds of
//! `o`'s best-response values there (a best response to a mixture gains at most the mixture of best responses), so
//! the re-solving gadget at the leaf with these opponent cfvs and `p`'s average belief is feasible and safe.
//!
//! # Layout and determinism
//! Internal nodes are stored in preorder, so a subtree is a contiguous range of nodes and of strategy entries
//! (`nh * ne` per node, row-major by hand). With `threads > 1`, large subtrees are walked in parallel (scoped
//! threads on disjoint slices) and children's results are combined in edge order, so results are bit-identical
//! whatever the number of threads.

use std::collections::HashMap;
use std::thread;
use std::time::Instant;

use crate::action::decode;
use crate::hands::{BINOM, hand_size, hands, play_draw, unplayed};
use crate::markov::{key, public};
use crate::pisolve::{PiSolver, outcome};
use crate::state::State;

/// Which public nodes end the window (besides terminal positions).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Leaves {
    /// The opponent's hand is public and the mover's deck holds at most one card (ply 15 of the real game): one exact
    /// value per mover hand (the best action for that hand, then the known last draw).
    Frontier,
    /// Both decks empty (ply 16 of the real game): every hand is public, one exact value.
    Public,
    /// No cut: the window runs to the end of the game (small variants).
    Terminal,
}

/// CFR variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algo {
    /// Discounted CFR, alpha 1.5, beta 0, gamma 2.
    Dcfr,
    /// Predictive CFR+: regret matching+ on accumulated regrets plus the last instantaneous regret; quadratic
    /// averaging.
    PcfrPlus,
    /// CFR+: regret matching+, linear averaging.
    CfrPlus,
}

impl Algo {
    pub fn parse(name: &str) -> Option<Algo> {
        match name.to_ascii_lowercase().as_str() {
            "dcfr" => Some(Algo::Dcfr),
            "pcfr+" | "pcfrplus" | "pcfr" => Some(Algo::PcfrPlus),
            "cfr+" | "cfrplus" => Some(Algo::CfrPlus),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Algo::Dcfr => "dcfr",
            Algo::PcfrPlus => "pcfr+",
            Algo::CfrPlus => "cfr+",
        }
    }
}

/// Index of hand `h ⊆ u` among the `|h|`-subsets of `u` in increasing global rank: the colex rank of `h`
/// compressed to the positions of `u` (compression preserves colex order).
pub fn local_rank(h: u16, u: u16) -> usize {
    debug_assert!(h & !u == 0, "hand {h:#x} outside {u:#x}");
    let (mut r, mut chosen, mut pos) = (0usize, 0usize, 0usize);
    let mut bits = u;
    while bits != 0 {
        let b = bits.trailing_zeros();
        bits &= bits - 1;
        if h >> b & 1 != 0 {
            chosen += 1;
            r += BINOM[pos][chosen] as usize;
        }
        pos += 1;
    }
    r
}

/// The possible hands of a player with a given unplayed set, in increasing rank (local layout).
#[derive(Clone, Debug)]
pub struct HandSet {
    pub unplayed: u16,
    pub masks: Vec<u16>,
}

/// Play-and-draw operator for one player's unplayed set `U` and played animal `c`, in local indices: row `r` is the
/// hand `from[r]` (which holds `c`), its successors `to[r * stride..(r + 1) * stride]`, each with probability `prob`.
#[derive(Clone, Debug)]
pub struct Trans {
    pub animal: u8,
    pub from: Vec<u16>,
    pub to: Vec<u16>,
    pub stride: usize,
    pub prob: f64,
    pub from_n: usize,
    pub to_n: usize,
}

impl Trans {
    fn new(u: u16, k: usize, c: u8, from_masks: &[u16]) -> Trans {
        let u2 = u & !(1 << c);
        let deck = u.count_ones() as usize - hand_size(u, k);
        let to_masks: Vec<u16> = hands(u2, k).collect();
        let mut t = Trans {
            animal: c,
            from: Vec::new(),
            to: Vec::new(),
            stride: deck.max(1),
            prob: if deck > 0 { 1.0 / deck as f64 } else { 1.0 },
            from_n: from_masks.len(),
            to_n: to_masks.len(),
        };
        for (i, &h) in from_masks.iter().enumerate() {
            if h >> c & 1 == 0 {
                continue;
            }
            t.from.push(i as u16);
            for next in play_draw(h, u, c) {
                let j = local_rank(next, u2);
                debug_assert_eq!(to_masks[j], next);
                t.to.push(j as u16);
            }
        }
        debug_assert_eq!(t.to.len(), t.from.len() * t.stride);
        t
    }

    /// `out[h'] += prob * weight(h)` over every row `h -> h'` (reach through the play and the draw).
    #[inline]
    fn push(&self, weight: impl Fn(usize) -> f64, out: &mut [f64]) {
        for (r, &h) in self.from.iter().enumerate() {
            let x = weight(h as usize) * self.prob;
            if x != 0.0 {
                for &j in &self.to[r * self.stride..(r + 1) * self.stride] {
                    out[j as usize] += x;
                }
            }
        }
    }

    /// `f(h, prob * sum_h' value(h'))` for every row (expectation over the draw).
    #[inline]
    fn pull(&self, value: &[f64], mut f: impl FnMut(usize, f64)) {
        for (r, &h) in self.from.iter().enumerate() {
            let s: f64 =
                self.to[r * self.stride..(r + 1) * self.stride].iter().map(|&j| value[j as usize]).sum();
            f(h as usize, self.prob * s);
        }
    }
}

/// Where an edge leads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Child {
    /// Internal node index.
    Node(u32),
    /// Terminal position: player 0's payoff.
    Terminal(f64),
    /// Frontier (or public) leaf: index into [`Window::frontier`].
    Frontier(u32),
    /// Value leaf (depth-limited window): index into [`Window::value_leaves`]; values from outside.
    Value(u32),
}

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    pub action: u8,
    pub animal: u8,
    pub trans: u32,
    pub child: Child,
}

/// A distinct frontier position: player 0's value per hand of its mover, `values[off..off + n]` (local layout).
#[derive(Clone, Copy, Debug)]
pub struct FrontierLeaf {
    pub mover: u8,
    pub off: u32,
    pub n: u32,
}

/// A value leaf: a non-terminal, non-frontier node at the cut depth. External values of player `p` (per hand of
/// `p` at the leaf, local layout) live at `off[p]..off[p] + n[p]` of a leaf-values buffer of length
/// [`Window::n_leaf_values`].
#[derive(Clone, Copy, Debug)]
pub struct ValueLeaf {
    /// The public position.
    pub state: State,
    /// Parent node and the action that led here.
    pub parent: u32,
    pub action: u8,
    pub mover: u8,
    pub hs: [u32; 2],
    pub off: [u32; 2],
    pub n: [u32; 2],
}

#[derive(Clone, Copy, Debug)]
pub struct Node {
    /// The public position (every unplayed animal in hand, decks empty).
    pub state: State,
    /// Parent node (`u32::MAX` for the root) and the action that led here.
    pub parent: u32,
    pub action: u8,
    pub mover: u8,
    pub depth: u8,
    pub n_edges: u8,
    /// Hand sets of players 0 and 1.
    pub hs: [u32; 2],
    pub edge_start: u32,
    /// Preorder: the subtree is `nodes[id..end]`.
    pub end: u32,
    /// First strategy entry: `nh(mover) * n_edges` entries, row-major by hand.
    pub entry: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WindowStats {
    pub internal: usize,
    pub terminal_leaves: usize,
    /// Frontier (or public) leaves of the tree, and distinct frontier positions (solved once each).
    pub frontier_leaves: usize,
    pub frontier_positions: usize,
    /// Value leaves (depth-limited windows; 0 otherwise).
    pub value_leaves: usize,
    pub max_depth: usize,
    /// Perfect-information search nodes spent on frontier values.
    pub pi_nodes: u64,
    pub tree_seconds: f64,
    pub frontier_seconds: f64,
}

/// Exact evaluation of a profile on the window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Expl {
    /// Best-response value of each player against the other's strategy (in that player's win probability).
    pub br: [f64; 2],
    /// Player 0's expected value under the profile (NaN when not computed).
    pub value: f64,
    pub nashconv: f64,
    pub exploitability: f64,
}

/// The window tree with its leaf values. Immutable once built; solver state lives in [`Cfr`].
pub struct Window {
    pub k: usize,
    pub leaves: Leaves,
    pub root_state: State,
    pub root: Child,
    pub root_hs: [u32; 2],
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub handsets: Vec<HandSet>,
    pub trans: Vec<Trans>,
    pub frontier: Vec<FrontierLeaf>,
    pub frontier_states: Vec<State>,
    pub values: Vec<f64>,
    /// Depth (moves below the root) at which value leaves cut the tree, if any.
    pub cut: Option<usize>,
    pub value_leaves: Vec<ValueLeaf>,
    /// Length of a leaf-values (or leaf-reaches) buffer: both players' hands at every value leaf.
    pub n_leaf_values: usize,
    pub n_entries: usize,
    pub stats: WindowStats,
    scratch_len: usize,
}

struct Builder {
    k: usize,
    leaves: Leaves,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    handsets: Vec<HandSet>,
    hs_index: HashMap<u16, u32>,
    trans: Vec<Trans>,
    tr_index: HashMap<(u16, u8), u32>,
    frontier: Vec<FrontierLeaf>,
    fr_index: HashMap<u64, u32>,
    frontier_states: Vec<State>,
    n_values: usize,
    cut: Option<usize>,
    value_leaves: Vec<ValueLeaf>,
    n_leaf_values: usize,
    n_entries: usize,
    terminal_leaves: usize,
    frontier_leaves: usize,
    max_depth: usize,
    max_need: usize,
}

impl Builder {
    fn handset(&mut self, u: u16) -> u32 {
        if let Some(&i) = self.hs_index.get(&u) {
            return i;
        }
        let i = self.handsets.len() as u32;
        self.handsets.push(HandSet { unplayed: u, masks: hands(u, self.k).collect() });
        self.hs_index.insert(u, i);
        i
    }

    fn trans_id(&mut self, u: u16, c: u8) -> u32 {
        if let Some(&i) = self.tr_index.get(&(u, c)) {
            return i;
        }
        let hs = self.handset(u);
        let t = Trans::new(u, self.k, c, &self.handsets[hs as usize].masks);
        let i = self.trans.len() as u32;
        self.trans.push(t);
        self.tr_index.insert((u, c), i);
        i
    }

    fn is_leaf(&self, s: &State) -> bool {
        let p = s.to_move as usize;
        let n = |q: usize| unplayed(s, q).count_ones() as usize;
        match self.leaves {
            Leaves::Frontier => n(1 - p) <= self.k && n(p) <= self.k + 1,
            Leaves::Public => n(0) <= self.k && n(1) <= self.k,
            Leaves::Terminal => false,
        }
    }

    fn frontier_id(&mut self, s: &State) -> u32 {
        let key = key(s);
        if let Some(&i) = self.fr_index.get(&key) {
            return i;
        }
        let p = s.to_move as usize;
        let n = hands(unplayed(s, p), self.k).count();
        let i = self.frontier.len() as u32;
        self.frontier.push(FrontierLeaf { mover: p as u8, off: self.n_values as u32, n: n as u32 });
        self.n_values += n;
        self.frontier_states.push(*s);
        self.fr_index.insert(key, i);
        i
    }

    fn node(&mut self, s: State, parent: u32, action: u8, depth: usize) -> Result<Child, String> {
        if s.is_over() {
            self.terminal_leaves += 1;
            return Ok(Child::Terminal(outcome(&s) as f64 / 2.0));
        }
        if self.is_leaf(&s) {
            self.frontier_leaves += 1;
            return Ok(Child::Frontier(self.frontier_id(&s)));
        }
        if self.cut == Some(depth) {
            let u = [unplayed(&s, 0), unplayed(&s, 1)];
            let hs = [self.handset(u[0]), self.handset(u[1])];
            let n = hs.map(|h| self.handsets[h as usize].masks.len() as u32);
            let off = [self.n_leaf_values as u32, (self.n_leaf_values + n[0] as usize) as u32];
            self.n_leaf_values += (n[0] + n[1]) as usize;
            let i = self.value_leaves.len() as u32;
            self.value_leaves.push(ValueLeaf { state: s, parent, action, mover: s.to_move, hs, off, n });
            return Ok(Child::Value(i));
        }
        self.max_depth = self.max_depth.max(depth);
        let id = self.nodes.len() as u32;
        let p = s.to_move as usize;
        let u = [unplayed(&s, 0), unplayed(&s, 1)];
        let hs = [self.handset(u[0]), self.handset(u[1])];
        let legal = s.legal_actions();
        let ne = legal.len();
        let edge_start = self.edges.len() as u32;
        let mut animals = 0u16;
        for a in legal {
            let animal = decode(a).animal;
            animals |= 1 << animal;
            let trans = self.trans_id(u[p], animal);
            self.edges.push(Edge { action: a, animal, trans, child: Child::Terminal(0.0) });
        }
        let masks = &self.handsets[hs[p] as usize].masks;
        if let Some(h) = masks.iter().find(|&&h| h & animals == 0) {
            return Err(format!("hand {h:#x} of player {p} has no legal action at {s:?}"));
        }
        let nh = masks.len();
        let nmax = self.handsets[hs[0] as usize].masks.len().max(self.handsets[hs[1] as usize].masks.len());
        self.max_need = self.max_need.max(2 * nh * ne + 3 * nmax);
        self.nodes.push(Node {
            state: s,
            parent,
            action,
            mover: p as u8,
            depth: depth.min(255) as u8,
            n_edges: ne as u8,
            hs,
            edge_start,
            end: 0,
            entry: self.n_entries,
        });
        self.n_entries += nh * ne;
        for (j, a) in legal.enumerate() {
            let mut c = s;
            c.apply(a);
            let child = self.node(c, id, a, depth + 1)?;
            self.edges[edge_start as usize + j].child = child;
        }
        self.nodes[id as usize].end = self.nodes.len() as u32;
        Ok(Child::Node(id))
    }
}

/// Mutable solver arrays restricted to a subtree: entries `[base, base + reg.len())`.
struct Slices<'a> {
    base: usize,
    reg: &'a mut [f64],
    avg: &'a mut [f64],
    pred: &'a mut [f64],
}

impl<'a> Slices<'a> {
    /// The sub-range `[lo, hi)` of entries (absolute indices).
    fn sub(&mut self, lo: usize, hi: usize) -> Slices<'_> {
        let (a, b) = (lo - self.base, hi - self.base);
        Slices { base: lo, reg: &mut self.reg[a..b], avg: &mut self.avg[a..b], pred: &mut self.pred[a..b] }
    }

    /// Splits off the first `len` entries.
    fn split(self, len: usize) -> (Slices<'a>, Slices<'a>) {
        let (r0, r1) = self.reg.split_at_mut(len);
        let (a0, a1) = self.avg.split_at_mut(len);
        let (p0, p1) = self.pred.split_at_mut(len);
        (
            Slices { base: self.base, reg: r0, avg: a0, pred: p0 },
            Slices { base: self.base + len, reg: r1, avg: a1, pred: p1 },
        )
    }
}

fn normalize(r: &[f64]) -> Vec<f64> {
    let s: f64 = r.iter().sum();
    assert!(s > 0.0 && s.is_finite(), "a range must have positive finite mass");
    r.iter().map(|x| x / s).collect()
}

/// Adds instantaneous regret `r` to entry `x` under `algo` (DCFR discounts `dpos` / `dneg`; PCFR+ keeps `r` as
/// the next prediction).
#[inline]
fn update_regret(algo: Algo, reg: &mut [f64], pred: &mut [f64], x: usize, r: f64, dpos: f64, dneg: f64) {
    match algo {
        Algo::Dcfr => {
            let y = reg[x] + r;
            reg[x] = if y > 0.0 { y * dpos } else { y * dneg };
        }
        Algo::CfrPlus => reg[x] = (reg[x] + r).max(0.0),
        Algo::PcfrPlus => {
            reg[x] = (reg[x] + r).max(0.0);
            pred[x] = r;
        }
    }
}

/// Regret matching over the legal actions of each hand: `x+ / sum(x+)`, uniform when no entry is positive.
#[inline]
fn regret_match(x: &[f64], extra: Option<&[f64]>, masks: &[u16], animals: &[u8], sig: &mut [f64]) {
    let ne = animals.len();
    for (h, &hm) in masks.iter().enumerate() {
        let row = h * ne;
        let (mut s, mut nl) = (0.0, 0usize);
        for e in 0..ne {
            if hm >> animals[e] & 1 != 0 {
                let v = x[row + e] + extra.map_or(0.0, |p| p[row + e]);
                if v > 0.0 {
                    s += v;
                }
                nl += 1;
            }
        }
        for e in 0..ne {
            sig[row + e] = if hm >> animals[e] & 1 == 0 {
                0.0
            } else if s > 0.0 {
                let v = x[row + e] + extra.map_or(0.0, |p| p[row + e]);
                if v > 0.0 { v / s } else { 0.0 }
            } else {
                1.0 / nl as f64
            };
        }
    }
}

impl Window {
    /// Builds the window below `root` (a real or public position) for hand size `k`. `threads` parallelises the
    /// frontier values (exact integers, so the result does not depend on it).
    pub fn build(root: &State, k: usize, leaves: Leaves, threads: usize) -> Result<Window, String> {
        Window::build_cut(root, k, leaves, None, threads)
    }

    /// [`Window::build`] with value leaves: every node `cut` moves below the root that is neither over nor a
    /// `leaves` leaf ends the window as a value leaf (depth-limited solving: values from outside, see the module
    /// documentation). `cut` must be at least 1.
    pub fn build_cut(
        root: &State,
        k: usize,
        leaves: Leaves,
        cut: Option<usize>,
        threads: usize,
    ) -> Result<Window, String> {
        if k == 0 {
            return Err("hand size k must be at least 1".into());
        }
        if cut == Some(0) {
            return Err("the cut depth must be at least 1 (the root cannot be a value leaf)".into());
        }
        for p in 0..2 {
            if root.deck_len[p] > 0 && root.hands[p].count_ones() as usize != k {
                return Err(format!(
                    "player {p} holds {} animals with a non-empty deck, not k = {k}",
                    root.hands[p].count_ones()
                ));
            }
        }
        let t0 = Instant::now();
        let s = public(root);
        let mut b = Builder {
            k,
            leaves,
            nodes: Vec::new(),
            edges: Vec::new(),
            handsets: Vec::new(),
            hs_index: HashMap::new(),
            trans: Vec::new(),
            tr_index: HashMap::new(),
            frontier: Vec::new(),
            fr_index: HashMap::new(),
            frontier_states: Vec::new(),
            n_values: 0,
            cut,
            value_leaves: Vec::new(),
            n_leaf_values: 0,
            n_entries: 0,
            terminal_leaves: 0,
            frontier_leaves: 0,
            max_depth: 0,
            max_need: 0,
        };
        let root_hs = [b.handset(unplayed(&s, 0)), b.handset(unplayed(&s, 1))];
        let nmax =
            b.handsets[root_hs[0] as usize].masks.len().max(b.handsets[root_hs[1] as usize].masks.len());
        let root_child = b.node(s, u32::MAX, 0, 0)?;
        let tree_seconds = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let (values, pi_nodes) = frontier_values(&b.frontier_states, &b.frontier, b.n_values, k, threads);
        let stats = WindowStats {
            internal: b.nodes.len(),
            terminal_leaves: b.terminal_leaves,
            frontier_leaves: b.frontier_leaves,
            frontier_positions: b.frontier.len(),
            value_leaves: b.value_leaves.len(),
            max_depth: b.max_depth,
            pi_nodes,
            tree_seconds,
            frontier_seconds: t1.elapsed().as_secs_f64(),
        };
        Ok(Window {
            k,
            leaves,
            root_state: s,
            root: root_child,
            root_hs,
            nodes: b.nodes,
            edges: b.edges,
            handsets: b.handsets,
            trans: b.trans,
            frontier: b.frontier,
            frontier_states: b.frontier_states,
            values,
            cut,
            value_leaves: b.value_leaves,
            n_leaf_values: b.n_leaf_values,
            n_entries: b.n_entries,
            stats,
            scratch_len: (b.max_depth + 2) * (b.max_need + 3 * nmax) + 16,
        })
    }

    /// Possible hands of `player` at the root (local layout).
    pub fn root_hands(&self, player: usize) -> &[u16] {
        &self.handsets[self.root_hs[player] as usize].masks
    }

    /// Possible hands of `player` at internal node `n`.
    pub fn node_hands(&self, n: usize, player: usize) -> &[u16] {
        &self.handsets[self.nodes[n].hs[player] as usize].masks
    }

    pub fn node_edges(&self, n: usize) -> &[Edge] {
        let node = &self.nodes[n];
        &self.edges[node.edge_start as usize..][..node.n_edges as usize]
    }

    /// Public actions from the root to internal node `n`.
    pub fn history(&self, mut n: usize) -> Vec<u8> {
        let mut out = Vec::new();
        while self.nodes[n].parent != u32::MAX {
            out.push(self.nodes[n].action);
            n = self.nodes[n].parent as usize;
        }
        out.reverse();
        out
    }

    /// The internal node reached by `history` from the root, if any.
    pub fn find(&self, history: &[u8]) -> Option<usize> {
        let mut child = self.root;
        for &a in history {
            let Child::Node(n) = child else { return None };
            child = self.node_edges(n as usize).iter().find(|e| e.action == a)?.child;
        }
        match child {
            Child::Node(n) => Some(n as usize),
            _ => None,
        }
    }

    /// End (exclusive) of the strategy entries of the subtree of internal node `n`.
    fn entry_end(&self, n: usize) -> usize {
        let end = self.nodes[n].end as usize;
        if end < self.nodes.len() { self.nodes[end].entry } else { self.n_entries }
    }

    fn animals(&self, n: usize) -> ([u8; 64], usize) {
        let mut a = [0u8; 64];
        let edges = self.node_edges(n);
        for (i, e) in edges.iter().enumerate() {
            a[i] = e.animal;
        }
        (a, edges.len())
    }

    /// The uniform profile: each hand plays its legal actions uniformly.
    pub fn uniform_profile(&self) -> Vec<f64> {
        let zeros = vec![0.0; self.n_entries];
        self.normalized(&zeros)
    }

    /// `table` normalised per (node, hand) over the legal actions; rows without positive mass become uniform.
    /// Entries for illegal actions are ignored. Fails on negative or non-finite entries.
    pub fn sanitize(&self, table: &[f64]) -> Result<Vec<f64>, String> {
        if table.len() != self.n_entries {
            return Err(format!("strategy table has {} entries, the window {}", table.len(), self.n_entries));
        }
        if let Some(i) = table.iter().position(|x| !x.is_finite() || *x < 0.0) {
            return Err(format!("strategy entry {i} is {} (must be finite and non-negative)", table[i]));
        }
        Ok(self.normalized(table))
    }

    fn normalized(&self, table: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.n_entries];
        for n in 0..self.nodes.len() {
            let node = &self.nodes[n];
            let (animals, ne) = self.animals(n);
            let masks = self.node_hands(n, node.mover as usize);
            let e0 = node.entry;
            regret_match(
                &table[e0..e0 + masks.len() * ne],
                None,
                masks,
                &animals[..ne],
                &mut out[e0..e0 + masks.len() * ne],
            );
        }
        out
    }

    /// Counterfactual values at a leaf for player `i`, given the opponent's reach `ro` at the leaf.
    /// `out` is over `i`'s hands at the leaf. `ext`: the leaf-values buffer for value leaves (owner's values; may be
    /// empty when the window has none).
    #[inline]
    fn leaf_cfv(&self, i: usize, leaf: Child, ro: &[f64], ext: &[f64], out: &mut [f64]) {
        let ui = |u: f64| if i == 0 { u } else { 1.0 - u };
        match leaf {
            Child::Value(j) => {
                assert!(ext.len() == self.n_leaf_values, "value leaves need external values");
                let leaf = &self.value_leaves[j as usize];
                let vals = &ext[leaf.off[i] as usize..][..leaf.n[i] as usize];
                let tot: f64 = ro.iter().sum();
                for (o, &v) in out.iter_mut().zip(vals) {
                    *o = v * tot;
                }
            }
            Child::Terminal(u) => {
                let tot: f64 = ro.iter().sum();
                out.fill(ui(u) * tot);
            }
            Child::Frontier(f) => {
                let leaf = self.frontier[f as usize];
                let vals = &self.values[leaf.off as usize..][..leaf.n as usize];
                if leaf.mover as usize == i {
                    let tot: f64 = ro.iter().sum();
                    for (o, &v) in out.iter_mut().zip(vals) {
                        *o = ui(v) * tot;
                    }
                } else {
                    let x: f64 = ro.iter().zip(vals).map(|(r, &v)| r * ui(v)).sum();
                    out.fill(x);
                }
            }
            Child::Node(_) => unreachable!("not a leaf"),
        }
    }

    fn par_min(&self, threads: usize) -> u32 {
        if threads <= 1 { u32::MAX } else { ((self.nodes.len() / (8 * threads)).max(256)) as u32 }
    }

    /// Root counterfactual values of player `i` (local layout) against the opponent's root reach `ro`:
    /// best-response values and, if `follow`, the values of following `profile`.
    pub fn root_cfvs(
        &self,
        i: usize,
        profile: &[f64],
        ro: &[f64],
        follow: bool,
        threads: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        self.root_cfvs_ext(i, profile, ro, follow, &[], threads)
    }

    /// [`Window::root_cfvs`] with fixed value-leaf values `ext` (a leaf-values buffer; empty without value leaves).
    pub fn root_cfvs_ext(
        &self,
        i: usize,
        profile: &[f64],
        ro: &[f64],
        follow: bool,
        ext: &[f64],
        threads: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let nh = self.root_hands(i).len();
        let (mut best, mut fol) = (vec![0.0; nh], vec![0.0; nh]);
        match self.root {
            Child::Node(r) => {
                let ctx = EvalCtx { w: self, prof: profile, i, follow, ext, par_min: self.par_min(threads) };
                let mut scratch = vec![0.0; self.scratch_len];
                ctx.walk(r as usize, ro, &mut best, &mut fol, &mut scratch);
            }
            leaf => {
                self.leaf_cfv(i, leaf, ro, ext, &mut best);
                fol.copy_from_slice(&best);
            }
        }
        (best, fol)
    }

    /// A pure best response of player `i` to `profile` (opponent root reach `ro`): `profile` with `i`'s entries
    /// replaced by the first best action of each (node, hand). Its value is the best-response value.
    pub fn best_response(&self, i: usize, profile: &[f64], ro: &[f64]) -> Vec<f64> {
        self.best_response_ext(i, profile, ro, &[])
    }

    /// [`Window::best_response`] with fixed value-leaf values `ext`.
    pub fn best_response_ext(&self, i: usize, profile: &[f64], ro: &[f64], ext: &[f64]) -> Vec<f64> {
        let mut out = profile.to_vec();
        if let Child::Node(r) = self.root {
            let mut cfv = vec![0.0; self.root_hands(i).len()];
            self.br_record(i, profile, r as usize, ro, ext, &mut cfv, &mut out);
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn br_record(
        &self,
        i: usize,
        prof: &[f64],
        n: usize,
        ro: &[f64],
        ext: &[f64],
        out: &mut [f64],
        table: &mut [f64],
    ) {
        let node = &self.nodes[n];
        let ne = node.n_edges as usize;
        let edges = self.node_edges(n);
        let masks = self.node_hands(n, node.mover as usize);
        let nh = masks.len();
        if node.mover as usize == i {
            let mut q = vec![0.0; nh * ne];
            for (e, edge) in edges.iter().enumerate() {
                let tr = &self.trans[edge.trans as usize];
                let mut c = vec![0.0; tr.to_n];
                match edge.child {
                    Child::Node(ch) => self.br_record(i, prof, ch as usize, ro, ext, &mut c, table),
                    leaf => self.leaf_cfv(i, leaf, ro, ext, &mut c),
                }
                tr.pull(&c, |h, v| q[h * ne + e] = v);
            }
            for (h, &hm) in masks.iter().enumerate() {
                let mut best = (f64::NEG_INFINITY, 0);
                for (e, edge) in edges.iter().enumerate() {
                    if hm >> edge.animal & 1 != 0 && q[h * ne + e] > best.0 {
                        best = (q[h * ne + e], e);
                    }
                }
                out[h] = best.0;
                for e in 0..ne {
                    table[node.entry + h * ne + e] = (e == best.1) as u8 as f64;
                }
            }
        } else {
            out.fill(0.0);
            for (e, edge) in edges.iter().enumerate() {
                let tr = &self.trans[edge.trans as usize];
                let mut cro = vec![0.0; tr.to_n];
                tr.push(|h| ro[h] * prof[node.entry + h * ne + e], &mut cro);
                let mut c = vec![0.0; out.len()];
                match edge.child {
                    Child::Node(ch) => self.br_record(i, prof, ch as usize, &cro, ext, &mut c, table),
                    leaf => self.leaf_cfv(i, leaf, &cro, ext, &mut c),
                }
                for (o, x) in out.iter_mut().zip(&c) {
                    *o += x;
                }
            }
        }
    }

    /// Exact best responses (and player 0's value, if `with_value`) of `profile` with root `ranges`
    /// (normalised here).
    pub fn exploitability(
        &self,
        profile: &[f64],
        ranges: &[Vec<f64>; 2],
        with_value: bool,
        threads: usize,
    ) -> Expl {
        self.exploitability_ext(profile, ranges, with_value, &[], threads)
    }

    /// [`Window::exploitability`] in the depth-limited game whose value leaves pay the fixed values `ext`.
    pub fn exploitability_ext(
        &self,
        profile: &[f64],
        ranges: &[Vec<f64>; 2],
        with_value: bool,
        ext: &[f64],
        threads: usize,
    ) -> Expl {
        let r = [normalize(&ranges[0]), normalize(&ranges[1])];
        let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let (b0, f0) = self.root_cfvs_ext(0, profile, &r[1], with_value, ext, threads);
        let (b1, _) = self.root_cfvs_ext(1, profile, &r[0], false, ext, threads);
        let br = [dot(&r[0], &b0), dot(&r[1], &b1)];
        let nashconv = br[0] + br[1] - 1.0;
        Expl {
            br,
            value: if with_value { dot(&r[0], &f0) } else { f64::NAN },
            nashconv,
            exploitability: nashconv / 2.0,
        }
    }
}

impl Window {
    /// Public actions from the root to value leaf `j`.
    pub fn leaf_history(&self, j: usize) -> Vec<u8> {
        let leaf = &self.value_leaves[j];
        let mut h = self.history(leaf.parent as usize);
        h.push(leaf.action);
        h
    }

    /// Possible hands of `player` at value leaf `j` (local layout).
    pub fn leaf_hands(&self, j: usize, player: usize) -> &[u16] {
        &self.handsets[self.value_leaves[j].hs[player] as usize].masks
    }

    /// Both players' reaches at every value leaf under `profile` from the root `ranges` (as given, not
    /// normalised): the root range times the player's own action probabilities and draws (leaf-values layout).
    pub fn leaf_reaches(&self, profile: &[f64], ranges: &[Vec<f64>; 2]) -> Vec<f64> {
        let mut out = vec![0.0; self.n_leaf_values];
        if let Child::Node(r) = self.root {
            self.reach_rec(profile, r as usize, [&ranges[0], &ranges[1]], &mut out);
        }
        out
    }

    fn reach_rec(&self, prof: &[f64], n: usize, r: [&[f64]; 2], out: &mut [f64]) {
        let node = &self.nodes[n];
        let p = node.mover as usize;
        let ne = node.n_edges as usize;
        for (e, edge) in self.node_edges(n).iter().enumerate() {
            if matches!(edge.child, Child::Terminal(_) | Child::Frontier(_)) {
                continue;
            }
            let tr = &self.trans[edge.trans as usize];
            let mut rp = vec![0.0; tr.to_n];
            tr.push(|h| r[p][h] * prof[node.entry + h * ne + e], &mut rp);
            let mut cr = r;
            cr[p] = &rp;
            match edge.child {
                Child::Node(c) => self.reach_rec(prof, c as usize, cr, out),
                Child::Value(j) => {
                    let leaf = &self.value_leaves[j as usize];
                    for q in 0..2 {
                        out[leaf.off[q] as usize..][..leaf.n[q] as usize].copy_from_slice(cr[q]);
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    /// Copies the profile `src_prof` of window `src` into `dest` (a profile of `self`): `src`'s internal node
    /// `src_at` sits at internal node `at` of `self`, and below them every node of either window must be the node of
    /// the other with the same history (same public position, actions and hands) and leaves must agree, except that
    /// a value leaf of either window may be an internal node of the other (nothing below it is copied). Entries of
    /// the common nodes are copied. This assembles a full-window profile from a depth-limited trunk and re-solved
    /// leaf subtrees, and extracts a trunk or a subtree from a full profile. Returns the number of entries copied.
    pub fn embed(
        &self,
        dest: &mut [f64],
        src: &Window,
        src_prof: &[f64],
        at: usize,
        src_at: usize,
    ) -> Result<usize, String> {
        if dest.len() != self.n_entries || src_prof.len() != src.n_entries {
            return Err("profile lengths do not match the windows".into());
        }
        if at >= self.nodes.len() || src_at >= src.nodes.len() {
            return Err("no such internal node".into());
        }
        self.embed_rec(dest, src, src_prof, at, src_at)
    }

    fn embed_rec(
        &self,
        dest: &mut [f64],
        src: &Window,
        sp: &[f64],
        m: usize,
        n: usize,
    ) -> Result<usize, String> {
        let (a, b) = (&self.nodes[m], &src.nodes[n]);
        let nh = self.node_hands(m, a.mover as usize).len();
        if key(&a.state) != key(&b.state)
            || a.n_edges != b.n_edges
            || nh != src.node_hands(n, b.mover as usize).len()
        {
            return Err(format!("node {n} of the source is not node {m} of the target"));
        }
        let len = nh * a.n_edges as usize;
        dest[a.entry..a.entry + len].copy_from_slice(&sp[b.entry..b.entry + len]);
        let mut count = len;
        for (ea, eb) in self.node_edges(m).iter().zip(src.node_edges(n)) {
            if ea.action != eb.action {
                return Err(format!("actions differ at node {m}"));
            }
            match (ea.child, eb.child) {
                (Child::Node(x), Child::Node(y)) => {
                    count += self.embed_rec(dest, src, sp, x as usize, y as usize)?
                }
                (_, Child::Value(_)) | (Child::Value(_), _) | (Child::Terminal(_), Child::Terminal(_)) => {}
                (Child::Frontier(_), Child::Frontier(_)) => {}
                (x, y) => return Err(format!("children differ below node {m}: {x:?} vs {y:?}")),
            }
        }
        Ok(count)
    }
}

/// Exact frontier values (player 0's win probability per mover hand), in parallel over contiguous chunks.
fn frontier_values(
    states: &[State],
    leaves: &[FrontierLeaf],
    n_values: usize,
    k: usize,
    threads: usize,
) -> (Vec<f64>, u64) {
    let mut values = vec![0.0; n_values];
    if states.is_empty() {
        return (values, 0);
    }
    let solve = |states: &[State], leaves: &[FrontierLeaf], out: &mut [f64], base: usize| -> u64 {
        // About 700 search nodes per frontier position (stage-1 benchmark): size the table to the work, 8 MiB at most.
        let log2 = (64 - (states.len() as u64 * 512).leading_zeros()).clamp(10, 18);
        let mut solver = PiSolver::new(log2);
        for (s, leaf) in states.iter().zip(leaves) {
            let vals = solver.frontier_values(s, k);
            debug_assert_eq!(vals.len(), leaf.n as usize);
            for (j, (_, v)) in vals.into_iter().enumerate() {
                out[leaf.off as usize - base + j] = v as f64 / 2.0;
            }
        }
        solver.stats.nodes
    };
    let threads = threads.max(1).min(states.len());
    if threads == 1 {
        let nodes = solve(states, leaves, &mut values, 0);
        return (values, nodes);
    }
    // Contiguous chunks (siblings share table entries); leaves are laid out in order, so value ranges are too.
    let chunk = states.len().div_ceil(threads);
    let mut nodes = 0;
    thread::scope(|sc| {
        let mut rest: &mut [f64] = &mut values;
        let mut handles = Vec::new();
        for (cs, cl) in states.chunks(chunk).zip(leaves.chunks(chunk)) {
            let base = cl[0].off as usize;
            let len = cl.iter().map(|l| l.n as usize).sum::<usize>();
            let (mine, tail) = std::mem::take(&mut rest).split_at_mut(len);
            rest = tail;
            handles.push(sc.spawn(move || solve(cs, cl, mine, base)));
        }
        for h in handles {
            nodes += h.join().expect("frontier worker panicked");
        }
    });
    (values, nodes)
}

/// Read-only walk: best-response (and profile-following) cfvs of player `i`.
struct EvalCtx<'a> {
    w: &'a Window,
    prof: &'a [f64],
    i: usize,
    follow: bool,
    /// Value-leaf values (empty without value leaves).
    ext: &'a [f64],
    par_min: u32,
}

impl EvalCtx<'_> {
    /// cfvs of `i` at node `n` over `i`'s hands, given the opponent's reach `ro`.
    fn walk(&self, n: usize, ro: &[f64], best: &mut [f64], fol: &mut [f64], scratch: &mut [f64]) {
        let w = self.w;
        let node = &w.nodes[n];
        let p = node.mover as usize;
        let ne = node.n_edges as usize;
        let edges = w.node_edges(n);
        let masks = w.node_hands(n, p);
        let nh = masks.len();
        let sig = &self.prof[node.entry..node.entry + nh * ne];
        let parallel = node.end - n as u32 >= self.par_min;
        if p == self.i {
            let (qb, rest) = scratch.split_at_mut(nh * ne);
            let (qf, rest) = rest.split_at_mut(nh * ne);
            let mut combine = |e: usize, cb: &[f64], cf: &[f64]| {
                let tr = &w.trans[edges[e].trans as usize];
                tr.pull(cb, |h, v| qb[h * ne + e] = v);
                if self.follow {
                    tr.pull(cf, |h, v| qf[h * ne + e] = v);
                }
            };
            if parallel {
                let results = self.children_parallel(n, |_e| ro.to_vec(), |_e, tr| tr.to_n);
                for (e, (cb, cf)) in results.iter().enumerate() {
                    combine(e, cb, cf);
                }
            } else {
                for (e, edge) in edges.iter().enumerate() {
                    let ton = w.trans[edge.trans as usize].to_n;
                    let (cb, rest) = rest.split_at_mut(ton);
                    let (cf, rest) = rest.split_at_mut(ton);
                    match edge.child {
                        Child::Node(c) => self.walk(c as usize, ro, cb, cf, rest),
                        leaf => {
                            w.leaf_cfv(self.i, leaf, ro, self.ext, cb);
                            cf.copy_from_slice(cb);
                        }
                    }
                    combine(e, cb, cf);
                }
            }
            for (h, &hm) in masks.iter().enumerate() {
                let row = h * ne;
                let (mut b, mut f) = (f64::NEG_INFINITY, 0.0);
                for e in 0..ne {
                    if hm >> edges[e].animal & 1 != 0 {
                        b = b.max(qb[row + e]);
                        if self.follow {
                            f += sig[row + e] * qf[row + e];
                        }
                    }
                }
                best[h] = b;
                fol[h] = f;
            }
        } else {
            best.fill(0.0);
            fol.fill(0.0);
            let nhi = best.len();
            let mut combine = |cb: &[f64], cf: &[f64]| {
                for h in 0..nhi {
                    best[h] += cb[h];
                }
                if self.follow {
                    for h in 0..nhi {
                        fol[h] += cf[h];
                    }
                }
            };
            let child_reach = |e: usize, out: &mut [f64]| {
                let tr = &w.trans[edges[e].trans as usize];
                out.fill(0.0);
                tr.push(|h| ro[h] * sig[h * ne + e], out);
            };
            if parallel {
                let results = self.children_parallel(
                    n,
                    |e| {
                        let mut r = vec![0.0; w.trans[edges[e].trans as usize].to_n];
                        child_reach(e, &mut r);
                        r
                    },
                    |_e, _tr| nhi,
                );
                for (cb, cf) in &results {
                    combine(cb, cf);
                }
            } else {
                for (e, edge) in edges.iter().enumerate() {
                    let ton = w.trans[edge.trans as usize].to_n;
                    let (cro, rest) = scratch.split_at_mut(ton);
                    let (cb, rest) = rest.split_at_mut(nhi);
                    let (cf, rest) = rest.split_at_mut(nhi);
                    child_reach(e, cro);
                    match edge.child {
                        Child::Node(c) => self.walk(c as usize, cro, cb, cf, rest),
                        leaf => {
                            w.leaf_cfv(self.i, leaf, cro, self.ext, cb);
                            cf.copy_from_slice(cb);
                        }
                    }
                    combine(cb, cf);
                }
            }
        }
    }

    /// Child cfvs of every edge of `n`, internal children walked on scoped threads. `reach(e)` is the opponent's
    /// reach at child `e`; `len(e, trans)` the length of `i`'s cfv vector there.
    fn children_parallel(
        &self,
        n: usize,
        reach: impl Fn(usize) -> Vec<f64>,
        len: impl Fn(usize, &Trans) -> usize,
    ) -> Vec<(Vec<f64>, Vec<f64>)> {
        let w = self.w;
        let edges = w.node_edges(n);
        let mut results: Vec<(Vec<f64>, Vec<f64>)> = edges
            .iter()
            .enumerate()
            .map(|(e, edge)| {
                let l = len(e, &w.trans[edge.trans as usize]);
                (vec![0.0; l], vec![0.0; l])
            })
            .collect();
        let reaches: Vec<Vec<f64>> = (0..edges.len()).map(&reach).collect();
        thread::scope(|sc| {
            for ((edge, (cb, cf)), ro) in edges.iter().zip(results.iter_mut()).zip(&reaches) {
                match edge.child {
                    Child::Node(c) => {
                        sc.spawn(move || {
                            let mut scratch = vec![0.0; w.scratch_len];
                            self.walk(c as usize, ro, cb, cf, &mut scratch);
                        });
                    }
                    leaf => {
                        w.leaf_cfv(self.i, leaf, ro, self.ext, cb);
                        cf.copy_from_slice(cb);
                    }
                }
            }
        });
        results
    }
}

/// One alternating half-iteration of CFR for player `i`.
struct CfrCtx<'a> {
    w: &'a Window,
    algo: Algo,
    i: usize,
    /// Average-strategy weight of this iteration, and DCFR's discounts of positive / negative regrets.
    wt: f64,
    dpos: f64,
    dneg: f64,
    /// Value-leaf values (empty without value leaves).
    ext: &'a [f64],
    par_min: u32,
}

impl CfrCtx<'_> {
    fn strategy(&self, reg: &[f64], pred: &[f64], masks: &[u16], animals: &[u8], sig: &mut [f64]) {
        let extra = if self.algo == Algo::PcfrPlus { Some(pred) } else { None };
        regret_match(reg, extra, masks, animals, sig);
    }

    /// Updates `i`'s regrets and average below `n`; writes `i`'s cfvs at `n` (over `i`'s hands) to `out`.
    /// `m` covers the entries of a range of subtrees containing `n`'s.
    fn walk(&self, m: &mut Slices, n: usize, ri: &[f64], ro: &[f64], out: &mut [f64], scratch: &mut [f64]) {
        let w = self.w;
        let node = &w.nodes[n];
        let p = node.mover as usize;
        let ne = node.n_edges as usize;
        let edges = w.node_edges(n);
        let masks = w.node_hands(n, p);
        let nh = masks.len();
        let (animals, _) = w.animals(n);
        let animals = &animals[..ne];
        let e0 = node.entry - m.base;
        let (sig, scratch) = scratch.split_at_mut(nh * ne);
        self.strategy(&m.reg[e0..e0 + nh * ne], &m.pred[e0..e0 + nh * ne], masks, animals, sig);
        let parallel = node.end - n as u32 >= self.par_min;
        if p == self.i {
            let (q, rest) = scratch.split_at_mut(nh * ne);
            q.fill(0.0);
            let child_reach = |e: usize, out: &mut [f64]| {
                let tr = &w.trans[edges[e].trans as usize];
                out.fill(0.0);
                tr.push(|h| ri[h] * sig[h * ne + e], out);
            };
            {
                let mut combine = |e: usize, ccfv: &[f64]| {
                    w.trans[edges[e].trans as usize].pull(ccfv, |h, v| q[h * ne + e] = v);
                };
                if parallel {
                    let results = self.children_parallel(
                        m,
                        n,
                        |e| {
                            let mut r = vec![0.0; w.trans[edges[e].trans as usize].to_n];
                            child_reach(e, &mut r);
                            (r, ro.to_vec())
                        },
                        |e| w.trans[edges[e].trans as usize].to_n,
                    );
                    for (e, c) in results.iter().enumerate() {
                        combine(e, c);
                    }
                } else {
                    for (e, edge) in edges.iter().enumerate() {
                        let ton = w.trans[edge.trans as usize].to_n;
                        let (cr, rest) = rest.split_at_mut(ton);
                        let (ccfv, rest) = rest.split_at_mut(ton);
                        match edge.child {
                            Child::Node(c) => {
                                child_reach(e, cr);
                                self.walk(m, c as usize, cr, ro, ccfv, rest);
                            }
                            leaf => w.leaf_cfv(self.i, leaf, ro, self.ext, ccfv),
                        }
                        combine(e, ccfv);
                    }
                }
            }
            // Node values, then the regret and average updates.
            let reg = &mut m.reg[e0..e0 + nh * ne];
            let avg = &mut m.avg[e0..e0 + nh * ne];
            let pred = &mut m.pred[e0..e0 + nh * ne];
            for (h, &hm) in masks.iter().enumerate() {
                let row = h * ne;
                let mut v = 0.0;
                for e in 0..ne {
                    v += sig[row + e] * q[row + e];
                }
                out[h] = v;
                let wr = self.wt * ri[h];
                for (e, &animal) in animals.iter().enumerate() {
                    if hm >> animal & 1 != 0 {
                        let x = row + e;
                        update_regret(self.algo, reg, pred, x, q[x] - v, self.dpos, self.dneg);
                        avg[x] += wr * sig[x];
                    }
                }
            }
        } else {
            out.fill(0.0);
            let nhi = out.len();
            let child_reach = |e: usize, out: &mut [f64]| {
                let tr = &w.trans[edges[e].trans as usize];
                out.fill(0.0);
                tr.push(|h| ro[h] * sig[h * ne + e], out);
            };
            if parallel {
                let results = self.children_parallel(
                    m,
                    n,
                    |e| {
                        let mut r = vec![0.0; w.trans[edges[e].trans as usize].to_n];
                        child_reach(e, &mut r);
                        (ri.to_vec(), r)
                    },
                    |_e| nhi,
                );
                for c in &results {
                    for h in 0..nhi {
                        out[h] += c[h];
                    }
                }
            } else {
                for (e, edge) in edges.iter().enumerate() {
                    let ton = w.trans[edge.trans as usize].to_n;
                    let (cro, rest) = scratch.split_at_mut(ton);
                    let (ccfv, rest) = rest.split_at_mut(nhi);
                    child_reach(e, cro);
                    match edge.child {
                        Child::Node(c) => self.walk(m, c as usize, ri, cro, ccfv, rest),
                        leaf => w.leaf_cfv(self.i, leaf, cro, self.ext, ccfv),
                    }
                    for h in 0..nhi {
                        out[h] += ccfv[h];
                    }
                }
            }
        }
    }

    /// Child cfvs of every edge of `n` (internal children on scoped threads over disjoint slices).
    /// `reaches(e)` gives `(r_i, r_o)` at child `e`, `len(e)` the length of `i`'s cfv vector there.
    fn children_parallel(
        &self,
        m: &mut Slices,
        n: usize,
        reaches: impl Fn(usize) -> (Vec<f64>, Vec<f64>),
        len: impl Fn(usize) -> usize,
    ) -> Vec<Vec<f64>> {
        let w = self.w;
        let node = &w.nodes[n];
        let edges = w.node_edges(n);
        let own = w.node_hands(n, node.mover as usize).len() * node.n_edges as usize;
        let sub = m.sub(node.entry, w.entry_end(n));
        let (_, mut rest) = sub.split(own);
        // Children's subtrees follow the node's own entries, in edge order.
        let mut parts: Vec<Option<Slices>> = Vec::with_capacity(edges.len());
        for edge in edges {
            if let Child::Node(c) = edge.child {
                let c = c as usize;
                debug_assert_eq!(rest.base, w.nodes[c].entry);
                let (mine, tail) = rest.split(w.entry_end(c) - w.nodes[c].entry);
                rest = tail;
                parts.push(Some(mine));
            } else {
                parts.push(None);
            }
        }
        let mut results: Vec<Vec<f64>> = (0..edges.len()).map(|e| vec![0.0; len(e)]).collect();
        let reach: Vec<(Vec<f64>, Vec<f64>)> = (0..edges.len()).map(&reaches).collect();
        thread::scope(|sc| {
            for (((edge, out), (ri, ro)), part) in edges.iter().zip(results.iter_mut()).zip(&reach).zip(parts)
            {
                match (edge.child, part) {
                    (Child::Node(c), Some(mut mine)) => {
                        sc.spawn(move || {
                            let mut scratch = vec![0.0; w.scratch_len];
                            self.walk(&mut mine, c as usize, ri, ro, out, &mut scratch);
                        });
                    }
                    (leaf, _) => w.leaf_cfv(self.i, leaf, ro, self.ext, out),
                }
            }
        });
        results
    }
}

/// Re-solving gadget state: for each opponent hand, terminate (value `cfvs[h]`) or follow into the window.
#[derive(Clone, Debug)]
pub struct Gadget {
    /// The re-solving player (ours); the opponent is `1 - player`.
    pub player: usize,
    /// Opponent counterfactual values per opponent root hand (local layout), in units where our range sums to 1.
    pub cfvs: Vec<f64>,
    /// Opponent prior over hands (normalised; uniform over the possible hands by default).
    pub prior: Vec<f64>,
    /// `[follow, terminate]` per opponent hand.
    reg: Vec<f64>,
    avg: Vec<f64>,
    pred: Vec<f64>,
}

impl Gadget {
    fn follow(&self, algo: Algo, avg: bool) -> Vec<f64> {
        let n = self.cfvs.len();
        (0..n)
            .map(|h| {
                let (x, extra) = if avg {
                    ([self.avg[2 * h], self.avg[2 * h + 1]], [0.0; 2])
                } else if algo == Algo::PcfrPlus {
                    ([self.reg[2 * h], self.reg[2 * h + 1]], [self.pred[2 * h], self.pred[2 * h + 1]])
                } else {
                    ([self.reg[2 * h], self.reg[2 * h + 1]], [0.0; 2])
                };
                let f = (x[0] + extra[0]).max(0.0);
                let t = (x[1] + extra[1]).max(0.0);
                if f + t > 0.0 { f / (f + t) } else { 0.5 }
            })
            .collect()
    }

    /// Solves the gadget exactly when the window root is a leaf: we have no decisions there, so each opponent hand
    /// just compares its fixed follow value (the leaf cfv against our range `ours`) with its terminate value. Sets
    /// the current and average gadget strategies to that pure best choice (following on ties).
    fn solve_at_leaf(&mut self, w: &Window, ours: &[f64]) {
        let mut fol = vec![0.0; self.cfvs.len()];
        w.leaf_cfv(1 - self.player, w.root, ours, &[], &mut fol);
        for (h, (&f, &c)) in fol.iter().zip(&self.cfvs).enumerate() {
            let x = if f >= c { [1.0, 0.0] } else { [0.0, 1.0] };
            self.reg[2 * h..2 * h + 2].copy_from_slice(&x);
            self.avg[2 * h..2 * h + 2].copy_from_slice(&x);
            self.pred[2 * h..2 * h + 2].fill(0.0);
        }
    }
}

/// Gadget evaluation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GadgetExpl {
    /// Exploitability of the gadget game: (BR_opp + BR_ours - 1) / 2.
    pub exploitability: f64,
    /// Largest excess of the opponent's best-response cfv over the given cfv, over the opponent's hands.
    pub max_violation: f64,
    /// Prior-weighted mean of max(0, excess).
    pub mean_violation: f64,
}

/// Result of [`Cfr::solve`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SolveResult {
    pub iterations: u32,
    /// Exploitability of the average strategy on the window (or of the gadget game when re-solving).
    pub exploitability: f64,
    pub expl: Expl,
    pub gadget: Option<GadgetExpl>,
    pub seconds: f64,
}

/// CFR solver state for a window (regrets, average strategy, predictions).
#[derive(Clone, Debug)]
pub struct Cfr {
    pub algo: Algo,
    /// Root ranges (local layout, normalised).
    pub ranges: [Vec<f64>; 2],
    /// Iterations done.
    pub t: u32,
    pub threads: usize,
    reg: Vec<f64>,
    avg: Vec<f64>,
    pred: Vec<f64>,
    pub gadget: Option<Gadget>,
    /// Depth-limited solving state (windows with value leaves).
    pub ext: Option<ExtState>,
}

/// What the caller of a depth-limited half-iteration evaluates: per value leaf, both players' beliefs (normalised
/// reaches in the leaf-values layout; uniform over the feasible hands where a player's reach mass is 0) and reach
/// masses.
#[derive(Clone, Debug, Default)]
pub struct LeafQuery {
    pub beliefs: Vec<f64>,
    pub mass: Vec<[f64; 2]>,
}

impl LeafQuery {
    /// Beliefs and masses from raw reaches (leaf-values layout).
    pub fn from_reaches(w: &Window, reach: &[f64]) -> LeafQuery {
        let mut q = LeafQuery { beliefs: reach.to_vec(), mass: Vec::with_capacity(w.value_leaves.len()) };
        for leaf in &w.value_leaves {
            let m = [0, 1].map(|p| {
                let b = &mut q.beliefs[leaf.off[p] as usize..][..leaf.n[p] as usize];
                let mass: f64 = b.iter().sum();
                if mass > 0.0 {
                    b.iter_mut().for_each(|x| *x /= mass);
                } else {
                    b.fill(1.0 / b.len() as f64);
                }
                mass
            });
            q.mass.push(m);
        }
        q
    }
}

/// Depth-limited solving state: the alternating half-iterations and the averaged leaf values.
#[derive(Clone, Debug)]
pub struct ExtState {
    /// The player whose half-iteration is next (0 starts each iteration).
    pub phase: usize,
    /// The query of the half-iteration in progress (`ext_begin` called, `ext_end` not yet).
    pub query: Option<LeafQuery>,
    /// The values of the last completed half-iteration (leaf-values layout).
    pub values: Vec<f64>,
    /// Per leaf and player `o`: `sum_t w_t m_p^t v_o^t(h)` over `p`'s half-iterations (`p = 1 - o`), at `o`'s slots.
    acc: Vec<f64>,
    /// Per leaf: `sum_t w_t m_p^t` over `p`'s half-iterations, indexed by `p`.
    acc_mass: Vec<[f64; 2]>,
    /// Half-iterations completed.
    pub halves: u64,
}

impl Cfr {
    pub fn new(w: &Window, ranges: [Vec<f64>; 2], algo: Algo) -> Cfr {
        for (p, r) in ranges.iter().enumerate() {
            assert_eq!(r.len(), w.root_hands(p).len(), "range of player {p} has the wrong length");
        }
        Cfr {
            algo,
            ranges: [normalize(&ranges[0]), normalize(&ranges[1])],
            t: 0,
            threads: 1,
            reg: vec![0.0; w.n_entries],
            avg: vec![0.0; w.n_entries],
            pred: vec![0.0; w.n_entries],
            gadget: None,
            ext: None,
        }
    }

    /// The re-solving gadget for `player` with our range `ours` and the opponent's counterfactual values `cfvs`
    /// (per opponent root hand, local layout, with our range normalised), `prior` the opponent's gadget prior
    /// (uniform when `None`).
    pub fn new_gadget(
        w: &Window,
        player: usize,
        ours: Vec<f64>,
        cfvs: Vec<f64>,
        prior: Option<Vec<f64>>,
        algo: Algo,
    ) -> Cfr {
        let o = 1 - player;
        let no = w.root_hands(o).len();
        assert_eq!(cfvs.len(), no, "one cfv per opponent hand");
        let prior = normalize(&prior.unwrap_or_else(|| vec![1.0; no]));
        let mut ranges: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        ranges[player] = ours;
        ranges[o] = prior.clone();
        let mut cfr = Cfr::new(w, ranges, algo);
        cfr.gadget = Some(Gadget {
            player,
            cfvs,
            prior,
            reg: vec![0.0; 2 * no],
            avg: vec![0.0; 2 * no],
            pred: vec![0.0; 2 * no],
        });
        cfr
    }

    /// Average-strategy weight and DCFR's discounts of positive / negative regrets at iteration `t` (from 1).
    fn weights(&self) -> (f64, f64, f64) {
        let t = self.t as f64;
        match self.algo {
            Algo::Dcfr => (t * t, t.powf(1.5) / (t.powf(1.5) + 1.0), 0.5),
            Algo::PcfrPlus => (t * t, 1.0, 1.0),
            Algo::CfrPlus => (t, 1.0, 1.0),
        }
    }

    /// Restarts from the current regrets with new root `ranges` (a warm start for a re-solve whose ranges moved a
    /// little): the average strategy, the predictions and the iteration count start over, the regrets are kept.
    pub fn warm_restart(&mut self, w: &Window, ranges: [Vec<f64>; 2]) {
        for (p, r) in ranges.iter().enumerate() {
            assert_eq!(r.len(), w.root_hands(p).len(), "range of player {p} has the wrong length");
        }
        self.ranges = [normalize(&ranges[0]), normalize(&ranges[1])];
        self.avg.fill(0.0);
        self.pred.fill(0.0);
        self.t = 0;
    }

    /// Runs `iters` iterations. At a leaf root there is nothing to iterate (`t` stays 0); a re-solving gadget is
    /// then solved exactly instead (the opponent's best terminate/follow choice per hand). Windows with value leaves
    /// iterate through [`Cfr::ext_begin`] / [`Cfr::ext_end`] instead.
    pub fn iterate(&mut self, w: &Window, iters: u32) {
        assert!(w.value_leaves.is_empty(), "a window with value leaves iterates with ext_begin / ext_end");
        let Child::Node(root) = w.root else {
            if let Some(g) = self.gadget.as_mut() {
                g.solve_at_leaf(w, &self.ranges[g.player]);
            }
            return;
        };
        let mut scratch = vec![0.0; w.scratch_len];
        for _ in 0..iters {
            self.t += 1;
            let (wt, dpos, dneg) = self.weights();
            for i in 0..2 {
                let ctx = CfrCtx {
                    w,
                    algo: self.algo,
                    i,
                    wt,
                    dpos,
                    dneg,
                    ext: &[],
                    par_min: w.par_min(self.threads),
                };
                let (ri, ro) = match &self.gadget {
                    None => (self.ranges[i].clone(), self.ranges[1 - i].clone()),
                    Some(g) => {
                        let follow = g.follow(self.algo, false);
                        let opp: Vec<f64> = g.prior.iter().zip(&follow).map(|(p, f)| p * f).collect();
                        if i == g.player {
                            (self.ranges[i].clone(), opp)
                        } else {
                            (opp, self.ranges[g.player].clone())
                        }
                    }
                };
                let mut out = vec![0.0; w.root_hands(i).len()];
                let mut m = Slices { base: 0, reg: &mut self.reg, avg: &mut self.avg, pred: &mut self.pred };
                ctx.walk(&mut m, root as usize, &ri, &ro, &mut out, &mut scratch);
                // The opponent's gadget decision per hand: follow (value `out`) or terminate (value `cfvs`).
                if let Some(g) = self.gadget.as_mut().filter(|g| g.player != i) {
                    let sigma_f = g.follow(self.algo, false);
                    for h in 0..g.cfvs.len() {
                        let q = [out[h], g.cfvs[h]];
                        let s = [sigma_f[h], 1.0 - sigma_f[h]];
                        let v = s[0] * q[0] + s[1] * q[1];
                        for a in 0..2 {
                            update_regret(
                                self.algo,
                                &mut g.reg,
                                &mut g.pred,
                                2 * h + a,
                                q[a] - v,
                                dpos,
                                dneg,
                            );
                            g.avg[2 * h + a] += wt * g.prior[h] * s[a];
                        }
                    }
                }
            }
        }
    }

    /// Starts the next half-iteration of a depth-limited solve (a window with value leaves, or none: then it is an
    /// ordinary alternating iteration in two calls). Computes the current strategies and returns the leaf query:
    /// both players' beliefs and masses at every value leaf. Evaluate the value function there and pass both
    /// players' values to [`Cfr::ext_end`]. The first call starts the external mode.
    pub fn ext_begin(&mut self, w: &Window) -> &LeafQuery {
        assert!(self.gadget.is_none(), "depth-limited solving does not take a gadget");
        let st = self.ext.get_or_insert_with(|| ExtState {
            phase: 0,
            query: None,
            values: vec![0.0; w.n_leaf_values],
            acc: vec![0.0; w.n_leaf_values],
            acc_mass: vec![[0.0; 2]; w.value_leaves.len()],
            halves: 0,
        });
        assert!(st.query.is_none(), "ext_begin called twice without ext_end");
        if st.phase == 0 {
            self.t += 1;
        }
        let prof = self.current(w);
        let reach = w.leaf_reaches(&prof, &self.ranges);
        let st = self.ext.as_mut().expect("external state");
        st.query = Some(LeafQuery::from_reaches(w, &reach));
        st.query.as_ref().expect("query")
    }

    /// Ends the half-iteration started by [`Cfr::ext_begin`]: `values` (leaf-values layout, both players, each the
    /// owner's win probability per hand against the other's belief) become the value-leaf payoffs, and the regrets
    /// and average strategy of the half-iteration's player are updated.
    pub fn ext_end(&mut self, w: &Window, values: Vec<f64>) -> Result<(), String> {
        let st = self.ext.as_mut().filter(|e| e.query.is_some()).ok_or("ext_end without ext_begin")?;
        if values.len() != w.n_leaf_values {
            return Err(format!("{} leaf values, the window has {}", values.len(), w.n_leaf_values));
        }
        if let Some(x) = values.iter().position(|v| !v.is_finite()) {
            return Err(format!("leaf value {x} is {}", values[x]));
        }
        let query = st.query.take().expect("pending query");
        let i = st.phase;
        let o = 1 - i;
        let (wt, dpos, dneg) = self.weights();
        if let Child::Node(root) = w.root {
            let ctx = CfrCtx {
                w,
                algo: self.algo,
                i,
                wt,
                dpos,
                dneg,
                ext: &values,
                par_min: w.par_min(self.threads),
            };
            let mut scratch = vec![0.0; w.scratch_len];
            let mut out = vec![0.0; w.root_hands(i).len()];
            let mut m = Slices { base: 0, reg: &mut self.reg, avg: &mut self.avg, pred: &mut self.pred };
            ctx.walk(&mut m, root as usize, &self.ranges[i], &self.ranges[o], &mut out, &mut scratch);
        }
        let st = self.ext.as_mut().expect("external state");
        for (j, leaf) in w.value_leaves.iter().enumerate() {
            let x = wt * query.mass[j][i];
            st.acc_mass[j][i] += x;
            if x > 0.0 {
                let r = leaf.off[o] as usize..leaf.off[o] as usize + leaf.n[o] as usize;
                for (a, v) in st.acc[r.clone()].iter_mut().zip(&values[r]) {
                    *a += x * v;
                }
            }
        }
        st.values = values;
        st.phase = o;
        st.halves += 1;
        Ok(())
    }

    /// Per value leaf and player `p`: `p`'s values averaged over the other player's half-iterations with the weights
    /// of the other player's average strategy (`w_t m^t`), i.e. `p`'s values against the other's average strategy
    /// at the leaf in units where the other's reach there sums to 1 (see the module documentation). The values of
    /// the last half-iteration where the other player never reached the leaf. Leaf-values layout.
    pub fn avg_leaf_values(&self, w: &Window) -> Vec<f64> {
        let Some(st) = &self.ext else { return vec![0.0; w.n_leaf_values] };
        let mut out = st.values.clone();
        for (j, leaf) in w.value_leaves.iter().enumerate() {
            for p in 0..2 {
                let m = st.acc_mass[j][1 - p];
                if m > 0.0 {
                    let r = leaf.off[p] as usize..leaf.off[p] as usize + leaf.n[p] as usize;
                    for (x, a) in out[r.clone()].iter_mut().zip(&st.acc[r]) {
                        *x = a / m;
                    }
                }
            }
        }
        out
    }

    /// The average strategy (normalised profile table).
    pub fn average(&self, w: &Window) -> Vec<f64> {
        w.normalized(&self.avg)
    }

    /// The current strategy (regret matching on the accumulated regrets, plus predictions for PCFR+).
    pub fn current(&self, w: &Window) -> Vec<f64> {
        if self.algo == Algo::PcfrPlus {
            let x: Vec<f64> = self.reg.iter().zip(&self.pred).map(|(r, p)| r + p).collect();
            w.normalized(&x)
        } else {
            w.normalized(&self.reg)
        }
    }

    /// Exact exploitability of the average strategy on the window with the solver's ranges.
    pub fn exploitability(&self, w: &Window, with_value: bool) -> Expl {
        w.exploitability(&self.average(w), &self.ranges, with_value, self.threads)
    }

    /// Exact evaluation of the gadget game for the average strategies.
    pub fn gadget_exploitability(&self, w: &Window) -> GadgetExpl {
        let g = self.gadget.as_ref().expect("not a gadget solve");
        let (pl, o) = (g.player, 1 - g.player);
        let prof = self.average(w);
        let ours = &self.ranges[pl];
        let (br_o, _) = w.root_cfvs(o, &prof, ours, false, self.threads);
        let mut br_opp = 0.0;
        let (mut max_v, mut mean_v) = (f64::NEG_INFINITY, 0.0);
        for ((&prior, &br), &cfv) in g.prior.iter().zip(&br_o).zip(&g.cfvs) {
            br_opp += prior * br.max(cfv);
            max_v = max_v.max(br - cfv);
            mean_v += prior * (br - cfv).max(0.0);
        }
        let follow = g.follow(self.algo, true);
        let opp: Vec<f64> = g.prior.iter().zip(&follow).map(|(p, f)| p * f).collect();
        let (br_p, _) = w.root_cfvs(pl, &prof, &opp, false, self.threads);
        let mut br_ours: f64 = ours.iter().zip(&br_p).map(|(r, v)| r * v).sum();
        for ((&prior, &f), &cfv) in g.prior.iter().zip(&follow).zip(&g.cfvs) {
            br_ours += prior * (1.0 - f) * (1.0 - cfv);
        }
        GadgetExpl {
            exploitability: (br_opp + br_ours - 1.0) / 2.0,
            max_violation: max_v,
            mean_violation: mean_v,
        }
    }

    /// Iterates until the exploitability (of the window, or of the gadget game) is at most `target`, checking
    /// every `check_every` iterations, or until `max_iters` iterations in all.
    pub fn solve(&mut self, w: &Window, target: f64, max_iters: u32, check_every: u32) -> SolveResult {
        let t0 = Instant::now();
        let check_every = check_every.max(1);
        let mut res = SolveResult::default();
        loop {
            let n = check_every.min(max_iters.saturating_sub(self.t));
            self.iterate(w, n);
            let e = if self.gadget.is_some() {
                let g = self.gadget_exploitability(w);
                res.gadget = Some(g);
                g.exploitability
            } else {
                res.expl = self.exploitability(w, false);
                res.expl.exploitability
            };
            res.exploitability = e;
            if e <= target || self.t >= max_iters || !matches!(w.root, Child::Node(_)) {
                break;
            }
        }
        if self.gadget.is_none() {
            res.expl = self.exploitability(w, true);
        }
        res.iterations = self.t;
        res.seconds = t0.elapsed().as_secs_f64();
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animal::ALL_ANIMALS;
    use crate::hands::subsets;

    #[test]
    fn local_rank_is_the_position_in_hands() {
        let mut x = 0x9E37_79B9u64;
        for _ in 0..300 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (x >> 20) as u16 & ALL_ANIMALS;
            for m in 0..=u.count_ones() as usize {
                for (i, h) in subsets(u, m).enumerate() {
                    assert_eq!(local_rank(h, u), i);
                }
            }
        }
    }
}
