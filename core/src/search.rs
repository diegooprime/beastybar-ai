//! Deterministic perfect-information search over a fully determinized position (every deck order fixed, so draws
//! are known): negamax alpha-beta, depth-limited in plies, [`evaluate`] at the horizon.
//!
//! [`Search::root_values`] returns the exact depth-limited minimax value of every root action (each root action is
//! searched with a full window), so values can be averaged across sampled worlds.
//!
//! A searcher built [`Search::with_stop`] polls its stop flag every few thousand nodes and abandons the search once it
//! is set ([`Search::aborted`]): a deep search (say from the opening to the end) can always be interrupted.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::action::{ActionSet, NUM_ACTIONS};
use crate::eval::evaluate;
use crate::state::State;

/// More plies than any game has (each ply plays one of the 24 cards).
const UNLIMITED: u32 = 64;
const INF: f32 = f32::INFINITY;

/// The value of every legal root action for the player to move.
#[derive(Clone, Copy, Debug)]
pub struct RootValues {
    pub actions: ActionSet,
    /// Indexed by action; meaningful for `actions` only.
    pub values: [f32; NUM_ACTIONS],
}

/// A searcher: node counter, an optional transposition table and an optional stop flag, reusable across positions
/// and worlds.
pub struct Search {
    /// Positions visited (interior nodes and leaves), cumulative.
    pub nodes: u64,
    tt: Option<Tt>,
    stop: Option<Arc<AtomicBool>>,
    /// `nodes` count at which the stop flag is polled next.
    next_poll: u64,
    /// The current (or last) `root_values` call was cut short by the stop flag.
    aborted: bool,
}

/// Nodes between polls of the stop flag (well under a millisecond of search).
const POLL_NODES: u64 = 1 << 14;

impl Default for Search {
    fn default() -> Search {
        Search::new(DEFAULT_TT_BITS)
    }
}

/// Transposition table size used by agents: 2^14 entries of 24 bytes (384 KB). The table speeds up full-depth
/// endgame solves by ~12-25% and depth-6..8 searches by ~10-20%; small tables that stay in cache are fastest
/// (`cargo run --release --example bench_search`).
pub const DEFAULT_TT_BITS: u32 = 14;

impl Search {
    /// A searcher with a transposition table of 2^`tt_bits` entries, or none if `tt_bits` is 0.
    pub fn new(tt_bits: u32) -> Search {
        Search {
            nodes: 0,
            tt: (tt_bits > 0).then(|| Tt::new(tt_bits)),
            stop: None,
            next_poll: 0,
            aborted: false,
        }
    }

    /// This searcher, abandoning any search once `stop` is set.
    pub fn with_stop(mut self, stop: Arc<AtomicBool>) -> Search {
        self.stop = Some(stop);
        self
    }

    /// Whether the last `root_values` call was cut short by the stop flag; its values are then meaningless.
    pub fn aborted(&self) -> bool {
        self.aborted
    }

    /// Exact `depth`-ply minimax value (0 = to the end of the game) of every legal action at `root`, from the
    /// perspective of the player to move. Results do not depend on the table's contents (an aborted search stores
    /// nothing in it).
    pub fn root_values(&mut self, root: &State, depth: u32) -> RootValues {
        let rem = if depth == 0 { UNLIMITED } else { depth };
        let actions = root.legal_actions();
        let mut values = [f32::NAN; NUM_ACTIONS];
        self.aborted = false;
        self.next_poll = self.nodes;
        self.nodes += 1;
        for a in actions {
            let mut child = *root;
            child.apply(a);
            values[a as usize] = -self.negamax(&child, rem - 1, -INF, INF);
        }
        RootValues { actions, values }
    }

    /// Polls the stop flag every `POLL_NODES` nodes; true (for the rest of this `root_values` call) once it is set.
    #[inline]
    fn interrupted(&mut self) -> bool {
        if self.nodes >= self.next_poll {
            self.next_poll = self.nodes + POLL_NODES;
            self.aborted |= self.stop.as_ref().is_some_and(|stop| stop.load(Ordering::Relaxed));
        }
        self.aborted
    }

    /// Fail-soft alpha-beta: the exact value if it lies strictly inside (alpha, beta), else a bound on the
    /// right side of the window.
    fn negamax(&mut self, s: &State, rem: u32, mut alpha: f32, beta: f32) -> f32 {
        self.nodes += 1;
        if self.interrupted() {
            return 0.0;
        }
        if rem == 0 || s.is_over() {
            return evaluate(s, s.to_move);
        }
        // Each ply plays a card from a hand, so the game ends within (hands + decks) plies: any deeper horizon
        // gives the same value. Normalizing makes more transpositions hit.
        let rem = rem.min(unplayed(s));
        let me = s.to_move;
        let legal = s.legal_actions();

        if rem == 1 {
            // Children are leaves: no ordering or recursion.
            let mut best = -INF;
            for a in legal {
                let mut child = *s;
                child.apply(a);
                self.nodes += 1;
                let v = evaluate(&child, me);
                if v > best {
                    best = v;
                    if v >= beta {
                        break;
                    }
                }
            }
            return best;
        }

        let mut tt_move = NO_MOVE;
        let key = self.tt.as_ref().map(|_| key(s));
        if let (Some(tt), Some(key)) = (&self.tt, key)
            && let Some(e) = tt.probe(key)
        {
            if e.rem as u32 == rem {
                match e.flag {
                    EXACT => return e.value,
                    LOWER if e.value >= beta => return e.value,
                    UPPER if e.value <= alpha => return e.value,
                    _ => {}
                }
            }
            tt_move = e.best;
        }

        // Children ordered by static value for the mover, the table's best move first.
        let mut kids = [State::default(); NUM_ACTIONS];
        let mut order = [(0f32, 0u8, 0u8); NUM_ACTIONS];
        let mut n = 0;
        for a in legal {
            let mut child = *s;
            child.apply(a);
            let key = if a == tt_move { INF } else { evaluate(&child, me) };
            kids[n] = child;
            order[n] = (key, a, n as u8);
            n += 1;
        }
        let order = &mut order[..n];
        for i in 1..n {
            let x = order[i];
            let mut j = i;
            while j > 0 && order[j - 1].0 < x.0 {
                order[j] = order[j - 1];
                j -= 1;
            }
            order[j] = x;
        }

        let alpha0 = alpha;
        let mut best = -INF;
        let mut best_move = NO_MOVE;
        for &(_, a, i) in order.iter() {
            let v = -self.negamax(&kids[i as usize], rem - 1, -beta, -alpha);
            if v > best {
                best = v;
                best_move = a;
                if v > alpha {
                    alpha = v;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }

        // An aborted subtree's value is meaningless: never store it.
        if let (Some(tt), Some(key), false) = (&mut self.tt, key, self.aborted) {
            let flag = if best <= alpha0 {
                UPPER
            } else if best >= beta {
                LOWER
            } else {
                EXACT
            };
            tt.store(Entry { key, value: best, rem: rem as u8, flag, best: best_move });
        }
        best
    }
}

/// Cards left to play: an upper bound on the plies left in the game.
#[inline]
fn unplayed(s: &State) -> u32 {
    s.hands[0].count_ones() + s.hands[1].count_ones() + s.deck_len[0] as u32 + s.deck_len[1] as u32
}

// ---- transposition table ----

const NO_MOVE: u8 = u8::MAX;
const EMPTY: u8 = 0;
const EXACT: u8 = 1;
const LOWER: u8 = 2;
const UPPER: u8 = 3;

#[derive(Clone, Copy, Default)]
struct Entry {
    key: [u64; 2],
    value: f32,
    rem: u8,
    flag: u8,
    best: u8,
}

struct Tt {
    entries: Box<[Entry]>,
    mask: usize,
}

impl Tt {
    fn new(bits: u32) -> Tt {
        let n = 1usize << bits;
        Tt { entries: vec![Entry::default(); n].into_boxed_slice(), mask: n - 1 }
    }

    #[inline]
    fn index(&self, key: [u64; 2]) -> usize {
        let h = (key[0] ^ key[1].wrapping_mul(0x9E37_79B9_7F4A_7C15)).wrapping_mul(0xD6E8_FEB8_6659_FD93);
        (h >> 32) as usize & self.mask
    }

    #[inline]
    fn probe(&self, key: [u64; 2]) -> Option<&Entry> {
        let e = &self.entries[self.index(key)];
        (e.flag != EMPTY && e.key == key).then_some(e)
    }

    #[inline]
    fn store(&mut self, e: Entry) {
        let i = self.index(e.key);
        self.entries[i] = e;
    }
}

/// Everything that affects the value of a position, packed exactly into 127 bits: queue, hands, deck orders,
/// player to move, and the bar margin. The bar and THAT'S IT contents and arrival order never affect play.
#[inline]
fn key(s: &State) -> [u64; 2] {
    let mut lo = s.hands[0] as u64 | (s.hands[1] as u64) << 12;
    for i in 0..4 {
        lo |= (s.queue[i] as u64) << (24 + 5 * i);
    }
    lo |= (s.queue_len as u64) << 44 | (s.to_move as u64) << 47;
    lo |= ((s.scores[0] as i64 - s.scores[1] as i64 + 64) as u64) << 48;
    lo |= (s.deck_len[0] as u64) << 55 | (s.deck_len[1] as u64) << 59;
    let mut hi = 0u64;
    for p in 0..2 {
        for (i, &a) in s.decks[p].iter().enumerate() {
            hi |= (a as u64) << (32 * p + 4 * i);
        }
    }
    [lo, hi]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// Plain minimax, no pruning: the definition `root_values` must match exactly.
    fn minimax(s: &State, rem: u32) -> f32 {
        if rem == 0 || s.is_over() {
            return evaluate(s, s.to_move);
        }
        let mut best = -INF;
        for a in s.legal_actions() {
            let mut child = *s;
            child.apply(a);
            best = best.max(-minimax(&child, rem - 1));
        }
        best
    }

    fn brute_root(s: &State, depth: u32) -> Vec<(u8, f32)> {
        let rem = if depth == 0 { UNLIMITED } else { depth };
        s.legal_actions()
            .map(|a| {
                let mut child = *s;
                child.apply(a);
                (a, -minimax(&child, rem - 1))
            })
            .collect()
    }

    /// A random position `plies` plies into a random game, or None if the game ended first.
    fn random_position(rng: &mut Rng, plies: u32) -> Option<State> {
        let start = rng.below(2) as u8;
        let mut s = State::deal(rng, start);
        for _ in 0..plies {
            if s.is_over() {
                return None;
            }
            let legal = s.legal_actions();
            s.apply(legal.nth_action(rng.below(legal.len())));
        }
        (!s.is_over()).then_some(s)
    }

    fn check(search: &mut Search, s: &State, depth: u32) {
        let got = search.root_values(s, depth);
        let want = brute_root(s, depth);
        assert_eq!(got.actions.len(), want.len());
        for (a, v) in want {
            assert_eq!(got.values[a as usize], v, "action {a} depth {depth} in {s:?}");
        }
    }

    #[test]
    fn matches_brute_force_minimax() {
        let mut rng = Rng::new(11);
        // One searcher with a table across all positions (stale entries must never change a result), one without.
        let mut with_tt = Search::new(12);
        let mut without = Search::new(0);
        let mut checked = 0;
        while checked < 300 {
            let plies = rng.below(18) as u32;
            let Some(s) = random_position(&mut rng, plies) else { continue };
            for depth in 1..=4 {
                check(&mut with_tt, &s, depth);
                check(&mut without, &s, depth);
            }
            if checked % 5 == 0 {
                check(&mut with_tt, &s, 6);
            }
            // To the end, where brute force is affordable.
            if unplayed(&s) <= 9 {
                check(&mut with_tt, &s, 0);
                check(&mut without, &s, 0);
            }
            checked += 1;
        }
    }

    #[test]
    fn depth_semantics() {
        let mut rng = Rng::new(12);
        let s = random_position(&mut rng, 4).unwrap();
        let mut search = Search::new(0);
        // Depth 1: the static value after each move.
        let r = search.root_values(&s, 1);
        for a in r.actions {
            let mut child = s;
            child.apply(a);
            assert_eq!(r.values[a as usize], evaluate(&child, s.to_move));
        }
        // Depth beyond the end of the game equals depth 0.
        let s = random_position(&mut rng, 16).unwrap();
        let (full, deep) = (search.root_values(&s, 0), search.root_values(&s, unplayed(&s)));
        for a in full.actions {
            assert_eq!(full.values[a as usize], deep.values[a as usize]);
        }
    }

    #[test]
    fn a_stop_aborts_the_search_and_leaves_the_table_clean() {
        use std::time::{Duration, Instant};
        let stop = Arc::new(AtomicBool::new(false));
        let mut search = Search::default().with_stop(stop.clone());
        // From the opening, a search to the end of the game would take hours.
        let s = State::deal(&mut Rng::new(14), 0);
        let setter = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(100));
                stop.store(true, Ordering::Relaxed);
            })
        };
        let t = Instant::now();
        search.root_values(&s, 0);
        assert!(search.aborted());
        assert!(t.elapsed() < Duration::from_secs(5), "took {:?} to notice the stop", t.elapsed());
        setter.join().unwrap();
        // Still set: the next search aborts at once.
        let t = Instant::now();
        search.root_values(&s, 0);
        assert!(search.aborted() && t.elapsed() < Duration::from_millis(100));
        // Cleared: the table kept nothing from the aborted searches, so values equal a fresh searcher's.
        stop.store(false, Ordering::Relaxed);
        for depth in [3, 5, 7] {
            let (a, b) = (search.root_values(&s, depth), Search::default().root_values(&s, depth));
            assert!(!search.aborted());
            for x in a.actions {
                assert_eq!(a.values[x as usize].to_bits(), b.values[x as usize].to_bits(), "depth {depth}");
            }
        }
    }

    #[test]
    fn key_ignores_bar_order_and_separates_play_relevant_fields() {
        let mut rng = Rng::new(13);
        let s = random_position(&mut rng, 10).unwrap();
        let mut t = s;
        t.bar_order.reverse();
        t.thats_it_order.reverse();
        assert_eq!(key(&s), key(&t));
        let mut t = s;
        t.to_move ^= 1;
        assert_ne!(key(&s), key(&t));
        let mut t = s;
        t.scores[0] += 1;
        assert_ne!(key(&s), key(&t));
        let mut t = s;
        t.decks[0].swap(0, 1);
        assert_ne!(key(&s), key(&t));
    }
}
