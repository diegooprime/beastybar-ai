//! Exact perfect-information solver, for positions where both hands are public (both decks empty).
//!
//! Values are outcomes in half points: 0 loss, 1 draw, 2 win (`value / 2` is the game score 0, .5, 1).
//! [`PiSolver::value`] gives the value for player 0. The search is fail-soft negamax alpha-beta on the mover's
//! outcome with
//! - static bounds: points can still come only from cards in hands or in the queue, at most two per remaining
//!   five-animal check (`max_gains`);
//! - move ordering: the transposition table's best move, then the immediate score swing (points the move sends to
//!   the mover's bar minus the opponent's), then action index;
//! - a transposition table keyed on the Markov key, which determines these positions' futures exactly. It stores
//!   lower and upper bounds, has a fixed size (two-entry buckets: one keeps the larger subtree, one is always
//!   replaced) and is reused across solves, since an entry stays true forever.
//!
//! [`PiSolver::frontier_values`] handles the one hidden-information node the solver can still settle: the mover's
//! deck holds at most one card and the opponent's hand is public (ply 15 in the real game: the second mover's last
//! deck card is known by elimination). For each possible hand `H` of the mover, the value is the best over the
//! actions playable from `H` of the exact value after playing it and drawing the remaining card.

use crate::action::NUM_ACTIONS;
use crate::animal::{NUM_ANIMALS, POINTS, animal, owner};
use crate::hands::{hands, unplayed};
use crate::markov::{key, public};
use crate::state::State;

pub const LOSS: u8 = 0;
pub const DRAW: u8 = 1;
pub const WIN: u8 = 2;

/// Outcome for player 0 of a finished game (or of the current scores).
#[inline]
pub fn outcome(s: &State) -> u8 {
    sign3(s.scores[0] as i32 - s.scores[1] as i32) as u8
}

#[inline]
fn sign3(x: i32) -> i8 {
    (x > 0) as i8 + (x >= 0) as i8
}

/// Plain minimax for player 0: no pruning, no table. The reference the solver is tested against.
/// Treats deck draws as known (the stored order), so it is the true value only when both decks are empty.
pub fn minimax(s: &State, nodes: &mut u64) -> u8 {
    *nodes += 1;
    if s.is_over() {
        return outcome(s);
    }
    let maxing = s.to_move == 0;
    let mut best = if maxing { 0 } else { 2 };
    for a in s.legal_actions() {
        let mut c = *s;
        c.apply(a);
        let v = minimax(&c, nodes);
        best = if maxing { best.max(v) } else { best.min(v) };
    }
    best
}

#[derive(Clone, Copy, Debug, Default)]
struct Entry {
    key: u64,
    lo: i8,
    hi: i8,
    /// Best action + 1 (0: none).
    best: u8,
    /// Cards left in hands and queue: a proxy for subtree size.
    depth: u8,
}

/// Search statistics, cumulative until [`PiSolver::reset_stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Positions visited (each call of the search, terminal positions included).
    pub nodes: u64,
    /// Visits answered by a table entry without searching children.
    pub tt_cutoffs: u64,
}

pub struct PiSolver {
    table: Vec<[Entry; 2]>,
    shift: u32,
    pub stats: Stats,
}

impl PiSolver {
    /// A solver whose table has `2^log2_buckets` buckets of two 16-byte entries (`2^20`: 32 MiB).
    pub fn new(log2_buckets: u32) -> PiSolver {
        assert!((1..=32).contains(&log2_buckets));
        PiSolver {
            table: vec![[Entry::default(); 2]; 1 << log2_buckets],
            shift: 64 - log2_buckets,
            stats: Stats::default(),
        }
    }

    /// Empties the table (never needed for correctness).
    pub fn clear(&mut self) {
        self.table.fill([Entry::default(); 2]);
    }

    pub fn reset_stats(&mut self) {
        self.stats = Stats::default();
    }

    /// Exact value for player 0 (0 loss, 1 draw, 2 win) of a position with both decks empty.
    pub fn value(&mut self, s: &State) -> u8 {
        assert!(s.deck_len == [0, 0], "both hands must be public (decks empty)");
        let v = self.search(s, -1, 3) as u8;
        if s.to_move == 0 { v } else { 2 - v }
    }

    /// Value for the side to move of a position with both decks empty.
    pub fn value_for_mover(&mut self, s: &State) -> u8 {
        assert!(s.deck_len == [0, 0], "both hands must be public (decks empty)");
        self.search(s, -1, 3) as u8
    }

    /// Per-hand values at a frontier node, for player 0, one per possible hand of the mover in increasing rank.
    ///
    /// `s` is either the real position (the mover holds one of the hands, the deck its remaining card) or the
    /// public one ([`public`]): only the unplayed sets are used. Requires the opponent's hand to be public
    /// (deck empty, at most `k` unplayed animals) and the mover to have at most `k + 1` unplayed animals.
    /// A hand's value is the best, for the mover, over the actions playable from it of the exact value after
    /// playing it and drawing the remaining card (the same position whatever the hand: unplayed minus the card).
    pub fn frontier_values(&mut self, s: &State, k: usize) -> Vec<(u16, u8)> {
        let p = s.to_move as usize;
        let u = [unplayed(s, 0), unplayed(s, 1)];
        assert!(
            s.deck_len[1 - p] == 0 && u[1 - p].count_ones() as usize <= k,
            "frontier: the opponent's hand must be public"
        );
        assert!(u[p].count_ones() as usize <= k + 1, "frontier: the mover's deck must hold at most one card");
        let root = public(s);
        let to_p0 = |v: i8| if p == 0 { v as u8 } else { 2 - v as u8 };
        if root.is_over() {
            let v = outcome(&root);
            return hands(u[p], k).map(|h| (h, v)).collect();
        }
        // The actions of the public position, best immediate swing first, with the bounds learned so far.
        struct Act {
            animal: u8,
            child: State,
            lo: i8,
            hi: i8,
        }
        let mut acts: Vec<(i8, Act)> = root
            .legal_actions()
            .map(|a| {
                let mut child = root;
                child.apply(a);
                let animal = crate::action::decode(a).animal;
                (-swing(&root, &child), Act { animal, child, lo: 0, hi: 2 })
            })
            .collect();
        acts.sort_by_key(|(neg_swing, _)| *neg_swing);
        let mut out = Vec::new();
        for h in hands(u[p], k) {
            let mut best = -1i8;
            for (_, act) in acts.iter_mut().filter(|(_, act)| h >> act.animal & 1 != 0) {
                if best == 2 {
                    break;
                }
                if act.hi <= best {
                    continue;
                }
                let v = if act.lo == act.hi {
                    act.lo
                } else {
                    // Only whether it beats `best` matters: window (best, 3) for the mover.
                    let r = 2 - self.search(&act.child, -1, 2 - best);
                    if r > best {
                        (act.lo, act.hi) = (r, r);
                    } else {
                        act.hi = act.hi.min(r);
                    }
                    r
                };
                best = best.max(v);
            }
            assert!(best >= 0, "hand {h:#x} has no legal action");
            out.push((h, to_p0(best)));
        }
        out
    }

    #[inline]
    fn index(&self, key: u64) -> usize {
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> self.shift) as usize
    }

    fn store(&mut self, key: u64, lo: i8, hi: i8, best: u8, depth: u8) {
        let i = self.index(key);
        let e = Entry { key, lo, hi, best, depth };
        let b = &mut self.table[i];
        if b[0].key == key {
            b[0] = e;
        } else if b[1].key == key {
            b[1] = e;
        } else if depth >= b[0].depth {
            b[1] = b[0];
            b[0] = e;
        } else {
            b[1] = e;
        }
    }

    /// Fail-soft negamax: the mover's value `v` in {0, 1, 2}, given `alpha < beta`. A result `r <= alpha` means
    /// `v <= r`, `r >= beta` means `v >= r`, anything between is exact.
    fn search(&mut self, s: &State, mut alpha: i8, mut beta: i8) -> i8 {
        self.stats.nodes += 1;
        let me = s.to_move as usize;
        let diff = s.scores[me] as i32 - s.scores[1 - me] as i32;
        if s.is_over() {
            return sign3(diff);
        }
        let gain = max_gains(s);
        let mut lo = sign3(diff - gain[1 - me]);
        let mut hi = sign3(diff + gain[me]);
        if lo >= beta || hi <= alpha || lo == hi {
            return if lo >= beta { lo } else { hi };
        }
        let key = key(s);
        let mut tt_best = 0;
        {
            let b = &self.table[self.index(key)];
            let hit = if b[0].key == key {
                Some(b[0])
            } else if b[1].key == key {
                Some(b[1])
            } else {
                None
            };
            if let Some(e) = hit {
                lo = lo.max(e.lo);
                hi = hi.min(e.hi);
                tt_best = e.best;
                if lo >= beta || hi <= alpha || lo == hi {
                    self.stats.tt_cutoffs += 1;
                    return if lo >= beta { lo } else { hi };
                }
            }
        }
        // Values outside [lo, hi] are impossible, so the window can shrink to them.
        alpha = alpha.max(lo - 1);
        beta = beta.min(hi + 1);
        let alpha0 = alpha;

        let legal = s.legal_actions();
        debug_assert!(!legal.is_empty(), "no legal action in an unfinished game");
        let mut order = [(0i8, 0u8); NUM_ACTIONS];
        let mut n = 0;
        for a in legal {
            let prio = if a + 1 == tt_best {
                i8::MAX
            } else {
                let mut c = *s;
                c.apply(a);
                swing(s, &c)
            };
            // Insertion sort: higher priority first, ties in action order.
            let mut j = n;
            while j > 0 && order[j - 1].0 < prio {
                order[j] = order[j - 1];
                j -= 1;
            }
            order[j] = (prio, a);
            n += 1;
        }

        let mut best = -1i8;
        let mut best_a = 0u8;
        for &(_, a) in &order[..n] {
            let mut c = *s;
            c.apply(a);
            let v = 2 - self.search(&c, 2 - beta, 2 - alpha);
            if v > best {
                best = v;
                best_a = a;
                if v > alpha {
                    alpha = v;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        let (elo, ehi) = if best <= alpha0 {
            (lo, hi.min(best))
        } else if best >= beta {
            (lo.max(best), hi)
        } else {
            (best, best)
        };
        let depth = (s.hands[0].count_ones() + s.hands[1].count_ones()) as u8 + s.queue_len;
        self.store(key, elo, ehi, best_a + 1, depth);
        best
    }
}

/// Upper bounds on the points each player can still score, for a position with both decks empty.
///
/// Points can only come from cards in hands or the queue, two per five-animal check. Each check takes 3 cards out
/// of play and leaves 2 in the queue, and none happens once fewer than 5 cards remain, so with `n` cards left at
/// most `(n - 2) / 3` checks remain: a player gains at most their best `2 (n - 2) / 3` cards' points.
#[inline]
fn max_gains(s: &State) -> [i32; 2] {
    // Animals worth 4, 3 and 2 points.
    const BY_POINTS: [u16; 3] = {
        let mut t = [0u16; 3];
        let mut a = 0;
        while a < NUM_ANIMALS {
            assert!(POINTS[a] >= 2 && POINTS[a] <= 4);
            t[4 - POINTS[a] as usize] |= 1 << a;
            a += 1;
        }
        t
    };
    let mut cards = s.hands;
    for &c in s.queue() {
        cards[owner(c) as usize] |= 1 << animal(c);
    }
    let n = (cards[0].count_ones() + cards[1].count_ones()) as i32;
    let entries = if n >= 5 { 2 * ((n - 2) / 3) } else { 0 };
    cards.map(|mine| {
        let (mut left, mut gain) = (entries, 0);
        for (i, mask) in BY_POINTS.iter().enumerate() {
            let take = left.min((mine & mask).count_ones() as i32);
            gain += take * (4 - i as i32);
            left -= take;
        }
        gain
    })
}

/// Points the last move sent to the mover's bar minus the opponent's (`before` to move, `after` one move later).
#[inline]
fn swing(before: &State, after: &State) -> i8 {
    let me = before.to_move as usize;
    let gain = |p: usize| after.scores[p] as i8 - before.scores[p] as i8;
    gain(me) - gain(1 - me)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hands::Variant;
    use crate::rng::Rng;

    fn public_position(rng: &mut Rng, variant: Variant, extra: usize) -> State {
        loop {
            let start = rng.below(2) as u8;
            let mut s = variant.deal(rng, start);
            while s.deck_len != [0, 0] && !s.is_over() {
                let legal = s.legal_actions();
                s.apply(legal.nth_action(rng.below(legal.len())));
            }
            for _ in 0..extra {
                if s.is_over() {
                    break;
                }
                let legal = s.legal_actions();
                s.apply(legal.nth_action(rng.below(legal.len())));
            }
            if !s.is_over() {
                return s;
            }
        }
    }

    #[test]
    fn solver_equals_minimax() {
        let mut rng = Rng::new(1);
        let mut solver = PiSolver::new(12);
        for i in 0..300 {
            let s = public_position(&mut rng, Variant::REAL, i % 3);
            assert_eq!(solver.value(&s), minimax(&s, &mut 0), "position {s:?}");
            if i % 2 == 0 {
                solver.clear();
            }
        }
    }

    #[test]
    fn tiny_table_still_exact() {
        let mut rng = Rng::new(2);
        let mut solver = PiSolver::new(1);
        for _ in 0..100 {
            let s = public_position(&mut rng, Variant::REAL, 0);
            assert_eq!(solver.value(&s), minimax(&s, &mut 0));
        }
    }

    #[test]
    fn mover_value_is_consistent() {
        let mut rng = Rng::new(3);
        let mut solver = PiSolver::new(10);
        for _ in 0..100 {
            let s = public_position(&mut rng, Variant::REAL, 1);
            let v = solver.value(&s);
            let m = solver.value_for_mover(&s);
            assert_eq!(if s.to_move == 0 { m } else { 2 - m }, v);
        }
    }
}
