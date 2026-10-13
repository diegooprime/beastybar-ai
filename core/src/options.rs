//! Option features: what each animal would do if played now (inputs of the PBS value network, bbar/value/pbs.py v2).
//!
//! For a public state (its key, `markov.rs`), each player `p` (the player to move first, then the other) and each
//! animal `p` still has unplayed: the immediate effect of `p` playing it on this queue; for the player not to move,
//! as if it were their turn (a counterfactual). Every legal parameter choice (parrot target, kangaroo hop, chameleon
//! copy with its parameter) is played by the engine itself (`State::apply` on the public position: play, act,
//! recurring actions, five-animal check; decks are empty, so nothing is drawn) and the best and the worst choice for
//! `p` are summarized. "self" is `p`, "opp" the other player; points are the animals' points (RULES.md).
//!
//! Choices are ranked for `p` lexicographically by (self's minus opp's points entering the bar, opp's minus self's
//! points bounced, self's minus opp's points in the front two of the resulting queue), ties to the lower action
//! index. Bounced means sent to THAT'S IT this move, by an action or by the five-animal check.
//!
//! One key gives `2 x 12 x F` floats: [player (to move, other)][animal id][feature], 0 for animals the player does
//! not hold. Per (player, animal), [`F`] = 2 + 2 [`CHOICE`] features:
//!
//! | index         | content                                                                     |
//! |---------------|-----------------------------------------------------------------------------|
//! | 0             | held: 1 if the player has the animal unplayed                               |
//! | 1             | legal parameter choices / 8 (0 if the game is over or none is legal)        |
//! | 2..2+C        | the best choice's fields ([`CHOICE_NAMES`])                                 |
//! | 2+C..2+2C     | the worst choice's fields                                                   |
//!
//! Choice fields, in order: points entering the bar for self, for opp (/ 8); points bounced of self, of opp (/ 8);
//! 1 if a five-animal check happened; resulting queue length / 4; the played card's resulting queue position
//! ((index from the front + 1) / 4, 0 if it left the queue); 1 if it entered the bar; 1 if it was bounced; self's
//! and opp's points in the front two of the resulting queue (/ 8).
//!
//! Only the key's queue, unplayed sets and side to move are read (not the hand sizes or the score).

use crate::action::decode;
use crate::animal::{NUM_ANIMALS, card};
use crate::hands::MASK_POINTS;
use crate::markov;
use crate::state::State;

/// Fields per choice.
pub const CHOICE: usize = 11;
/// Features per (player, animal).
pub const F: usize = 2 + 2 * CHOICE;
/// Floats per key: 2 players x 12 animals x F.
pub const ROW: usize = 2 * NUM_ANIMALS * F;

pub const CHOICE_NAMES: [&str; CHOICE] = [
    "bar_self",
    "bar_opp",
    "bounced_self",
    "bounced_opp",
    "five_check",
    "queue_len",
    "played_pos",
    "played_bar",
    "played_bounced",
    "front2_self",
    "front2_opp",
];

/// The names of the F features of a (player, animal) row.
pub fn feature_names() -> Vec<String> {
    let mut out = vec!["held".to_string(), "choices".to_string()];
    for which in ["best", "worst"] {
        out.extend(CHOICE_NAMES.iter().map(|n| format!("{which}_{n}")));
    }
    out
}

/// Points of `player`'s cards in a card mask.
#[inline]
fn points(cards: u32, player: usize) -> i32 {
    MASK_POINTS[(cards >> (NUM_ANIMALS * player)) as usize & 0xfff] as i32
}

#[derive(Clone, Copy)]
struct Choice {
    fields: [f32; CHOICE],
    rank: [i32; 3],
}

/// The fields of one choice: `before` the move, `after` it, played by `p` with card `played`.
fn choice(before: &State, after: &State, p: usize, played: u8) -> Choice {
    let o = 1 - p;
    let bar = after.bar & !before.bar;
    let out = after.thats_it & !before.thats_it;
    let q = after.queue();
    let front = q[..q.len().min(2)].iter().fold(0u32, |m, &c| m | 1 << c);
    let (bs, bo) = (points(bar, p), points(bar, o));
    let (xs, xo) = (points(out, p), points(out, o));
    let (fs, fo) = (points(front, p), points(front, o));
    let pos = q.iter().position(|&c| c == played).map_or(0.0, |i| (i + 1) as f32 / 4.0);
    Choice {
        fields: [
            bs as f32 / 8.0,
            bo as f32 / 8.0,
            xs as f32 / 8.0,
            xo as f32 / 8.0,
            (bar != 0) as u8 as f32,
            q.len() as f32 / 4.0,
            pos,
            (bar >> played & 1) as f32,
            (out >> played & 1) as f32,
            fs as f32 / 8.0,
            fo as f32 / 8.0,
        ],
        rank: [bs - bo, xo - xs, fs - fo],
    }
}

/// The public position of a key: its queue, every unplayed animal in hand, decks empty, no score.
fn public_position(key: u64) -> (State, usize, [u16; 2]) {
    let k = markov::decode(key);
    let mut s = State::default();
    let n = k.queue_len as usize;
    s.queue[..n].copy_from_slice(&k.queue[..n]);
    s.queue_len = k.queue_len;
    s.hands = k.unplayed;
    (s, k.to_move as usize, k.unplayed)
}

/// The option features of one key into `out` (`ROW` floats; module documentation).
pub fn option_features(key: u64, out: &mut [f32]) {
    assert_eq!(out.len(), ROW, "option_features: output of {ROW} floats");
    out.fill(0.0);
    let (mut s, mover, unplayed) = public_position(key);
    for (rel, p) in [(0, mover), (1, 1 - mover)] {
        s.to_move = p as u8;
        let mut best: [Option<Choice>; NUM_ANIMALS] = [None; NUM_ANIMALS];
        let mut worst: [Option<Choice>; NUM_ANIMALS] = [None; NUM_ANIMALS];
        let mut count = [0u8; NUM_ANIMALS];
        for action in s.legal_actions() {
            let a = decode(action).animal;
            let mut t = s;
            t.apply(action);
            let c = choice(&s, &t, p, card(p as u8, a));
            let a = a as usize;
            count[a] += 1;
            // Strict comparisons: ties go to the lower action index (actions come in increasing order).
            if best[a].is_none_or(|b| c.rank > b.rank) {
                best[a] = Some(c);
            }
            if worst[a].is_none_or(|w| c.rank < w.rank) {
                worst[a] = Some(c);
            }
        }
        for a in 0..NUM_ANIMALS {
            if unplayed[p] >> a & 1 == 0 {
                continue;
            }
            let row = &mut out[(rel * NUM_ANIMALS + a) * F..][..F];
            row[0] = 1.0;
            row[1] = count[a] as f32 / 8.0;
            if let (Some(b), Some(w)) = (best[a], worst[a]) {
                row[2..2 + CHOICE].copy_from_slice(&b.fields);
                row[2 + CHOICE..].copy_from_slice(&w.fields);
            }
        }
    }
}

/// [`option_features`] of many keys into `out` (`keys.len() * ROW` floats) on `threads` threads (0 = every core).
pub fn option_features_many(keys: &[u64], out: &mut [f32], threads: usize) {
    assert_eq!(out.len(), keys.len() * ROW, "option_features_many: output of keys x {ROW} floats");
    let threads = if threads == 0 { std::thread::available_parallelism().map_or(1, |n| n.get()) } else { threads };
    let threads = threads.min(keys.len() / 256).max(1);
    let run = |ks: &[u64], os: &mut [f32]| {
        for (&k, o) in ks.iter().zip(os.chunks_mut(ROW)) {
            option_features(k, o);
        }
    };
    if threads == 1 {
        return run(keys, out);
    }
    let per = keys.len().div_ceil(threads);
    std::thread::scope(|scope| {
        for (ks, os) in keys.chunks(per).zip(out.chunks_mut(per * ROW)) {
            scope.spawn(move || run(ks, os));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animal::{POINTS, animal, owner};
    use crate::hands::{Variant, unplayed};
    use crate::markov::public_key;
    use crate::rng::Rng;

    fn random_state(rng: &mut Rng, plies: usize) -> State {
        let start = rng.below(2) as u8;
        let mut s = Variant::REAL.deal(rng, start);
        for _ in 0..plies {
            if s.is_over() {
                break;
            }
            let legal = s.legal_actions();
            s.apply(legal.nth_action(rng.below(legal.len())));
        }
        s
    }

    /// The fields of a choice from the engine's own bookkeeping on a real position (scores, arrival orders), written
    /// independently of `choice`.
    fn fields_from_play(s: &State, t: &State, p: usize, played: u8) -> ([f32; CHOICE], [i32; 3]) {
        let o = 1 - p;
        let gained = |x: usize| (t.scores[x] - s.scores[x]) as i32;
        let nb = s.thats_it.count_ones() as usize;
        let mut bounced = [0i32; 2];
        for &c in &t.thats_it_order()[nb..] {
            bounced[owner(c) as usize] += POINTS[animal(c) as usize] as i32;
        }
        let mut front = [0i32; 2];
        for &c in t.queue().iter().take(2) {
            front[owner(c) as usize] += POINTS[animal(c) as usize] as i32;
        }
        let in_bar = t.bar_order()[s.bar.count_ones() as usize..].contains(&played);
        let out = t.thats_it_order()[nb..].contains(&played);
        let pos = match t.queue().iter().position(|&c| c == played) {
            Some(i) => (i + 1) as f32 / 4.0,
            None => 0.0,
        };
        let five = t.bar.count_ones() > s.bar.count_ones();
        let f = [
            gained(p) as f32 / 8.0,
            gained(o) as f32 / 8.0,
            bounced[p] as f32 / 8.0,
            bounced[o] as f32 / 8.0,
            five as u8 as f32,
            t.queue_len as f32 / 4.0,
            pos,
            in_bar as u8 as f32,
            out as u8 as f32,
            front[p] as f32 / 8.0,
            front[o] as f32 / 8.0,
        ];
        (f, [gained(p) - gained(o), bounced[o] - bounced[p], front[p] - front[o]])
    }

    /// On real positions (hidden hands, decks, scores, history): for the player to move, each animal in the real
    /// hand, played with every parameter on the real position (`apply` also draws), and for the other player the
    /// same with the turn handed over: best and worst by the documented order equal the key's features. Also checks
    /// the zero rows and the held flags.
    #[test]
    fn features_match_play_on_real_positions() {
        let mut rng = Rng::new(17);
        let mut out = vec![0f32; ROW];
        let mut checked = 0;
        for _ in 0..3000 {
            let plies = rng.below(20);
            let s = random_state(&mut rng, plies);
            option_features(public_key(&s), &mut out);
            let mover = s.to_move as usize;
            for (rel, p) in [(0, mover), (1, 1 - mover)] {
                let mut real = s;
                real.to_move = p as u8;
                let u = unplayed(&s, p);
                for a in 0..NUM_ANIMALS as u8 {
                    let row = &out[(rel * NUM_ANIMALS + a as usize) * F..][..F];
                    assert_eq!(row[0], (u >> a & 1) as f32);
                    if u >> a & 1 == 0 {
                        assert!(row.iter().all(|&x| x == 0.0));
                        continue;
                    }
                    if real.hands[p] >> a & 1 == 0 {
                        continue; // in the deck: not playable in the real position
                    }
                    let mut choices: Vec<([f32; CHOICE], [i32; 3])> = Vec::new();
                    for action in real.legal_actions() {
                        if decode(action).animal != a {
                            continue;
                        }
                        let mut t = real;
                        t.apply(action);
                        choices.push(fields_from_play(&real, &t, p, card(p as u8, a)));
                    }
                    assert_eq!(row[1], choices.len() as f32 / 8.0);
                    if choices.is_empty() {
                        assert!(row[2..].iter().all(|&x| x == 0.0));
                        continue;
                    }
                    // max_by_key keeps the last maximum, min_by_key the first minimum: reverse for the best.
                    let best = choices.iter().rev().max_by_key(|c| c.1).unwrap();
                    let worst = choices.iter().min_by_key(|c| c.1).unwrap();
                    assert_eq!(&row[2..2 + CHOICE], &best.0[..]);
                    assert_eq!(&row[2 + CHOICE..], &worst.0[..]);
                    checked += 1;
                }
            }
        }
        assert!(checked > 10_000, "{checked}");
    }

    #[test]
    fn many_equals_one_by_one() {
        let mut rng = Rng::new(3);
        let keys: Vec<u64> = (0..3000).map(|_| public_key(&random_state(&mut rng, 14))).collect();
        let mut one = vec![0f32; keys.len() * ROW];
        for (k, o) in keys.iter().zip(one.chunks_mut(ROW)) {
            option_features(*k, o);
        }
        for threads in [1, 3, 0] {
            let mut many = vec![1f32; keys.len() * ROW];
            option_features_many(&keys, &mut many, threads);
            assert_eq!(one, many);
        }
        assert_eq!(feature_names().len(), F);
    }
}
