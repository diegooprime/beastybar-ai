//! Network inputs, layout v1: the policy observation of the player to move, and privileged critic features.
//!
//! The **observation** uses only what the player to move knows: their own hand, their own deck as a set (never its
//! order), everything public (queue, bar, THAT'S IT, scores, deck sizes, who started, the opponent's unplayed animals
//! as a set, and the opponent's hand once their deck is empty), and the public move history. "Mine" and "theirs"
//! are relative to the player to move. All entries are 0/1 except the scalars (scores/20 can exceed 1: a player's animals total 36 points).
//!
//! ```text
//! offset  size  content
//!    0     12   my hand                                  (animal id = strength - 1)
//!   12     12   my deck (unplayed animals not in my hand, as a set)
//!   24     12   opponent's unplayed animals (hand + deck)
//!   36     12   opponent's hand if public (their deck is empty), else 0
//!   48     12   my animals in the bar
//!   60     12   opponent's animals in the bar
//!   72     12   my animals in THAT'S IT
//!   84     12   opponent's animals in THAT'S IT
//!   96     56   queue, front first: 4 slots x (animal one-hot 12, mine, theirs); empty slots 0
//!  152     25   ply one-hot (ply k = k moves already played; the starting player moves at even plies)
//!  177      8   scalars: my score/20, opp score/20, (my - opp)/20, my deck len/8, opp deck len/8, i started,
//!               queue len/4, cards left in hands + decks + queue/24
//!  185   2256   history: 24 slots, most recent move first; slot = (by me, by opponent, action one-hot 36,
//!               queue before that move as 4 x 14 like above) = 94; unused slots 0
//! 2441          OBS_DIM
//! ```
//!
//! The **privileged** features are hidden information, for critics only (never for a policy):
//! `[0:12]` opponent's hand, `[12:24]` my next draw one-hot (top of my deck), `[24:36]` opponent's next draw one-hot.

use crate::animal::{NUM_ANIMALS, animal, owner};
use crate::state::State;

pub const OBS_VERSION: u32 = 1;

pub const MY_HAND: usize = 0;
pub const MY_DECK: usize = 12;
pub const OPP_UNPLAYED: usize = 24;
pub const OPP_HAND_PUBLIC: usize = 36;
pub const MY_BAR: usize = 48;
pub const OPP_BAR: usize = 60;
pub const MY_THATS_IT: usize = 72;
pub const OPP_THATS_IT: usize = 84;
pub const QUEUE: usize = 96;
pub const QUEUE_SLOTS: usize = 4;
/// Animal one-hot (12), mine, theirs.
pub const QUEUE_SLOT: usize = NUM_ANIMALS + 2;
pub const PLY: usize = QUEUE + QUEUE_SLOTS * QUEUE_SLOT;
/// Ply one-hot slots 0..=24 (decisions happen at plies 0..=23).
pub const PLY_SLOTS: usize = 25;
pub const SCALARS: usize = PLY + PLY_SLOTS;
pub const NUM_SCALARS: usize = 8;
pub const HISTORY: usize = SCALARS + NUM_SCALARS;
/// Every card is played at most once, so a game has at most 24 moves.
pub const MAX_PLIES: usize = 24;
pub const HISTORY_SLOTS: usize = MAX_PLIES;
/// Within a history slot: by me, by opponent, action one-hot, queue before the move.
pub const HIST_BY_ME: usize = 0;
pub const HIST_BY_OPP: usize = 1;
pub const HIST_ACTION: usize = 2;
pub const HIST_QUEUE: usize = HIST_ACTION + crate::action::NUM_ACTIONS;
pub const HISTORY_SLOT: usize = HIST_QUEUE + QUEUE_SLOTS * QUEUE_SLOT;
pub const OBS_DIM: usize = HISTORY + HISTORY_SLOTS * HISTORY_SLOT;

pub const PRIV_OPP_HAND: usize = 0;
pub const PRIV_MY_DRAW: usize = 12;
pub const PRIV_OPP_DRAW: usize = 24;
pub const PRIV_DIM: usize = 36;

const _: () = assert!(PLY == 152 && SCALARS == 177 && HISTORY == 185 && HISTORY_SLOT == 94 && OBS_DIM == 2441);

/// Named `(offset, size)` blocks of the observation, exported to Python.
pub const OBS_LAYOUT: [(&str, usize, usize); 14] = [
    ("my_hand", MY_HAND, NUM_ANIMALS),
    ("my_deck", MY_DECK, NUM_ANIMALS),
    ("opp_unplayed", OPP_UNPLAYED, NUM_ANIMALS),
    ("opp_hand_public", OPP_HAND_PUBLIC, NUM_ANIMALS),
    ("my_bar", MY_BAR, NUM_ANIMALS),
    ("opp_bar", OPP_BAR, NUM_ANIMALS),
    ("my_thats_it", MY_THATS_IT, NUM_ANIMALS),
    ("opp_thats_it", OPP_THATS_IT, NUM_ANIMALS),
    ("queue", QUEUE, QUEUE_SLOTS * QUEUE_SLOT),
    ("ply", PLY, PLY_SLOTS),
    ("scalars", SCALARS, NUM_SCALARS),
    ("history", HISTORY, HISTORY_SLOTS * HISTORY_SLOT),
    // Privileged features, in their own vector.
    ("priv_opp_hand", PRIV_OPP_HAND, NUM_ANIMALS),
    ("priv_draws", PRIV_MY_DRAW, 2 * NUM_ANIMALS),
];

/// One move of the public history: who moved, the action, and the queue just before it (at most 4 cards between
/// turns).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Ply {
    pub player: u8,
    pub action: u8,
    pub queue: [u8; QUEUE_SLOTS],
    pub queue_len: u8,
}

/// The public move history of one game, oldest first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct History {
    plies: [Ply; MAX_PLIES],
    len: u8,
}

impl History {
    /// Records `action`, about to be played in `before`. Call it before [`State::apply`].
    pub fn push(&mut self, before: &State, action: u8) {
        let q = before.queue();
        assert!(q.len() <= QUEUE_SLOTS, "a queue holds at most 4 animals between turns");
        assert!((self.len as usize) < MAX_PLIES, "a game has at most {MAX_PLIES} moves");
        let mut queue = [0; QUEUE_SLOTS];
        queue[..q.len()].copy_from_slice(q);
        self.plies[self.len as usize] = Ply { player: before.to_move, action, queue, queue_len: q.len() as u8 };
        self.len += 1;
    }

    pub fn clear(&mut self) {
        *self = History::default();
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Moves in the order played.
    #[inline]
    pub fn plies(&self) -> &[Ply] {
        &self.plies[..self.len as usize]
    }
}

/// Moves already played: every move plays exactly one card from a hand, and draws keep hand + deck constant.
#[inline]
pub fn ply(s: &State) -> usize {
    let left: u32 = (0..2).map(|p| s.hands[p].count_ones() + s.deck_len[p] as u32).sum();
    2 * NUM_ANIMALS - left as usize
}

/// Animal bitmask of player `p`'s deck.
#[inline]
pub fn deck_mask(s: &State, p: usize) -> u16 {
    s.deck(p).iter().fold(0, |m, &a| m | 1 << a)
}

#[inline]
fn set_animals(out: &mut [f32], offset: usize, mut mask: u16) {
    while mask != 0 {
        out[offset + mask.trailing_zeros() as usize] = 1.0;
        mask &= mask - 1;
    }
}

/// A queue (front first, at most 4 cards) as 4 x (animal one-hot, mine, theirs), relative to `me`.
#[inline]
fn set_queue(out: &mut [f32], offset: usize, queue: &[u8], me: u8) {
    for (i, &c) in queue.iter().enumerate() {
        let slot = offset + i * QUEUE_SLOT;
        out[slot + animal(c) as usize] = 1.0;
        out[slot + NUM_ANIMALS + (owner(c) != me) as usize] = 1.0;
    }
}

/// Writes the observation of the player to move in `s` into `out` (length [`OBS_DIM`]), given the game's public
/// history up to `s`. Every entry is written.
pub fn encode_obs(s: &State, h: &History, out: &mut [f32]) {
    out.fill(0.0);
    encode_obs_zeroed(s, h, out);
}

/// [`encode_obs`] for an `out` that is already all zeros.
pub fn encode_obs_zeroed(s: &State, h: &History, out: &mut [f32]) {
    assert_eq!(out.len(), OBS_DIM, "observation buffer length");
    let me = s.to_move;
    let (m, o) = (me as usize, 1 - me as usize);
    let ply = ply(s);
    debug_assert_eq!(ply, h.len(), "history length must equal the ply");

    set_animals(out, MY_HAND, s.hands[m]);
    set_animals(out, MY_DECK, deck_mask(s, m));
    set_animals(out, OPP_UNPLAYED, s.hands[o] | deck_mask(s, o));
    if s.deck_len[o] == 0 {
        set_animals(out, OPP_HAND_PUBLIC, s.hands[o]);
    }
    for (cards, mine, theirs) in [(s.bar, MY_BAR, OPP_BAR), (s.thats_it, MY_THATS_IT, OPP_THATS_IT)] {
        let mut bits = cards;
        while bits != 0 {
            let c = bits.trailing_zeros() as u8;
            bits &= bits - 1;
            let base = if owner(c) == me { mine } else { theirs };
            out[base + animal(c) as usize] = 1.0;
        }
    }
    set_queue(out, QUEUE, s.queue(), me);
    out[PLY + ply.min(PLY_SLOTS - 1)] = 1.0;

    let (my_score, opp_score) = (s.scores[m] as f32, s.scores[o] as f32);
    let left: u32 = (0..2).map(|p| s.hands[p].count_ones() + s.deck_len[p] as u32).sum::<u32>() + s.queue_len as u32;
    let scalars = [
        my_score / 20.0,
        opp_score / 20.0,
        (my_score - opp_score) / 20.0,
        s.deck_len[m] as f32 / 8.0,
        s.deck_len[o] as f32 / 8.0,
        (s.start == me) as u8 as f32,
        s.queue_len as f32 / 4.0,
        left as f32 / 24.0,
    ];
    out[SCALARS..SCALARS + NUM_SCALARS].copy_from_slice(&scalars);

    for (j, p) in h.plies().iter().rev().enumerate() {
        let slot = HISTORY + j * HISTORY_SLOT;
        out[slot + if p.player == me { HIST_BY_ME } else { HIST_BY_OPP }] = 1.0;
        out[slot + HIST_ACTION + p.action as usize] = 1.0;
        set_queue(out, slot + HIST_QUEUE, &p.queue[..p.queue_len as usize], me);
    }
}

/// Writes the privileged (hidden-information) features of the player to move into `out` (length [`PRIV_DIM`]).
pub fn encode_privileged(s: &State, out: &mut [f32]) {
    assert_eq!(out.len(), PRIV_DIM, "privileged buffer length");
    out.fill(0.0);
    let (m, o) = (s.to_move as usize, 1 - s.to_move as usize);
    set_animals(out, PRIV_OPP_HAND, s.hands[o]);
    if let Some(&a) = s.deck(m).first() {
        out[PRIV_MY_DRAW + a as usize] = 1.0;
    }
    if let Some(&a) = s.deck(o).first() {
        out[PRIV_OPP_DRAW + a as usize] = 1.0;
    }
}

/// The legal-action mask of the player to move as 0/1 bytes (length [`crate::NUM_ACTIONS`]).
pub fn encode_mask(s: &State, out: &mut [u8]) {
    let legal = s.legal_actions();
    for (a, x) in out.iter_mut().enumerate() {
        *x = legal.contains(a as u8) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animal::{ALL_ANIMALS, card};
    use crate::rng::Rng;
    use crate::state::DECK_CAP;

    /// Random games with their histories: every position before a move, with the history up to it.
    fn positions(games: usize, seed: u64) -> Vec<(State, History)> {
        let mut rng = Rng::new(seed);
        let mut out = Vec::new();
        for g in 0..games {
            let mut s = State::deal(&mut rng, (g % 2) as u8);
            let mut h = History::default();
            while !s.is_over() {
                out.push((s, h));
                let legal = s.legal_actions();
                let a = legal.nth_action(rng.below(legal.len()));
                h.push(&s, a);
                s.apply(a);
            }
            out.push((s, h));
        }
        out
    }

    fn obs(s: &State, h: &History) -> Vec<f32> {
        let mut v = vec![f32::NAN; OBS_DIM];
        encode_obs(s, h, &mut v);
        v
    }

    fn mask_of(v: &[f32], offset: usize) -> u16 {
        (0..NUM_ANIMALS).filter(|&a| v[offset + a] == 1.0).fold(0, |m, a| m | 1 << a)
    }

    /// Same public information and the same hand for the player to move, with the hidden information redrawn: the
    /// mover's deck reshuffled, the opponent's unplayed animals re-split between hand and deck, and their deck
    /// reshuffled.
    fn redraw_hidden(s: &State, rng: &mut Rng) -> State {
        let mut v = *s;
        let (m, o) = (s.to_move as usize, 1 - s.to_move as usize);
        let mut own: Vec<u8> = s.deck(m).to_vec();
        rng.shuffle(&mut own);
        v.decks[m] = [0; DECK_CAP];
        v.decks[m][..own.len()].copy_from_slice(&own);

        let hand_size = s.hands[o].count_ones() as usize;
        let unplayed = s.hands[o] | deck_mask(s, o);
        let mut animals: Vec<u8> = (0..NUM_ANIMALS as u8).filter(|&a| unplayed >> a & 1 != 0).collect();
        rng.shuffle(&mut animals);
        v.hands[o] = animals[..hand_size].iter().fold(0, |h, &a| h | 1 << a);
        v.decks[o] = [0; DECK_CAP];
        v.decks[o][..animals.len() - hand_size].copy_from_slice(&animals[hand_size..]);
        v
    }

    #[test]
    fn layout_offsets() {
        assert_eq!((PLY, SCALARS, HISTORY, HISTORY_SLOT, OBS_DIM), (152, 177, 185, 94, 2441));
        let mut end = 0;
        for &(_, off, size) in &OBS_LAYOUT[..12] {
            assert_eq!(off, end, "observation blocks are contiguous");
            end = off + size;
        }
        assert_eq!(end, OBS_DIM);
    }

    #[test]
    fn observation_ignores_hidden_information() {
        let mut rng = Rng::new(7);
        let mut changed_hidden = 0;
        for (s, h) in positions(300, 1) {
            if s.is_over() {
                continue;
            }
            let base = obs(&s, &h);
            for _ in 0..4 {
                let v = redraw_hidden(&s, &mut rng);
                assert_eq!(v.hands[s.to_move as usize], s.hands[s.to_move as usize]);
                if v != s {
                    changed_hidden += 1;
                }
                assert_eq!(obs(&v, &h), base, "observation depends on hidden information");
                // The privileged features are the hidden information.
                let (mut p1, mut p2) = (vec![0.0; PRIV_DIM], vec![0.0; PRIV_DIM]);
                encode_privileged(&s, &mut p1);
                encode_privileged(&v, &mut p2);
                if v.hands[1 - s.to_move as usize] != s.hands[1 - s.to_move as usize] {
                    assert_ne!(p1, p2);
                }
            }
        }
        assert!(changed_hidden > 1000, "the test must actually perturb hidden information");
    }

    /// Decodes the observation back into what the mover knows and compares it with the state and history.
    #[test]
    fn encoder_round_trip() {
        for (s, h) in positions(300, 2) {
            let v = obs(&s, &h);
            assert!(v.iter().all(|x| x.is_finite() && (-2.0..=2.0).contains(x)));
            let (m, o) = (s.to_move as usize, 1 - s.to_move as usize);
            let me = s.to_move;
            assert_eq!(mask_of(&v, MY_HAND), s.hands[m]);
            assert_eq!(mask_of(&v, MY_DECK), deck_mask(&s, m));
            assert_eq!(mask_of(&v, OPP_UNPLAYED), s.hands[o] | deck_mask(&s, o));
            assert_eq!(mask_of(&v, OPP_HAND_PUBLIC), if s.deck_len[o] == 0 { s.hands[o] } else { 0 });
            // Every animal of each player is in exactly one place: hand, deck, bar, THAT'S IT or queue.
            let queue_mine = s.queue().iter().filter(|&&c| owner(c) == me).fold(0u16, |x, &c| x | 1 << animal(c));
            let queue_theirs = s.queue().iter().filter(|&&c| owner(c) != me).fold(0u16, |x, &c| x | 1 << animal(c));
            let mine: Vec<u16> = [MY_HAND, MY_DECK, MY_BAR, MY_THATS_IT].iter().map(|&b| mask_of(&v, b)).collect();
            assert_eq!(mine.iter().fold(queue_mine, |x, &p| x | p), ALL_ANIMALS);
            assert_eq!(mine.iter().map(|p| p.count_ones()).sum::<u32>() + queue_mine.count_ones(), 12);
            let theirs = [OPP_UNPLAYED, OPP_BAR, OPP_THATS_IT].iter().map(|&b| mask_of(&v, b)).collect::<Vec<_>>();
            assert_eq!(theirs.iter().fold(queue_theirs, |x, &p| x | p), ALL_ANIMALS);
            // Queue slots decode to the queue's cards.
            let mut queue = Vec::new();
            for i in 0..QUEUE_SLOTS {
                let slot = &v[QUEUE + i * QUEUE_SLOT..QUEUE + (i + 1) * QUEUE_SLOT];
                let total: f32 = slot.iter().sum();
                if total == 0.0 {
                    continue;
                }
                assert_eq!(total, 2.0);
                let a = slot[..NUM_ANIMALS].iter().position(|&x| x == 1.0).unwrap() as u8;
                let theirs = slot[NUM_ANIMALS + 1] == 1.0;
                queue.push(card(if theirs { 1 - me } else { me }, a));
            }
            assert_eq!(queue, s.queue());
            // Ply, scalars.
            let ply_onehot = &v[PLY..PLY + PLY_SLOTS];
            assert_eq!(ply_onehot.iter().sum::<f32>(), 1.0);
            assert_eq!(ply_onehot.iter().position(|&x| x == 1.0).unwrap(), h.len());
            assert_eq!(ply(&s), h.len());
            assert_eq!((v[SCALARS] * 20.0).round() as u8, s.scores[m]);
            assert_eq!((v[SCALARS + 1] * 20.0).round() as u8, s.scores[o]);
            assert_eq!(v[SCALARS + 5] == 1.0, s.start == me);
            assert_eq!((v[SCALARS + 6] * 4.0).round() as u8, s.queue_len);
            // History: most recent first; the mover parity follows the ply convention.
            for (j, p) in h.plies().iter().rev().enumerate() {
                let slot = &v[HISTORY + j * HISTORY_SLOT..HISTORY + (j + 1) * HISTORY_SLOT];
                let k = h.len() - 1 - j;
                assert_eq!(p.player, s.start ^ (k % 2) as u8, "starting player moves at even plies");
                assert_eq!(slot[HIST_BY_ME] == 1.0, p.player == me);
                assert_eq!(slot[HIST_BY_ME] + slot[HIST_BY_OPP], 1.0);
                let acts = &slot[HIST_ACTION..HIST_QUEUE];
                assert_eq!(acts.iter().sum::<f32>(), 1.0);
                assert_eq!(acts.iter().position(|&x| x == 1.0).unwrap(), p.action as usize);
                assert_eq!(slot[HIST_QUEUE..].iter().sum::<f32>(), 2.0 * p.queue_len as f32);
            }
            for j in h.len()..HISTORY_SLOTS {
                assert!(v[HISTORY + j * HISTORY_SLOT..HISTORY + (j + 1) * HISTORY_SLOT].iter().all(|&x| x == 0.0));
            }
        }
    }

    #[test]
    fn history_replays_the_game() {
        // Replaying the recorded actions from the deal reproduces the position and every recorded queue.
        let mut rng = Rng::new(3);
        for g in 0..200 {
            let deal = State::deal(&mut rng, (g % 2) as u8);
            let mut s = deal;
            let mut h = History::default();
            while !s.is_over() {
                let legal = s.legal_actions();
                let a = legal.nth_action(rng.below(legal.len()));
                h.push(&s, a);
                s.apply(a);
            }
            let mut r = deal;
            for p in h.plies() {
                assert_eq!(p.player, r.to_move);
                assert_eq!(&p.queue[..p.queue_len as usize], r.queue());
                r.apply(p.action);
            }
            assert_eq!(r, s);
        }
    }

    #[test]
    fn privileged_features() {
        for (s, _) in positions(50, 4) {
            let mut p = vec![f32::NAN; PRIV_DIM];
            encode_privileged(&s, &mut p);
            let (m, o) = (s.to_move as usize, 1 - s.to_move as usize);
            assert_eq!(mask_of(&p, PRIV_OPP_HAND), s.hands[o]);
            assert_eq!(mask_of(&p, PRIV_MY_DRAW), s.deck(m).first().map_or(0, |&a| 1 << a));
            assert_eq!(mask_of(&p, PRIV_OPP_DRAW), s.deck(o).first().map_or(0, |&a| 1 << a));
        }
    }
}
