//! The engine for the site (site/bb.wasm): one game held in the module, driven through a small C ABI.
//! JavaScript side: site/engine.js. Nothing here changes the rules or the agents: positions are `bbcore::State`,
//! players see only `bbcore::view::view`, and the search is `bbcore::search::Search`.
//!
//! State layout written by `bb_state` / `bb_preview` (75 bytes at `bb_out`):
//!   0 to_move, 1 start, 2 over, 3 score0, 4 score1, 5 queue_len, 6..11 queue (card ids, front first),
//!   11..15 hands (two u16 animal masks, little-endian), 15 deck_len0, 16 deck_len1,
//!   17 bar_len, 18..34 bar (arrival order), 34 thats_it_len, 35..59 thats_it (arrival order),
//!   59..67 deck0, 67..75 deck1 (animal ids, top first).
//! Card id = owner * 12 + animal; animal id = strength - 1.

use bbcore::agents::Spec;
use bbcore::search::Search;
use bbcore::view::view;
use bbcore::{ActionSet, NUM_ACTIONS, Rng, State};
use std::sync::Mutex;

const OUT: usize = 75;

struct Game {
    state: State,
    history: Vec<u8>,
    past: Vec<State>,
}

static GAME: Mutex<Option<Game>> = Mutex::new(None);
static BUF: Mutex<[u8; OUT]> = Mutex::new([0; OUT]);
static INPUT: Mutex<[u8; 16]> = Mutex::new([0; 16]);
static VALUES: Mutex<[f32; NUM_ACTIONS]> = Mutex::new([0.0; NUM_ACTIONS]);

fn seed(hi: u32, lo: u32) -> u64 {
    (hi as u64) << 32 | lo as u64
}

fn write(s: &State) -> u32 {
    let mut b = BUF.lock().unwrap();
    b.fill(0);
    b[0] = s.to_move;
    b[1] = s.start;
    b[2] = s.is_over() as u8;
    b[3] = s.scores[0];
    b[4] = s.scores[1];
    b[5] = s.queue_len;
    b[6..6 + s.queue().len()].copy_from_slice(s.queue());
    b[11..13].copy_from_slice(&s.hands[0].to_le_bytes());
    b[13..15].copy_from_slice(&s.hands[1].to_le_bytes());
    b[15] = s.deck_len[0];
    b[16] = s.deck_len[1];
    let bar = s.bar_order();
    b[17] = bar.len() as u8;
    b[18..18 + bar.len()].copy_from_slice(bar);
    let out = s.thats_it_order();
    b[34] = out.len() as u8;
    b[35..35 + out.len()].copy_from_slice(out);
    b[59..59 + s.deck(0).len()].copy_from_slice(s.deck(0));
    b[67..67 + s.deck(1).len()].copy_from_slice(s.deck(1));
    OUT as u32
}

fn with<T>(f: impl FnOnce(&mut Game) -> T) -> T {
    let mut g = GAME.lock().unwrap();
    f(g.as_mut().expect("bb_new first"))
}

/// A new game: a uniformly random deal from the seed; `start` moves first.
#[unsafe(no_mangle)]
pub extern "C" fn bb_new(seed_hi: u32, seed_lo: u32, start: u32) {
    let mut rng = Rng::new(seed(seed_hi, seed_lo));
    let state = State::deal(&mut rng, (start & 1) as u8);
    *GAME.lock().unwrap() = Some(Game { state, history: Vec::with_capacity(24), past: Vec::with_capacity(24) });
}

/// Where `bb_setup` reads: 0 player to move, 1 queue length, 2..6 queue (card ids, front first),
/// 6..10 hands (two u16 animal masks, little-endian).
#[unsafe(no_mangle)]
pub extern "C" fn bb_in() -> *mut u8 {
    INPUT.lock().unwrap().as_mut_ptr()
}

/// A constructed position (for the rules demos): the queue and hands at `bb_in`; every animal a player has
/// neither in hand nor in the queue goes to their deck, weakest on top (at most 8).
#[unsafe(no_mangle)]
pub extern "C" fn bb_setup() {
    let i = *INPUT.lock().unwrap();
    let to_move = i[0] & 1;
    let n = (i[1] as usize).min(4);
    let hands = [u16::from_le_bytes([i[6], i[7]]) & 0xFFF, u16::from_le_bytes([i[8], i[9]]) & 0xFFF];
    let mut decks = [[0u8; 8]; 2];
    let mut lens = [0usize; 2];
    for p in 0..2 {
        for a in 0..12u8 {
            let queued = i[2..2 + n].contains(&(p as u8 * 12 + a));
            if hands[p] >> a & 1 == 0 && !queued && lens[p] < 8 {
                decks[p][lens[p]] = a;
                lens[p] += 1;
            }
        }
    }
    let mut state = State::new(to_move, hands, [&decks[0][..lens[0]], &decks[1][..lens[1]]]);
    state.queue[..n].copy_from_slice(&i[2..2 + n]);
    state.queue_len = n as u8;
    *GAME.lock().unwrap() = Some(Game { state, history: Vec::new(), past: Vec::new() });
}

/// Where `bb_state` and `bb_preview` write.
#[unsafe(no_mangle)]
pub extern "C" fn bb_out() -> *const u8 {
    BUF.lock().unwrap().as_ptr()
}

/// Where `bb_think` writes one value per action (f32, 36 of them; NaN for illegal actions).
#[unsafe(no_mangle)]
pub extern "C" fn bb_values() -> *const f32 {
    VALUES.lock().unwrap().as_ptr()
}

/// Writes the current position; returns its length in bytes.
#[unsafe(no_mangle)]
pub extern "C" fn bb_state() -> u32 {
    with(|g| write(&g.state))
}

/// Legal actions as a 36-bit mask: low 32 bits, then the high 4.
#[unsafe(no_mangle)]
pub extern "C" fn bb_legal_lo() -> u32 {
    with(|g| g.state.legal_actions().0 as u32)
}

#[unsafe(no_mangle)]
pub extern "C" fn bb_legal_hi() -> u32 {
    with(|g| (g.state.legal_actions().0 >> 32) as u32)
}

/// Writes the position `action` would lead to (the game is unchanged); 0 if the action is illegal.
#[unsafe(no_mangle)]
pub extern "C" fn bb_preview(action: u32) -> u32 {
    with(|g| {
        if action >= NUM_ACTIONS as u32 || !g.state.legal_actions().contains(action as u8) {
            return 0;
        }
        let mut s = g.state;
        s.apply(action as u8);
        write(&s)
    })
}

/// Plays `action`; 1 if it was legal and played, else 0.
#[unsafe(no_mangle)]
pub extern "C" fn bb_apply(action: u32) -> u32 {
    with(|g| {
        if action >= NUM_ACTIONS as u32 || !g.state.legal_actions().contains(action as u8) {
            return 0;
        }
        g.past.push(g.state);
        g.state.apply(action as u8);
        g.history.push(action as u8);
        1
    })
}

/// Takes back the last move; 1 if there was one.
#[unsafe(no_mangle)]
pub extern "C" fn bb_undo() -> u32 {
    with(|g| match g.past.pop() {
        Some(s) => {
            g.state = s;
            g.history.pop();
            1
        }
        None => 0,
    })
}

/// The move of an agent for the player to move, who sees only their own view of the position.
/// kind 0: random, 1: greedy, 2: PIMC with `worlds` sampled worlds searched `depth` plies (`pimc:w=W,d=D`).
/// For PIMC the summed value of every legal action is left at `bb_values`.
#[unsafe(no_mangle)]
pub extern "C" fn bb_think(kind: u32, worlds: u32, depth: u32, seed_hi: u32, seed_lo: u32) -> u32 {
    with(|g| {
        let mut rng = Rng::new(seed(seed_hi, seed_lo));
        let mover = g.state.to_move;
        let seen = view(&g.state, mover, &mut rng);
        let mut values = VALUES.lock().unwrap();
        values.fill(f32::NAN);
        let action = match kind {
            0 => Spec::Random.build().act(&seen, &g.history, &mut rng),
            1 => Spec::Greedy.build().act(&seen, &g.history, &mut rng),
            _ => pimc(&seen, worlds.max(1), depth.max(1), &mut rng, &mut values),
        };
        action as u32
    })
}

/// `bbcore::agents::Pimc` for a view, also reporting what it summed: same worlds, same search, same tie-break.
fn pimc(seen: &State, worlds: u32, depth: u32, rng: &mut Rng, out: &mut [f32; NUM_ACTIONS]) -> u8 {
    let legal = seen.legal_actions();
    if legal.len() == 1 {
        let a = legal.nth_action(0);
        out[a as usize] = 0.0;
        return a;
    }
    let mut search = Search::default();
    let mut sum = [0.0f32; NUM_ACTIONS];
    let me = seen.to_move as usize;
    if seen.deck_len[me] <= 1 && seen.deck_len[1 - me] == 0 {
        // Perfect information for the mover: one world, searched to the end.
        sum = search.root_values(seen, 0).values;
    } else {
        for _ in 0..worlds {
            let world = view(seen, seen.to_move, rng);
            let r = search.root_values(&world, depth);
            for a in legal {
                sum[a as usize] += r.values[a as usize] / worlds as f32;
            }
        }
    }
    for a in legal {
        out[a as usize] = sum[a as usize];
    }
    let best = legal.map(|a| sum[a as usize]).fold(f32::NEG_INFINITY, f32::max);
    let ties = ActionSet(legal.filter(|&a| sum[a as usize] == best).fold(0, |set, a| set | 1 << a));
    ties.nth_action(rng.below(ties.len()))
}
