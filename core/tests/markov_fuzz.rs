//! Fuzz test of the Markov key: positions reached by different histories with equal keys must have the same
//! legal actions, the same successor keys for every action, and the same outcome when over.
//!
//! From random roots (random deals of several variants, random play to a random ply) every line of play is
//! explored to a small depth; each position whose key was already seen under this root (necessarily by another
//! history) is compared with the first one. Public positions (both decks empty, e.g. every real position from ply
//! 16 on, or the all-unplayed-in-hand public representation) are compared on [`key`] alone with deterministic
//! successors; positions with hidden hands on [`full_key`] (key plus both hands), with successors over every draw.
//!
//!     cargo test --release --test markov_fuzz                 # default: 30k pairs of distinct states
//!     cargo test --release --test markov_fuzz -- --ignored    # 1M pairs of distinct states

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use bbcore::hands::Variant;
use bbcore::markov::{full_key, key, public, successors};
use bbcore::pisolve::outcome;
use bbcore::{Rng, State};

#[derive(Default, Debug)]
struct Tally {
    nodes: u64,
    /// Positions compared with an earlier position of equal key reached by another history.
    pairs: u64,
    /// ...of which the two `State`s differ (bar or THAT'S IT order, absolute scores, deck order, ...).
    distinct: u64,
    /// ...of which both decks are empty, compared on the 64-bit key alone.
    public_pairs: u64,
    /// Pairs whose futures differ, and the first such difference.
    failures: u64,
    first_failure: Option<String>,
}

type KeyFn = fn(&State) -> u128;

/// The key under test: [`key`] when both decks are empty (it must identify the position alone), else
/// [`full_key`] (key plus the hidden hands).
fn markov(s: &State) -> u128 {
    if s.deck_len == [0, 0] { key(s) as u128 } else { full_key(s) }
}

fn successor_keys(kf: KeyFn, s: &State, action: u8) -> Vec<u128> {
    let mut keys: Vec<u128> = successors(s, action).map(|c| kf(&c)).collect();
    keys.sort_unstable();
    keys
}

/// Same legal actions, same successor keys (over every draw) for every action, same outcome when over.
fn compare(kf: KeyFn, a: &State, b: &State) -> Result<(), String> {
    if a.legal_actions() != b.legal_actions() {
        return Err(format!("legal actions differ:\n{a:?}\n{b:?}"));
    }
    if a.is_over() != b.is_over() || (a.is_over() && outcome(a) != outcome(b)) {
        return Err(format!("end or outcome differs:\n{a:?}\n{b:?}"));
    }
    for action in a.legal_actions() {
        if successor_keys(kf, a, action) != successor_keys(kf, b, action) {
            return Err(format!("successor keys differ after action {action}:\n{a:?}\n{b:?}"));
        }
    }
    Ok(())
}

fn explore(kf: KeyFn, s: &State, depth: usize, seen: &mut HashMap<u128, State>, tally: &mut Tally) {
    tally.nodes += 1;
    match seen.entry(kf(s)) {
        Entry::Occupied(e) => {
            tally.pairs += 1;
            tally.public_pairs += (s.deck_len == [0, 0]) as u64;
            tally.distinct += (e.get() != s) as u64;
            if let Err(msg) = compare(kf, e.get(), s) {
                tally.failures += 1;
                tally.first_failure.get_or_insert(msg);
            }
        }
        Entry::Vacant(e) => {
            e.insert(*s);
        }
    }
    if depth == 0 || s.is_over() {
        return;
    }
    for a in s.legal_actions() {
        let mut c = *s;
        c.apply(a);
        explore(kf, &c, depth - 1, seen, tally);
    }
}

fn random_root(rng: &mut Rng, variant: Variant) -> State {
    let start = rng.below(2) as u8;
    let mut s = variant.deal(rng, start);
    let plies = rng.below(2 * variant.n());
    for _ in 0..plies {
        if s.is_over() {
            break;
        }
        let legal = s.legal_actions();
        s.apply(legal.nth_action(rng.below(legal.len())));
    }
    s
}

/// Explores random roots until `min_distinct` pairs of positions with equal keys but different `State`s have
/// been compared.
fn fuzz(kf: KeyFn, min_distinct: u64, seed: u64) -> Tally {
    let variants = [
        Variant::REAL,
        Variant::new(0b1111_1111_1100, 4), // 10 animals, no skunk or parrot
        Variant::new(0b1011_1011_1011, 3), // 9 animals
        Variant::new(0b0001_1111_1000, 2), // 6 animals: monkey, chameleon, seal, zebra, giraffe, snake
    ];
    let mut rng = Rng::new(seed);
    let mut tally = Tally::default();
    let mut seen = HashMap::new();
    let mut roots = 0;
    while tally.distinct < min_distinct {
        let variant = variants[roots % variants.len()];
        let mut root = random_root(&mut rng, variant);
        roots += 1;
        // Alternate between the real position and its public representation (every unplayed animal in hand),
        // whose wider branching keeps the depth smaller.
        let depth = if roots % 2 == 0 {
            root = public(&root);
            3
        } else {
            4
        };
        seen.clear();
        explore(kf, &root, depth, &mut seen, &mut tally);
        if root.deck_len != [0, 0] {
            // The same root with the decks reshuffled: different draws, and equal keys must not care.
            let mut shuffled = root;
            for p in 0..2 {
                let n = shuffled.deck_len[p] as usize;
                rng.shuffle(&mut shuffled.decks[p][..n]);
            }
            explore(kf, &shuffled, depth, &mut seen, &mut tally);
        }
    }
    println!(
        "{roots} roots: {} nodes, {} pairs, {} of distinct states, {} public, {} failures",
        tally.nodes, tally.pairs, tally.distinct, tally.public_pairs, tally.failures
    );
    assert!(tally.public_pairs * 10 > tally.pairs, "too few public pairs: {tally:?}");
    tally
}

fn assert_sound(tally: &Tally) {
    assert_eq!(tally.failures, 0, "{}", tally.first_failure.as_deref().unwrap_or(""));
}

#[test]
fn equal_keys_have_equal_futures() {
    assert_sound(&fuzz(markov, 30_000, 1));
}

/// The fuzzer catches keys that drop something the future depends on.
#[test]
fn broken_keys_are_caught() {
    fn no_score(s: &State) -> u128 {
        markov(s) & !(127 << 55)
    }
    fn queue_as_a_set(s: &State) -> u128 {
        let mut t = *s;
        let n = t.queue_len as usize;
        t.queue[..n].sort_unstable();
        markov(&t)
    }
    fn no_hidden_hands(s: &State) -> u128 {
        key(s) as u128
    }
    for (name, kf) in [
        ("no_score", no_score as KeyFn),
        ("queue_as_a_set", queue_as_a_set),
        ("no_hidden_hands", no_hidden_hands),
    ] {
        let tally = fuzz(kf, 2_000, 3);
        assert!(tally.failures > 0, "{name} not caught: {tally:?}");
    }
}

#[test]
#[ignore = "long: run with --release -- --ignored"]
fn equal_keys_have_equal_futures_1m() {
    assert_sound(&fuzz(markov, 1_000_000, 2));
}
