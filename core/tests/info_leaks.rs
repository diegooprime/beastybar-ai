//! Adversarial checks for information leaks and seat/perspective bugs in views, agents, search and the arena.
//!
//! - Information-set invariance: two true states that player p cannot tell apart must give p exactly the same view
//!   and the same decision (same RNG), through the arena's own `play_game` pipeline. Oracles must NOT be invariant
//!   (the check has power).
//! - Seat mirror: relabelling the players (cards, hands, decks, scores, mover, starter) must not change legal moves,
//!   transitions, evaluation, search values, views or decisions.
//! - Transposition-table history: an agent that has searched other positions (including the true state of the deal it
//!   is playing, as in a duplicate pair) decides exactly like a fresh one.
//! - Duplicate pairs: A really plays the other seat's cards in game 2i+1 and its points are A's.
//! - Joint uniformity of views late in the game (opponent hand x opponent deck order x own deck order).
//! - The RNG side channel: an agent that brute-forces the seed behind its RNG learns nothing when the match seed is
//!   random (bbar.arena's default), and agent code stays out of the arena (trust model, agents.rs).

use std::collections::HashMap;

use bbcore::agents::{Agent, Spec, parse};
use bbcore::animal::NUM_ANIMALS;
use bbcore::arena::{Control, deal, play_game, play_match};
use bbcore::eval::evaluate;
use bbcore::search::Search;
use bbcore::view::{deck_mask, view};
use bbcore::{Rng, State};

// ---------------------------------------------------------------------------------------------------------------
// helpers

/// Plays `plies` uniformly random plies from a fresh deal; None if the game ended first.
fn position(rng: &mut Rng, plies: usize) -> Option<State> {
    let start = rng.below(2) as u8;
    let mut s = State::deal(rng, start);
    for _ in 0..plies {
        if s.is_over() {
            return None;
        }
        let legal = s.legal_actions();
        s.apply(legal.nth_action(rng.below(legal.len())));
    }
    (!s.is_over() && s.legal_actions().len() > 1).then_some(s)
}

fn mask_to_vec(mask: u16) -> Vec<u8> {
    (0..NUM_ANIMALS as u8).filter(|&a| mask >> a & 1 != 0).collect()
}

/// Another true state in the same information set for player `p`: p's deck order permuted, the opponent's unplayed
/// animals re-split into hand and deck, the opponent's deck permuted. Everything p knows is unchanged.
fn same_info_set(s: &State, p: u8, rng: &mut Rng) -> State {
    let mut t = *s;
    let (me, opp) = (p as usize, 1 - p as usize);
    let mut own: Vec<u8> = s.deck(me).to_vec();
    rng.shuffle(&mut own);
    t.decks[me] = [0; 8];
    t.decks[me][..own.len()].copy_from_slice(&own);

    let hand_size = s.hands[opp].count_ones() as usize;
    let mut unplayed = mask_to_vec(s.hands[opp] | deck_mask(s, opp));
    rng.shuffle(&mut unplayed);
    t.hands[opp] = unplayed[..hand_size].iter().fold(0, |h, &a| h | 1 << a);
    t.decks[opp] = [0; 8];
    t.decks[opp][..unplayed.len() - hand_size].copy_from_slice(&unplayed[hand_size..]);
    t
}

fn flip(c: u8) -> u8 {
    if (c as usize) < NUM_ANIMALS { c + NUM_ANIMALS as u8 } else { c - NUM_ANIMALS as u8 }
}

fn flip_mask(m: u32) -> u32 {
    let lo = m & 0xFFF;
    let hi = (m >> 12) & 0xFFF;
    lo << 12 | hi
}

/// The same position with the players' labels swapped.
fn mirror(s: &State) -> State {
    let mut m = *s;
    for i in 0..s.queue_len as usize {
        m.queue[i] = flip(s.queue[i]);
    }
    m.hands = [s.hands[1], s.hands[0]];
    m.decks = [s.decks[1], s.decks[0]];
    m.deck_len = [s.deck_len[1], s.deck_len[0]];
    m.bar = flip_mask(s.bar);
    m.thats_it = flip_mask(s.thats_it);
    for i in 0..s.bar.count_ones() as usize {
        m.bar_order[i] = flip(s.bar_order[i]);
    }
    for i in 0..s.thats_it.count_ones() as usize {
        m.thats_it_order[i] = flip(s.thats_it_order[i]);
    }
    m.scores = [s.scores[1], s.scores[0]];
    m.to_move = 1 - s.to_move;
    m.start = 1 - s.start;
    m
}

/// Delegates its first decision to `inner` and records it; afterwards plays the first legal action (cheap).
struct Probe {
    inner: Box<dyn Agent>,
    first: Option<(State, u8)>,
}

impl Probe {
    fn new(spec: &str) -> Probe {
        Probe { inner: parse(spec).unwrap(), first: None }
    }
}

impl Agent for Probe {
    fn act(&mut self, input: &State, history: &[u8], rng: &mut Rng) -> u8 {
        if self.first.is_none() {
            let a = self.inner.act(input, history, rng);
            self.first = Some((*input, a));
            return a;
        }
        input.legal_actions().nth_action(0)
    }
}

/// The first decision (and the input it was given) of seat `s.to_move`, via the arena's `play_game`.
fn arena_first_decision(s: State, spec: &str, oracle: bool, seed: u64) -> (State, u8) {
    let p = s.to_move as usize;
    let mut probes = [Probe::new(spec), Probe::new("random")];
    if p == 1 {
        probes.swap(0, 1);
    }
    let mut oracles = [false; 2];
    oracles[p] = oracle;
    // The probed seat's RNG is `seed` whichever seat it is.
    let mut rngs = [Rng::new(seed ^ 0xABCD), Rng::new(seed ^ 0xABCD)];
    rngs[p] = Rng::new(seed);
    {
        let [p0, p1] = &mut probes;
        play_game(s, [p0 as &mut dyn Agent, p1 as &mut dyn Agent], oracles, &mut rngs);
    }
    probes[p].first.unwrap()
}

// ---------------------------------------------------------------------------------------------------------------
// information-set invariance

#[test]
fn views_and_rng_streams_are_identical_across_an_information_set() {
    let mut rng = Rng::new(101);
    let mut checked = 0;
    while checked < 3000 {
        let plies = rng.below(16);
        let Some(s) = position(&mut rng, plies) else { continue };
        for p in 0..2u8 {
            let t = same_info_set(&s, p, &mut rng);
            for _ in 0..4 {
                let seed = rng.next_u64();
                let (mut r1, mut r2) = (Rng::new(seed), Rng::new(seed));
                assert_eq!(
                    view(&s, p, &mut r1),
                    view(&t, p, &mut r2),
                    "view differs within p's information set"
                );
                // The RNG handed on to the agent afterwards is in the same state too.
                assert_eq!(
                    r1.next_u64(),
                    r2.next_u64(),
                    "view consumed a hidden-information-dependent RNG amount"
                );
            }
        }
        checked += 1;
    }
}

#[test]
fn non_oracle_decisions_are_identical_across_an_information_set() {
    let mut rng = Rng::new(102);
    let specs = ["random", "greedy", "pimc:w=4,d=2", "pimc:w=3,d=4", "pimc:w=2,d=6"];
    let mut checked = 0;
    let mut hidden_differs = 0;
    while checked < 600 {
        let plies = rng.below(16);
        let Some(s) = position(&mut rng, plies) else { continue };
        let p = s.to_move;
        let t = same_info_set(&s, p, &mut rng);
        if t != s {
            hidden_differs += 1;
        }
        for spec in specs {
            let seed = rng.next_u64();
            let (in_s, a_s) = arena_first_decision(s, spec, false, seed);
            let (in_t, a_t) = arena_first_decision(t, spec, false, seed);
            assert_eq!(in_s, in_t, "{spec}: input differs within the information set");
            assert_eq!(a_s, a_t, "{spec}: decision depends on hidden information at {s:?} vs {t:?}");
        }
        checked += 1;
    }
    assert!(hidden_differs > 500);
}

#[test]
fn late_exact_solve_path_is_also_invariant() {
    // Plies 12..16 exercise the `determined` exact-solve branch (pimc solves to the end once it knows everything)
    // and the d=24 (to the end) path; the decision must still depend only on what the mover knows.
    let mut rng = Rng::new(103);
    let mut checked = 0;
    while checked < 300 {
        let plies = 12 + rng.below(5);
        let Some(s) = position(&mut rng, plies) else { continue };
        let p = s.to_move;
        let t = same_info_set(&s, p, &mut rng);
        for spec in ["pimc:w=3,d=2", "pimc:w=2,d=24"] {
            let seed = rng.next_u64();
            let (_, a_s) = arena_first_decision(s, spec, false, seed);
            let (_, a_t) = arena_first_decision(t, spec, false, seed);
            assert_eq!(a_s, a_t, "{spec} at ply {plies}: decision depends on hidden information");
        }
        checked += 1;
    }
}

#[test]
fn the_invariance_check_has_power_oracles_and_true_state_inputs_do_vary() {
    // The oracles' decisions do vary within the information set: the invariance tests above would catch an agent
    // that uses hidden information. (pimc is invariant even when handed the true state: it re-views internally.)
    let mut rng = Rng::new(104);
    let mut differs = HashMap::new();
    let mut n = 0;
    while n < 300 {
        let plies = rng.below(12);
        let Some(s) = position(&mut rng, plies) else { continue };
        let p = s.to_move;
        let t = same_info_set(&s, p, &mut rng);
        for (spec, oracle) in [("pimc-hand:w=3,d=4", true), ("cheat:d=4", true)] {
            let seed = rng.next_u64();
            let (_, a_s) = arena_first_decision(s, spec, oracle, seed);
            let (_, a_t) = arena_first_decision(t, spec, oracle, seed);
            *differs.entry(spec).or_insert(0) += (a_s != a_t) as u32;
        }
        // Defence in depth: pimc handed the true state (as if the arena leaked) still decides identically.
        let seed = rng.next_u64();
        let (_, a_s) = arena_first_decision(s, "pimc:w=3,d=4", true, seed);
        let (_, a_t) = arena_first_decision(t, "pimc:w=3,d=4", true, seed);
        assert_eq!(a_s, a_t);
        n += 1;
    }
    for (spec, d) in &differs {
        assert!(*d > 10, "{spec}: only {d}/300 decisions changed with the hidden information");
    }
}

// ---------------------------------------------------------------------------------------------------------------
// seat mirror

#[test]
fn mirror_is_an_involution_and_rules_are_seat_symmetric() {
    let mut rng = Rng::new(201);
    for _ in 0..2000 {
        let plies = rng.below(20);
        let Some(s) = position(&mut rng, plies) else { continue };
        let m = mirror(&s);
        assert_eq!(mirror(&m), s);
        assert_eq!(m.legal_actions(), s.legal_actions());
        assert_eq!(m.is_over(), s.is_over());
        for a in s.legal_actions() {
            let (mut x, mut y) = (s, m);
            x.apply(a);
            y.apply(a);
            assert_eq!(mirror(&x), y, "apply is not seat-symmetric for action {a}");
        }
    }
}

#[test]
fn evaluation_and_search_are_seat_symmetric() {
    let mut rng = Rng::new(202);
    let mut search = Search::new(12);
    let mut checked = 0;
    while checked < 300 {
        let plies = rng.below(18);
        let Some(s) = position(&mut rng, plies) else { continue };
        let m = mirror(&s);
        for p in 0..2u8 {
            assert_eq!(evaluate(&s, p), evaluate(&m, 1 - p));
            assert_eq!(evaluate(&s, p), -evaluate(&s, 1 - p));
        }
        for depth in [1, 2, 3, 4] {
            let (a, b) = (search.root_values(&s, depth), search.root_values(&m, depth));
            assert_eq!(a.actions, b.actions);
            for x in a.actions {
                assert_eq!(a.values[x as usize], b.values[x as usize], "depth {depth} action {x}");
            }
        }
        checked += 1;
    }
}

#[test]
fn search_values_are_from_the_movers_point_of_view_in_both_seats() {
    // A one-move win for the mover must get a positive value whichever seat moves: find terminal-in-one positions.
    let mut rng = Rng::new(203);
    let mut found = [0; 2];
    let mut search = Search::new(0);
    while found[0] < 50 || found[1] < 50 {
        let start = rng.below(2) as u8;
        let mut s = State::deal(&mut rng, start);
        while !s.is_over() {
            let legal = s.legal_actions();
            let p = s.to_move;
            let r = search.root_values(&s, 1);
            for a in legal {
                let mut c = s;
                c.apply(a);
                let margin = c.scores[p as usize] as f32 - c.scores[1 - p as usize] as f32;
                if c.is_over() {
                    // Terminal: value = 1000 * sign(margin) + margin, from the mover's point of view.
                    let sign = if margin > 0.0 {
                        1.0
                    } else if margin < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    let want = 1000.0 * sign + margin;
                    assert_eq!(r.values[a as usize], want, "seat {p} action {a}");
                    found[p as usize] += 1;
                }
            }
            s.apply(legal.nth_action(rng.below(legal.len())));
        }
    }
}

#[test]
fn views_and_decisions_are_seat_symmetric() {
    let mut rng = Rng::new(204);
    let mut checked = 0;
    while checked < 300 {
        let plies = rng.below(16);
        let Some(s) = position(&mut rng, plies) else { continue };
        let m = mirror(&s);
        let p = s.to_move;
        let seed = rng.next_u64();
        assert_eq!(mirror(&view(&s, p, &mut Rng::new(seed))), view(&m, 1 - p, &mut Rng::new(seed)));
        // pimc-hand is left out: `shuffle_decks` draws seat 0's deck order before seat 1's, so at a fixed seed its
        // worlds are not the mirror images of each other (their distribution is; not a leak or a bias).
        for spec in ["random", "greedy", "pimc:w=3,d=3", "pimc:w=2,d=5", "cheat:d=3"] {
            let oracle = Spec::parse(spec).unwrap().is_oracle();
            let seed = rng.next_u64();
            let (_, a) = arena_first_decision(s, spec, oracle, seed);
            let (_, b) = arena_first_decision(m, spec, oracle, seed);
            assert_eq!(a, b, "{spec}: seat {p} decides differently from its mirror");
        }
        checked += 1;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// transposition-table history (agents are reused across games, including both games of a duplicate pair)

#[test]
fn searched_history_never_changes_root_values() {
    let mut rng = Rng::new(301);
    let mut warm = Search::default();
    let mut checked = 0;
    while checked < 200 {
        let plies = rng.below(16);
        let Some(s) = position(&mut rng, plies) else { continue };
        // Warm the table on the true state, its mirror and a few other worlds of the same information set.
        for d in [2, 4, 6] {
            warm.root_values(&s, d);
            warm.root_values(&mirror(&s), d);
        }
        for _ in 0..3 {
            let w = view(&s, s.to_move, &mut rng);
            warm.root_values(&w, 5);
        }
        for _ in 0..4 {
            let w = view(&s, s.to_move, &mut rng);
            for d in [3, 4, 6] {
                let (a, b) = (warm.root_values(&w, d), Search::default().root_values(&w, d));
                for x in a.actions {
                    assert_eq!(a.values[x as usize].to_bits(), b.values[x as usize].to_bits());
                }
            }
        }
        checked += 1;
    }
}

#[test]
fn agent_decisions_do_not_depend_on_what_it_played_before() {
    // Duplicate pair: the same agent object plays seat 0 of a deal and then seat 1 of the same deal. Its decisions in
    // the second game must equal those of a fresh agent.
    for spec in ["pimc:w=4,d=4", "pimc:w=2,d=6"] {
        for i in 0..6u32 {
            let d = deal(77, i);
            let seeds = [11 + i as u64, 23 + i as u64];
            let run = |agent_a: &mut Box<dyn Agent>| {
                let mut b = parse("greedy").unwrap();
                let mut rngs = [Rng::new(seeds[0]), Rng::new(seeds[1])];
                play_game(d, [b.as_mut(), agent_a.as_mut()], [false, false], &mut rngs).1
            };
            let mut fresh = parse(spec).unwrap();
            let fresh_moves = run(&mut fresh);
            let mut used = parse(spec).unwrap();
            {
                // Game 2i: `used` plays seat 0 of the same deal (sees seat 0's true hand and draws).
                let mut b = parse("greedy").unwrap();
                let mut rngs = [Rng::new(5), Rng::new(6)];
                play_game(d, [used.as_mut(), b.as_mut()], [false, false], &mut rngs);
            }
            assert_eq!(run(&mut used), fresh_moves, "{spec} deal {i}: history changed its play");
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// duplicate pairs: seats and points

/// Whether `a` maximizes the 1-ply evaluation for the mover (what greedy plays; hidden information cannot change
/// this value, so it can be checked on the true state).
fn is_greedy_move(s: &State, a: u8) -> bool {
    let p = s.to_move;
    let value = |x: u8| {
        let mut c = *s;
        c.apply(x);
        evaluate(&c, p)
    };
    let best = s.legal_actions().map(value).fold(f32::NEG_INFINITY, f32::max);
    value(a) == best
}

#[test]
fn a_plays_the_other_seats_cards_in_the_second_game_and_gets_its_points() {
    for (a, b, greedy_is_a) in [("greedy", "random", true), ("random", "greedy", false)] {
        let records =
            play_match(&Spec::parse(a).unwrap(), &Spec::parse(b).unwrap(), 60, 31, 3, &Control::default());
        assert_eq!(records.len(), 120);
        let mut non_greedy_by_random = 0;
        let mut a_wins_by_seat = [0; 2];
        for r in &records {
            let mut s = deal(31, r.deal);
            assert_eq!(s.start, r.start);
            for &m in &r.moves {
                let a_moves = s.to_move == r.a_seat;
                if a_moves == greedy_is_a {
                    assert!(is_greedy_move(&s, m), "the greedy agent is not in the seat the record says");
                } else if !is_greedy_move(&s, m) {
                    non_greedy_by_random += 1;
                }
                s.apply(m);
            }
            assert!(s.is_over());
            assert_eq!(r.a_points, s.scores[r.a_seat as usize]);
            assert_eq!(r.b_points, s.scores[1 - r.a_seat as usize]);
            a_wins_by_seat[r.a_seat as usize] += (r.a_points > r.b_points) as u32;
        }
        assert!(non_greedy_by_random > 100, "the random agent never deviated: seats not distinguished");
        // Greedy wins most games in BOTH seats: points are attributed to the right agent in both games of a pair.
        for (seat, &w) in a_wins_by_seat.iter().enumerate() {
            if greedy_is_a {
                assert!(w > 40, "A=greedy won only {w}/60 in seat {seat}");
            } else {
                assert!(w < 20, "A=random won {w}/60 in seat {seat}");
            }
        }
    }
}

#[test]
fn both_games_of_a_pair_start_from_the_same_deal() {
    // The record's moves of both games replay from the same `deal(seed, i)`, and A's first own move in game 2i+1 is
    // drawn from seat 1's initial hand (the cards B held in game 2i).
    let records = play_match(
        &Spec::parse("random").unwrap(),
        &Spec::parse("random").unwrap(),
        40,
        8,
        2,
        &Control::default(),
    );
    for pair in records.chunks(2) {
        let (g0, g1) = (&pair[0], &pair[1]);
        assert_eq!((g0.deal, g0.a_seat, g1.deal, g1.a_seat), (g1.deal, 0, g0.deal, 1));
        let d = deal(8, g0.deal);
        for g in [g0, g1] {
            let mut s = d;
            for &m in &g.moves {
                if s.to_move == g.a_seat && s.hands == d.hands {
                    // A's first move: played from A's seat's initial hand.
                    let animal = bbcore::decode(m).animal;
                    assert!(d.hands[g.a_seat as usize] >> animal & 1 != 0);
                }
                s.apply(m);
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------
// view uniformity, jointly over every hidden component

fn chi_square(counts: &[u64]) -> f64 {
    let total: u64 = counts.iter().sum();
    let e = total as f64 / counts.len() as f64;
    counts.iter().map(|&c| (c as f64 - e).powi(2) / e).sum()
}

fn assert_uniform(counts: &[u64], what: &str) {
    let df = (counts.len() - 1) as f64;
    let chi2 = chi_square(counts);
    assert!(chi2 < df + 6.0 * (2.0 * df).sqrt(), "{what}: chi-square {chi2:.1}, {df} degrees of freedom");
}

#[test]
fn views_are_jointly_uniform_over_hidden_configurations() {
    let mut rng = Rng::new(401);
    // Late game: every hidden configuration (opponent hand x opponent deck order x own deck order) is equally likely.
    for plies in [10usize, 11, 12, 13] {
        let s = loop {
            if let Some(s) = position(&mut rng, plies) {
                break s;
            }
        };
        let p = s.to_move;
        let (me, opp) = (p as usize, 1 - p as usize);
        let u = (s.hands[opp] | deck_mask(&s, opp)).count_ones() as u64;
        let k = s.hands[opp].count_ones() as u64;
        let fact = |n: u64| (1..=n).product::<u64>();
        // C(u, k) hands x (u - k)! opponent deck orders x own deck orders.
        let n_cells = fact(u) / fact(k) * fact(s.deck_len[me] as u64);
        let mut cells: HashMap<(u16, [u8; 8], [u8; 8]), u64> = HashMap::new();
        for _ in 0..n_cells * 300 {
            let v = view(&s, p, &mut rng);
            *cells.entry((v.hands[opp], v.decks[opp], v.decks[me])).or_insert(0) += 1;
        }
        assert_eq!(cells.len() as u64, n_cells, "ply {plies}: some hidden configurations never sampled");
        let counts: Vec<u64> = cells.values().copied().collect();
        assert_uniform(&counts, &format!("ply {plies}: joint hidden configuration ({n_cells} cells)"));
    }
    // Early: the opponent hand marginal over all C(12, 4) = 495 and C(11, 4) = 330 hands.
    for plies in [0usize, 1] {
        let s = loop {
            if let Some(s) = position(&mut rng, plies) {
                break s;
            }
        };
        let p = s.to_move;
        let opp = 1 - p as usize;
        let mut hands: HashMap<u16, u64> = HashMap::new();
        for _ in 0..1_000_000 {
            *hands.entry(view(&s, p, &mut rng).hands[opp]).or_insert(0) += 1;
        }
        assert_eq!(hands.len(), if plies == 0 { 495 } else { 330 });
        let counts: Vec<u64> = hands.values().copied().collect();
        assert_uniform(&counts, &format!("ply {plies}: opponent hand"));
    }
}

// ---------------------------------------------------------------------------------------------------------------
// RNG side channel and the trust model

/// A deliberately malicious non-oracle agent: at its first move it brute-forces a small match seed and the deal index
/// from the RNG the arena hands it, then knows the whole deal (the opponent's hand and both deck orders).
struct Snoop {
    known: Option<State>,
}

impl Agent for Snoop {
    fn act(&mut self, v: &State, history: &[u8], rng: &mut Rng) -> u8 {
        if self.known.is_none() && history.len() <= 1 {
            // Draws view() consumed: (own deck - 1) + (opponent's unplayed - 1), both public.
            let (me, opp) = (v.to_move as usize, 1 - v.to_move as usize);
            let k = (v.deck_len[me] as usize).saturating_sub(1)
                + ((v.hands[opp] | deck_mask(v, opp)).count_ones() as usize).saturating_sub(1);
            let probe = rng.clone().next_u64();
            'search: for seed in 0..64u64 {
                for i in 0..2000u64 {
                    for g in 0..2u64 {
                        // arena.rs: seat RNG = Rng(derive_seed(seed, [GAME = 2, i, g, seat])).
                        let mut r = Rng::new(bbcore::arena::derive_seed(seed, &[2, i, g, me as u64]));
                        for _ in 0..k {
                            r.next_u64();
                        }
                        if r.next_u64() == probe {
                            self.known = Some(deal(seed, i as u32));
                            break 'search;
                        }
                    }
                }
            }
        }
        v.legal_actions().nth_action(0)
    }
}

/// What the snoop learned in each game of deal `i` of a match with `seed`, played exactly as arena.rs `play_match`
/// does (same deal, same per-seat RNGs), and the true deal.
fn snoop(seed: u64, i: u32) -> ([Option<State>; 2], State) {
    let d = deal(seed, i);
    let known = std::array::from_fn(|g| {
        let g = g as u64;
        let mut snoop = Snoop { known: None };
        let mut other = parse("random").unwrap();
        let mut rngs: [Rng; 2] =
            std::array::from_fn(|s| Rng::new(bbcore::arena::derive_seed(seed, &[2, i as u64, g, s as u64])));
        let agents: [&mut dyn Agent; 2] =
            if g == 0 { [&mut snoop, other.as_mut()] } else { [other.as_mut(), &mut snoop] };
        play_game(d, agents, [false, false], &mut rngs);
        snoop.known
    });
    (known, d)
}

#[test]
fn a_malicious_agent_cannot_recover_the_deal_from_its_rng() {
    // bbar.arena's default match seed: 64 random bits, which no agent can enumerate; and the keyed derivation means
    // the RNG an agent holds does not give the seed back (arena.rs `seat_seeds_do_not_give_the_match_seed_back`).
    let mut rng = Rng::new(501);
    for _ in 0..4 {
        let seed = rng.next_u64();
        let (known, d) = snoop(seed, 1234);
        for (g, k) in known.iter().enumerate() {
            let opp = 1 - g;
            assert!(
                k.is_none(),
                "game {g}: a non-oracle agent recovered the opponent's hand {:#05x} (true {:#05x}) and deck {:?} \
                 (true {:?})",
                k.unwrap().hands[opp],
                d.hands[opp],
                k.unwrap().deck(opp),
                d.deck(opp)
            );
        }
    }
}

#[test]
fn the_snoop_has_power_when_the_seed_is_guessable() {
    // The same snoop does recover the deal when the match seed is small and known, as with an explicit `--seed 2`:
    // explicit seeds are for agents that follow the trust model (agents.rs), never for untrusted agent code.
    let (known, d) = snoop(2, 1234);
    assert_eq!(known, [Some(d), Some(d)]);
}

#[test]
fn agent_code_stays_out_of_the_arena() {
    // Trust model (agents.rs): agents learn about the game only through `act`. Their code must not reach the deal or
    // the seed derivation (arena.rs), nor read memory it is not given (unsafe). Comments and test modules are exempt.
    let src = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
    for file in ["agents.rs", "search.rs", "eval.rs", "view.rs"] {
        let text = std::fs::read_to_string(format!("{src}/{file}")).unwrap();
        let code = text.split("#[cfg(test)]").next().unwrap();
        for (n, line) in code.lines().enumerate() {
            let line = line.split("//").next().unwrap();
            for banned in ["arena", "derive_seed", "siphash", "unsafe"] {
                assert!(!line.contains(banned), "{file}:{}: agent code uses {banned:?}: {line}", n + 1);
            }
        }
    }
}
