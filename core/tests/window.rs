//! The window solver (window.rs) against independent computations.
//!
//! - Tree: public actions are the union of the hands' legal actions; nodes are the public positions.
//! - Evaluation: a profile's value equals a brute-force evaluation that enumerates both players' root hands and
//!   every draw explicitly (no reach vectors, no transposed transitions); the best-response value equals the
//!   brute-force value of the extracted best-response strategy and dominates every strategy tried.
//! - Solving: exploitability falls to the target; frontier, public and terminal leaves give the same game value;
//!   results are bit-identical across runs and thread counts.
//! - Re-solving gadget: with opponent cfvs from an equilibrium, the re-solved strategy is as unexploitable and no
//!   opponent hand gains.

use bbcore::animal::*;
use bbcore::hands::{Variant, hands, unplayed};
use bbcore::markov::public;
use bbcore::window::{Algo, Cfr, Child, Leaves, Window};
use bbcore::{Rng, State};

fn random_position(rng: &mut Rng, variant: Variant, ply: usize) -> State {
    loop {
        let start = rng.below(2) as u8;
        let mut s = variant.deal(rng, start);
        for _ in 0..ply {
            if s.is_over() {
                break;
            }
            let legal = s.legal_actions();
            s.apply(legal.nth_action(rng.below(legal.len())));
        }
        if !s.is_over() && variant.ply(&s) == ply {
            return s;
        }
    }
}

fn variant(animals: &[u8], k: usize) -> Variant {
    Variant::new(animals.iter().fold(0, |m, &a| m | 1 << a), k)
}

fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A fixed non-uniform profile: weights 1..4 hashed from (node, hand, edge), normalised over the legal actions.
fn hashed_profile(w: &Window, salt: u64) -> Vec<f64> {
    let table: Vec<f64> = (0..w.n_entries).map(|i| 1.0 + (mix(i as u64 ^ salt) % 4) as f64).collect();
    w.sanitize(&table).unwrap()
}

/// Random root ranges with some zeros.
fn random_ranges(w: &Window, rng: &mut Rng) -> [Vec<f64>; 2] {
    std::array::from_fn(|p| {
        let n = w.root_hands(p).len();
        let mut r: Vec<f64> =
            (0..n).map(|_| if rng.below(4) == 0 { 0.0 } else { rng.below(100) as f64 + 1.0 }).collect();
        if r.iter().sum::<f64>() == 0.0 {
            r[0] = 1.0;
        }
        let s: f64 = r.iter().sum();
        r.iter().map(|x| x / s).collect()
    })
}

/// Player 0's value with both hands known (local indices `h`), by explicit enumeration of every draw.
fn brute_value(w: &Window, prof: &[f64], child: Child, h: [usize; 2]) -> f64 {
    match child {
        Child::Terminal(u) => u,
        Child::Frontier(f) => {
            let leaf = w.frontier[f as usize];
            w.values[leaf.off as usize + h[leaf.mover as usize]]
        }
        Child::Value(_) => unreachable!("no value leaves here"),
        Child::Node(n) => {
            let n = n as usize;
            let node = &w.nodes[n];
            let p = node.mover as usize;
            let ne = node.n_edges as usize;
            let mut v = 0.0;
            for (e, edge) in w.node_edges(n).iter().enumerate() {
                let pr = prof[node.entry + h[p] * ne + e];
                if pr == 0.0 {
                    continue;
                }
                let tr = &w.trans[edge.trans as usize];
                let row = tr.from.iter().position(|&x| x as usize == h[p]).expect("legal action has a row");
                for &next in &tr.to[row * tr.stride..(row + 1) * tr.stride] {
                    let mut hn = h;
                    hn[p] = next as usize;
                    v += pr * tr.prob * brute_value(w, prof, edge.child, hn);
                }
            }
            v
        }
    }
}

fn brute_ev(w: &Window, prof: &[f64], r: &[Vec<f64>; 2]) -> f64 {
    let mut v = 0.0;
    for (h0, &p0) in r[0].iter().enumerate() {
        for (h1, &p1) in r[1].iter().enumerate() {
            if p0 * p1 > 0.0 {
                v += p0 * p1 * brute_value(w, prof, w.root, [h0, h1]);
            }
        }
    }
    v
}

#[test]
fn tree_is_the_public_tree() {
    let mut rng = Rng::new(1);
    for ply in [12, 13] {
        for _ in 0..5 {
            let s = random_position(&mut rng, Variant::REAL, ply);
            let w = Window::build(&s, 4, Leaves::Frontier, 1).unwrap();
            assert_eq!(w.root_state, public(&s));
            for (n, node) in w.nodes.iter().enumerate() {
                let p = node.mover as usize;
                assert_eq!(node.state.to_move as usize, p);
                // The public actions are the union over the mover's possible hands of each hand's legal actions.
                let mut union = 0u64;
                for h in hands(unplayed(&node.state, p), 4) {
                    let mut t = node.state;
                    t.hands[p] = h;
                    union |= t.legal_actions().0;
                }
                let acts: u64 = w.node_edges(n).iter().fold(0, |m, e| m | 1 << e.action);
                assert_eq!(acts, union);
                assert_eq!(w.history(n).len(), node.depth as usize);
                assert_eq!(w.find(&w.history(n)), Some(n));
                for e in w.node_edges(n) {
                    let mut c = node.state;
                    c.apply(e.action);
                    if let Child::Node(ch) = e.child {
                        assert_eq!(w.nodes[ch as usize].state, c);
                        assert_eq!(w.nodes[ch as usize].parent as usize, n);
                    } else {
                        // Leaves: over, or a frontier node (opponent hand public, mover deck at most 1).
                        let m = c.to_move as usize;
                        let u = |q: usize| unplayed(&c, q).count_ones() as usize;
                        assert!(c.is_over() || (u(1 - m) <= 4 && u(m) <= 5));
                    }
                }
            }
        }
    }
}

#[test]
fn evaluation_matches_brute_force() {
    let mut rng = Rng::new(2);
    let cases: Vec<(Variant, usize, Leaves)> = vec![
        (Variant::REAL, 13, Leaves::Frontier),
        (Variant::REAL, 12, Leaves::Frontier),
        (Variant::REAL, 13, Leaves::Public),
        (variant(&[PARROT, MONKEY, SEAL, GIRAFFE, LION], 2), 1, Leaves::Terminal),
        (variant(&[SKUNK, KANGAROO, CHAMELEON, SEAL, CROCODILE], 3), 1, Leaves::Terminal),
        (variant(&[CHAMELEON, MONKEY, GIRAFFE, LION], 2), 0, Leaves::Terminal),
        (variant(&[KANGAROO, CHAMELEON, ZEBRA, GIRAFFE, CROCODILE, HIPPO, LION], 4), 5, Leaves::Frontier),
    ];
    let mut checked = 0;
    for (case, &(v, ply, leaves)) in cases.iter().enumerate() {
        for rep in 0..3 {
            let s = random_position(&mut rng, v, ply);
            let w = Window::build(&s, v.k(), leaves, 1).unwrap();
            if w.stats.internal > 3000 {
                continue;
            }
            let ranges = random_ranges(&w, &mut rng);
            for (pi, prof) in [w.uniform_profile(), hashed_profile(&w, rng.next_u64())].iter().enumerate() {
                let e = w.exploitability(prof, &ranges, true, 1);
                let ev = brute_ev(&w, prof, &ranges);
                assert!(
                    (e.value - ev).abs() < 1e-12,
                    "case {case}.{rep} profile {pi}: value {} vs brute {ev}",
                    e.value
                );
                // Best responses: their brute-force value equals the BR value, which dominates the profile.
                for i in 0..2 {
                    let br = w.best_response(i, prof, &ranges[1 - i]);
                    let v_br = brute_ev(&w, &br, &ranges);
                    let v_br_i = if i == 0 { v_br } else { 1.0 - v_br };
                    assert!(
                        (v_br_i - e.br[i]).abs() < 1e-12,
                        "case {case}: BR{i} {} vs brute {v_br_i}",
                        e.br[i]
                    );
                    let v_i = if i == 0 { ev } else { 1.0 - ev };
                    assert!(e.br[i] >= v_i - 1e-12);
                }
                assert!(e.nashconv >= -1e-12);
                checked += 1;
            }
        }
    }
    assert!(checked >= 20, "only {checked} cases");
}

#[test]
fn solving_reaches_the_target_and_leaf_rules_agree() {
    let mut rng = Rng::new(3);
    for algo in [Algo::Dcfr, Algo::PcfrPlus, Algo::CfrPlus] {
        for _ in 0..4 {
            let ply = 12 + rng.below(2);
            let s = random_position(&mut rng, Variant::REAL, ply);
            let mut values = Vec::new();
            for leaves in [Leaves::Frontier, Leaves::Public] {
                let w = Window::build(&s, 4, leaves, 1).unwrap();
                let ranges = random_ranges(&w, &mut rng.clone());
                let mut cfr = Cfr::new(&w, ranges, algo);
                let r = cfr.solve(&w, 1e-7, 50_000, 10);
                assert!(r.exploitability <= 1e-7, "{algo:?} stopped at {}", r.exploitability);
                // The game value lies in [1 - BR_1, BR_0], an interval of width NashConv.
                assert!(1.0 - r.expl.br[1] <= r.expl.value + 1e-12 && r.expl.value <= r.expl.br[0] + 1e-12);
                values.push((r.expl.value, r.expl.nashconv));
            }
            let (a, b) = (values[0], values[1]);
            assert!((a.0 - b.0).abs() <= a.1 + b.1 + 1e-12, "frontier {a:?} vs public {b:?}");
        }
    }
    // Mini variant from the start: terminal vs frontier leaves.
    for (animals, k) in
        [(&[CHAMELEON, MONKEY, GIRAFFE, LION][..], 2), (&[SKUNK, MONKEY, CHAMELEON, SEAL, GIRAFFE][..], 2)]
    {
        let v = variant(animals, k);
        let s = v.deal(&mut rng, 1);
        let mut values = Vec::new();
        for leaves in [Leaves::Terminal, Leaves::Frontier, Leaves::Public] {
            let w = Window::build(&s, k, leaves, 1).unwrap();
            let ranges = [vec![1.0; w.root_hands(0).len()], vec![1.0; w.root_hands(1).len()]];
            let mut cfr = Cfr::new(&w, ranges, Algo::Dcfr);
            let r = cfr.solve(&w, 1e-8, 50_000, 10);
            assert!(r.exploitability <= 1e-8);
            values.push(r.expl.value);
        }
        assert!((values[0] - values[1]).abs() < 4e-8 && (values[0] - values[2]).abs() < 4e-8, "{values:?}");
    }
}

#[test]
fn deterministic_across_runs_and_threads() {
    let mut rng = Rng::new(4);
    // Large enough that the threaded runs walk subtrees in parallel (subtrees of at least 256 nodes).
    let (s, w1) = loop {
        let s = random_position(&mut rng, Variant::REAL, 11);
        let w = Window::build(&s, 4, Leaves::Frontier, 1).unwrap();
        if w.stats.internal >= 800 {
            break (s, w);
        }
    };
    let w4 = Window::build(&s, 4, Leaves::Frontier, 4).unwrap();
    assert_eq!(w1.values, w4.values);
    let ranges = random_ranges(&w1, &mut rng);
    for algo in [Algo::Dcfr, Algo::PcfrPlus] {
        let mut runs = Vec::new();
        for (w, threads) in [(&w1, 1), (&w1, 1), (&w4, 4), (&w1, 3)] {
            let mut cfr = Cfr::new(w, ranges.clone(), algo);
            cfr.threads = threads;
            cfr.iterate(w, 25);
            let avg = cfr.average(w);
            let e = w.exploitability(&avg, &ranges, true, threads);
            runs.push((avg, cfr.current(w), e));
        }
        for r in &runs[1..] {
            assert!(r.0.iter().zip(&runs[0].0).all(|(a, b)| a.to_bits() == b.to_bits()), "average differs");
            assert!(r.1.iter().zip(&runs[0].1).all(|(a, b)| a.to_bits() == b.to_bits()), "current differs");
            assert_eq!(r.2.br[0].to_bits(), runs[0].2.br[0].to_bits());
            assert_eq!(r.2.br[1].to_bits(), runs[0].2.br[1].to_bits());
            assert_eq!(r.2.value.to_bits(), runs[0].2.value.to_bits());
        }
    }
}

/// Re-solving with opponent cfvs from an equilibrium of the same window.
#[test]
fn gadget_resolve_keeps_the_guarantee() {
    let mut rng = Rng::new(5);
    let cases =
        [(Variant::REAL, 12), (Variant::REAL, 11), (variant(&[CHAMELEON, MONKEY, GIRAFFE, LION], 2), 0)];
    for &(v, ply) in &cases {
        let s = random_position(&mut rng, v, ply);
        let leaves = if ply == 0 { Leaves::Terminal } else { Leaves::Frontier };
        let w = Window::build(&s, v.k(), leaves, 1).unwrap();
        let ranges = random_ranges(&w, &mut rng);
        let eps = 1e-7;
        let mut cfr = Cfr::new(&w, ranges.clone(), Algo::Dcfr);
        let r = cfr.solve(&w, eps, 100_000, 10);
        assert!(r.exploitability <= eps);
        let star = cfr.average(&w);
        for player in 0..2 {
            let o = 1 - player;
            // The opponent's best-response cfvs against our equilibrium strategy, per opponent hand.
            let (cfvs, _) = w.root_cfvs(o, &star, &ranges[player], false, 1);
            let tau = 1e-7;
            let mut g = Cfr::new_gadget(&w, player, ranges[player].clone(), cfvs.clone(), None, Algo::Dcfr);
            let rg = g.solve(&w, tau, 100_000, 10);
            let ge = rg.gadget.unwrap();
            assert!(ge.exploitability <= tau, "gadget stopped at {}", ge.exploitability);
            let ours = g.average(&w);
            // The opponent's best-response cfvs against the re-solved strategy do not increase (up to the bound
            // 2 tau / prior(h) implied by the gadget's exploitability; the prior is uniform).
            let (after, _) = w.root_cfvs(o, &ours, &ranges[player], false, 1);
            let slack = 2.0 * tau * cfvs.len() as f64 + 1e-12;
            for h in 0..cfvs.len() {
                assert!(after[h] <= cfvs[h] + slack, "hand {h}: {} > {}", after[h], cfvs[h]);
            }
            // Our re-solved strategy with the opponent's equilibrium strategy: as unexploitable as the equilibrium.
            let mut mixed = star.clone();
            for node in &w.nodes {
                if node.mover as usize == player {
                    let len = w.handsets[node.hs[player] as usize].masks.len() * node.n_edges as usize;
                    mixed[node.entry..node.entry + len].copy_from_slice(&ours[node.entry..node.entry + len]);
                }
            }
            // NashConv grows by at most sum_h r_o(h) max(0, after - cfvs) <= 2 tau max_h r_o(h) / prior(h).
            let ratio = ranges[o].iter().fold(0.0f64, |m, &x| m.max(x)) * cfvs.len() as f64;
            let e = w.exploitability(&mixed, &ranges, false, 1);
            assert!(
                e.exploitability <= eps + tau * ratio + 1e-12,
                "re-solved exploitability {}",
                e.exploitability
            );
        }
    }
}

#[test]
fn external_profiles_are_sanitized() {
    let mut rng = Rng::new(6);
    let s = random_position(&mut rng, Variant::REAL, 13);
    let w = Window::build(&s, 4, Leaves::Frontier, 1).unwrap();
    let u = w.uniform_profile();
    assert_eq!(w.sanitize(&vec![0.0; w.n_entries]).unwrap(), u);
    assert_eq!(w.sanitize(&vec![7.0; w.n_entries]).unwrap(), u);
    assert!(w.sanitize(&vec![-1.0; w.n_entries]).is_err());
    assert!(w.sanitize(&vec![0.0; w.n_entries + 1]).is_err());
    for (n, node) in w.nodes.iter().enumerate() {
        let ne = node.n_edges as usize;
        for (h, &mask) in w.node_hands(n, node.mover as usize).iter().enumerate() {
            let row = &u[node.entry + h * ne..][..ne];
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            for (e, edge) in w.node_edges(n).iter().enumerate() {
                assert_eq!(row[e] > 0.0, mask >> edge.animal & 1 != 0);
            }
        }
    }
}

/// Re-solving at a leaf root (a real ply-15 frontier root, a ply-16 public root, a finished game, a mini-variant
/// frontier root): we have no decisions, so the gadget is solved exactly without iterations. Regression: the opponent's
/// terminate/follow choice used to stay at its 50/50 default, reporting a gadget exploitability of
/// `sum_h prior(h) |br(h) - cfv(h)| / 2` (0.05 for shifts of 0.2) instead of 0.
#[test]
fn gadget_at_a_leaf_root_is_exact() {
    let mut rng = Rng::new(7);
    let finished = |rng: &mut Rng| {
        let start = rng.below(2) as u8;
        let mut s = Variant::REAL.deal(rng, start);
        while !s.is_over() {
            let legal = s.legal_actions();
            s.apply(legal.nth_action(rng.below(legal.len())));
        }
        s
    };
    let mini = variant(&[SKUNK, KANGAROO, ZEBRA, CROCODILE, SEAL], 3);
    let cases: Vec<(State, usize, Leaves)> = vec![
        (random_position(&mut rng, Variant::REAL, 15), 4, Leaves::Frontier),
        (random_position(&mut rng, Variant::REAL, 15), 4, Leaves::Frontier),
        (random_position(&mut rng, Variant::REAL, 16), 4, Leaves::Public),
        (finished(&mut rng), 4, Leaves::Frontier),
        (random_position(&mut rng, mini, 3), 3, Leaves::Frontier),
    ];
    for (case, (s, k, leaves)) in cases.iter().enumerate() {
        let w = Window::build(s, *k, *leaves, 1).unwrap();
        assert!(!matches!(w.root, Child::Node(_)), "case {case}: root is not a leaf");
        assert_eq!(matches!(w.root, Child::Terminal(_)), case == 3);
        let ranges = random_ranges(&w, &mut rng);
        let prof = w.uniform_profile();
        for (player, ours) in ranges.iter().enumerate() {
            let (br, _) = w.root_cfvs(1 - player, &prof, ours, false, 1);
            let no = br.len();
            let prior: Vec<f64> = (0..no).map(|_| 1.0 + rng.below(9) as f64).collect();
            let psum: f64 = prior.iter().sum();
            for shift in [0.2, -0.2, 0.0] {
                // Per hand: br + shift (the opponent terminates or follows), or br exactly (a tie).
                let cfvs: Vec<f64> = br
                    .iter()
                    .map(|&b| if rng.below(4) == 0 { b } else { (b + shift).clamp(0.0, 1.0) })
                    .collect();
                let mut g =
                    Cfr::new_gadget(&w, player, ours.clone(), cfvs.clone(), Some(prior.clone()), Algo::Dcfr);
                for max_iters in [100_000, 0] {
                    let r = g.solve(&w, 1e-9, max_iters, 10);
                    let ge = r.gadget.unwrap();
                    assert_eq!(r.iterations, 0);
                    assert!(
                        ge.exploitability.abs() <= 1e-12,
                        "case {case} player {player} shift {shift}: gadget exploitability {}",
                        ge.exploitability
                    );
                    assert!(ge.exploitability <= 1e-9, "a caller testing the target must see success");
                    // The violations only depend on our (empty) strategy: the leaf cfvs against the given ones.
                    let max_v = br.iter().zip(&cfvs).map(|(b, c)| b - c).fold(f64::NEG_INFINITY, f64::max);
                    let mean_v: f64 =
                        br.iter().zip(&cfvs).zip(&prior).map(|((b, c), p)| p / psum * (b - c).max(0.0)).sum();
                    assert_eq!(ge.max_violation, max_v);
                    assert!((ge.mean_violation - mean_v).abs() <= 1e-15);
                }
            }
        }
    }
}
