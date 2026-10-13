//! Depth-limited windows (window.rs, value leaves and the external half-iterations).
//!
//! - The cut tree is the full tree above the cut, with value leaves exactly at the cut depth.
//! - Without value leaves, the external half-iterations are the ordinary solver, bit for bit.
//! - Leaf reaches and leaf cfvs: with leaf values taken from the full window (each leaf subtree's values under a
//!   profile, at the beliefs that profile induces), the depth-limited window reproduces the full window's value.
//! - `embed` moves profiles between a full window, its trunk and its leaf subtrees losslessly.
//! - Averaged leaf values: constant leaf values average to themselves; changing ones average with the weights of the
//!   average strategy (iteration weight x the other player's reach mass).

use bbcore::hands::Variant;
use bbcore::window::{Algo, Cfr, Child, LeafQuery, Leaves, Window};
use bbcore::{Rng, State};

fn random_position(rng: &mut Rng, ply: usize) -> State {
    loop {
        let start = rng.below(2) as u8;
        let mut s = Variant::REAL.deal(rng, start);
        for _ in 0..ply {
            if s.is_over() {
                break;
            }
            let legal = s.legal_actions();
            s.apply(legal.nth_action(rng.below(legal.len())));
        }
        if !s.is_over() && Variant::REAL.ply(&s) == ply {
            return s;
        }
    }
}

fn random_ranges(w: &Window, rng: &mut Rng) -> [Vec<f64>; 2] {
    std::array::from_fn(|p| {
        let n = w.root_hands(p).len();
        let mut r: Vec<f64> =
            (0..n).map(|_| if rng.below(3) == 0 { 0.0 } else { rng.below(100) as f64 + 1.0 }).collect();
        if r.iter().sum::<f64>() == 0.0 {
            r[0] = 1.0;
        }
        r
    })
}

fn hashed_profile(w: &Window, salt: u64) -> Vec<f64> {
    let mut x = salt | 1;
    let table: Vec<f64> = (0..w.n_entries)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            1.0 + (x % 5) as f64
        })
        .collect();
    w.sanitize(&table).unwrap()
}

#[test]
fn cut_tree_is_the_full_tree_above_the_cut() {
    let mut rng = Rng::new(1);
    for (ply, cut) in [(11, 1), (11, 2), (12, 2), (10, 1)] {
        let s = random_position(&mut rng, ply);
        let full = Window::build(&s, 4, Leaves::Frontier, 1).unwrap();
        let dl = Window::build_cut(&s, 4, Leaves::Frontier, Some(cut), 1).unwrap();
        assert!(!dl.value_leaves.is_empty());
        let above = full.nodes.iter().filter(|n| (n.depth as usize) < cut).count();
        assert_eq!(dl.nodes.len(), above, "ply {ply} cut {cut}");
        for (n, node) in dl.nodes.iter().enumerate() {
            let m = full.find(&dl.history(n)).expect("trunk node in the full window");
            assert_eq!(full.nodes[m].state, node.state);
            for (ea, eb) in dl.node_edges(n).iter().zip(full.node_edges(m)) {
                assert_eq!(ea.action, eb.action);
                match ea.child {
                    Child::Value(j) => {
                        let leaf = &dl.value_leaves[j as usize];
                        let Child::Node(c) = eb.child else {
                            panic!("a value leaf is internal in the full tree")
                        };
                        assert_eq!(full.nodes[c as usize].state, leaf.state);
                        assert_eq!(full.nodes[c as usize].depth as usize, cut);
                        assert_eq!(dl.leaf_history(j as usize), full.history(c as usize));
                    }
                    Child::Node(_) => assert!(matches!(eb.child, Child::Node(_))),
                    other => assert_eq!(other, eb.child),
                }
            }
        }
        // Leaf-values layout: both players' hands at every leaf, contiguous.
        let mut off = 0;
        for (j, leaf) in dl.value_leaves.iter().enumerate() {
            for p in 0..2 {
                assert_eq!(leaf.off[p] as usize, off);
                assert_eq!(leaf.n[p] as usize, dl.leaf_hands(j, p).len());
                off += leaf.n[p] as usize;
            }
        }
        assert_eq!(off, dl.n_leaf_values);
    }
    let s = random_position(&mut rng, 12);
    assert!(Window::build_cut(&s, 4, Leaves::Frontier, Some(0), 1).is_err());
}

#[test]
fn external_mode_without_value_leaves_is_the_ordinary_solver() {
    let mut rng = Rng::new(2);
    for (ply, cut) in [(12, Some(3)), (13, Some(2)), (11, None)] {
        let s = random_position(&mut rng, ply);
        let w = Window::build(&s, 4, Leaves::Frontier, 1).unwrap();
        let dl = Window::build_cut(&s, 4, Leaves::Frontier, cut, 1).unwrap();
        assert!(dl.value_leaves.is_empty() && dl.n_entries == w.n_entries);
        let ranges = random_ranges(&w, &mut rng);
        for algo in [Algo::Dcfr, Algo::PcfrPlus] {
            let mut a = Cfr::new(&w, ranges.clone(), algo);
            a.iterate(&w, 30);
            let mut b = Cfr::new(&dl, ranges.clone(), algo);
            for _ in 0..60 {
                let q = b.ext_begin(&dl);
                assert!(q.beliefs.is_empty());
                b.ext_end(&dl, Vec::new()).unwrap();
            }
            assert_eq!(a.t, b.t);
            let (x, y) = (a.average(&w), b.average(&dl));
            assert!(x.iter().zip(&y).all(|(p, q)| p.to_bits() == q.to_bits()));
        }
    }
}

/// The leaf subtree of `full` below its node `m`, as a window of its own, with `profile` restricted to it.
fn subtree(full: &Window, m: usize, profile: &[f64]) -> (Window, Vec<f64>) {
    let sub = Window::build(&full.nodes[m].state, 4, Leaves::Frontier, 1).unwrap();
    let mut prof = sub.uniform_profile();
    let n = sub.embed(&mut prof, full, profile, 0, m).unwrap();
    assert_eq!(n, sub.n_entries, "the subtree covers the leaf window");
    (sub, prof)
}

#[test]
fn leaf_values_from_the_full_window_reproduce_its_value() {
    let mut rng = Rng::new(3);
    for (ply, cut) in [(11, 1), (10, 2), (12, 1)] {
        let s = random_position(&mut rng, ply);
        let full = Window::build(&s, 4, Leaves::Frontier, 1).unwrap();
        let dl = Window::build_cut(&s, 4, Leaves::Frontier, Some(cut), 1).unwrap();
        let ranges = random_ranges(&full, &mut rng);
        for prof in [full.uniform_profile(), hashed_profile(&full, rng.next_u64())] {
            let e_full = full.exploitability(&prof, &ranges, true, 1);
            let mut trunk = dl.uniform_profile();
            assert!(dl.embed(&mut trunk, &full, &prof, 0, 0).unwrap() > 0);
            let q = LeafQuery::from_reaches(&dl, &dl.leaf_reaches(&trunk, &ranges));
            let mut vals = vec![0.0; dl.n_leaf_values];
            for (j, leaf) in dl.value_leaves.iter().enumerate() {
                let m = full.find(&dl.leaf_history(j)).unwrap();
                let (sub, sp) = subtree(&full, m, &prof);
                for p in 0..2 {
                    let b = &q.beliefs[leaf.off[1 - p] as usize..][..leaf.n[1 - p] as usize];
                    let (_, fol) = sub.root_cfvs(p, &sp, b, true, 1);
                    vals[leaf.off[p] as usize..][..leaf.n[p] as usize].copy_from_slice(&fol);
                }
            }
            let e_dl = dl.exploitability_ext(&trunk, &ranges, true, &vals, 1);
            assert!((e_dl.value - e_full.value).abs() < 1e-12, "{} vs {}", e_dl.value, e_full.value);
            // The full profile back from the trunk and the leaf subtrees.
            let mut back = full.uniform_profile();
            let mut n = full.embed(&mut back, &dl, &trunk, 0, 0).unwrap();
            for j in 0..dl.value_leaves.len() {
                let m = full.find(&dl.leaf_history(j)).unwrap();
                let (sub, sp) = subtree(&full, m, &prof);
                n += full.embed(&mut back, &sub, &sp, m, 0).unwrap();
            }
            assert_eq!(n, full.n_entries);
            assert_eq!(back, prof);
        }
    }
}

#[test]
fn constant_leaf_values_average_to_themselves() {
    let mut rng = Rng::new(4);
    let s = random_position(&mut rng, 11);
    let dl = Window::build_cut(&s, 4, Leaves::Frontier, Some(2), 1).unwrap();
    let ranges = random_ranges(&dl, &mut rng);
    // A zero-sum leaf payoff that does not depend on the hands: player 0 gets c_j, player 1 gets 1 - c_j.
    let mut vals = vec![0.0; dl.n_leaf_values];
    for (j, leaf) in dl.value_leaves.iter().enumerate() {
        let c = (j % 7) as f64 / 7.0;
        vals[leaf.off[0] as usize..][..leaf.n[0] as usize].fill(c);
        vals[leaf.off[1] as usize..][..leaf.n[1] as usize].fill(1.0 - c);
    }
    let mut cfr = Cfr::new(&dl, ranges.clone(), Algo::Dcfr);
    assert!(cfr.ext_end(&dl, vals.clone()).is_err(), "ext_end needs ext_begin");
    for _ in 0..200 {
        cfr.ext_begin(&dl);
        assert!(cfr.ext_end(&dl, vec![0.5; 3]).is_err(), "wrong length");
        cfr.ext_end(&dl, vals.clone()).unwrap();
    }
    assert_eq!(cfr.t, 100);
    let avg = cfr.avg_leaf_values(&dl);
    assert!(avg.iter().zip(&vals).all(|(a, v)| (a - v).abs() < 1e-12));
    // With fixed (zero-sum) leaf values this is CFR on a fixed game: the average converges to its equilibrium.
    let e = dl.exploitability_ext(&cfr.average(&dl), &ranges, false, &vals, 1);
    assert!(e.exploitability < 1e-4, "{}", e.exploitability);
}

#[test]
fn leaf_values_average_with_the_average_strategy_weights() {
    // Values that change every half-iteration: the average for player o at leaf j must be
    // sum_t w_t m_i^t v_o^t / sum_t w_t m_i^t over the half-iterations of i = 1 - o (w_t = t^2 for DCFR, t for CFR+).
    let mut rng = Rng::new(9);
    let s = random_position(&mut rng, 11);
    let dl = Window::build_cut(&s, 4, Leaves::Frontier, Some(2), 1).unwrap();
    let ranges = random_ranges(&dl, &mut rng);
    for (algo, weight) in [(Algo::Dcfr, 2), (Algo::CfrPlus, 1)] {
        let mut cfr = Cfr::new(&dl, ranges.clone(), algo);
        let n = dl.value_leaves.len();
        let (mut num, mut den) = (vec![0.0; dl.n_leaf_values], vec![[0.0; 2]; n]);
        let mut last = Vec::new();
        for half in 0..80 {
            let mass = cfr.ext_begin(&dl).mass.clone();
            let i = half % 2;
            let wt = (cfr.t as f64).powi(weight);
            let vals: Vec<f64> = (0..dl.n_leaf_values).map(|_| rng.below(1000) as f64 / 1000.0).collect();
            for (j, leaf) in dl.value_leaves.iter().enumerate() {
                let o = 1 - i;
                den[j][o] += wt * mass[j][i];
                for k in leaf.off[o] as usize..leaf.off[o] as usize + leaf.n[o] as usize {
                    num[k] += wt * mass[j][i] * vals[k];
                }
            }
            cfr.ext_end(&dl, vals.clone()).unwrap();
            last = vals;
        }
        let avg = cfr.avg_leaf_values(&dl);
        let mut checked = 0;
        for (j, leaf) in dl.value_leaves.iter().enumerate() {
            for o in 0..2 {
                for k in leaf.off[o] as usize..leaf.off[o] as usize + leaf.n[o] as usize {
                    let want = if den[j][o] > 0.0 { num[k] / den[j][o] } else { last[k] };
                    assert!((avg[k] - want).abs() < 1e-12, "{algo:?} leaf {j} player {o}: {} vs {want}", avg[k]);
                    checked += usize::from(den[j][o] > 0.0);
                }
            }
        }
        assert!(checked > 0);
    }
}
