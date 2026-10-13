//! Window solver performance on the real game.
//!
//!     cargo run --release --example window_bench [ply] [positions] [target] [threads] [algos] [seed]
//!
//! Samples public positions at `ply` by uniformly random play from random deals, builds the frontier window below
//! each (uniform root ranges) and solves it to `target` exploitability with each algorithm (comma-separated:
//! dcfr, pcfr+, cfr+). Reports window sizes, build time (tree, frontier values) and solve time / iterations
//! (mean, p50, max).

use std::time::Instant;

use bbcore::hands::Variant;
use bbcore::window::{Algo, Cfr, Leaves, Window};
use bbcore::{Rng, State};

fn sample(rng: &mut Rng, ply: usize) -> State {
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

fn stats(xs: &mut [f64]) -> String {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    let mean = xs.iter().sum::<f64>() / n as f64;
    format!("mean {mean:>9.4}  p50 {:>9.4}  max {:>9.4}", xs[n / 2], xs[n - 1])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let ply: usize = args.get(1).map_or(12, |s| s.parse().unwrap());
    let n: usize = args.get(2).map_or(10, |s| s.parse().unwrap());
    let target: f64 = args.get(3).map_or(1e-4, |s| s.parse().unwrap());
    let threads: usize = args.get(4).map_or(1, |s| s.parse().unwrap());
    let algos: Vec<Algo> = args
        .get(5)
        .map_or("dcfr,pcfr+".to_string(), |s| s.clone())
        .split(',')
        .map(|a| Algo::parse(a).unwrap())
        .collect();
    let seed: u64 = args.get(6).map_or(1, |s| s.parse().unwrap());
    let mut rng = Rng::new(seed);
    let positions: Vec<State> = (0..n).map(|_| sample(&mut rng, ply)).collect();
    println!("ply {ply}, {n} positions, target {target:e}, threads {threads}");
    let mut sizes = Vec::new();
    let mut build = Vec::new();
    let mut solves: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = algos.iter().map(|_| Default::default()).collect();
    for (i, s) in positions.iter().enumerate() {
        let t0 = Instant::now();
        let w = Window::build(s, 4, Leaves::Frontier, threads).unwrap();
        let bt = t0.elapsed().as_secs_f64();
        build.push(bt);
        sizes.push((w.stats.internal, w.stats.frontier_leaves, w.stats.frontier_positions, w.n_entries));
        let ranges = [vec![1.0; w.root_hands(0).len()], vec![1.0; w.root_hands(1).len()]];
        let mut line = format!(
            "pos {i:>3}: internal {:>7} leaves {:>8} (frontier {:>8}, distinct {:>7}) entries {:>9} hands {}x{} build {:.3}s (frontier {:.3}s)",
            w.stats.internal,
            w.stats.terminal_leaves + w.stats.frontier_leaves,
            w.stats.frontier_leaves,
            w.stats.frontier_positions,
            w.n_entries,
            w.root_hands(0).len(),
            w.root_hands(1).len(),
            bt,
            w.stats.frontier_seconds
        );
        for (a, &algo) in algos.iter().enumerate() {
            let mut cfr = Cfr::new(&w, ranges.clone(), algo);
            cfr.threads = threads;
            let r = cfr.solve(&w, target, 100_000, 10);
            solves[a].0.push(r.seconds);
            solves[a].1.push(r.iterations as f64);
            solves[a].2.push(r.exploitability);
            line += &format!(
                " | {} {:>5} it {:.3}s e {:.1e} v {:.4}",
                algo.name(),
                r.iterations,
                r.seconds,
                r.exploitability,
                r.expl.value
            );
        }
        println!("{line}");
    }
    let mut internal: Vec<f64> = sizes.iter().map(|s| s.0 as f64).collect();
    let mut leaves: Vec<f64> = sizes.iter().map(|s| s.1 as f64).collect();
    println!("internal nodes   {}", stats(&mut internal));
    println!("frontier leaves  {}", stats(&mut leaves));
    println!("build seconds    {}", stats(&mut build));
    for (a, algo) in algos.iter().enumerate() {
        println!("{:<6} seconds    {}", algo.name(), stats(&mut solves[a].0));
        println!("{:<6} iterations {}", algo.name(), stats(&mut solves[a].1));
    }
}
