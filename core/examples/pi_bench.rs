//! Perfect-information solver performance on the real game.
//!
//!     cargo run --release --example pi_bench [positions] [seed] [log2_buckets]
//!
//! Samples positions by random play from random deals: ply-16 positions (both hands public) and ply-15 frontier
//! nodes (the second mover's last deck card is known by elimination). Reports nodes per solve (mean, p50, p99,
//! max) and solves per second, with the transposition table cleared before every solve ("cold") and kept across
//! solves ("warm"), plus plain-minimax node counts for comparison. Last, what a window solve rooted at ply 12
//! needs: the frontier values of every public ply-15 node below a public ply-12 root, sharing one table.

use std::time::Instant;

use bbcore::hands::Variant;
use bbcore::markov::public;
use bbcore::pisolve::{PiSolver, minimax};
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

struct Summary {
    nodes: Vec<u64>,
    secs: f64,
}

impl Summary {
    fn print(&mut self, what: &str) {
        self.nodes.sort_unstable();
        let n = self.nodes.len();
        let mean = self.nodes.iter().sum::<u64>() as f64 / n as f64;
        println!(
            "{what:<34} n {n:>6}  nodes mean {mean:>9.1}  p50 {:>7}  p99 {:>8}  max {:>8}  {:>9.0} solves/s",
            self.nodes[n / 2],
            self.nodes[n * 99 / 100],
            self.nodes[n - 1],
            n as f64 / self.secs
        );
    }
}

fn run(positions: &[State], solver: &mut PiSolver, cold: bool, frontier: bool) -> Summary {
    let mut nodes = Vec::with_capacity(positions.len());
    let mut secs = 0.0;
    for s in positions {
        if cold {
            solver.clear();
        }
        solver.reset_stats();
        let t = Instant::now();
        if frontier {
            std::hint::black_box(solver.frontier_values(s, 4));
        } else {
            std::hint::black_box(solver.value(s));
        }
        secs += t.elapsed().as_secs_f64();
        nodes.push(solver.stats.nodes);
    }
    Summary { nodes, secs }
}

fn main() {
    let args: Vec<u64> = std::env::args().skip(1).map(|a| a.parse().expect("numeric argument")).collect();
    let n = *args.first().unwrap_or(&20_000) as usize;
    let seed = *args.get(1).unwrap_or(&1);
    let log2 = *args.get(2).unwrap_or(&18) as u32;
    let mut rng = Rng::new(seed);
    let ply16: Vec<State> = (0..n).map(|_| sample(&mut rng, 16)).collect();
    let ply15: Vec<State> = (0..n).map(|_| sample(&mut rng, 15)).collect();
    println!("{n} positions per set, seed {seed}, table 2^{log2} buckets ({} MiB)", (32u64 << log2) >> 20);

    let mut solver = PiSolver::new(log2);
    run(&ply16, &mut solver, true, false).print("ply 16 value, cold table");
    solver.clear();
    run(&ply16, &mut solver, false, false).print("ply 16 value, warm table");
    run(&ply15, &mut solver, true, true).print("ply 15 frontier (5 hands), cold");
    solver.clear();
    run(&ply15, &mut solver, false, true).print("ply 15 frontier (5 hands), warm");

    let m = n.min(2_000);
    let t = Instant::now();
    let nodes: Vec<u64> = ply16[..m]
        .iter()
        .map(|s| {
            let mut nodes = 0;
            minimax(s, &mut nodes);
            nodes
        })
        .collect();
    Summary { nodes, secs: t.elapsed().as_secs_f64() }.print("ply 16 plain minimax");
    let t = Instant::now();
    let nodes: Vec<u64> = ply15[..m]
        .iter()
        .map(|s| {
            let mut nodes = 0;
            minimax(&public(s), &mut nodes);
            nodes
        })
        .collect();
    Summary { nodes, secs: t.elapsed().as_secs_f64() }.print("ply 15 public minimax (all hands)");

    // Window leaves: every public ply-15 node below a public ply-12 root.
    fn frontier_nodes(s: &State, out: &mut Vec<State>) {
        if s.is_over() {
            return;
        }
        if Variant::REAL.ply(s) == 15 {
            out.push(*s);
            return;
        }
        for a in s.legal_actions() {
            let mut c = *s;
            c.apply(a);
            frontier_nodes(&c, out);
        }
    }
    let roots = (n / 200).max(5);
    let (mut leaves, mut nodes, mut secs) = (Vec::new(), Vec::new(), 0.0);
    for _ in 0..roots {
        let root = public(&sample(&mut rng, 12));
        let mut fs = Vec::new();
        frontier_nodes(&root, &mut fs);
        solver.clear();
        solver.reset_stats();
        let t = Instant::now();
        for f in &fs {
            std::hint::black_box(solver.frontier_values(f, 4));
        }
        secs += t.elapsed().as_secs_f64();
        leaves.push(fs.len() as u64);
        nodes.push(solver.stats.nodes);
    }
    let mean = |v: &[u64]| v.iter().sum::<u64>() as f64 / v.len() as f64;
    println!(
        "ply 12 window leaves ({roots} roots)        frontier nodes per root mean {:.0} max {}, search nodes per root \
         mean {:.0}, {:.1} ms per root, {:.0} frontier solves/s",
        mean(&leaves),
        leaves.iter().max().unwrap(),
        mean(&nodes),
        1e3 * secs / roots as f64,
        leaves.iter().sum::<u64>() as f64 / secs
    );
}
