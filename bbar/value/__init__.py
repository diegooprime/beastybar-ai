"""Public-belief-state value network (experiment E2 of docs/PLAN.md): encoding, exact labels, training, evaluation.

    bbar/value/pbs.py    PBS encoding (public features, v2 with option features from core/src/options.rs, + both
                         ranges), hand-slot tables, root-range samplers
    bbar/value/sample.py public states from games of a policy mix
    bbar/value/gen.py    exact labels: per-hand values of both players from exact window solves (npz shards)
    bbar/value/net.py    the value networks (--arch mlp, hand, opt; networks of PBS versions 1 and 2)
    bbar/value/train.py  trainer
    bbar/value/dl.py     depth-limited solving: leaf value functions (network, exact oracle), lockstep batched solves
                         (solve_many) and the composite profile
    bbar/value/e2.py     E2: exploitability of depth-limited solving with network leaves, in the full window
    bbar/value/cascade.py  labels at earlier plies from depth-limited solves valued by the network; the cascade
"""
