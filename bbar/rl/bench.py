"""Throughput of the blueprint pipeline on this machine; run it first on a new pod.

    python -m bbar.rl.bench [--device cuda] [--n-envs 4096] [--rollout-steps 16] [--width 512 --blocks 4] ...

Reports (one JSON line): env encoding in Rust alone (observe + privileged + step with random legal actions) at
n_envs, then trainer decisions/s split into rollout (env + policy on the device) and learn (critic pass, GAE, PPO
epochs), for the network and batch given by the same flags as bbar.rl.mmd.
"""
import dataclasses
import json
import tempfile

import torch

from bbar._core import env_bench
from bbar.rl.mmd import Config, Trainer, build_parser


def main(argv=None):
    p = build_parser()
    p.add_argument("--updates", type=int, default=5, help="timed updates (after 3 warm-up updates)")
    p.add_argument("--env-steps", type=int, default=50, help="steps of the Rust-only env benchmark")
    args = p.parse_args(argv)
    cfg = Config(**{f.name: getattr(args, f.name) for f in dataclasses.fields(Config)})
    out = {"device": cfg.device, "n_envs": cfg.n_envs, "rollout_steps": cfg.rollout_steps, "width": cfg.width,
           "blocks": cfg.blocks, "minibatch": cfg.minibatch, "epochs": cfg.epochs}
    b = env_bench(cfg.n_envs, args.env_steps, 0, cfg.env_threads)
    out["env_observe_per_s"] = round(b["decisions"] / b["observe"])
    out["env_steps_per_s"] = round(b["decisions"] / (b["observe"] + b["privileged"] + b["step"]))
    with tempfile.TemporaryDirectory() as tmp:
        cfg = dataclasses.replace(cfg, runs_dir=f"{tmp}/runs", ckpt_dir=f"{tmp}/ckpt", name="bench")
        tr = Trainer(cfg)
        out["device"] = str(tr.device)
        for _ in range(3):
            tr.train_step()
        recs = [tr.train_step() for _ in range(args.updates)]
        tr.metrics.close()
    per_update = cfg.n_envs * cfg.rollout_steps
    rollout = sum(r["rollout_s"] for r in recs) / len(recs)
    learn = sum(r["learn_s"] for r in recs) / len(recs)
    out.update(decisions_per_update=per_update, rollout_s=round(rollout, 4), learn_s=round(learn, 4),
               rollout_decisions_per_s=round(per_update / rollout), learn_decisions_per_s=round(per_update / learn),
               trainer_decisions_per_s=round(per_update / (rollout + learn)),
               hours_per_1e9=round(1e9 / (per_update / (rollout + learn)) / 3600, 2))
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name()
        out["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
