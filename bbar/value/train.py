"""Train the PBS value network on exact labels (bbar/value/gen.py shards).

    python -m bbar.value.train --name NAME --data data/value/A [data/value/B ...] [--steps 20000] [--batch 1024]
        [--width 512 --blocks 3 --expansion 1] [--lr 1e-3] [--device auto] [--val-permille 50] [--test-permille 50]

Model: bbar/value/net.py (residual MLP, sigmoid outputs, zero-sum shift unless --no-zero-sum). Loss per label:
range-weighted Huber over both players' hands, sum_p sum_h b_p(h) huber(v_p(h) - y_p(h)) / 2, plus --all-weight times
the mean Huber over all feasible hands (hands outside the ranges still matter: they are the values of deviations).
Two held-out splits of whole public states (hash bucket of the public key, so range draws of one state never
straddle a split): the validation split (--val-permille) picks best.pt; the test split (--test-permille, the next
buckets) is used once, to report best.pt's error, which the selection does not bias. Metrics, overall and per ply:
range-weighted MAE and RMSE of the per-hand values (rw_mae, rw_rmse), MAE over all feasible hands (all_mae), and the
error of the game value b_m . v_m against the label (gv_mae, gv_rmse); the constant-0.5 predictor's rw_mae is logged
once for scale.

Outputs: runs/value/NAME/metrics.jsonl (config line, then train and val lines, and a final test line: best.pt's
metrics on the test split), checkpoints/value/NAME/{latest.pt, best.pt} (best by overall val rw_mae; best.pt also
holds its test metrics) with the hyperparameters, PBS version, data files and provenance.
"""
import argparse
import glob
import json
import math
import shlex
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bbar import provenance
from bbar.value import pbs
from bbar.value.net import ValueNet

ROOT = Path(__file__).resolve().parents[2]


def shard_paths(sources):
    out = []
    for s in sources:
        p = Path(s)
        out += sorted(glob.glob(str(p / "*.npz"))) if p.is_dir() else sorted(glob.glob(s))
    return [x for x in out if not x.endswith(".tmp.npz")]


def load_labels(sources):
    """Concatenated label arrays of every shard, with the PBS public features and feasibility masks."""
    paths = shard_paths(sources)
    if not paths:
        raise SystemExit(f"no shards in {sources}")
    cols = {k: [] for k in ("key", "start", "ply", "ranges", "values", "game_value", "exploitability", "kinds")}
    for p in paths:
        with np.load(p) as z:
            for k in cols:
                cols[k].append(z[k])
    d = {k: np.concatenate(v) for k, v in cols.items()}
    d["pub"] = pbs.public_features(d["key"], d["start"])
    d["mask"] = pbs.masks_of_keys(d["key"])
    return d, paths


def split(keys, val_permille, test_permille=0, salt=0x9E3779B97F4A7C15):
    """(val, test) bool [n] by a hash bucket of the public key in [0, 1000) (all range draws of a state on the same
    side): validation below val_permille, test in [val_permille, val_permille + test_permille), training above."""
    h = ((keys.astype(np.uint64) * np.uint64(salt)) >> np.uint64(40)) % np.uint64(1000)
    return h < np.uint64(val_permille), (h >= np.uint64(val_permille)) & (h < np.uint64(val_permille + test_permille))


def huber(x, delta):
    a = x.abs()
    return torch.where(a < delta, 0.5 * x * x, delta * (a - 0.5 * delta))


def losses(pred, y, r, mask, delta, all_weight):
    h = huber(pred - y, delta)
    rw = (r * h).sum((1, 2)) / 2
    m = mask.float()
    al = (m * h).sum((1, 2)) / m.sum((1, 2)).clamp(min=1)
    return (rw + all_weight * al).mean()


@torch.no_grad()
def evaluate(net, data, idx, device, batch=4096):
    """Per-label errors on rows idx: dict of numpy arrays."""
    net.eval()
    out = {k: [] for k in ("rw_abs", "rw_sq", "all_abs", "gv_err", "const_rw_abs")}
    for i in range(0, len(idx), batch):
        j = idx[i:i + batch]
        pub = torch.as_tensor(data["pub"][j], device=device)
        r = torch.as_tensor(data["ranges"][j], device=device)
        y = torch.as_tensor(data["values"][j], device=device)
        m = torch.as_tensor(data["mask"][j], device=device).float()
        v = net(pub, r).clamp(0, 1)
        e = v - y
        out["rw_abs"].append(((r * e.abs()).sum((1, 2)) / 2).cpu().numpy())
        out["rw_sq"].append(((r * e * e).sum((1, 2)) / 2).cpu().numpy())
        out["all_abs"].append(((m * e.abs()).sum((1, 2)) / m.sum((1, 2))).cpu().numpy())
        gv = (r[:, 0] * v[:, 0]).sum(1)
        out["gv_err"].append((gv - torch.as_tensor(data["game_value"][j], device=device)).cpu().numpy())
        out["const_rw_abs"].append(((r * (0.5 - y).abs()).sum((1, 2)) / 2).cpu().numpy())
    net.train()
    return {k: np.concatenate(v) for k, v in out.items()}


def summarize(err, plies):
    def stats(sel):
        return dict(n=int(sel.sum()), rw_mae=float(err["rw_abs"][sel].mean()),
                    rw_rmse=float(math.sqrt(err["rw_sq"][sel].mean())), all_mae=float(err["all_abs"][sel].mean()),
                    gv_mae=float(np.abs(err["gv_err"][sel]).mean()),
                    gv_rmse=float(math.sqrt((err["gv_err"][sel] ** 2).mean())))
    out = {"all": stats(np.ones(len(plies), bool))}
    for p in sorted(set(plies.tolist())):
        out[str(p)] = stats(plies == p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bbar.value.train", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", nargs="+", required=True, help="shard directories or globs")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--expansion", type=int, default=1)
    ap.add_argument("--no-zero-sum", action="store_true")
    ap.add_argument("--huber", type=float, default=0.1, help="Huber delta")
    ap.add_argument("--all-weight", type=float, default=0.1, help="weight of the all-feasible-hands term")
    ap.add_argument("--val-permille", type=int, default=50,
                    help="validation share of public states (checkpoint selection), per mille")
    ap.add_argument("--test-permille", type=int, default=50,
                    help="test share of public states (reporting best.pt only; 0 = none), per mille")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite runs/value/NAME and checkpoints/value/NAME")
    a = ap.parse_args(argv)
    if a.val_permille <= 0 or a.test_permille < 0 or a.val_permille + a.test_permille >= 1000:
        ap.error("need 0 < --val-permille, 0 <= --test-permille and their sum < 1000")

    from bbar.match import resolve_device
    device = resolve_device(a.device)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    run_dir, ck_dir = ROOT / "runs" / "value" / a.name, ROOT / "checkpoints" / "value" / a.name
    if (run_dir / "metrics.jsonl").exists() and not a.force:
        ap.error(f"{run_dir} exists: pick another --name or --force")
    run_dir.mkdir(parents=True, exist_ok=True)
    ck_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data, paths = load_labels(a.data)
    held, test = split(data["key"], a.val_permille, a.test_permille)
    tr_idx, va_idx, te_idx = np.flatnonzero(~held & ~test), np.flatnonzero(held), np.flatnonzero(test)
    if not len(tr_idx) or not len(va_idx) or (a.test_permille and not len(te_idx)):
        ap.error(f"{len(tr_idx)} training, {len(va_idx)} validation and {len(te_idx)} test labels: need each "
                 "(--test-permille 0 drops the test split)")
    n_states = len(np.unique(data["key"]))
    print(f"{len(data['key'])} labels ({n_states} public states) from {len(paths)} shards in {time.time() - t0:.1f}s: "
          f"{len(tr_idx)} train, {len(va_idx)} validation, {len(te_idx)} test; "
          f"plies {np.bincount(data['ply']).nonzero()[0].tolist()}; device {device}", flush=True)
    # Training tensors on the device (a few KB per label).
    dev = {k: torch.as_tensor(data[k][tr_idx]).to(device) for k in ("pub", "ranges", "values", "mask")}

    net = ValueNet(a.width, a.blocks, a.expansion, zero_sum=not a.no_zero_sum).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    nparams = sum(p.numel() for p in net.parameters())
    revision = provenance.revision()
    config = dict(vars(a), device=str(device), params=nparams, data_files=[str(p) for p in paths],
                  n_train=len(tr_idx), n_val=len(va_idx), n_test=len(te_idx), revision=revision,
                  host=socket.gethostname(),
                  command=shlex.join(["python", *sys.orig_argv[1:]]) if argv is None else
                  shlex.join(["python", "-m", "bbar.value.train", *argv]), pbs_version=pbs.PBS_VERSION)
    log = open(run_dir / "metrics.jsonl", "w")

    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()
    write(dict(kind="config", **config))

    def lr_at(step):
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        x = (step - a.warmup) / max(1, a.steps - a.warmup)
        return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * min(1.0, x)))

    def save(path, step, metrics):
        torch.save(dict(model=net.state_dict(), hparams=net.hparams, pbs_version=pbs.PBS_VERSION, step=step,
                        metrics=metrics, config=config), path)

    best, best_step = float("inf"), None
    va_plies = data["ply"][va_idx]
    err0 = evaluate(net, data, va_idx, device)
    print(f"held out: constant 0.5 predictor rw_mae {err0['const_rw_abs'].mean():.4f}", flush=True)
    write(dict(kind="baseline", const_rw_mae=float(err0["const_rw_abs"].mean())))
    running, t_last = 0.0, time.time()
    n_tr = len(tr_idx)
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        b = torch.as_tensor(rng.integers(0, n_tr, a.batch), device=device)
        pred = net(dev["pub"][b], dev["ranges"][b])
        loss = losses(pred, dev["values"][b], dev["ranges"][b], dev["mask"][b], a.huber, a.all_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        running += loss.item()
        if step % a.log_every == 0:
            rec = dict(kind="train", step=step, loss=running / a.log_every, lr=lr_at(step),
                       steps_per_sec=a.log_every / (time.time() - t_last))
            write(rec)
            running, t_last = 0.0, time.time()
        if step % a.eval_every == 0 or step == a.steps:
            m = summarize(evaluate(net, data, va_idx, device), va_plies)
            write(dict(kind="val", step=step, **m))
            al = m["all"]
            print(f"step {step}: val rw_mae {al['rw_mae']:.4f} rw_rmse {al['rw_rmse']:.4f} all_mae "
                  f"{al['all_mae']:.4f} gv_mae {al['gv_mae']:.4f}  " +
                  " ".join(f"p{p}:{v['rw_mae']:.4f}" for p, v in m.items() if p != "all"), flush=True)
            save(ck_dir / "latest.pt", step, m)
            if al["rw_mae"] < best:
                best, best_step = al["rw_mae"], step
                save(ck_dir / "best.pt", step, m)
    test_mae = None
    if len(te_idx) and best_step is not None:
        # best.pt's error on labels used neither for training nor for picking it.
        ck = torch.load(ck_dir / "best.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(ck["model"])
        m = summarize(evaluate(net, data, te_idx, device), data["ply"][te_idx])
        write(dict(kind="test", step=best_step, **m))
        ck["test_metrics"] = m
        torch.save(ck, ck_dir / "best.pt")
        test_mae = m["all"]["rw_mae"]
    print(f"done in {time.time() - t0:.0f}s; best.pt: step {best_step}, val rw_mae {best:.4f} (selection), test rw_mae "
          f"{'n/a' if test_mae is None else f'{test_mae:.4f}'}; checkpoints in {ck_dir}", flush=True)
    return best


if __name__ == "__main__":
    main()
