"""Train the PBS value network on exact labels (bbar/value/gen.py shards) and cascade labels (bbar/value/cascade.py).

    python -m bbar.value.train --name NAME --data data/value/A [data/value/B ...] [--steps 20000] [--batch 1024]
        [--arch mlp|hand|opt] [--pbs 2] [--width 512 --blocks 3 --expansion 1] [--lr 1e-3] [--device auto]
        [--val-permille 50] [--test-permille 50] [--minutes M] [--train-permille 1000]

Model: bbar/value/net.py (--arch; sigmoid outputs, zero-sum shift unless --no-zero-sum), reading the public features
of PBS version --pbs (1: 134 features, 2: with the option features; opt needs 2). Loss per label:
range-weighted Huber over both players' hands, sum_p sum_h b_p(h) huber(v_p(h) - y_p(h)) / 2, plus --all-weight times
the mean Huber over all feasible hands (hands outside the ranges still matter: they are the values of deviations).
That second term needs labels that are right for every hand: a shard whose meta says all_hands_exact = false
(cascade shards: their values for hands with zero or tiny range mass are not gen.py's, bbar/value/cascade.py) gets
--cascade-all-weight (default 0) instead, and its labels count in all_mae only when that weight is positive.
Two held-out splits of whole public states (hash bucket of the public key, so range draws of one state never
straddle a split): the validation split (--val-permille) picks best.pt; the test split (--test-permille, the next
buckets) is used once, to report best.pt's error, which the selection does not bias. Metrics, overall and per ply:
range-weighted MAE and RMSE of the per-hand values (rw_mae, rw_rmse), MAE over all feasible hands (all_mae, None
where no label of the selection has all-hands values), and the error of the game value b_m . v_m against the label
(gv_mae, gv_rmse); the constant-0.5 predictor's rw_mae is logged once for scale.

Budget: --steps, and with --minutes also a wall-clock limit on the training steps (evaluations not counted): the
learning-rate schedule (warmup, then cosine to --min-lr) follows whichever is further along, and training stops when
either runs out (equal-time comparisons of architectures). --train-permille keeps that share of the training public
states (another hash of the key: the same states for every run, nested as it grows; validation and test unchanged):
data-scaling slices.

Outputs: runs/value/NAME/metrics.jsonl (config line, then train and val lines, a test line: best.pt's metrics on the
test split, and a final done line), checkpoints/value/NAME/{latest.pt, best.pt} (best by overall val rw_mae) with the
hyperparameters, PBS version, data files and provenance. Checkpoints are written atomically; best.pt is rewritten
whenever validation improves, and once more at the end with its test metrics and complete=True: only then is the
training finished (is_complete; bbar/value/cascade.py's --resume relies on it). A new run of NAME deletes NAME's old
checkpoints before its first step.
"""
import argparse
import glob
import json
import math
import os
import shlex
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

from bbar import provenance
from bbar.value import pbs
from bbar.value.net import ARCHS, make_value_net

ROOT = Path(__file__).resolve().parents[2]


def shard_paths(sources):
    out = []
    for s in sources:
        p = Path(s)
        out += sorted(glob.glob(str(p / "*.npz"))) if p.is_dir() else sorted(glob.glob(s))
    return [x for x in out if not x.endswith(".tmp.npz")]


def all_hands_exact(meta):
    """Whether a shard's values are labels for every feasible hand (gen.py), not only on the ranges (cascade.py)."""
    return bool(meta.get("all_hands_exact", meta.get("kind") != "cascade-labels"))


def load_labels(sources, pbs_version=pbs.PBS_VERSION):
    """Concatenated label arrays of every shard, with the public features of PBS version `pbs_version`, feasibility
    masks and all_exact (bool per label: its shard's all_hands_exact)."""
    paths = shard_paths(sources)
    if not paths:
        raise SystemExit(f"no shards in {sources}")
    cols = {k: [] for k in ("key", "start", "ply", "ranges", "values", "game_value", "exploitability", "kinds",
                            "all_exact")}
    for p in paths:
        with np.load(p) as z:
            for k in cols:
                if k != "all_exact":
                    cols[k].append(z[k])
            meta = json.loads(z["meta"].item()) if "meta" in z.files else {}
            cols["all_exact"].append(np.full(len(z["key"]), all_hands_exact(meta)))
    d = {k: np.concatenate(v) for k, v in cols.items()}
    d["pub"] = pbs.public_features(d["key"], d["start"], pbs_version)
    d["mask"] = pbs.masks_of_keys(d["key"])
    return d, paths


def split(keys, val_permille, test_permille=0, salt=0x9E3779B97F4A7C15):
    """(val, test) bool [n] by a hash bucket of the public key in [0, 1000) (all range draws of a state on the same
    side): validation below val_permille, test in [val_permille, val_permille + test_permille), training above."""
    h = ((keys.astype(np.uint64) * np.uint64(salt)) >> np.uint64(40)) % np.uint64(1000)
    return h < np.uint64(val_permille), (h >= np.uint64(val_permille)) & (h < np.uint64(val_permille + test_permille))


def train_subset(keys, permille, salt=0xD1B54A32D192ED03):
    """bool [n]: the public states kept for training at `permille` (a hash of the key independent of split's; all
    range draws of a state together; the kept sets are nested as permille grows)."""
    h = ((keys.astype(np.uint64) * np.uint64(salt)) >> np.uint64(40)) % np.uint64(1000)
    return h < np.uint64(permille)


def huber(x, delta):
    a = x.abs()
    return torch.where(a < delta, 0.5 * x * x, delta * (a - 0.5 * delta))


def losses(pred, y, r, mask, delta, all_weight):
    """Mean loss of a batch; all_weight: a number or a tensor [batch] (per-label weight of the all-hands term)."""
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


def summarize(err, plies, all_exact=None):
    """Metrics overall and per ply; all_mae only over labels with all_exact (default: all), None if there are none."""
    all_exact = np.ones(len(plies), bool) if all_exact is None else np.asarray(all_exact, bool)

    def stats(sel):
        sa = sel & all_exact
        return dict(n=int(sel.sum()), rw_mae=float(err["rw_abs"][sel].mean()),
                    rw_rmse=float(math.sqrt(err["rw_sq"][sel].mean())),
                    all_mae=float(err["all_abs"][sa].mean()) if sa.any() else None,
                    gv_mae=float(np.abs(err["gv_err"][sel]).mean()),
                    gv_rmse=float(math.sqrt((err["gv_err"][sel] ** 2).mean())))
    out = {"all": stats(np.ones(len(plies), bool))}
    for p in sorted(set(plies.tolist())):
        out[str(p)] = stats(plies == p)
    return out


def all_weights(all_exact, all_weight, cascade_all_weight):
    """Per-label weight of the all-feasible-hands term: all_weight where the label's values are right for every hand,
    cascade_all_weight where only on the ranges (module docstring)."""
    return np.where(np.asarray(all_exact, bool), all_weight, cascade_all_weight)


def save_atomic(obj, path):
    """torch.save through a temporary file: an interruption never leaves a truncated checkpoint at `path`."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def is_complete(path):
    """Whether `path` is the best.pt of a training that ran to its end (complete=True, written last)."""
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:       # missing, or not a checkpoint
        return False
    return isinstance(ck, dict) and ck.get("complete") is True


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
    ap.add_argument("--arch", default="mlp", choices=sorted(ARCHS), help="mlp: 990-way linear head; hand: shared "
                    "per-hand head; opt: per-hand head built from the animals' option features (bbar/value/net.py)")
    ap.add_argument("--pbs", type=int, default=pbs.PBS_VERSION, choices=sorted(pbs.PUB_DIMS),
                    help="PBS version of the public features the network reads (2: with option features)")
    ap.add_argument("--hand-width", type=int, default=256, help="--arch hand, opt: width of the per-hand head")
    ap.add_argument("--tok-width", type=int, default=128, help="--arch opt: width of the per-animal tokens")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--expansion", type=int, default=1)
    ap.add_argument("--no-zero-sum", action="store_true")
    ap.add_argument("--huber", type=float, default=0.1, help="Huber delta")
    ap.add_argument("--all-weight", type=float, default=0.1, help="weight of the all-feasible-hands term")
    ap.add_argument("--cascade-all-weight", type=float, default=0.0,
                    help="weight of that term for labels whose values are right on the ranges only (shards with "
                         "all_hands_exact false: cascade labels)")
    ap.add_argument("--val-permille", type=int, default=50,
                    help="validation share of public states (checkpoint selection), per mille")
    ap.add_argument("--test-permille", type=int, default=50,
                    help="test share of public states (reporting best.pt only; 0 = none), per mille")
    ap.add_argument("--minutes", type=float, default=0,
                    help="also stop after this many minutes of training steps (0 = --steps only)")
    ap.add_argument("--train-permille", type=int, default=1000,
                    help="share of the training public states used, per mille (data-scaling slices)")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite runs/value/NAME and checkpoints/value/NAME")
    a = ap.parse_args(argv)
    if a.val_permille <= 0 or a.test_permille < 0 or a.val_permille + a.test_permille >= 1000:
        ap.error("need 0 < --val-permille, 0 <= --test-permille and their sum < 1000")
    if not 0 < a.train_permille <= 1000:
        ap.error("--train-permille must be in 1..1000")
    if a.arch == "opt" and a.pbs < 2:
        ap.error("--arch opt reads the option features: needs --pbs 2")

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
    data, paths = load_labels(a.data, a.pbs)
    held, test = split(data["key"], a.val_permille, a.test_permille)
    keep = train_subset(data["key"], a.train_permille)
    tr_idx, va_idx, te_idx = np.flatnonzero(~held & ~test & keep), np.flatnonzero(held), np.flatnonzero(test)
    if not len(tr_idx) or not len(va_idx) or (a.test_permille and not len(te_idx)):
        ap.error(f"{len(tr_idx)} training, {len(va_idx)} validation and {len(te_idx)} test labels: need each "
                 "(--test-permille 0 drops the test split)")
    n_states = len(np.unique(data["key"]))
    n_tr_states = len(np.unique(data["key"][tr_idx]))
    print(f"{len(data['key'])} labels ({n_states} public states) from {len(paths)} shards in {time.time() - t0:.1f}s: "
          f"{len(tr_idx)} train ({n_tr_states} states, {a.train_permille} permille kept), {len(va_idx)} validation, "
          f"{len(te_idx)} test; PBS v{a.pbs} ({pbs.PUB_DIMS[a.pbs]} public features); "
          f"plies {np.bincount(data['ply']).nonzero()[0].tolist()}; device {device}", flush=True)
    # Training tensors on the device (a few KB per label).
    dev = {k: torch.as_tensor(data[k][tr_idx]).to(device) for k in ("pub", "ranges", "values", "mask")}
    all_w = torch.as_tensor(all_weights(data["all_exact"][tr_idx], a.all_weight, a.cascade_all_weight),
                            dtype=torch.float32).to(device)
    # all_mae counts a label when the all-hands term trains on it.
    all_counted = data["all_exact"] | (a.cascade_all_weight > 0)
    n_partial = int((~data["all_exact"]).sum())
    if n_partial:
        print(f"{n_partial} labels with values right on the ranges only (cascade): all-hands weight "
              f"{a.cascade_all_weight}", flush=True)

    extra = {"hand": {"hand_width": a.hand_width}, "opt": {"hand_width": a.hand_width, "tok_width": a.tok_width}}
    net = make_value_net(a.arch, width=a.width, blocks=a.blocks, expansion=a.expansion, zero_sum=not a.no_zero_sum,
                         pub_dim=pbs.PUB_DIMS[a.pbs], **extra.get(a.arch, {})).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    nparams = sum(p.numel() for p in net.parameters())
    revision = provenance.revision()
    config = dict(vars(a), device=str(device), params=nparams, data_files=[str(p) for p in paths],
                  n_train=len(tr_idx), n_train_states=n_tr_states, n_val=len(va_idx), n_test=len(te_idx),
                  n_partial=n_partial, revision=revision,
                  host=socket.gethostname(),
                  command=shlex.join(["python", *sys.orig_argv[1:]]) if argv is None else
                  shlex.join(["python", "-m", "bbar.value.train", *argv]), pbs_version=a.pbs)
    log = open(run_dir / "metrics.jsonl", "w")
    for old in ("best.pt", "latest.pt"):     # a previous run's checkpoints must not pass for this run's
        (ck_dir / old).unlink(missing_ok=True)

    def write(rec):
        log.write(json.dumps(rec) + "\n")
        log.flush()
    write(dict(kind="config", **config))

    budget = 60 * a.minutes

    def progress(step, seconds):
        """How far training is along its budget, in [0, 1]: steps or, with --minutes, time, whichever is further."""
        return min(1.0, max(step / a.steps, seconds / budget if budget else 0.0))

    def lr_at(step, seconds=0.0):
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        p0 = progress(a.warmup, 0.0)    # the cosine runs from the end of the warmup to the end of the budget
        x = (progress(step, seconds) - p0) / max(1e-9, 1 - p0)
        return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * min(1.0, x)))

    def save(path, step, metrics):
        save_atomic(dict(model=net.state_dict(), hparams=net.hparams, pbs_version=a.pbs, step=step,
                         metrics=metrics, config=config, complete=False), path)

    best, best_step = float("inf"), None
    va_plies = data["ply"][va_idx]
    err0 = evaluate(net, data, va_idx, device)
    print(f"held out: constant 0.5 predictor rw_mae {err0['const_rw_abs'].mean():.4f}", flush=True)
    write(dict(kind="baseline", const_rw_mae=float(err0["const_rw_abs"].mean())))
    running, t_last, train_seconds, step = 0.0, time.time(), 0.0, 0
    n_tr = len(tr_idx)
    while True:
        step += 1
        t_step = time.time()
        lr = lr_at(step, train_seconds)
        for g in opt.param_groups:
            g["lr"] = lr
        b = torch.as_tensor(rng.integers(0, n_tr, a.batch), device=device)
        pred = net(dev["pub"][b], dev["ranges"][b])
        loss = losses(pred, dev["values"][b], dev["ranges"][b], dev["mask"][b], a.huber, all_w[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        running += loss.item()
        train_seconds += time.time() - t_step
        last = progress(step, train_seconds) >= 1.0
        if step % a.log_every == 0:
            rec = dict(kind="train", step=step, loss=running / a.log_every, lr=lr,
                       steps_per_sec=a.log_every / (time.time() - t_last), train_seconds=train_seconds)
            write(rec)
            running, t_last = 0.0, time.time()
        if step % a.eval_every == 0 or last:
            m = summarize(evaluate(net, data, va_idx, device), va_plies, all_counted[va_idx])
            write(dict(kind="val", step=step, **m))
            al = m["all"]
            all_mae = "n/a" if al["all_mae"] is None else f"{al['all_mae']:.4f}"
            print(f"step {step}: val rw_mae {al['rw_mae']:.4f} rw_rmse {al['rw_rmse']:.4f} all_mae "
                  f"{all_mae} gv_mae {al['gv_mae']:.4f}  " +
                  " ".join(f"p{p}:{v['rw_mae']:.4f}" for p, v in m.items() if p != "all"), flush=True)
            save(ck_dir / "latest.pt", step, m)
            if al["rw_mae"] < best:
                best, best_step = al["rw_mae"], step
                save(ck_dir / "best.pt", step, m)
            t_last = time.time()
        if last:
            break
    test_mae = None
    if best_step is None:
        raise SystemExit(f"{a.name}: no finite validation error in {step} steps: no best.pt")
    ck = torch.load(ck_dir / "best.pt", map_location="cpu", weights_only=False)
    if len(te_idx):
        # best.pt's error on labels used neither for training nor for picking it.
        net.load_state_dict(ck["model"])
        m = summarize(evaluate(net, data, te_idx, device), data["ply"][te_idx], all_counted[te_idx])
        write(dict(kind="test", step=best_step, **m))
        ck["test_metrics"] = m
        test_mae = m["all"]["rw_mae"]
    ck["complete"] = True       # written last: the training ran to its end
    save_atomic(ck, ck_dir / "best.pt")
    write(dict(kind="done", step=step, best_step=best_step, seconds=time.time() - t0, train_seconds=train_seconds))
    print(f"done in {time.time() - t0:.0f}s; best.pt: step {best_step}, val rw_mae {best:.4f} (selection), test rw_mae "
          f"{'n/a' if test_mae is None else f'{test_mae:.4f}'}; checkpoints in {ck_dir}", flush=True)
    return best


if __name__ == "__main__":
    main()
