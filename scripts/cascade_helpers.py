"""Add helper pods to a cascade running on a hub pod (scripts/cascade.sh), from the laptop.

    python3 scripts/cascade_helpers.py --hub ID --log runs/cascade4.log --helpers ID1 ID2 [--minutes 30]
        [--exact ID --exact-dir data/value/x9 --into data/value/v3]

The cascade is sequential: label a level with the current network, train on everything, next level. Helpers make
each level bigger. This script polls the hub's log; when the hub starts labelling a level (its
"[cascade] level [A, B]: python -m bbar.value.cascade labels ... --net NET --out DIR ... --minutes M" line) it copies
NET to every helper, runs the same labels command there with its own seed, no --force and a shorter time limit, and
when the helpers are done copies their shards into the hub's DIR before the hub's own limit ends, so the hub's next
training includes them. With --exact, shards from an exact-label pod (bbar.value.gen) are also copied into the hub's
--into directory (a DATA directory of the cascade run) at every level. Pods only ever receive files and commands;
nothing runs on the laptop but rsync and ssh.
"""
import argparse
import re
import shlex
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
POD = str(REPO / "scripts" / "pod.sh")
STAGE = REPO / "backups" / "cascade-helpers"


def sh(*args, check=True):
    r = subprocess.run(args, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"{args}: {r.stderr.strip()[-500:]}")
    return r.stdout


def ssh(pod, cmd, check=True):
    return sh(POD, "ssh", pod, cmd, check=check)


def target(pod):
    out = sh("runpodctl", "ssh", "info", pod)
    ip = re.search(r'"ip":\s*"([\d.]+)"', out).group(1)
    port = re.search(r'"port":\s*(\d+)', out).group(1)
    return f"root@{ip}", port


def rsync(src, dst, pod_side, pod):
    """Copy between the laptop and a pod; pod_side 'src' or 'dst' says which path is on the pod."""
    host, port = target(pod)
    key = Path.home() / ".runpod/ssh/RunPod-Key-Go"
    e = f"ssh -i {key} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p {port}"
    remote = f"{host}:/workspace/beastybar-ai/"
    a = remote + src if pod_side == "src" else src
    b = remote + dst if pod_side == "dst" else dst
    sh("rsync", "-az", "--exclude", "*.tmp.npz", "-e", e, a, b)


def level_line(hub, log):
    out = ssh(hub, f"grep -E '^\\[cascade\\] level' {shlex.quote(log)} | tail -n 1", check=False).strip()
    return out or None


def helper_command(hub_cmd, seed, minutes, workers):
    args = shlex.split(hub_cmd.split(": ", 1)[1])
    out, skip = [], False
    for i, x in enumerate(args):
        if skip:
            skip = False
            continue
        if x == "--force":
            continue
        if x in ("--seed", "--minutes", "--workers"):
            skip = True
            continue
        out.append(x)
    return shlex.join(out + ["--seed", str(seed), "--minutes", str(minutes), "--workers", str(workers)])


def rel(path):
    return str(path).replace("/workspace/beastybar-ai/", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hub", required=True)
    ap.add_argument("--log", default="runs/cascade4.log")
    ap.add_argument("--helpers", nargs="*", default=[])
    ap.add_argument("--workers", nargs="*", type=int, default=None, help="label workers per helper (default 4)")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--exact", default=None, help="pod generating exact labels (bbar.value.gen)")
    ap.add_argument("--exact-dir", default="data/value/x9")
    ap.add_argument("--into", default="data/value/v3", help="hub DATA directory that receives the exact shards")
    ap.add_argument("--skip-current", action="store_true", help="wait for the hub's next level")
    a = ap.parse_args()
    workers = a.workers or [4] * len(a.helpers)
    seen = level_line(a.hub, a.log) if a.skip_current else None
    while True:
        line = level_line(a.hub, a.log)
        if line and line != seen:
            seen = line
            m = re.search(r"--net (\S+)", line)
            o = re.search(r"--out (\S+)", line)
            net, outdir = rel(m.group(1)), rel(o.group(1))
            print(f"{time.strftime('%H:%M:%S')} hub level: {line.split(':')[0]}  net {net}  out {outdir}", flush=True)
            STAGE.mkdir(parents=True, exist_ok=True)
            local_net = STAGE / net
            local_net.parent.mkdir(parents=True, exist_ok=True)
            rsync(net, str(local_net), "src", a.hub)
            t0 = time.time()
            for k, (pod, w) in enumerate(zip(a.helpers, workers), start=1):
                ssh(pod, f"mkdir -p {Path(net).parent}")
                rsync(str(local_net), net, "dst", pod)
                cmd = helper_command(line, 1000 + 100 * k + int(t0) % 100, a.minutes, w)
                ssh(pod, f"mkdir -p runs && nohup bash -c {shlex.quote(cmd)} > runs/helper-{Path(outdir).name}.log "
                         f"2>&1 &")
                print(f"  helper {pod}: started ({w} workers)", flush=True)
            # wait for the helpers to finish (they stop after --minutes plus the batch in flight)
            deadline = t0 + 60 * (a.minutes + 8)
            while time.time() < deadline:
                busy = [p for p in a.helpers
                        if ssh(p, "pgrep -fc 'python -m [b]bar.value.cascade labels' || true", check=False).strip()
                        not in ("", "0")]
                if not busy:
                    break
                time.sleep(30)
            local_out = STAGE / outdir
            local_out.mkdir(parents=True, exist_ok=True)
            for pod in a.helpers:
                rsync(outdir + "/", str(local_out) + "/", "src", pod)
            n = len(list(local_out.glob("*.npz")))
            rsync(str(local_out) + "/", outdir + "/", "dst", a.hub)
            print(f"  {time.strftime('%H:%M:%S')} copied {n} helper shards into the hub's {outdir}", flush=True)
            if a.exact:
                local_x = STAGE / a.exact_dir
                local_x.mkdir(parents=True, exist_ok=True)
                rsync(a.exact_dir + "/", str(local_x) + "/", "src", a.exact)
                rsync(str(local_x) + "/", a.into + "/", "dst", a.hub)
                print(f"  copied {len(list(local_x.glob('*.npz')))} exact shards into the hub's {a.into}", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    main()
