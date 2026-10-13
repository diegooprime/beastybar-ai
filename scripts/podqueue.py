"""Keep pods busy: start queued jobs on pods as they go idle (runs on the laptop).

    python3 scripts/podqueue.py QUEUE [--every 60] [--hours 4]

QUEUE is a text file with one job per line, `POD[,POD...] NAME COMMAND...`; blank lines and lines starting with #
are skipped. Every --every seconds, each pod with no job running (no `python -m bbar...` or `scripts/*.sh`
process) takes the first line that names it (the first idle pod of a list takes a line naming several), which is
removed from the file and started with `scripts/pod.sh bg POD NAME COMMAND` (COMMAND is a shell command line; its
output goes to runs/NAME.log on the pod). Edit the file at any time to add or reorder jobs.
"""
import argparse
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
POD = str(REPO / "scripts" / "pod.sh")
# bracketed first letters: the pattern must not match the ssh command that carries it
BUSY = "pgrep -fc 'python -m [b]bar|scripts/[a-z0-9_]*[.]sh' || true"


def idle(pod):
    r = subprocess.run([POD, "ssh", pod, BUSY], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    return r.stdout.strip() == "0"   # unreachable or unknown counts as busy


def jobs(queue):
    out = []
    for line in queue.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            pods, name, cmd = s.split(None, 2)
            out.append((line, pods.split(","), name, cmd))
    return out


def remove(queue, line):
    lines = queue.read_text().splitlines()
    lines.remove(line)
    queue.write_text("".join(x + "\n" for x in lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("queue", type=Path)
    ap.add_argument("--every", type=float, default=60, help="seconds between checks")
    ap.add_argument("--hours", type=float, default=4, help="stop after this long")
    a = ap.parse_args()
    end = time.time() + 3600 * a.hours
    while time.time() < end:
        state = {}
        for line, pods, name, cmd in jobs(a.queue):
            for pod in pods:
                if pod not in state:
                    state[pod] = idle(pod)
                if state[pod]:
                    subprocess.run([POD, "bg", pod, name, cmd], stdin=subprocess.DEVNULL, check=True)
                    remove(a.queue, line)
                    state[pod] = False
                    print(f"{time.strftime('%H:%M:%S')} {pod}: started {name}", flush=True)
                    break
        time.sleep(a.every)


if __name__ == "__main__":
    main()
