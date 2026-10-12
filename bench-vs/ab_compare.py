#!/usr/bin/env python3
"""Interleaved A/B comparison of two versions of the bench-vs suites.

Ratios measured across separate sessions are unreliable on a laptop: the same incumbent moved
up to 35% within one session (docs/incumbent-benchmarks.md, Round 6). This script runs version A
and version B alternately, several times each, in one session, and reports the median of each
side and the ratio B / A for every benchmark name.

Usage (from the repository root):

    python bench-vs/ab_compare.py --a HEAD --b WORKTREE --suites text_vs --runs 2

`--a` and `--b` are git refs, or `WORKTREE` for the current working tree (including uncommitted
changes). A git ref is checked out into a temporary worktree and removed afterwards.
"""

import argparse
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

TIME = re.compile(r"time:\s+\[(\S+) (\S+) (\S+) (\S+) (\S+) (\S+)\]")
UNIT_NS = {"ps": 1e-3, "ns": 1.0, "µs": 1e3, "us": 1e3, "ms": 1e6, "s": 1e9}


def parse_medians(text):
    """Map each benchmark name to its criterion median, in nanoseconds."""
    medians = {}
    previous = ""
    for line in text.splitlines():
        match = TIME.search(line)
        if match:
            # Criterion puts a long name on its own line, and the timing on the next line.
            name = line[: match.start()].strip() or previous
            medians[name] = float(match.group(3)) * UNIT_NS[match.group(4)]
        elif line.strip() and "time:" not in line and "p =" not in line:
            previous = line.strip()
    return medians


def repo_root():
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True
    )
    return Path(out.stdout.strip())


def run_bench(directory, suite):
    proc = subprocess.run(
        ["cargo", "bench", "-p", "bench-vs", "--bench", suite],
        cwd=directory,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        sys.exit(f"bench {suite} failed in {directory}:\n{proc.stderr[-3000:]}")
    return parse_medians(proc.stdout + proc.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--a", required=True, help="git ref, or WORKTREE for the working tree")
    parser.add_argument("--b", required=True, help="git ref, or WORKTREE for the working tree")
    parser.add_argument("--suites", nargs="+", required=True, help="bench-vs suite names")
    parser.add_argument("--runs", type=int, default=2, help="interleaved runs per side")
    args = parser.parse_args()

    root = repo_root()
    temporary = []

    def directory_for(ref):
        if ref == "WORKTREE":
            return root
        path = Path(tempfile.mkdtemp(prefix="ab-")) / "tree"
        subprocess.run(
            ["git", "-C", str(root), "worktree", "add", "--detach", str(path), ref],
            check=True,
            capture_output=True,
        )
        temporary.append(path)
        return path

    try:
        dir_a = directory_for(args.a)
        dir_b = directory_for(args.b)
        samples = {}  # (suite, name) -> {"A": [...], "B": [...]}
        for suite in args.suites:
            for run in range(args.runs):
                for side, directory in (("A", dir_a), ("B", dir_b)):
                    print(f"[{suite}] run {run + 1}/{args.runs} side {side}", file=sys.stderr)
                    for name, value in run_bench(directory, suite).items():
                        samples.setdefault((suite, name), {"A": [], "B": []})[side].append(value)
    finally:
        for path in temporary:
            subprocess.run(
                ["git", "-C", str(root), "worktree", "remove", "--force", str(path.parent / "tree")],
                capture_output=True,
            )
            shutil.rmtree(path.parent, ignore_errors=True)

    print("| suite | benchmark | A median (ns) | B median (ns) | B / A | A runs | B runs |")
    print("|---|---|---|---|---|---|---|")
    for (suite, name), sides in samples.items():
        if not sides["A"] or not sides["B"]:
            continue
        a = statistics.median(sides["A"])
        b = statistics.median(sides["B"])
        spread_a = " / ".join(f"{v:.1f}" for v in sides["A"])
        spread_b = " / ".join(f"{v:.1f}" for v in sides["B"])
        print(f"| {suite} | {name} | {a:.2f} | {b:.2f} | {b / a:.2f} | {spread_a} | {spread_b} |")


if __name__ == "__main__":
    main()
