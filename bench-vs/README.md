# bench-vs

Unpublished head-to-head benchmarks: tpt-simd kernels against incumbent crates.
Results and the method are in [../docs/incumbent-benchmarks.md](../docs/incumbent-benchmarks.md).

## Run recipe

Run each suite **twice, back to back**, and compare only ratios measured inside the
same run. On the dev machine the same incumbent moved 3–4× between sessions, so
absolute times from different runs cannot be compared.

```sh
for run in 1 2; do
  cargo bench -p bench-vs --bench hash_vs
done
```

Use `runtime-dispatch` as configured in `Cargo.toml`: the SIMD kernels are chosen at
runtime, the same as the incumbents. Do not set `RUSTFLAGS` for these runs.

## Interleaved A/B comparison

Two versions of a suite can be compared in one session with `ab_compare.py`. It runs
version A and version B alternately, several times each, and reports the median of each
side and the ratio B / A. This is the preferred way to show a change, since the incumbent
can move by 30% or more between sessions.

```sh
python bench-vs/ab_compare.py --a HEAD --b WORKTREE --suites text_vs --runs 2
```

`WORKTREE` means the current working tree, including uncommitted changes. A git ref is
built in a temporary worktree and removed afterwards. Each side builds once, so the first
run takes a while.
