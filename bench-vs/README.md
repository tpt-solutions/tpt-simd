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
