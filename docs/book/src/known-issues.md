# Known issues

Documented issues and their status.

## Appendix B doctests

All spec Appendix B examples **compile and pass as doctests** (verified
`cargo test --doc` on `tpt-simd-butterfly`, `tpt-simd-complex`,
`tpt-simd-fixed`, `tpt-simd-dot`, `tpt-simd-permute`). They are embedded in
the crate READMEs / doc comments, so `cargo doc --workspace --no-deps` with
`RUSTDOCFLAGS=-Dwarnings` stays clean.

## Performance regressions

- `dot_product_i16` (256 lanes) meets the 8x target only against the indexed
  scalar baseline (measured 1.55x against the auto-vectorised baseline).
- `butterfly_f32` on baseline SSE2 is ~5x slower than an iterator loop; the
  8-lane array backend lowers poorly to 128-bit registers. Fix: compile with
  `-C target-feature=+avx2` for 8-lane f32 kernels, or specialise `F32x4`.
- Horizontal reductions (`sum_f32`, other single-register reductions) miss
  the 4-8x target; a single-register reduction is only a few cycles.
- `tpt-simd-blas` `runtime-dispatch` is implemented; `reduce`, `math`,
  `sparse`, `rng` currently use portable code only.

## Unsafe audit

`cargo test --doc` runs; Miri and `cargo-show-asm` audits are pending tools.
