# Known issues

Documented issues and their status.

## Appendix B doctests

The spec's Appendix B snippets are checked by the integration test
`tpt-simd/tests/appendix_b.rs` (`cargo test -p tpt-simd --test appendix_b`),
which resolves their imports through the umbrella crate and asserts each
documented result. They are not doctests, so the book text is not compiled
by `mdbook test`. The snippets needed two spec fixes, recorded in `spec.txt`:
`Fixed` multiplies with `*` (wrapping, ties away), and `F32x8` must be
imported. Per-crate doc comments carry their own examples in addition.

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
