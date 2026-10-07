# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release.
- Level-1 routines for `f32` and `f64`: `axpy`, `scal`, `dot`, `nrm2`, `asum` (`_f32` / `_f64` suffixes), written to auto-vectorise, with `W` independent partial sums (32 for `f32`, 16 for `f64`), a fixed pairwise reduction and the tail added last.
- Column-major `gemv_*` (`y = alpha*A*x + beta*y`) and `gemv_t_*` (transposed), with explicit leading dimension.
- Packed, register- and cache-blocked `gemm_*` (`C = alpha*A*B + beta*C`), with `gemm_with_workspace_*` and `gemm_workspace_len_*` for caller-provided packing buffers (no allocation).
- AVX2+FMA `gemm` microkernels (16x4 `f32`, 8x4 `f64`) on x86_64, selected at compile time via target features, or at run time with the `runtime-dispatch` feature (detected once and cached); portable lane-array microkernel elsewhere.
- `reference` module with naive scalar implementations used as the test oracle and benchmark baseline.
- BLAS special cases: `beta == 0` overwrites the output, `alpha == 0` skips reading inputs, `k == 0` and zero dimensions are valid; descriptive panics for bad dimensions, leading dimensions, lengths and workspaces.
- Feature flags `alloc` (default), `std`, `runtime-dispatch`, `nightly` and `scalar-only`.
- `#![no_std]`; `unsafe` confined to the AVX2+FMA kernels and forbidden when they are not compiled.
- Unit and property tests (`src/tests.rs`) and Criterion benchmarks (`benches/blas.rs`).

### Notes

- Results are Tier 2 (ADR 0003): reassociated reductions and, on the AVX2+FMA path, fused multiply-add, so they differ from the naive reference by a few ulps and can differ slightly between the portable and AVX2+FMA paths.
- `nrm2_*` does no overflow/underflow scaling.
