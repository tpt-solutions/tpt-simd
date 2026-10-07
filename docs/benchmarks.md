# Benchmarks (Phases 1–2)

Machine: x86_64 with AVX2 + FMA (no AVX-512), Windows 11, Rust 1.97 stable,
criterion. Numbers are from a shared, loaded machine, so expect a few percent
(more for single-register ops) of noise. "native" = `RUSTFLAGS="-C
target-cpu=native"`; "default" = baseline x86_64 (SSE2).

Reproduce: `cargo bench -p <crate>` (add the RUSTFLAGS above for native).

## How to read these

Two scalar baselines matter:

* **indexed scalar**: `for i in 0..n { out[i] = f(a[i], b[i]) }`, which LLVM often
  fails to vectorise.
* **iterator scalar**: `zip`/`iter` loops, which LLVM auto-vectorises.

The spec's speedup targets (§6.2) are met against the first baseline and, for
the simple element-wise ops, roughly **matched** (not beaten) against the
second. That is an honest result: for these primitives tpt-simd's value is a
portable, tested, consistently-semantic API, with real wins only where
intrinsics beat LLVM (i16 dot product, fixed-point multiply, FMA complex multiply).

## Results

| Crate / op | vs scalar (indexed / naive) | vs auto-vectorised scalar | Target | Met? |
|---|---|---|---|---|
| complex: `complex_mul_f32` (native, FMA) | 3.3x | 1.6x | 3x | vs naive only |
| complex: `complex_mul_f32` (default, no FMA) | 0.7x | 0.5x | 3x | no (software fma; use core `a * b`, 1.1x) |
| fixed: `wrapping_mul` Q16.16 (native, AVX2 path) | 10.7x | n/a | 2x | yes |
| fixed: `saturating_mul` (native, AVX2 path) | 3.9x | n/a | 2x | yes |
| fixed: `saturating_mul` (default, portable) | 3.3x | n/a | 2x | yes |
| fixed: `from_f32` | 0.7–1.0x | n/a | n/a | no gain |
| saturate: add_i16 / add_i8 / pack | 33–59x | 1.0–1.2x | 4x | vs indexed only |
| saturate: mul_i16 | 10–16x | 1.15–1.3x | 4x | vs indexed only |
| butterfly: f32 (native) | 9x | 1.0x | 2x | vs indexed only |
| butterfly: f32 (default, SSE2) | 1.4x | **0.18x** | 2x | no — see known issues |
| butterfly: i16 | 16–21x | 1.0x | 2x | vs indexed only |
| butterfly: complex twiddle f32 | 4–5x | n/a | 2x | yes |
| dot: `dot_product_i16` 256 (native, AVX2) | 1.55x | (baseline is auto-vectorised) | 8x | **no** |
| dot: `dot_product_f32` 256–4096 | 5.7–12.5x | vs strict serial IEEE loop | 8x | yes (reassociated) |
| dot: `dot_product_saturating_i16` | 3.4–3.9x | n/a | n/a | n/a |
| dot: complex 64x8 | 0.93–1.0x | n/a | n/a | no gain |
| horizontal: sum_f32 x1024 vectors (default) | 2.6x | n/a | 4–8x | **no** |
| horizontal: other single-register reductions | 0.7–2.1x | n/a | 4–8x | **no** |

## Known issues / follow-ups

* `butterfly_f32` on baseline SSE2 is ~5x slower than an iterator loop: the
  8-lane array backend does not lower well to 128-bit registers. Candidate
  fixes: a 4-lane specialisation in `tpt-simd-vector`, or documenting that
  8-lane f32 kernels want `-C target-feature=+avx2`.
* The spec's 8x (dot i16) and 4–8x (horizontal) targets are not reachable
  against baselines LLVM already vectorises; a single-register reduction is
  only a few cycles. The numbers above should replace the spec's cycle table.
* `cargo-show-asm` audits have not been run (needs the tool installed).

# Benchmarks (Phases 3–6)

Same machine and method as above, **native only** (`-C target-cpu=native`),
criterion with a short run (`--warm-up-time 0.5 --measurement-time 1`), so
treat figures as indicative (±10–20%). Times are per benchmark iteration as
defined in each crate's `benches/*.rs` (iteration sizes differ per row, so
compare only within a row). Reproduce with `cargo bench -p <crate> --bench <name>`.

| Crate / op | scalar | tpt | speedup | Notes |
|---|---|---|---|---|
| permute: `transpose_8x8_i16` | 11.9 ns | 5.1 ns | 2.3x | target ≥5x **not met** (SSE2 network; no AVX2 transpose) |
| permute: `transpose_4x4_f32` | 2.5 ns | 5.6 ns | 0.45x | scalar wins (LLVM already optimal) |
| permute: `interleave_stereo_i16` | 7.7 µs | 0.47 µs | 16x | vs indexed scalar |
| rounding: floor / ceil / trunc | 1.4 / 6.3 / 2.8 µs | 0.45 / 0.48 / 0.58 µs | 3–13x | `vroundps` |
| rounding: `round_ties_even` | 4.3 µs | 0.47 µs | 9x | |
| rounding: `round` (half away) | 5.0 µs | 6.1 µs | 0.8x | no gain |
| rounding: `round_to_nearest_even_i32` | 17.6 µs | 13.0 µs | 1.35x | |
| shift: left / logical / arithmetic / rotate | 2.5–5.2 µs | 0.57–0.78 µs | 3.2–8x | |
| shift: `shift_with_rounding` | 3.7 µs | 2.7 µs | 1.4x | 64-bit intermediate |
| shift: per-lane variable | 0.59–0.63 µs | 0.51–0.60 µs | ~1.1x | scalar baseline auto-vectorises |
| convolve: `convolve_1d_f32` | 355 µs | 177 µs | 2.0x | |
| convolve: `fir_filter_i16` | 145 µs | 62.6 µs | 2.3x | |
| convolve: `convolve_2d_separable_f32` | 275 µs | 356 µs | 0.77x | **slower**; follow-up |
| interpolate: linear / cubic | 17.2 / 52.7 ns | 13.2 / 49.6 ns | 1.3x / 1.06x | |
| interpolate: lanczos3 | 1.01 µs | 1.25 µs | 0.81x | **slower** (polynomial sinc) |
| window: `apply_window_f32` | 335 ns | 353 ns | 0.95x | memory bound / auto-vectorised |
| window: hamming via `cos_approx` | 28.3 µs (libm `cosf`) | 86.5 µs | 0.33x | **slower than libm**; polynomial cos not worth it as-is |
| blend: `blend_f32` (mask) | 3.39 µs (branchy) | 4.73 µs | 0.72x | **slower** |
| blend: `select_f32` (sign bit) | 3.39 µs (branchy) | 1.17 µs | 2.9x | |
| compare: count `gt` i32 / f32 | 0.29 / 0.96 µs | 5.6 / 7.9 µs | 0.05x / 0.12x | **much slower**: mask materialisation + `mask_count`; scalar loop auto-vectorises |
| gather: 256-entry table | 1.0 µs | 10.8 µs | 0.09x | hardware gather loses to scalar loads |
| gather: 4M-entry table | 7.6 µs | 59.6 µs | 0.13x | as above; `checked` ≈ unchecked at 4M |
| scatter: 256 / 4M | 4.5 / 61.8 µs | 4.5 / 56.7 µs | 1.0x | no AVX-512 on this machine: scalar path |

## Phase 3–6 optimisation pass (second run, native)

Root cause of most regressions above: `Simd::mul_add`, `round` and `floor` in
`tpt-simd-vector` are per-lane `libm` calls (`fmaf` is an indirect call that
never becomes a vector instruction). Crates now avoid them in hot paths (magic
number rounding, hand-rolled polynomials, `_mm256_fmadd_ps` under
`cfg(target_feature)`). Results are bit-identical to the scalar references
where documented. Scalar baselines drift ±2x between runs on this machine, so
compare ratios only.

| Crate / op | scalar | tpt | speedup | Was |
|---|---|---|---|---|
| window: hamming (4096) vs libm `cosf` | 15.8 µs | 3.0 µs | 5.3x | 0.33x |
| window: blackman_into | n/a | 4.9 µs | n/a | 126 µs |
| window: `apply_window_f32` | 116 ns | 135 ns | 0.86x | memory bound, no gain |
| convolve: `convolve_1d_f32` | 335 µs | 5.6 µs | 60x | 2.0x (baseline is a strict-order serial FP sum) |
| convolve: `convolve_2d_separable_f32` | 184 µs | 16.4 µs | 11x | 0.77x |
| convolve: `fir_filter_i16` | 68.9 µs | 23.0 µs | 3.0x | 2.3x |
| interpolate: lanczos3 | 475 ns | 269 ns | 1.8x | 0.81x |
| interpolate: linear / cubic | 13.5 / 43.8 ns | 12.8 / 35.6 ns | 1.05x / 1.2x | |
| blend: `blend_f32` (mask, AVX2 `vblendvps`) | 3.17 µs (branchy) | 1.10 µs | 2.9x | 0.72x |
| blend: fused `blend_gt_f32` | 3.17 µs | 0.49 µs | 6.4x | new |
| blend: `select_f32` | 3.17 µs | 0.52 µs | 6.1x | 2.9x |
| compare: count gt i32, `cmp_gt` + `mask_count` | 314 ns | 736 ns | 0.43x | 0.05x |
| compare: count gt f32, `cmp_gt` + `mask_count` | 315 ns | 851 ns | 0.37x | 0.12x |
| compare: fused `count_gt_i32` / `_f32` | 314 / 315 ns | 130 / 98 ns | 2.4x / 3.2x | new |

## Mask redesign (`SimdMask` as integer lanes)

`SimdMask<T, N>` used to be `[bool; N]`; it is now `[T::MaskLane; N]` (a signed
integer of `T`'s width, all ones = set), so compares, `& | ^ !` and `select`
stay in vector registers and the AVX2 compare/blend/movemask paths load and
store the lanes directly. Public API unchanged (`from_raw`/`to_raw` added).
Native, 4096 elements, two runs each (machine is noisy; ranges shown):

| Op | scalar | `cmp_gt` + `mask_count` | fused | Was (mask path) |
|---|---|---|---|---|
| count_gt i32 | 405–581 ns | 332–387 ns | 303 ns | 736 ns (0.43x scalar) |
| count_gt f32 | 1.04–1.14 µs | 337–345 ns | 280–297 ns | 851 ns (0.37x scalar) |
| blend_f32 (mask) vs branchy | 4.6–5.0 µs | 1.18–1.31 µs (3.5–4x) | 1.10–1.14 µs | 2.9x |

Default (SSE2, no flags): per-vector compare+count is about on par with scalar
(i32) and ~1.2x scalar time (f32); the portable `SimdMask::to_bitmask` loop is
still slow there (~2 µs / 4096).

## Phase 3–6 known issues / follow-ups

* **compare (per-vector)**: fixed by the mask redesign below. Fused
  `count_*`/`blend_gt_*` helpers are still marginally fastest.
* **Without `target-cpu=native`**: convolve's FMA falls back to per-lane
  `libm::fmaf` (correct, not faster). Window/interpolate need no flags. Not timed.
* **vector crate `std` feature**: with `--features std`, `mul_add`/`round`/`floor`/
  `ceil`/`trunc`/`sqrt` use the `std` intrinsics instead of `libm` (bit-identical:
  these ops are exactly specified). Complex multiply via the portable fused path,
  native, 1024 elements: 2.89 µs -> 0.56 µs (5x). Without `std` the `libm`
  per-lane calls remain (`core` has no float math methods on stable). Enable
  `std` (and `-C target-cpu=native` for FMA) for performance.
* **gather**: documented honestly in the crate: `vpgatherdd` is slower than
  scalar loads here.
* **permute**: 4x4 f32 transpose slower than scalar; 8x8 i16 only 2.3x (target 5x).
* `blend_imm_*` remain on the portable lane loop (const immediates can't feed
  `_mm256_blend_ps` on stable).
* No `cargo-show-asm` audit has been done for any of these.

# Benchmarks (Phase 10: blas, reduce)

Native (`-C target-cpu=native`), criterion `--warm-up-time 0.5 --measurement-time 1`,
noisy shared machine (ratios indicative; timings swung 2-5x between runs).
The "naive" baselines are plain single-accumulator / strided column-major triple
loops, i.e. what tpt-math currently does; strict IEEE order stops LLVM
vectorising them. The SIMD versions reassociate (documented tolerance, ADR 0001
relaxed for these crates).

| tpt-simd-blas | naive | tpt | speedup | GFLOP/s |
|---|---|---|---|---|
| gemm f32 64 / 256 / 512 | 127 µs / 24.8 ms / 231 ms | 18.7 µs / 0.83 ms / 7.3 ms | 6.8x / 29.8x / 31.5x | 28 / 40 / 37 |
| gemm f64 64 / 256 / 512 | 289 µs / 24.3 ms / 323 ms | 35.8 µs / 1.64 ms / 13.2 ms | 8.1x / 14.8x / 24.5x | 15 / 20 / 20 |
| gemv f32 1024 N / T | 4.52 / 1.29 ms | 157 / 295 µs | 28.8x / 4.4x | 13 / 7 |
| dot f32 4096 / 1M | | | 4.4x / 2.8x | 8 / 5 (1M memory bound) |
| nrm2 / asum f32 4096 | | | 4.2x / 5.2x | |
| axpy f32 | | | 1.0x | already auto-vectorised |

| tpt-simd-reduce (f32) | n=1024 naive -> tpt | n=65536 naive -> tpt |
|---|---|---|
| sum / mean / sum_squares / norm | ~500 -> ~80-135 ns (~6x) | ~40 -> ~6 µs (~7x) |
| sum_compensated vs scalar Kahan | 2.2 -> ~1 µs (~2x) | 150 -> ~60 µs (~2.5x) |
| variance / covariance | 1.1 -> 0.3 µs / 1.9 -> 0.5 µs | 80 -> 19 µs / 140 -> 34 µs |
| argmax | 1.15 -> 0.29 µs (~4x) | 72 -> 9 µs (~8x) |
| min / max, i32 sum/min/max | parity | parity (LLVM already vectorises; plain folds kept) |

Notes: gemm without `target-cpu=native` uses the portable non-FMA kernel
(untimed). Complex BLAS, cargo-show-asm and tpt-math wiring are not done.

## Phase 10: math, rng, sparse

Native, noisy machine (other builds running; single runs swung up to 10x, so
figures are best-of-runs / interleaved and indicative).

| Crate / op | vs baseline | Notes |
|---|---|---|
| math: exp / ln / sin / cos / tanh / erf (f32 slices) | 4.7-9.9x / 2.5x / 4.3x / 5.3x / 5.6-8.9x / 3.2-3.9x vs libm scalar loop | max ULP 0.97 / 0.79 / 3.4 / 3.4 / 1.28 / 2.64; sin/cos NaN for abs(x) > 1e5; no FMA, bit-identical across targets |
| rng: uniform f32 / normal f32 / normal f64 | 2.5x / 7.5x / 5.2x vs scalar xoshiro / libm Box-Muller | |
| rng: raw u64 / uniform f64 | 1.1-1.4x | AVX2 lacks 64-bit rotate/multiply |
| sparse: SpMV CSR (Poisson / random) | 1.2-1.4x (2x on 128 nnz/row) | memory/latency bound |
| sparse: fused `cg_update` | 1.5x vs 3 passes (n=65536), 1.1x at 4M | memory bound at large n |

Sparse finding: hardware gather beat the default 4-accumulator kernel by 5-15%
on f32 rows of 12+ entries when `x` fits in L2 (isolated gather loops were slower
than scalar loads, see Phase 3-6 above). Not adopted: f32-only, AVX2-only,
needs `unsafe`, within noise. Pitfall: a `[[T; 8]; 4]` accumulator spilled to
memory (10x slower); flat `[T; 32]` fixed it.

## tpt-math baseline vs tpt-simd (Phase 10 prerequisite)

Standalone bench crate (kept outside both repos) comparing tpt-math
`tpt-math-linalg-dense` 0.1.0 (generic `Scalar`, `from_fn` + strided triple loop,
`DMatrix * DMatrix` consumes its operands so the bench clones them: O(n^2),
negligible) against `tpt-simd-blas` / on the same f64 data. Plain build vs
`-C target-cpu=native`; machine noisy.

| Op (f64) | tpt-math | tpt-simd plain | speedup | tpt-simd native | speedup |
|---|---|---|---|---|---|
| gemm 64 | 144 µs | 39.6 µs | 3.6x | 17.3 µs | 9.8x |
| gemm 256 | 12.2 ms | 2.73 ms | 4.5x | 0.87 ms | 18x |
| gemm 512 | 327 ms | 20.1 ms | 16x | 6.2 ms | 57x |
| dot 1024 / 65536 | 451 ns / 36.7 µs | 115 ns / 12.3 µs | 3.9x / 3.0x | 103 ns / 9.0 µs | 4.4x / 3.4x |
| norm 1024 / 65536 | 502 ns / 33.6 µs | 107 ns / 6.7 µs | 4.7x / 5.0x | 88 ns / 6.4 µs | 4.8x / 5.6x |
| LU `solve` 64 / 256 | 117 µs / 4.57 ms | not yet wired (no tpt-simd LU) | | | |

Adoption notes: `DMatrix`/`DVector` keep their `Vec` private and expose no
slice accessor, so tpt-math needs `as_slice()`/`as_mut_slice()` before it can
call the kernels. LU/Cholesky/QR need a blocked/`axpy`-based inner loop in
tpt-simd (or tpt-math calling `axpy`/`gemm` from its own factorisations).

### End to end through tpt-math (`simd` feature wired into `DMatrix`/`DVector`)

`tpt-math-linalg-dense` now has an off-by-default `simd` feature (and
`simd-runtime`) that routes `DMatrix * DMatrix`, `DMatrix * DVector`, `dot` and
`norm` to `tpt-simd-blas` for f32/f64. Same bench harness, plain build (no
target flags), tpt-math's own API:

| f64 op | feature off | `simd` | speedup | `simd-runtime` | speedup |
|---|---|---|---|---|---|
| `DMatrix * DMatrix` 64 | 174 µs | 39 µs | 4.4x | 16.9 µs | 10x |
| 256 | 12.7 ms | 2.71 ms | 4.7x | 1.51 ms | 8.4x |
| 512 | 337 ms | 19.1 ms | 18x | 8.15 ms | 41x |
| `dot` 1024 / 65536 | 988 ns / 59 µs | 124 ns / 13.1 µs | 8x / 4.5x | 107 ns / 10.4 µs | 9x / 5.7x |
| `norm` 1024 / 65536 | 930 ns / 59.9 µs | 104 ns / 7.5 µs | 8.9x / 8x | 90 ns / 6.2 µs | 10x / 9.7x |
| LU `solve` 64 / 256 | 152 µs / 5.1 ms | unchanged (not wired) | | | |

(Machine noisy: the feature-off baseline itself varied 1.5-2x between runs.)
tpt-math workspace: fmt, clippy `-D warnings` and all 71 test suites pass with
the feature off and on, so no test depends on bit-exact scalar sums. Enabling the
feature adds `T: 'static` to those operations; `tpt-math-linalg-sparse`
`conjugate_gradient`/`bicgstab` needed `+ 'static` (added unconditionally).
