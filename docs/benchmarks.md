# Benchmarks (Phases 1–2)

Machine: x86_64 with AVX2 + FMA and AVX-512 F/BW/DQ/VL (Tiger Lake i5-1135G7;
the AVX-512 numbers below come from `-C target-cpu=native` runs), Windows 11, Rust 1.97 stable,
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
| scatter: 256 / 4M | 4.5 / 61.8 µs | 4.5 / 56.7 µs | 1.0x | pre-AVX-512 run (scalar path); AVX-512 numbers in the Phase 9 section |

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

### LU / Cholesky kernels and `solve` / `inverse` in tpt-math

`tpt-simd-blas` now has `getrf`/`getrs`/`trsm`/`potrf`/`potrs` (f32 and f64,
LAPACK-style, blocked on top of gemm). Kernel benchmarks, f64, plain build with
`runtime-dispatch`, vs the crate's naive reference:

| n | LU factor+solve naive -> tpt | Cholesky naive -> tpt |
|---|---|---|
| 64 | 25.1 -> 20.6 µs (1.2x) | 23.2 -> 11.1 µs (2.1x) |
| 256 | 1.21 -> 0.76 ms (1.6x) | 2.01 -> 0.31 ms (6.4x) |
| 512 | 12.3 -> 6.2 ms (2.0x) | 28.2 -> 1.94 ms (14.5x) |

LU is only 1.2-2x over a decent naive column-oriented reference (panel factor and
row swaps dominate; room to improve). Cholesky is a clear win.

Through tpt-math (`DMatrix<f64>::solve` / `inverse`, which used a `Vec<Vec<f64>>`
LU), plain build, tpt-math's own API:

| op | feature off | `simd` | `simd-runtime` |
|---|---|---|---|
| `solve` 64 | 248 µs | 22 µs (11x) | 17.8 µs (14x) |
| `solve` 256 | 4.13 ms | 0.90 ms (4.6x) | 0.56 ms (7.4x) |
| `inverse` 64 | 199 µs | 71 µs (2.8x) | 66.8 µs (3.0x) |
| `inverse` 256 | 35.9 ms | 3.18 ms (11x) | 2.21 ms (16x) |

The singularity rule is unchanged (non-finite pivot or |pivot| < 1e-12 =>
`Err(Singular)`), checked on the diagonal of U after `getrf`. All 71 tpt-math
suites pass with the features off and on.

### tpt-math-stats with the `simd` feature (mean / variance)

`tpt-math-stats` gained an off-by-default `simd` feature that uses
`tpt-simd-reduce::sum_compensated_f64` for `mean` and (chunked, no sample-sized
temporary) `variance`. Plain build, 1M and 1k samples around 1e6:

| op | feature off | `simd` | speedup |
|---|---|---|---|
| mean 1024 / 1M | 1.29 µs / 1.68 ms | 1.11 µs / 1.31 ms | 1.16x / 1.29x |
| variance 1024 / 1M | 3.79 µs / 4.01 ms | 3.08 µs / 3.46 ms | 1.23x / 1.16x |

Modest: these loops are largely memory-bound at 1M elements (a first version that
allocated a deviations buffer was *slower* than scalar at 1M, 5.8 vs 3.9 ms, so it
was replaced by 1024-element stack chunks). Worth enabling only if mean/variance show
up in a profile. The Monte Carlo and sampler crates are generic over an `Rng` trait,
so vectorising them (tpt-simd-rng, tpt-simd-math) needs an API change, not just a
feature flag, and is not done.

# Phase 4 performance-target review (horizontal, dot, permute, complex)

Native (`-C target-cpu=native`), criterion `--warm-up-time 0.5 --measurement-time 1`,
noisy machine (ratios indicative). `cargo-show-asm` is not installed; hot paths
were checked by writing the intrinsic version by hand and timing it instead.

| Crate / op | Before | After | Verdict |
|---|---|---|---|
| permute: `transpose_8x8_i16`, 256 independent blocks | SSE2 network 2.7 ns/block vs scalar 12.2 ns (4.6x) | **AVX2 network 1.74 ns/block (7.0x)** | target >=5x met |
| permute: `transpose_8x8_i16`, one block in a serial in-place loop | 5.1 ns vs 11.9 ns (2.3x) | 4.2 ns vs 12.1 ns (2.9x) | latency-bound (store -> load forwarding per iteration) |
| complex: `complex_mul_f32`, 1024 elements, zip loops | auto-vectorised unfused 92.9 ns, auto-vectorised fused (`mul_add`) 79.1 ns, tpt 81.7 ns | unchanged | parity (1.14x / 0.97x) |
| dot: `dot_product_i16` 256 / 4096 | scalar 7.4 / 90 ns, tpt 4.7 / 64 ns (1.6x / 1.4x) | unchanged (4 accumulators tried: no gain, reverted) | load-bound; LLVM emits `vpmaddwd` for the scalar loop too |
| horizontal: `horizontal_sum_f32` x1024 | scalar 1.03 µs, tpt 0.68 µs | hand-written AVX intrinsics: 0.68 µs (identical) | LLVM already emits the optimal shuffle chain |

Changes: AVX2 `transpose_8x8_i16` (two rows per `__m256i`: 4+4 `vpunpck`, 4
`vpermq`, 12 shuffles instead of 24; compile-time `cfg(target_feature = "avx2")`,
SSE2 path kept, bit-identical, existing unit and proptest tests pass). New
benches: `transpose_8x8_i16_x256/{scalar,simd}` (throughput form) and the
`scalar_zip_*` / `tpt_complex_mul_f32_chunks` complex rows.

Why the other three targets are not reachable:

* **complex**: the earlier "1.6x vs auto-vectorised" came from a harness that
  `black_box`ed every 8-lane slice (about 4 ns of fixed overhead per vector). The
  indexed `scalar` loop (1.6 µs) is not vectorised either, because the `Vec` length
  checks block it. A real zip loop is auto-vectorised and runs at the same speed
  as tpt (4 loads, 2 stores and 4 FP ops per 8 complex numbers; load/store bound).
  The 3x target holds only against a baseline LLVM cannot vectorise (3.3-4.5x vs
  `scalar_noautovec`, ~20x vs the indexed loop in a tight slice loop).
* **dot i16**: 256 x i16 is 32 loads of 32 B, about 16 cycles on two load ports, so
  4.7 ns is near the L1 ceiling, and LLVM turns the wrapping scalar loop into
  `vpmaddwd` as well. 8x over auto-vectorised code is not physically available.
* **horizontal**: a single-register reduction is 3 shuffles + 3 adds (about 1 ns).
  The scalar baseline is a strict-order serial add chain (hence 1.5-2.6x on
  `f32`), or is itself vectorised (integer sums/max: parity). Intrinsics do not help.

### Phase 10 wiring pass 2: sparse solvers, new baselines, skipped items

Same machine and harness as above (criterion, plain build, no target flags, noisy;
ranges are two runs). Matrix: 2D Poisson 5-point stencil, f64.

`tpt-math-linalg-sparse` gained an off-by-default `simd` feature: `f32`/`f64`
`conjugate_gradient` and `bicgstab` run on `tpt-simd-sparse` (`spmv_csr`,
`cg_update`, `xpay`, `bicgstab_p_update`, `axpy_sqnorm`; buffers allocated once,
the CSR view validated once per solve, `col_idx` narrowed to `u32` at construction).
The scalar loops cloned `DVector`s several times per iteration, which is where part
of the win comes from.

| op (f64 Poisson) | feature off | `simd` | speedup |
|---|---|---|---|
| CG, 4096 unknowns | 4.3-4.8 ms | 1.99 ms | 2.1-2.4x |
| CG, 16384 unknowns | 32-37 ms | 15.2 ms | 2.1-2.5x |
| BiCGSTAB, 16384 unknowns | 41-45 ms | 20.2 ms | 2.0-2.3x |
| `CsrMatrix::matvec` 4096 / 65536 (not routed) | 13.4-13.9 / 222-231 µs | 14.9 / 268 µs | 0.9x / 0.83x |

Standalone `matvec` was tried and **not** adopted: `CsrView::try_new` re-validates
O(nnz) indices on every call (`unsafe_code = "forbid"` in tpt-math rules out
`new_unchecked`) and a fresh output `Vec` is allocated, so it was 7-17% slower than
the scalar loop. Only the solvers (one validation per solve) use the kernels.
Tolerance: reductions reassociate and the residual test uses `sqrt(r.r)`, so iterates
agree to rounding and the iteration count near `tol` can differ by one. Tests
(`poisson_solvers_and_matvec_f64_f32`, non-convergence cases) run with the feature on
and off. Whole tpt-math workspace: fmt, clippy `-D warnings`, 691 tests off / 692 on
(`tpt-math-linalg-dense/simd-runtime`, `tpt-math-linalg-sparse/simd`,
`tpt-math-stats/simd`), zero failures.

New baseline benches (previously missing; scalar only, no `simd` path):

| bench | time |
|---|---|
| Monte Carlo `integrate(x^2)`, 10^7 samples, SplitMix64 | 10.05 ms (~1.0 ns/sample) |
| Monte Carlo `estimate_mean(Standard)`, 10^7 samples (80 MB buffer) | 30.1 ms |
| CG Poisson 4096 / 16384 | see table above |

Not wired, with reasons:

* **Cholesky / QR in `tpt-math-linalg-dense`**: `DMatrix<f64>` has no Cholesky or QR
  (only LU `solve`/`inverse`). There is no scalar reference to keep, so wiring
  `potrf`/`potrs` would mean adding new public API; not done. The only Cholesky and
  QR in tpt-math are in `tpt-math-linalg-complex` (Hermitian, `Vec<Vec<Complex<f64>>>`,
  Householder QR for the eigen solver).
* **`tpt-math-linalg-complex`**: `tpt-simd-blas` had no complex kernels when this pass was done, so it was
  skipped. They have since been added (see "tpt-simd-blas complex kernels" below);
  wiring is still open (needs a copy from `Vec<Complex<T>>` into planes).
* **`tpt-math-stats` / `tpt-math-prob-dist` with `tpt-simd-math`**: `tpt-simd-math` is
  f32 only; every function in both crates is f64 (no f32 anywhere), so nothing can be
  wired without an API change. Nothing measured.
* **Monte Carlo / sampler vectorisation (`tpt-simd-rng`)**: needs an API change. At
  minimum: (1) a bulk method on `Rng` (e.g. `fill_u64` / `fill_f64(&mut [f64])`, default
  impl looping `next_u64`) so a lane-parallel generator can fill buffers; (2) a batch
  method on `Distribution` (`sample_into(&self, rng, &mut [T])`) so `integrate` /
  `estimate_mean` stop calling `sample` once per draw; (3) f64 transforms, since
  `tpt-simd-math` is f32 only (an f32 path changes accuracy and results); (4) accept that
  lane-parallel streams are not bit-identical to `SplitMix64` for a given seed, i.e.
  reproducibility changes (ADR 0003 tier 3). At ~1 ns/sample the scalar generator is
  already cheap, so the ceiling is a few x at best.

## Benchmarks for aligned, mul, select, matrix

Native (`-C target-cpu=native`), criterion `--warm-up-time 0.5 --measurement-time 1`,
median-ish of one run on a loaded machine (treat as ±15%, more for the sub-microsecond
rows). 4096 elements per iteration unless noted. "scalar_idx" is an indexed loop,
"scalar_iter" a zip/iterator loop that LLVM auto-vectorises. Reproduce with
`cargo bench -p <crate> --bench <crate-name>`.

| bench | scalar | simd | verdict |
|---|---|---|---|
| mul: `mul_hi_i16` | 4.5 us idx / 230 ns iter | 162 ns | 1.4x vs auto-vectorised, 28x vs indexed |
| mul: `mul_widen_i16` | 642 ns iter | 653 ns | parity |
| mul: `mul_lo_i32` | 716 ns iter | 716 ns | parity |
| mul: `mul_add_sub_f32` | 8.7 us idx | 21.5 us | **2.5x slower** (see below) |
| mul: `complex_mul_f32` (split layout) | 14.4 us idx, 13.9 us idx+fma | 6.5 us | 2.2x faster |
| mul: `complex_mul_f32_portable` | | 20.5 us | 1.4x slower than scalar (reference only) |
| select: `select_i32` (8 lanes, 512 vectors) | 2.4 us | 5.4 us | **2.2x slower** |
| select: `select_lanes_f32` | 5.9 us | 4.6 us | 1.3x faster |
| select: `select_from_slice_i32` (checked / try, 256-entry table) | 6.1 us | 8.0 / 6.6 us | slower / parity |
| matrix: `mat4x4_mul_f32` x256 | 5.1 us (unfused) | 56 us | **11x slower** |
| matrix: `mat8x8_mul_i16` x256 | 52 us | 53 us | parity |
| matrix: `mat4x4_transpose_f32` x256 | 557 ns | 352 ns | 1.6x faster |
| matrix: `mat8x8_transpose_i16` x256 | 4.9 us | 1.4 us | 3.5x faster |
| matrix: `mat3x3_inverse_f32` x256 | n/a (scalar only) | 5.1 us | ~20 ns per matrix |
| aligned: `copy_scale` scalar iter | 75 ns | n/a | auto-vectorised baseline wins |
| aligned: `copy_scale` unaligned `from_slice`/`copy_to_slice` | | 2.2-2.5 us | |
| aligned: `copy_scale` `load_aligned`/`store_aligned` | | 590-620 ns | |
| aligned: `copy_scale` `try_load_aligned` | | 465-485 ns | |
| aligned: alloc zeroed / filled 4096 f32 | Vec 500-600 ns / 580-1000 ns | 510-1400 ns / 510-710 ns | within noise of `Vec` |
| aligned: alloc `from_slice` 4096 f32 | Vec 710-775 ns | 817-830 ns | ~10-15% slower |
| aligned: alloc zeroed 8 f32 | Vec 63-65 ns | 70-72 ns | ~10% slower (aligned allocator path) |

Notes (honest reading):

* **aligned**: the wrapper cost is small, but the large gap between the unchecked
  `from_slice` loop (2.2 us) and the checked `load_aligned` loop (0.6 us) is not a
  hardware aligned-vs-unaligned effect (modern x86 treats them the same on aligned data;
  the two paths run the same `Simd::from_slice` underneath). It is a codegen/inlining
  difference between the loops, so do not read it as "alignment makes loads 4x faster".
  The auto-vectorised scalar loop is far faster than either explicit loop here (75 ns).
  `AlignedBuf` allocation is the same order as `Vec`; the benefit is the alignment
  guarantee, not speed. Alloc rows are noisy (the allocator dominates).
* **mul**: wins only where intrinsics/special lowering exist (`mul_hi_i16`,
  `complex_mul_f32`). `mul_add_sub_f32` is slower than scalar: it builds its sign vector
  with `Simd::from_fn` and uses `Simd::mul_add` every call, which does not lower to a
  clean vector sequence. Candidate for an optimisation pass (hoist the sign constant,
  use an `fmaddsub` intrinsic).
* **select**: `select_i32` goes through `to_array`/`from_fn` scalar indexing, so it is
  slower than a plain scalar loop; a `vpermd`-based path would be the real fix.
* **matrix**: `mat4x4_mul_f32` is 11x slower than the scalar reference (219 ns vs 20 ns
  per matrix): the `Simd::splat(..).mul_add(..)` chain is evidently not lowering to
  vector FMA. 8x8 i16 multiply (i64 accumulation) is parity. Transposes win.

### tpt-simd-blas complex kernels (split re/im, `benches/complex.rs`)

Native (`-C target-cpu=native`), criterion on a noisy shared machine (gemm re-run with
`--measurement-time 3`; ratios indicative). Baseline is `reference::*_c32/c64`: naive
split-plane loops (strided triple loop for gemm). GFLOP/s counts 8 flops per complex
multiply-add.

| bench | naive | tpt (split) | speedup | GFLOP/s (tpt) |
|---|---|---|---|---|
| cgemm c32 64 | 234-334 µs | 64 µs | ~3.7-5x | 32 |
| cgemm c32 256 | 38-69 ms | 3.2-3.9 ms | ~10-20x | 34-42 |
| cgemm c64 64 | 480-600 µs | 131-140 µs | ~3.7-4.3x | 16 |
| cgemm c64 256 | 42-52 ms | 6.7 ms (alpha != 1: 8.1 ms) | ~6x | 20 |
| cgemv c32 1024, N | 14.6 ms | 0.48 ms | ~30x | 17 |
| cgemv c32 1024, H (conj-transpose) | 1.95 ms | 1.05 ms | 1.9x | 8 |
| dotc c32 4096 | 5.0 µs | 3.8 µs | 1.3x | 8.6 |
| nrm2 / asum c32 4096 | 9.8 / 6.1 µs | 3.1 / 1.1 µs | 3.1x / 5.4x | |
| axpy c32 4096 | 1.51 µs | 1.46 µs | 1.0x (already vectorised) | |

Interleaved (`&[[T; 2]]`) entry points on the same machine: `gemm_il` c32 256 4.0 ms vs
3.2-3.9 ms split (the plane conversion plus allocations cost ~3-25% at n=64..256, shrinking
as `O(mk + kn + mn)` against `O(mnk)`); `gemv_il` N 1.0 ms vs 0.48 ms split (2x slower,
so keep matrices split for hot gemv); `dotc_il` 2.3 µs vs 3.8 µs split (interleaved is
faster here: each element is read once); `axpy_il` 3.5 µs vs 1.46 µs split.

Notes: complex gemm is four real packed gemms (4M) and so inherits the real kernel's
throughput (28-43 / 15-20 GFLOP/s real f32 / f64 in the table above). Complex level 1/2
are portable auto-vectorised code (no new intrinsics); `dotu`/`dotc` carry four
accumulators and are close to memory-bound at 4096 elements. Conversion from tpt-math's
non-`repr(C)` `Complex<T>` is a separate caller-side pass and is not included above.
No `cargo-show-asm` audit of the complex kernels has been done. 3M was not implemented
(norm-wise-only error bound on the imaginary part).

## After: mul, select, matrix fast paths

Root cause of the slow rows above: `Simd::mul_add` is a per-lane `libm::fmaf` call
without the `std` feature (and a libc `fmaf` call without hardware FMA), and the
select kernels used scalar `to_array`/`from_fn` indexing. Fixes (same bench harness,
`--warm-up-time 0.5 --measurement-time 1`, one run, treat as +-15%; the machine was
less loaded than for the "before" run, so compare the ratio to the scalar row, not the
absolute time):

| bench | scalar | simd, default build | simd, `target-cpu=native` |
|---|---|---|---|
| matrix: `mat4x4_mul_f32` x256 | 2.8 us / 2.5 us native | 1.7 us (1.6x faster) | 1.26 us (2.0x faster) |
| mul: `mul_add_sub_f32` | 4.5 us / 3.5 us native | 5.3 us (needs FMA; software `fmaf`) | 0.44 us (7.9x faster) |
| mul: `complex_mul_f32` | 6.5 us idx | 12.8 us (no FMA: portable path) | 2.1 us (3x faster) |
| mul: `complex_mul_f32_portable` | | 12.8 us | 11.7 us (reference only, software `fmaf`) |
| select: `select_i32` | 2.6 us | 0.91 us (2.9x faster) | 0.35 us (7.6x faster) |
| select: `select_lanes_f32` | 2.5 us | 0.92 us | 0.23 us (10.8x faster) |
| select: `select_from_slice_i32` checked / try | 2.5 us | 1.14 / 1.01 us | 1.35 / 1.24 us (~2x faster) |

What changed:

* `mat4x4_mul_f32` is now unfused mul+add in the scalar reference's order, so it is
  **bit-identical** to `mat4x4_mul_scalar_f32` (the proptest now asserts bits, not a
  tolerance) and vectorises on baseline SSE2 (explicit `_mm_mul_ps`/`_mm_add_ps` with a
  broadcast, plus the portable lane-loop as reference). Numeric policy in the crate docs
  was updated: it used to say "fused, differs from scalar by rounding".
* `mul_add_sub_f32` has an AVX2+FMA path (sign-flip of odd lanes of `c`, one
  `vfmadd`), bit-identical to the portable fused rule. Without hardware FMA it is still a
  software `fmaf` per lane (no vector sequence can be correctly fused); that is unchanged
  and the doc says so.
* `select_i32` / `select_lanes_f32` use `vpermd` / `vpermps` on AVX2 (hardware uses
  exactly the low 3 index bits, matching `& 7`); portable path kept and compared in tests.
  The default-build numbers are likewise a different run (the earlier run was on a
  loaded machine); trust the native column for the AVX2 effect.
  earlier noisy run, so trust the native column.
* `select_from_slice_i32` is unchanged (it is `tpt-simd-gather`); the earlier "slightly
  slower" reading was load noise, it is about 2x faster than the scalar table lookup.
* Remaining: `complex_mul_f32` and `mul_add_sub_f32` without hardware FMA stay at
  software-`fmaf` speed by design (bit-identical results on every target). A root-cause
  fix for everything built on `Simd::mul_add` (tpt-simd-vector choosing a hardware FMA
  where `target_feature = "fma"` is on) would speed up all other crates' portable paths.

# Session 2026-10-12: AVX-512BW dot, matrix i16, and the Q4 summary

## AVX-512BW `dot_product_i16` (new path)

`tpt-simd-dot` now has an AVX-512BW kernel (`x86::dot_i16_avx512bw`): the AVX2
kernel with 512-bit `vpmaddwd`, two 64-element loads per operand per
iteration. It is selected only when `avx512bw` is enabled at compile time
(`target-cpu=native` on this host). Results are identical: each `vpmaddwd`
lane is an exact `i32` pair sum and all additions wrap, so the summation order
does not change the result. Verified by the existing unit and proptests (both
configurations) and by the golden digest (`tpt-simd/tests/cross_arch_golden.rs`).

A/B run on the same machine with the arm64 emulation container paused (criterion,
`--warm-up-time 1 --measurement-time 3`, `i16_` filter), Tiger Lake i5-1135G7:

| Bench | scalar (indexed) | portable | tpt, AVX2+FMA | tpt, native (AVX-512BW) |
|---|---|---|---|---|
| `i16_256` | 24.6 ns | 71.9 ns | 16.2 ns | 15.5 ns |
| `i16_4096` | 276.5 ns | 413.4 ns | 158.6 ns | 129.4 ns |

* At 4096 elements, AVX-512BW is 1.23x faster than the AVX2 path and 2.1x
  faster than the indexed scalar loop.
* At 256 elements there is no gain. The work is 4 iterations, so the fixed
  cost dominates: the horizontal reduction (store 16 lanes, then 16 scalar
  adds) and the scalar tail. A shuffle-based in-register reduction would be
  the next step.
* Earlier runs (overlapping a container build) were inconsistent. Only the
  paused-container A/B above is used for comparisons.

## `matrix::mat8x8_mul_i16`: codegen fix

The explicit `Simd<i64, 8>` kernel compiled to 309 AVX2 instructions. AVX2 has
no 64-bit lane multiply, so each product was emulated with `vpmuludq`, and the
clamp went through a scalar per-lane `map`. The scalar reference
(`mat8x8_mul_scalar_i16`) auto-vectorises to 131. The kernel is now written in
the reference's loop shape, which compiles to 125 instructions. Bit-identical
(matrix unit and proptest suites pass). Timing was not measured for this
change; the instruction count is the evidence. See `docs/asm-audit.md`,
second pass.

## Q4: why the speedup targets are not met

The spec's targets (section 6.2) assumed baselines that LLVM does not produce
in practice. The causes, with the measurements behind them:

1. **The baseline is already vectorised.** For simple element-wise and
   reduction kernels the iterator loop auto-vectorises, so the explicit kernel
   has nothing left to win: `saturate` (parity with the zip loop), `horizontal`
   (LLVM already emits the optimal shuffle tree; hand-written AVX gave 0.68 µs,
   the same), `dot_product_i16` (LLVM emits `vpmaddwd` for the scalar loop too),
   `complex_mul` (parity with the zip loop; load/store bound).
2. **Some kernels are bound by memory or L1.** `dot_product_i16` at 256 elements
   is 32 loads of 32 bytes, close to the two-load-port ceiling. `window` and
   `sparse` SpMV are memory bound. Doubling the instruction throughput does not
   help here.
3. **Fixed overheads dominate short inputs.** A single-register reduction is
   about 1 ns of work. Store-to-load forwarding and the reduction tail set the
   floor for `transpose_8x8_i16` on one block (2.3x to 2.9x) and for short dots.
4. **Codegen pitfalls in our own code.** These were the largest avoidable losses,
   found in this session's audit:
   * i64 lane arithmetic on AVX2 (no 64-bit multiply): `mat8x8_mul_i16` was 2.5x
     the instructions of its scalar reference. Fixed.
   * Per-lane libm calls. Without hardware FMA, `complex_mul_f32` and
     `mul_add_sub_f32` call `libm::fmaf` once per lane (16 and 8 calls per
     call). On aarch64 this was the case for every build, because no NEON path
     existed. Fixed for aarch64 (NEON `vfmaq_f32`); the x86 default build without
     `+fma` still has this cost.
   * Scalar lowering of 8-lane arrays on 128-bit targets: `butterfly_f32` on SSE2
     (`docs/benchmarks.md`), and on aarch64 `butterfly_f32`, `butterfly_i16`,
     `transpose_8x8_i16` and `blend_i32` compiled to scalar code (`csel`, `strh`,
     one scalar `fadd` per lane). Fixed on aarch64 with NEON intrinsics.
   * Per-lane branches: `round_f32` and `floor_f32` on aarch64 compiled to about
     200 instructions with 30 branches. A single `vrnd*` instruction does the
     same job. Fixed.
5. **The default build is SSE2.** ADR 0001 makes ISA selection compile-time, so
   a user who does not pass `-C target-cpu` gets the SSE2 code. Several results
   above depend on the ISA: `butterfly_f32` is 0.18x of an iterator loop on
   SSE2, and the 256-block 8x8 transpose is 4.6x on SSE2 but 7.0x with AVX2. That is a deliberate trade-off (ADR
   0003 adds runtime dispatch only for blas), and it is the main reason a
   default-build benchmark looks worse than the native one.
6. **Determinism costs throughput.** `mat4x4_mul_f32` and the complex kernels
   avoid FMA contraction where the portable path must be bit-identical, so they
   issue a separate multiply and add. ADR 0003 relaxes this only for reductions.
7. **The measurement environment is noisy.** The machine is shared, so run-to-run
   spread is 10–50%. Comparisons in this file are made back-to-back, and results
   are listed as ratios where that is the honest unit.

Consequence: the remaining 3–8x targets are not reachable against the
baselines LLVM produces. The realistic gains are the codegen fixes in item 4
(which cannot change the ratio against an auto-vectorised baseline, but do
remove slow paths on targets that lack a vector backend) and the explicit
ISA paths on targets that lack them. The spec's cycle table should be replaced
by the measured ratios in this file.

# Phase 9: AVX-512 tuning pass (dot i16)

Scope: the two existing AVX-512 kernels (`dot_i16_avx512bw`, `scatter_*`). Other
AVX-512 specialisations are not started.

Changes to `dot_i16_avx512bw` (`tpt-simd-dot/src/x86.rs`):

* The remainder of 1..=31 elements is loaded with one zero-masked
  `_mm512_maskz_loadu_epi16` per operand instead of a scalar loop. Masked-off
  lanes are not accessed and add 0, so the result is unchanged.
* The 512-bit accumulator is folded to 256 bits in-register before storing, so
  the final reduction is 8 scalar adds instead of 16.

Results are unchanged: the wrapping `i32` sum does not depend on order. Verified
by the dot unit tests (which sweep every tail length 0..=130) and proptests,
under both `target-cpu=native` and the default build.

A/B on the same machine (criterion, `--warm-up-time 1 --measurement-time 3`).
The bench now includes `i16_255`, the worst-case tail (7×32 + 31). The scalar
baseline drifts by up to 1.5x between runs on this machine, so compare ratios
within a run:

| Bench (native) | before (tpt) | after (tpt) | speedup |
|---|---|---|---|
| `i16_255` | 49.8 ns | 22.2 ns | 2.2x |
| `sat_i16_255` | (not re-run) | | |

`i16_256` and `i16_4096` are multiples of 32 and have no tail, so they are
unaffected in principle; a single noisy run gave 14.1 ns and 133 ns (the earlier
run recorded 15.5 ns and 129 ns).

Codegen check (`cargo asm`, native): the masked tail lowers to
`vmovdqu16 zmm {k1}{z}` loads, and the reduction is a `vextracti64x4` fold to
256 bits followed by 128-bit adds. No scalar tail loop remains.

Scatter AVX-512 (`scatter_i32`, `_mm256_i32scatter_epi32`, `avx512f`+`avx512vl`),
clean A/B from one native run (criterion `--warm-up-time 1 --measurement-time 3`):

| Bench | scalar | tpt (AVX-512) | ratio |
|---|---|---|---|
| `scatter_i32_256` | 6.1 µs | 6.7 µs | 0.9x (no gain) |
| `scatter_i32_4194304` | 51.3 µs | 35.4 µs | 1.45x |

Crossover sweep (`crossover_i32_*` in `tpt-simd-scatter/benches/scatter.rs`,
same native build, 1024 random index sets of 8 per iteration). Ratio is
scalar time / tpt time, so above 1.0 means AVX-512 is faster:

| Array elements | run 1 (2 s) | run 2 (5 s) |
|---|---|---|
| 1 024 | 1.5x | 1.0x |
| 4 096 | 1.1x | 1.1x |
| 16 384 | 1.0x | 1.0x |
| 65 536 | 0.8x | 1.3x |
| 262 144 | 0.9x | 1.2x |
| 256 (`scatter_i32_256`) | 0.9x | 2.1x |
| 4 194 304 (`scatter_i32_4194304`) | 1.45x | — |

The two runs disagree by up to 2.3x at the same size, which is more than the
gain or loss being measured. No size shows a loss that holds across runs, and
16 K elements is parity in both. A size-based dispatch to scalar would not be
justified by these numbers, so none was added. The `vpscatterdd` path stays
as-is. Settling a crossover would need a quieter machine and more repetitions.

## AVX-512F gemm microkernels (tpt-simd-blas)

`tpt-simd-blas` now has AVX-512F f32 and f64 microkernels (`kernel_f32_512`,
`kernel_f64_512` in `tpt-simd-blas/src/x86.rs`). They are used when AVX-512F,
AVX2 and FMA are enabled at compile time (`target-cpu=native` on this host).
The f32 tile is 32x4, the f64 tile 16x4: twice as tall as the AVX2 tiles, with
the same 8 accumulators. The tile height is a compile-time choice
(`AVX512_TILES` in `tpt-simd-blas/src/lib.rs`), so the packing layout is fixed
per build. Builds without AVX-512F keep the AVX2 tiles (16x4 f32, 8x4 f64).

Correctness: the blas test suite (104 unit tests and the integration test)
passes in the default, AVX2+FMA and native AVX-512 builds. Clippy
`-D warnings` is clean in all three.

A/B (criterion, `--warm-up-time 1 --measurement-time 3`, `gemm_` filter, same
source; the AVX2 build is the baseline). The AVX-512 build also gets other
AVX-512 codegen, not just the kernel, so the gain is the combined effect.

| Bench | AVX2 + FMA | AVX-512F | speedup |
|---|---|---|---|
| `gemm_256/f32/tpt` | 647 µs | 480 µs | 1.35x |
| `gemm_512/f32/tpt` | 6.00 ms | 3.86 ms | 1.56x |
| `gemm_256/f64/tpt` | 1.31 ms | 1.12 ms | 1.17x |
| `gemm_512/f64/tpt` | 12.3 ms | 8.43 ms | 1.46x |

The 512 f32 case is about 69 GFLOP/s on this machine (2n^3 / time), up from
about 45. The spread on the AVX2 512 f32 run was wide (4.7 to 7.5 ms), so
treat the 1.56x as approximate.

## AVX-512F f32 dot (tpt-simd-dot)

`dot_product_f32` now uses an AVX-512F main loop when AVX-512F is enabled at
compile time (`dot_f32_blocks_512` in `tpt-simd-dot/src/x86.rs`). Two `__m512`
accumulators hold the portable layout's four 8-lane accumulators, and multiply
and add are rounded separately (no FMA), as in the portable path. The remainder
and final reduction are shared with the portable code (`portable::dot_f32_tail`),
so the result is bit-identical. The existing test
`f32_matches_documented_order_all_lengths` compares bits for every length 0..=100.

A/B in one native build (criterion, `--warm-up-time 1 --measurement-time 3`):

| Bench | portable | AVX-512F | speedup |
|---|---|---|---|
| `f32_256` | 38.4 ns | 33.8 ns | 1.14x |
| `f32_4096` | 401 ns | 298 ns | 1.35x |

The accumulation order is fixed by the documented contract, so each block still
feeds two dependent add chains. That limits the gain; a different order would
change the results.

## AVX-512F fused counters (tpt-simd-compare)

`count_gt/lt/eq_{i32,f32}` now use an AVX-512F path when AVX-512F is enabled at
compile time (`x86_512` module in `tpt-simd-compare/src/lib.rs`). Each 16
elements is one mask compare (`_mm512_cmp*_mask`) and a `popcnt` of the mask,
with four independent counters. The float compares use the ordered quiet
predicates (`_CMP_*_OQ`), so NaN never matches, as in the scalar operator.
Counts are exact integers, so results match the other paths bit for bit. The
existing fused-count tests and proptests (`i32_vs_scalar`, `f32_vs_scalar`, with
NaN and specials) pass in the native, AVX2 and default builds.

A/B in the same session (criterion, `count_gt_*/fused`). Machine noise is large
here: the scalar baseline alone moved from 449 to 803 ns across runs, so these
are approximate.

| Bench | AVX2 build | AVX-512F build (unrolled) |
|---|---|---|
| `count_gt_i32/fused` | 171 ns | 186 ns (range 182-237 across runs) |
| `count_gt_f32/fused` | 204 ns | 165 ns (range 165-209 across runs) |

Read this as: i32 is about parity with AVX2, f32 is a modest gain. A
single-accumulator AVX-512 version was measured too and was no better in the
A/B (i32 221 ns, f32 301 ns in that run), so the four-counter version was kept.

## AVX-512BW unpack_i8_to_i16 (tpt-simd-permute)

`unpack_i8_to_i16` had no explicit SIMD path. It relied on LLVM's autovectorised
loop, which was slower on the native AVX-512 build than on the AVX2 build of the
same source (144 ns vs 91 ns, criterion, 1024 bytes). The native build now uses an
explicit AVX-512BW kernel (`unpack_i8_to_i16_avx512bw`): one 32-byte load,
`vpmovsxbw` to 32 `i16`, one 64-byte store per iteration, scalar tail.

| Bench (1024 elements) | before (native) | after (native) |
|---|---|---|
| `unpack_i8_to_i16/lib` | 144 ns | 67 ns (2.1x; now faster than the AVX2 build's 91 ns) |

Correctness: `unpack_all_lengths_and_extremes` (every length 0..=130, full i8
range) and the existing proptest pass in the native, AVX2 and default builds.

Checked and not changed: `interleave_stereo_i16` and `deinterleave_stereo_i16`
are already faster on the native build than on AVX2 (254 vs 331 ns and 371 vs
435 ns), so autovectorisation is adequate there.

## AVX-512F convolve_1d_f32 (tpt-simd-convolve)

`convolve_1d_f32` is one `axpy` pass per kernel tap. `axpy` now processes whole
16-element blocks with `_mm512_fmadd_ps` when AVX-512F and FMA are enabled at
compile time (`axpy_512` in `tpt-simd-convolve/src/lib.rs`), then the existing
8-lane code handles the remainder. Each lane is independent and FMA rounds once,
so results are unchanged. The proptest `conv1d_matches_scalar` checks equality
with the scalar reference for input lengths up to 70, which covers the 16-lane
and tail blocks, and passes in the native, AVX2 and default builds.

A/B (criterion, `convolve_1d_f32/tpt`, same source):

| Build | time |
|---|---|
| AVX2 + FMA | 35.2 us (range 32-39) |
| AVX-512F + FMA (native) | 15.5 us (range 15-16) |

Only the native build changed, so the 2.3x is the 16-lane path. The scalar reference
(`convolve_1d_scalar_f32`, software `fmaf`) is about 570-730 us in these runs and
is not a useful baseline for speed.

## AVX-512F fir_filter_i16 (tpt-simd-convolve)

The Q15 FIR interior now has an AVX-512F path (`fir_interior_512` in
`tpt-simd-convolve/src/lib.rs`): eight outputs per iteration in `i64` lanes. Each
tap sign-extends eight `i16` inputs (`vpmovsxwq`), multiplies by the coefficient
(`vpmuldq`, exact in 64 bits), and accumulates (`vpaddq`). Rounding is
`(acc + 2^14) >> 15` (`vpsraq`), then clamping to the i16 range, then narrowing.
This is the same arithmetic as `q15_round_sat`. The 4-lane and scalar loops handle
what is left, as before.

Correctness: the existing `fir_matches_scalar` proptest (full-range `i16` inputs
and taps, lengths to 59) and a new `fir_long_extremes_match_scalar` test (lengths
to 300, tap counts 0, 1, 2, 7, 8, 9, 16, 17, extreme values that hit both clamps)
compare against the scalar reference. They pass in the native, AVX2 and default
builds.

A/B (criterion, `fir_filter_i16/tpt`, same source):

| Build | time |
|---|---|
| AVX2 + FMA | 51.7 us (range 50-54) |
| AVX-512F (native) | 15.7 us (range 15-16) |

The scalar reference is 164 us (AVX2 build) and 180 us (native build), so the
native path is about 11x faster than scalar and the AVX2 path about 3x.

## AVX-512F f32 sum (tpt-simd-reduce)

`sum_f32` (and so `pairwise_sum_f32`, whose leaves call `sum`) now takes its
full 32-element blocks from an AVX-512F kernel (`x86_512::f32_blocks` in
`tpt-simd-reduce/src/lib.rs`) when AVX-512F is enabled at compile time. Two
`__m512` registers hold the four 8-lane accumulators as `[a0|a1]` and `[a2|a3]`,
so each block is two adds, in the documented order. The `sum` loop then adds the
remaining 8-element vectors and the scalar tail as before, so results are
bit-identical. The crate's `#![forbid(unsafe_code)]` is now conditional on the
absence of this path, following ADR 0002 and the dot and blas crates.

Correctness: the existing tests use a tolerance, so a new test
`sum_f32_is_bit_identical_to_documented_order` compares `sum_f32` bit for bit with
a reference written from the documented order, for every length 0..=200. It passes
in the native, AVX2 and default builds.

A/B in one native build (criterion, `sum/tpt` rows; the "no hook" row has the
hook turned off in a temporary copy and the file restored afterwards):

| Bench | native, no hook (portable) | native, AVX-512F hook |
|---|---|---|
| `f32_1024/sum/tpt` | 115 ns | 33.5 ns (3.4x) |
| `f32_65536/sum/tpt` | 9.0 us | 3.55 us (2.5x) |

For reference, the AVX2 build of the same source runs `f32_1024` in 127 ns, so
the portable array-backed `Simd` code is slow in both builds. The gain above is
the explicit kernel, not just the wider target. f64 `sum` has no hook yet.

## AVX-512F f64 sum (tpt-simd-reduce)

The f64 counterpart of the f32 hook (`x86_512::f64_blocks`). Each 16-element
block takes two `__m512d` adds: `[a0|a1]` from elements 0..8 and `[a2|a3]` from
elements 8..16, in the documented order. The `sum` loop, the tail and the
results are unchanged, so the result is bit-identical. Checked by
`sum_f64_is_bit_identical_to_documented_order` (lengths 0..=200) in the native,
AVX2 and default builds.

A/B (criterion, `other_1024/sum_f64/tpt`, same session):

| Build | time |
|---|---|
| native, no hook (portable) | 107 ns |
| native, AVX-512F hook | 60 ns (1.8x) |
| AVX2 + FMA | 104 ns |

The hook-off row was measured by turning the hook off in a temporary copy
and restoring the file, checked byte for byte afterwards.

## AVX-512F blas level-1: dot, nrm2, asum (tpt-simd-blas)

The portable level-1 kernels accumulate into a 32-lane (f32) or 16-lane (f64)
array, which LLVM autovectorises. On the native AVX-512 build that codegen was
slower than on the AVX2 build for cache-resident data (dot 4096: 1.05 us native
vs 0.88 us AVX2, nrm2 4096: 1.10 us vs 0.72 us; the same kind of regression as
`unpack_i8_to_i16`).

`l1_f32` / `l1_f64` (`tpt-simd-blas/src/x86.rs`) are AVX-512F block kernels for
`x*y`, `x*x` and `|x|`. They use two 512-bit accumulators, with separate multiply
and add (no FMA), and the same lane layout, reduction and scalar tail as the
portable code. The results are bit-identical. The new test
`level1_matches_documented_order_bit_exact` checks this for lengths 0..=130 in
f32 and f64, and all three builds pass.

A/B in one session (criterion, `level1_*/tpt`, AVX2+FMA build then native AVX-512F):

| Bench | AVX2 + FMA | AVX-512F | speedup |
|---|---|---|---|
| `dot_f32` 4096 | 1.13 us | 225 ns | 5.0x |
| `nrm2_f32` 4096 | 1.04 us | 193 ns | 5.4x |
| `asum_f32` 4096 | 607 ns | 156 ns | 3.9x |
| `dot_f32` 1M | 460 us | 295 us | 1.6x |
| `nrm2_f32` 1M | 209 us | 110 us | 1.9x |
| `asum_f32` 1M | 249 us | 113 us | 2.2x |

The machine is noisy (the same benchmark moved by about 30% between runs), so
treat these as approximate. The 1M rows are closer to memory-bound than the 4096
rows, and they gain less.

## blas axpy / scal: tried, not kept (tpt-simd-blas)

An AVX-512F `axpy_f32` (`y += alpha*x`, 16 lanes) and `scal_f32` were written and
checked bit-exact (test `axpy_scal_bit_exact_elementwise`, lengths 0..=130, all
pass). They were then benchmarked in the same session against the autovectorised
native loop (criterion, `level1_*/axpy_f32/tpt`, hook turned off in a temporary
copy and restored byte for byte afterwards):

| Size | autovectorised (native) | AVX-512F kernel |
|---|---|---|
| 4096 | 296 ns | 363 ns (slower) |
| 1M | 408 us | 350 us (faster in this run; an earlier run gave 469 us) |

The cache-resident case is slower and the memory-bound case is inside the noise,
so the kernels were reverted. The autovectorised loop stays for `axpy` and `scal`.
`scal` was not benchmarked on its own. The bit-exact test stays because it is
still a valid check of the portable path.


## Phase 11 crates: criterion baseline (2026-10-15)

Benches: `tpt-simd-*/benches/kernels.rs` for the 13 Phase 11 crates, run with
`cargo bench -p <crate> --bench kernels` on the development host (Windows 11,
i5-1135G7, default `x86_64` target, so SSE2 only: no `target-cpu`, no explicit
SIMD paths in these crates). Figures are criterion medians from one run and are
noisy at the 10 to 20 per cent level. Treat them as a baseline, not a claim.

None of these crates has an explicit SIMD path yet (see the Phase 11 section in
`todo.md`). The `tpt` column is the crate's public kernel, which the compiler
autovectorises. The comparison is against a naive loop, `std`, or `format!`, as
noted per row. Read the rows where the crate is slower as findings, not as
regressions.

| crate | bench | median |
|---|---|---|
| bits | `popcount_u64` naive / tpt | 2.20 µs / 2.22 µs |
| bits | `prefix_sum_u32` naive / tpt | 1.16 µs / 1.16 µs |
| sort | `sort_u32` 1 000, std / tpt | 5.53 µs / 4.42 µs |
| sort | `sort_u32` 100 000, std / tpt | 1.24 ms / 0.47 ms |
| ntt | `modmul` `%` / Barrett / Montgomery | 3.14 µs / 2.96 µs / 10.85 µs |
| transpose | `u32` 64x64, naive / tpt | 1.91 µs / 2.13 µs |
| transpose | `u32` 512x512, naive / tpt | 675 µs / 148 µs |
| pixel | `rgb8_to_rgba8` 256K px, naive / tpt | 121 µs / 152 µs |
| pixel | `histogram_u8` 768 KiB, naive / tpt | 177 µs / 175 µs |
| hash | `adler32` 1 MiB, naive / tpt | 2.35 ms / 65 µs |
| hash | `crc32` 1 MiB, tpt | 475 µs |
| text | `utf8` validate ~1.1 MB, std / tpt | 330 µs / 324 µs |
| text | hex encode 256 KiB, format / tpt | 12.8 ms / 16.9 µs |
| search | `find_byte` 1 MiB, iter / tpt | 243 µs / 13.4 µs |
| search | `count_byte` 1 MiB, iter / tpt | 310 µs / 14.2 µs |
| search | `find` substring 1 MiB, windows / tpt | 433 µs / 15.3 µs |
| geom | `distance_sq` 16K points, AoS naive / SoA tpt | 8.46 µs / 2.81 µs |
| dct | DCT-II N=256, table matvec / tpt | 21.7 µs / 21.4 µs |
| distance | inner product f32 1024, iter / tpt | 428 ns / 120 ns |
| distance | L2 squared f32 1024, iter / tpt | 459 ns / 147 ns |
| audio | FIR 32 taps, 64K samples, naive / tpt | 1.06 ms / 0.92 ms |
| audio | `mix_into` 64K samples, naive / tpt | 5.46 µs / 5.41 µs |
| half | f32 to f16 64K, `half` per element / tpt | 65.5 µs / 66.0 µs |

Where the crate is slower or level, and the reason as far as it is known:

* **`ntt` Montgomery `mul`** is 3.5x slower than `%` in this loop. `mul` converts
  `b` into Montgomery form on every call (`mul_mont(a, to_mont(b))`, a second
  reduction), and the modulus is a runtime value. Barrett is level
  with `%`. For fixed multipliers (twiddles) use `mul_mont`, the one-reduction path: 2.09 µs
  for the same 4096 products, against 2.88 µs for `%` in the same run.
* **`transpose` 64x64** is slightly slower than the naive loop, which the compiler
  handles well at that size. The gap reverses at 512x512 (4.6x faster).
* **`pixel` `rgb8_to_rgba8`** is slower than the naive per-pixel copy. The compiler
  vectorises the naive loop, and the crate's version is not better. Candidate for a
  real SIMD path, or for removing the claim.
* **`pixel` `histogram_u8`** and **`text` UTF-8 validation** are level with the
  baseline. Their gains would come from explicit paths, not from autovectorisation.
* **`crc32`** has no baseline in the bench (a bitwise reference is far too slow to
  time usefully). Compare against `crc32fast` when the oracle decision is made.

The bench workflow (`.github/workflows/bench.yml`) uploads its criterion output as a
workflow artifact. Nothing commits these numbers automatically, so this table is updated by hand.


## tpt-simd-dct: FFT-based DCT-II against the table (2026-10-16)

`cargo bench -p tpt-simd-dct --bench kernels -- dct2_fast_vs_table`, one run, same host as the
Phase 11 baseline. Forward transform of one block, orthonormal DCT-II, f32 in and out:

| N | table `Dct` (O(N²) memory and time) | `FastDct` (2N-point FFT) | speed-up |
|---|---|---|---|
| 256 | 21.5 µs | 4.2 µs | 5.1x |
| 1024 | 435 µs | 19.1 µs | 23x |
| 4096 | 9.21 ms | 123 µs | 75x |

The table costs 4 bytes times N² in memory (64 MiB at N = 4096), which is the larger
practical problem. `FastDct` uses O(N) memory. It is faster from N = 256 at least, and the
table is the reference for sizes that are not powers of two.

## tpt-simd-blas refresh after the NEON level-1 change (2026-10-16)

`RUSTFLAGS="-C target-cpu=native" cargo bench -p tpt-simd-blas --bench blas`, one full run on the
x86_64 host (AVX-512 build, the same configuration as the sections above). The NEON
`dot`/`nrm2`/`asum` kernels are aarch64-only and compile out on this build, so this run checks
that the x86 numbers did not move. aarch64 was not timed (emulated runs are correctness-only).
Median times, `naive` = `tpt_simd_blas::reference`.

| Bench | naive | tpt | speed-up |
|---|---|---|---|
| `gemm_64` f32 | 133 µs | 8.2 µs (6.9 µs with workspace) | 16x (19x) |
| `gemm_64` f64 | 143 µs | 26.5 µs | 5.4x |
| `gemm_256` f32 | 21.6 ms | 331 µs (296 µs with workspace) | 65x (73x) |
| `gemm_256` f64 | 9.12 ms | 707 µs | 12.9x |
| `gemm_512` f32 | 77.3 ms | 2.71 ms (2.32 ms with workspace) | 28x (33x) |
| `gemm_512` f64 | 291 ms | 5.63 ms | 52x |
| `gemv_1024` f32 `n` | 2.37 ms | 75.7 µs | 31x |
| `gemv_1024` f32 `t` | 622 µs | 166 µs | 3.7x |
| `level1_4096` `dot_f32` | 1.80 µs | 481–495 ns | 3.7x |
| `level1_4096` `nrm2_f32` | 1.76 µs | 402 ns | 4.4x |
| `level1_4096` `asum_f32` | 1.75 µs | 312 ns | 5.6x |
| `level1_4096` `dot_f64` | 1.78 µs | 605 ns | 2.9x |
| `level1_1048576` `dot_f32` | 464 µs | 160 µs | 2.9x |
| `level1_1048576` `nrm2_f32` | 460 µs | 112 µs | 4.1x |
| `level1_1048576` `asum_f32` | 477 µs | 103 µs | 4.6x |
| `lu_solve_f64_512` | 15.6 ms | 5.11 ms | 3.0x |
| `getrf_f64_512` | 15.4 ms | 5.02 ms | 3.1x |
| `getrf_f64_64` | 25.5 µs | 18.2 µs | 1.4x |
| `cholesky_f64_512` | 31.0 ms | 1.92 ms | 16x |
| `cholesky_f32_512` | 22.7 ms | 1.01 ms | 22x |
| `lu_solve_f32_512` | 6.26 ms | 3.63 ms (range 2.8–4.2 ms) | 1.7x |

Caveats, in the order they matter:

* **Noise.** A `python` process on the host had accumulated about 70,000 CPU-seconds during the run.
  Repeat runs of `level1_4096/dot_f32/tpt` gave 481, 495 and 490 ns, but one gave 1.16 µs. Treat
  each figure as approximate. The `level1_1048576` `axpy_f32` and `dot_f64` rows are not
  reliable: their criterion ranges overlap the naive ones (`axpy` tpt 232 µs, range 203–262 µs;
  `dot_f64` 1M tpt 1.85 ms, naive 1.26 ms, range 0.96–1.56 ms), so no speed-up or slow-down is
  claimed for them.
* **Not the same as the AVX-512 A/B table above.** That table records `dot_f32` 4096 at 225 ns.
  Today's run gives about 490 ns for the same bench on the same build path, so the earlier
  absolute numbers do not reproduce on this machine state. The x86 code path changed only by
  compiled-out `cfg` branches, but this run does not prove the earlier figure was correct.
* **No aarch64 timings.** The NEON level-1 kernels and the `lu`/`cholesky` NEON gemm path were
  not measured.
