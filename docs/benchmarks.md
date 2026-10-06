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

## Phase 3–6 known issues / follow-ups

* **compare**: the `SimdMask` array backend is a poor fit for count/reduce
  patterns; add fused `count_*`/`movemask`-style helpers or lower to
  `_mm256_cmpgt_epi32` + `movemask` intrinsics.
* **gather**: documented honestly in the crate: `vpgatherdd` is slower than
  scalar loads here. Prefer scalar loops unless indices are dependent on SIMD
  data already in registers.
* **blend (mask)**: lane-loop lowering is not reaching `vblendvps`; try an
  explicit `core::arch` path.
* **window cos**: the degree-12 polynomial is slower than `libm::cosf`; revisit
  (lower degree, vectorised across lanes properly) or drop in favour of libm.
* **2D separable convolve / lanczos**: slower than scalar; likely strided
  vertical pass and per-call weight computation.
* No `cargo-show-asm` audit has been done for any of these.
