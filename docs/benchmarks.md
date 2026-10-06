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
