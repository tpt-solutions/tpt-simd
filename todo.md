# tpt-simd — Project Todo

Owner: TPT Solutions · License: `MIT OR Apache-2.0` · Spec: [spec.txt](spec.txt)

## Project decisions (locked)

- [x] License: dual MIT OR Apache-2.0, copyright TPT Solutions
- [x] Dependencies must be MIT-compatible (permissive; **no Apache-2.0-only deps**)
- [x] Backend: `core::simd` behind `nightly` feature; stable path via own `tpt-simd-vector` crate
- [x] 22 crates = spec's 21 + new `tpt-simd-vector` (stable backend types)
- [x] `no_std` + `alloc` (`std` optional), MSRV = stable-2
- [x] Targets in v0.1: x86_64 (SSE/AVX/AVX2/AVX-512), ARM64 (NEON/SVE), RISC-V RVV, scalar fallback
- [x] GitHub + GitHub Actions; tpt-kinetix and tpt-cadence are existing separate repos
- [x] Duplicate fns get one home crate, re-exported elsewhere:
  - `pack_i16_to_i8` → `tpt-simd-saturate` (re-exported by permute)
  - `select_f32` → `tpt-simd-blend` (re-exported by select)
  - `ComplexSimd` type → `tpt-simd-core`/`vector`; `complex_mul` lives in `tpt-simd-mul`; `tpt-simd-complex` builds on it (breaks mul↔complex cycle)

## Per-crate "definition of done" (apply to every crate)

- [ ] Scalar reference implementation
- [ ] SIMD implementation(s) for each target in scope
- [ ] Unit tests vs scalar reference (incl. NaN/inf/overflow/underflow/empty/tail lengths)
- [ ] proptest property tests (SIMD == scalar)
- [ ] Doc comment on every public item (perf notes, example, `# Safety` on unsafe fns)
- [ ] criterion benchmark vs scalar (and vs manual intrinsics / `wide` where relevant)
- [ ] `cargo-show-asm` check of hot functions
- [ ] `no_std` build passes; clippy + rustfmt clean; `cargo doc` clean
- [ ] Dual-license headers/metadata (`license = "MIT OR Apache-2.0"`)

---

## Phase 0 — Project setup (before Week 1)

### Repo & licensing
- [x] `git init`, default branch (`master`), `.gitignore`, `.gitattributes`
- [ ] Create GitHub repo under TPT Solutions org
- [x] Add `LICENSE-MIT` and `LICENSE-APACHE` (copyright TPT Solutions)
- [x] README with dual-license statement ("MIT or Apache-2.0, at your option") and contribution clause
- [x] `CONTRIBUTING.md` (contributions are dual-licensed unless stated), `SECURITY.md`
- [x] `cargo-deny` config: allow-list MIT, BSD-2/3, ISC, Zlib, Unicode, `MIT OR Apache-2.0` duals; deny Apache-2.0-only, GPL/LGPL/AGPL
- [x] Audit candidate deps (`wide`, `proptest`, `criterion`, `bytemuck`, etc.) for license compatibility
- [x] Decide dev-dependency policy (criterion/proptest are dev-only; confirm licenses anyway)
- [ ] Add NOTICE/THIRD-PARTY-LICENSES generation (`cargo-about` or similar)

### Workspace
- [x] Root `Cargo.toml` workspace with `[workspace.package]` (version, edition, license, authors, repository, rust-version)
- [x] Scaffold all 22 crate dirs with `Cargo.toml` + `src/lib.rs` (`#![no_std]`, `extern crate alloc`)
  - [x] tpt-simd-core
  - [x] tpt-simd-vector
  - [x] tpt-simd-complex
  - [x] tpt-simd-fixed
  - [x] tpt-simd-saturate
  - [x] tpt-simd-horizontal
  - [x] tpt-simd-permute
  - [x] tpt-simd-gather
  - [x] tpt-simd-scatter
  - [x] tpt-simd-aligned
  - [x] tpt-simd-mul
  - [x] tpt-simd-dot
  - [x] tpt-simd-rounding
  - [x] tpt-simd-shift
  - [x] tpt-simd-compare
  - [x] tpt-simd-blend
  - [x] tpt-simd-select
  - [x] tpt-simd-butterfly
  - [x] tpt-simd-convolve
  - [x] tpt-simd-interpolate
  - [x] tpt-simd-matrix
  - [x] tpt-simd-window
- [x] Optional umbrella crate `tpt-simd` re-exporting all crates (decide: yes/no) — yes; re-exports the implemented crates
- [ ] Feature flags: `std`, `alloc`, `nightly`, `sse2`/`avx2`/`avx512`/`neon`/`sve`/`rvv`, `scalar-only`
- [x] Decide runtime dispatch vs compile-time `target_feature` strategy (document in ADR) — compile-time `cfg(target_feature)`; see ADR 0001
- [x] `rust-toolchain.toml` (stable) + separate nightly job config (nightly CI job in ci.yml)
- [ ] Pin and test MSRV (stable-2) with `cargo-msrv`
- [x] Shared test-utils / scalar-reference helper crate (unpublished) for proptest strategies

### CI (GitHub Actions)
- [x] fmt + clippy + doc (`-D warnings`)
- [x] Test matrix: stable + nightly (`nightly` feature) × Linux/Windows/macOS — workflow written, not yet run on GitHub
- [x] `no_std` build check (e.g. `thumbv7em`/`wasm32-unknown-unknown`)
- [x] x86_64 runners with target-feature matrix (SSE2, AVX, AVX2) — written, not yet run
- [ ] AVX-512 testing via Intel SDE (or AVX-512 runner)
- [x] ARM64 testing (GitHub ARM runners / macOS-14) — runner defined, not yet run
- [ ] ARM SVE testing via QEMU (vector-length matrix)
- [ ] RISC-V RVV testing via QEMU (`riscv64gcv`)
- [x] Scalar-fallback job (`scalar-only` feature)
- [x] Miri job for unsafe code (where supported) — written, not yet run
- [x] `cargo-deny` license + advisory job
- [ ] Benchmark job with regression tracking (criterion baselines)
- [x] Caching, concurrency limits, release workflow

### Docs & spec fixes
- [x] Fix spec: "20 crates" vs 21 listed (now 22 with vector crate)
- [x] Fix spec: `core::simd` is nightly-only, not stable (spec.txt updated)
- [x] Fix spec typos ("Weekes"); resolve duplicate API entries (see decisions)
- [ ] Fix spec API inconsistencies: `Fixed::saturate`, `_mm256_mulhi_epi32` does not exist (use `_mm256_mul_epi32` + shifts), `ComplexSimd` `a * b` operator in Appendix B vs `.mul()` methods
- [x] Define `Simd`/`SimdMask` public API surface and naming (ADR)
- [x] ADRs: backend strategy, dispatch strategy, error/panic policy, unsafe policy
- [ ] Reserve crate names on crates.io (publish 0.0.0 placeholders)

---

## Phase 1 — Core infrastructure (Weeks 1–3)

### tpt-simd-core
- [x] `SimdElement` trait + impls (i8/i16/i32/i64/u8/u16/u32/f32/f64)
- [x] `SimdOps`/`SimdVector` traits
- [x] Runtime feature detection helpers (`std` + `no_std` paths)
- [x] Vector width constants per target
- [x] Type aliases: `F32x8`, `I32x8`, `I16x16`, `I8x32`, 128/256/512-bit families
- [x] Shared `ComplexSimd` type definition (see decisions)
- [ ] Backend selection plumbing — `nightly` feature reserved/no-op (ADR 0001) (`nightly` → `core::simd`, else `tpt-simd-vector`)
- [ ] Core-done checklist

### tpt-simd-vector (stable backend)
- [x] `Simd<T, N>` and `SimdMask<T, N>` types (array-backed)
- [x] Arithmetic/bitwise/comparison ops, splat, from_array/to_array, load/store (aligned + unaligned)
- [ ] `core::arch` fast paths: SSE2/AVX2 (x86_64), NEON (aarch64) — decided per-crate instead (fixed, mul, dot have AVX2 paths); none in vector; no NEON anywhere
- [x] Scalar fallback for all other targets
- [ ] Verify codegen: array backend auto-vectorizes; fast paths beat it — `mul_add`/`round`/`floor` are per-lane libm calls without the `std` feature; with `std` they use std intrinsics (5x on portable complex mul; see docs/benchmarks.md)
- [x] Layout/ABI tests (size, align, lane order)
- [ ] Tests mirroring `core::simd` behavior so the two backends are interchangeable
- [ ] Vector-crate-done checklist

### tpt-simd-complex (f32 first)
- [x] `new`, `add`, `sub`, `mul`, `div`, `conj`, `mag_sq`, `mag`, `phase`
- [x] `twiddle_mul`, `butterfly` (in-place)
- [x] Operator overloads (`+ - * /`) so Appendix B example compiles (in core)
- [x] FMA (`fmadd/fmsub`) and `addsub` fast path on x86; NEON equivalent
- [x] Interleaved ↔ split (SoA/AoS) conversions
- [x] f64 variant (stretch)
- [ ] Hit target: ≥3× complex multiply speedup vs scalar — 3.3× vs naive scalar, only 1.6× vs auto-vectorised; see docs/benchmarks.md
- [ ] Crate-done checklist

### tpt-simd-fixed (i32 first)
- [x] `Fixed<T, INT_BITS, FRAC_BITS, N>` with compile-time format checks (i32 and i16)
- [x] `add`, `sub`, `mul` (64-bit intermediate, correct rounding), `div`
- [x] `from_f32`, `to_f32`, `saturate` — `saturate` replaced by explicit `saturating_*` ops (ADR 0002)
- [x] Format conversion between Q formats (for cookbook)
- [x] Overflow semantics documented (wrapping vs saturating variants)
- [x] i16 variant (stretch)
- [x] Hit target: ≥2× vs manual fixed-point — met on mul/add (3.3–10.7×); `from_f32` no gain
- [ ] Crate-done checklist

### Phase 1 exit
- [ ] Workspace builds on all CI targets
- [ ] Core traits frozen for 0.1 review
- [ ] Complex + fixed tested vs scalar refs

---

## Phase 2 — Common patterns (Weeks 4–6)

### tpt-simd-horizontal
- [x] `horizontal_sum_i32`, `_f32`, `_i16` (→ i32)
- [x] `horizontal_max_i16`, `horizontal_min_f32`, `horizontal_product_f32`
- [x] Add missing min/max/sum variants for symmetry (decide scope) (min/max for i16/i32/f32, u8/i8 sums, generic `reduce_*`)
- [x] NaN handling policy documented for min/max
- [ ] Hit target: ≥4–8× vs scalar — NOT met (0.7–2.6×; LLVM already vectorises); see docs/benchmarks.md
- [ ] Crate-done checklist

### tpt-simd-dot
- [x] `dot_product_i16`, `dot_product_f32`, `dot_product_complex_f32`, `dot_product_saturating_i16`
- [x] Handle arbitrary lengths / tails; length-mismatch policy
- [x] Accumulation order/precision documented (f32 reassociation)
- [ ] Hit target: ≥8× vs scalar (256 × i16) — NOT met (1.55× native AVX2 `vpmaddwd`); f32 dot 5.7–12.5×
- [x] Verify FLAC LPC example (Appendix B.3) compiles and passes
- [ ] Crate-done checklist

### tpt-simd-butterfly
- [x] `butterfly_f32`, `butterfly_i16`, `butterfly_complex_f32`, `butterfly_with_twiddle_f32`
- [x] Fix spec: `butterfly_with_twiddle_f32` signature (real vectors + complex twiddle is ambiguous)
- [x] Verify Appendix B.5 example output
- [x] Hit target: ≥2× — vs indexed scalar only; parity with auto-vectorised; f32 on SSE2 default is a known slow case (docs/benchmarks.md)
- [ ] Crate-done checklist

### tpt-simd-saturate
- [x] `saturating_add_i16`, `saturating_sub_i16`, `saturating_add_i8`, `saturating_mul_i16`
- [x] `pack_i16_to_i8` (home crate), plus unsigned-saturating pack variants
- [x] Hit target: ≥4× — vs indexed scalar only; parity with auto-vectorised (docs/benchmarks.md)
- [ ] Crate-done checklist

### Phase 2 exit
- [x] Benchmark suite vs scalar (criterion) published in repo (per-crate `benches/`, results in docs/benchmarks.md)
- [ ] 3–8× speedups demonstrated (table from spec §6.2 reproduced)
- [x] Docs + examples for all four crates

---

## Phase 3 — Data shuffling (Weeks 7–8)

### tpt-simd-permute
- [x] `transpose_8x8_i16`, `transpose_4x4_f32`
- [x] `interleave_stereo_i16`, `deinterleave_stereo_i16`
- [x] `interleave_yuv420`
- [x] `unpack_i8_to_i16` (and re-export `pack_i16_to_i8` from saturate)
- [x] General permutation helper (`permutevar8x32` equivalent)
- [x] Length-mismatch and tail handling
- [ ] Verify Appendix B.4 example
- [ ] Hit target: ≥5× 8×8 transpose — NOT met (2.3×); see docs/benchmarks.md
- [ ] Crate-done checklist

### tpt-simd-gather
- [x] `gather_i32`, `gather_f32` (unsafe)
- [x] `gather_checked_i32` (panics on OOB)
- [x] Software fallback for CPUs without gather
- [x] Benchmark vs scalar (gather is often slow — document honestly) — 0.09–0.13× here; documented
- [ ] Crate-done checklist

### tpt-simd-scatter
- [x] `scatter_i32`, `scatter_f32` (unsafe)
- [x] `scatter_checked_i32`
- [ ] AVX-512 scatter path + AVX2/scalar fallback
- [x] Define behavior for duplicate indices
- [ ] Crate-done checklist

### tpt-simd-aligned
- [x] `Aligned32<T>`, `Aligned64<T>` (+ `Aligned16`)
- [x] `aligned_vec_f32` (fix spec: `Vec<Aligned32<f32>>` wastes space; consider aligned buffer type)
- [x] `is_aligned`
- [x] Safe aligned load/store wrappers
- [x] `alloc`-gated allocation helpers
- [ ] Crate-done checklist

### Phase 3 exit
- [ ] Integration tests with real codec data patterns (sample YUV/PCM fixtures — check fixture licenses)

---

## Phase 4 — Arithmetic variants (Weeks 9–10)

### tpt-simd-mul
- [x] `mul_hi_i16`, `mul_lo_i32`, `mul_hi_i32`
- [x] `mul_add_sub_f32` (`fmaddsub`)
- [x] `complex_mul_f32` (home for complex multiply; complex crate uses it)
- [x] Widening multiplies (i16→i32, i32→i64) (decide scope)
- [ ] Crate-done checklist

### tpt-simd-rounding
- [x] `round_f32`, `floor_f32`, `ceil_f32`, `trunc_f32`
- [x] `round_to_nearest_even_i32`, `round_with_bias_i32`
- [x] SSE2 fallback (no `roundps` before SSE4.1) — branch-free bit trick; NEON `vrndn` not written (LLVM autovectorises)
- [x] Define tie-breaking (half-away vs half-even) per function
- [ ] Crate-done checklist

### tpt-simd-shift
- [x] `shift_left_i32`, `shift_right_logical_i32`, `shift_right_arithmetic_i32`
- [x] `rotate_left_i32`, `shift_with_rounding_i32`
- [x] Out-of-range shift amount policy (≥32)
- [x] Per-lane variable shift variants (AVX2 `vpsllvd` etc. when compiled in)
- [ ] Crate-done checklist

### Phase 4 exit
- [ ] `cargo-show-asm` audit recorded for hot functions in all Phase 1–4 crates
- [ ] Fallback to explicit intrinsics where LLVM codegen is poor

---

## Phase 5 — Comparison & control flow (Weeks 11–12)

### tpt-simd-compare
- [x] `cmp_gt_i32`, `cmp_lt_i32`, `cmp_eq_i32`, `cmp_ne_i32`, `cmp_gt_f32`
- [x] Fill out symmetry (`ge`, `le`, f32 `lt/eq/ne/ge/le`, i16/i8) (decide scope)
- [x] Mask ops: `any`, `all`, `count`, `to_bitmask`
- [x] NaN comparison semantics documented
- [ ] Crate-done checklist

### tpt-simd-blend
- [x] `blend_i32`, `blend_f32`, `blend_i8`
- [x] `select_f32` (home crate)
- [x] Immediate-blend (const mask) variants
- [ ] Crate-done checklist

### tpt-simd-select
- [x] `select_i32`, `select_f32` (re-export from blend; this crate's index-based variant is distinct — rename to avoid clash)
- [x] `select_from_slice_i32` (uses gather; bounds policy)
- [ ] Crate-done checklist

### Phase 5 exit
- [x] Branch-elimination benchmarks showing benefit vs branchy scalar — `blend_f32` 2.9×, fused `blend_gt_f32` 6.4×, `select_f32` 6.1× (docs/benchmarks.md)

---

## Phase 6 — DSP primitives (Weeks 13–14)

### tpt-simd-convolve
- [x] `convolve_1d_f32` (output length / edge-mode policy)
- [x] `convolve_2d_separable_f32` (added caller `scratch` slice; fix spec: `&[[f32]]` is not valid Rust — use `&[f32]` + stride/width or slice of rows)
- [x] `fir_filter_i16` (saturation/rounding policy)
- [ ] Crate-done checklist

### tpt-simd-interpolate
- [x] `interpolate_linear_f32` (scalar), `interpolate_linear_simd_f32`
- [x] `interpolate_cubic_f32`
- [x] `interpolate_lanczos_f32` (kernel size `a`, `sinc` approximation)
- [ ] Crate-done checklist

### tpt-simd-matrix
- [x] `mat4x4_mul_f32`, `mat8x8_mul_i16`
- [x] `mat4x4_transpose_f32`, `mat8x8_transpose_i16` (reuse permute)
- [x] `mat3x3_inverse_f32` (singular-matrix → `None`, epsilon policy)
- [ ] Crate-done checklist

### tpt-simd-window
- [x] `hamming`, `hanning`, `blackman`, `kaiser` window generators (`alloc`)
- [x] `apply_window_f32`, `apply_window_simd_f32`
- [x] Cosine approximation (polynomial, max err 2e-6) — no LUT
- [x] Bessel I0 for Kaiser
- [x] Optional `no_alloc` variants writing into caller slices
- [ ] Crate-done checklist

### Phase 6 exit
- [ ] Integration tests: FFT and MDCT built from tpt-simd crates, validated vs rustfft / reference
- [ ] Cross-check conformance vs known-good outputs

---

## Phase 7 — Integration & polish (Weeks 15–16)

### tpt-kinetix (video; separate repo)
- [ ] Profile and list manual-SIMD hot loops (baseline numbers)
- [ ] Replace hot loops with tpt-simd (motion comp, transforms, etc.)
- [ ] Byte-exact conformance on reference AV1 streams (before/after)
- [ ] Measure speedup (target 20–30% AV1 decode)
- [ ] Verify ARM/NEON now works without separate code

### tpt-cadence (audio; separate repo)
- [ ] Profile baseline FFT/MDCT
- [ ] Replace manual complex math/shuffling with tpt-simd-complex/butterfly
- [ ] Adopt `Fixed` where applicable
- [ ] Bit-exact / tolerance conformance
- [ ] Measure speedup (target 40–50% FFT)

### Documentation
- [ ] API docs complete (every public fn; perf notes; `# Safety`)
- [ ] mdBook (or similar) site, hosted on GitHub Pages
- [ ] Cookbook:
  - [ ] "How to implement an FFT using tpt-simd"
  - [ ] "How to do motion compensation in a video codec"
  - [ ] "How to convert between fixed-point formats"
  - [ ] "How to apply a window function to audio data"
- [ ] Migration guide:
  - [ ] from manual intrinsics
  - [ ] from scalar code
  - [ ] from other SIMD crates (`wide`, `simba`, `std::simd`)
- [ ] All Appendix B examples compile as doctests

### Release
- [ ] Full benchmark report vs scalar / manual intrinsics / `wide` / `simba` / `rustfft`
- [ ] Performance-regression guard in CI
- [ ] Per-crate READMEs, keywords, categories, `rust-version`, license fields
- [ ] CHANGELOG, semver policy, release process
- [ ] `cargo publish --dry-run` for all crates; publish in dependency order
- [ ] Tag v0.1.0, GitHub release, announce (r/rust, users.rust-lang.org, This Week in Rust)

---

## Phase 8 — Cross-architecture hardening (v0.1 scope: everything in spec)

- [ ] x86_64: SSE / AVX / AVX2 paths verified on real hardware
- [ ] x86_64: AVX-512 specializations for all crates, tested via SDE/hardware
- [ ] ARM64: NEON paths for every crate (Appendix A NEON intrinsics list)
- [ ] ARM64: SVE implementation (scalable vector length; test several VLs in QEMU)
- [ ] RISC-V: RVV implementation (check `core::arch::riscv64` stabilization status; fallback if intrinsics unstable)
- [ ] Scalar fallback verified on every crate — `cargo check --workspace --exclude tpt-simd-testutil --no-default-features` passes on thumbv7em-none-eabihf, wasm32-unknown-unknown, riscv32imc-none-elf, aarch64-none-softfloat (compile only, not run)
- [ ] Test all vector widths (128/256/512-bit)
- [ ] Runtime dispatch correctness (right path picked, no UB on unsupported CPUs)
- [ ] Byte-exact cross-arch results where documented; document float-differences otherwise
- [ ] Unsafe audit + Miri pass
- [ ] Fuzzing (`cargo-fuzz`) of slice-based APIs

---

## Phase 9 — Future roadmap (post-v0.1)

### 6–12 months
- [ ] AVX-512 tuning pass beyond baseline specializations
- [ ] SVE tuning (predication, VL-agnostic loops)
- [ ] RVV tuning
- [ ] GPU compute backend: CUDA
- [ ] GPU compute backend: Metal
- [ ] GPU compute backend: Vulkan
- [ ] License-check all new GPU dependencies (MIT-compatible only)

### 12–24 months
- [ ] Machine-learning primitives (tensor ops)
- [ ] Image processing primitives (convolution, resampling)
- [ ] Halide integration for automatic scheduling
- [ ] JIT compilation for runtime optimization

---

## Phase 10 — tpt-math acceleration (numeric tier)

Goal: speed up [tpt-math](https://github.com/tpt-solutions/tpt-math) (31 crates, currently no SIMD; hand-written scalar loops, e.g. `DMatrix * DMatrix` in `tpt-math-linalg-dense` is a naive strided triple loop). New f32/f64 kernel crates sit on `tpt-simd-vector`/`core`; tpt-math adopts them behind an optional `simd` feature with the scalar path kept as the reference. Skip FFT (`tpt-math-signal-fft` wraps rustfft, already SIMD).

### Prerequisites / decisions
- [ ] Clone tpt-math and add baseline criterion benches first (gemm 64/256/1024, dot/norm, LU solve, CG on a Poisson matrix, Monte Carlo 10^7 samples); confirm gains justify the work and how much LLVM already autovectorises
- [ ] Decide dispatch for library consumers: ADR 0001 is compile-time `cfg(target_feature)`, so users without `-C target-cpu=native` silently get the slow path. Choose runtime dispatch (`is_x86_feature_detected!`, needs `std`) behind a feature, or document required build flags; write ADR 0003 — drafted as ADR 0003 (proposed; needs sign-off, not implemented)
- [ ] Decide float-determinism policy for reordered reductions and polynomial math (ADR 0001 requires bit-identical; relax to documented ULP/tolerance for `tpt-simd-math` and SIMD reductions); check tpt-math tests and formal-verification consumers for exact-value dependence — tiers drafted in ADR 0003 (proposed)
- [ ] Add new crates to the workspace; keep `no_std`, MIT/Apache-only deps

### tpt-simd-blas (highest value)
- [x] f32/f64 `axpy`, `scal`, `dot`, `nrm2`, `asum`
- [x] `gemv` (column-major, matches tpt-math storage)
- [x] Packed, register-blocked `gemm` microkernel (e.g. 8x4 f32 / 4x4 f64, FMA) with cache blocking
- [ ] Complex variants via split re/im (`ComplexSimd`)
- [x] Portable reference + AVX2/FMA path; NEON later
- [ ] Tests vs scalar reference (tails, NaN/inf, non-multiple sizes), proptest, criterion bench, `cargo-show-asm`
- [x] Target: gemm >= 3x scalar f32 on AVX2 — 5.5–32x vs naive strided loop (28–43 GFLOP/s), see docs/benchmarks.md
- [ ] Wire into `tpt-math-linalg-dense` (`DMatrix` mul, `DVector::dot`/`norm`, LU/Cholesky/QR inner loops) and `tpt-math-linalg-complex` behind `simd`

### tpt-simd-math
- [x] Vector `exp`, `ln`, `sin`, `cos`, `tanh`, `erf` (polynomial approximations, documented max ULP error) — f32 only; max ULP 0.8–3.4 (docs in crate)
- [x] Accuracy tests against `libm` over full range, special values (NaN/inf/subnormal)
- [ ] Wire into `tpt-math-stats`, `tpt-math-prob-dist`, `tpt-math-prob-monte-carlo`, `tpt-math-prob-sampler`, autodiff

### tpt-simd-rng
- [x] Lane-parallel xoshiro / Philox generators (independent streams per lane, reproducible seeding)
- [x] Vectorised uniform -> normal (Box-Muller or ziggurat) — Box-Muller, 5-7.5x vs libm scalar
- [x] Statistical quality checks (e.g. PractRand/TestU01-style smoke tests) — chi-square/moment/bit-balance smoke tests (not PractRand)
- [ ] Wire into `tpt-math-prob-sampler` and `tpt-math-prob-monte-carlo`

### tpt-simd-sparse
- [x] CSR/CSC SpMV using `tpt-simd-gather` — implemented with multi-accumulator loads (gather measured, not adopted; ~1.2–2x, memory bound)
- [x] Fused vector updates for CG / BiCGSTAB — ~1.5x vs 3 passes (cache-resident)
- [ ] Wire into `tpt-math-linalg-sparse`

### tpt-simd-reduce
- [x] Pairwise / compensated sum, mean, variance, covariance over slices — done in tpt-simd-reduce
- [x] min/max/argmin/argmax over slices — done (plain fold; LLVM parity)
- [x] Builds on `tpt-simd-horizontal`; wire into `tpt-math-stats` — crate done; tpt-math wiring not done

### Existing stubs that map directly to tpt-math
- [ ] `tpt-simd-convolve` (FIR) and `tpt-simd-window` -> `tpt-math-signal-filter`
- [ ] `tpt-simd-matrix` (3x3 / 4x4 multiply, transpose, inverse) -> `tpt-math-linalg-fixed`, `tpt-math-geometry`, `tpt-math-spatial`
- [ ] Later: IIR biquad cascades (serial dependency; batch across channels)

### Phase 10 exit
- [ ] End-to-end tpt-math benches with and without `simd` show target speedups
- [ ] tpt-math test suite passes with `simd` on and off (within documented tolerances)
- [ ] `no_std` builds pass in both repos

### Repo housekeeping found while scoping
- [x] Workspace `Cargo.toml` / `rust-toolchain.toml` were rewritten to an older shape (10 members, no `keywords`/`libm`/lints/path deps) and no longer parse; keep reconciled manifest with all real crates as members
- [x] `tpt-simd-dot`: `[[bench]] name = "dot"` declared but `benches/dot.rs` missing
- [x] `tpt-simd-core` docs link `docs/adr/0001-backend-strategy.md`; real file is `0001-backend-and-dispatch.md`
- [x] `tpt-simd` umbrella crate is still a 2-line stub with no re-exports
- [x] `tpt-simd-mul` is missing `mul_hi_i16`, `mul_lo_i32`, `mul_hi_i32`, `mul_add_sub_f32`
- [x] Tick Phase 0 / 1 / 2 checkboxes that are already done in code; commit the uncommitted butterfly/complex/fixed/horizontal/saturate/dot work

---

## Success metrics tracking

### Technical
- [ ] 3–8× speedup over scalar in hot loops
- [ ] 100% match with scalar reference implementations
- [ ] Works on x86_64, ARM64, RISC-V without code changes

### Ecosystem (measured 6 months after v0.1)
- [ ] Used by ≥3 major Rust codec projects
- [ ] 500+ GitHub stars
- [ ] 10,000+ crates.io downloads

### Business
- [ ] SIMD dev time reduced 70%
- [ ] SIMD-related bugs reduced 90%
- [ ] Code size reduced 50%

## Risk watchlist

- [ ] LLVM generates suboptimal code → show-asm audits, explicit-intrinsic fallbacks
- [ ] API too complex → start small, examples first
- [ ] Portability leaks → CI on all targets, careful `cfg`
- [ ] Performance regression → continuous benchmarks
- [ ] Maintenance burden → automated cross-arch CI, small focused crates
- [ ] Nightly `core::simd` API churn → keep behind `nightly` feature, pin nightly date in CI
- [ ] Stable intrinsics gaps (SVE/RVV/AVX-512 on stable) → track Rust releases, feature-gate
- [ ] License drift → `cargo-deny` gate on every PR
- [ ] Compile-time dispatch means library users miss SIMD without target flags → runtime dispatch or documented flags (Phase 10)
- [ ] SIMD float reordering changes results vs scalar → documented tolerances, scalar path stays default in tpt-math
