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
- [ ] Audit candidate deps (`wide`, `proptest`, `criterion`, `bytemuck`, etc.) for license compatibility
- [ ] Decide dev-dependency policy (criterion/proptest are dev-only; confirm licenses anyway)
- [ ] Add NOTICE/THIRD-PARTY-LICENSES generation (`cargo-about` or similar)

### Workspace
- [ ] Root `Cargo.toml` workspace with `[workspace.package]` (version, edition, license, authors, repository, rust-version)
- [ ] Scaffold all 22 crate dirs with `Cargo.toml` + `src/lib.rs` (`#![no_std]`, `extern crate alloc`)
  - [ ] tpt-simd-core
  - [ ] tpt-simd-vector
  - [ ] tpt-simd-complex
  - [ ] tpt-simd-fixed
  - [ ] tpt-simd-saturate
  - [ ] tpt-simd-horizontal
  - [ ] tpt-simd-permute
  - [ ] tpt-simd-gather
  - [ ] tpt-simd-scatter
  - [ ] tpt-simd-aligned
  - [ ] tpt-simd-mul
  - [ ] tpt-simd-dot
  - [ ] tpt-simd-rounding
  - [ ] tpt-simd-shift
  - [ ] tpt-simd-compare
  - [ ] tpt-simd-blend
  - [ ] tpt-simd-select
  - [ ] tpt-simd-butterfly
  - [ ] tpt-simd-convolve
  - [ ] tpt-simd-interpolate
  - [ ] tpt-simd-matrix
  - [ ] tpt-simd-window
- [ ] Optional umbrella crate `tpt-simd` re-exporting all crates (decide: yes/no)
- [ ] Feature flags: `std`, `alloc`, `nightly`, `sse2`/`avx2`/`avx512`/`neon`/`sve`/`rvv`, `scalar-only`
- [ ] Decide runtime dispatch vs compile-time `target_feature` strategy (document in ADR)
- [ ] `rust-toolchain.toml` (stable) + separate nightly job config
- [ ] Pin and test MSRV (stable-2) with `cargo-msrv`
- [ ] Shared test-utils / scalar-reference helper crate (unpublished) for proptest strategies

### CI (GitHub Actions)
- [ ] fmt + clippy + doc (`-D warnings`)
- [ ] Test matrix: stable + nightly (`nightly` feature) × Linux/Windows/macOS
- [ ] `no_std` build check (e.g. `thumbv7em`/`wasm32-unknown-unknown`)
- [ ] x86_64 runners with target-feature matrix (SSE2, AVX, AVX2)
- [ ] AVX-512 testing via Intel SDE (or AVX-512 runner)
- [ ] ARM64 testing (GitHub ARM runners / macOS-14)
- [ ] ARM SVE testing via QEMU (vector-length matrix)
- [ ] RISC-V RVV testing via QEMU (`riscv64gcv`)
- [ ] Scalar-fallback job (`scalar-only` feature)
- [ ] Miri job for unsafe code (where supported)
- [ ] `cargo-deny` license + advisory job
- [ ] Benchmark job with regression tracking (criterion baselines)
- [ ] Caching, concurrency limits, release workflow

### Docs & spec fixes
- [ ] Fix spec: "20 crates" vs 21 listed (now 22 with vector crate)
- [ ] Fix spec: `core::simd` is nightly-only, not stable
- [ ] Fix spec typos ("Weekes"); resolve duplicate API entries (see decisions)
- [ ] Fix spec API inconsistencies: `Fixed::saturate`, `_mm256_mulhi_epi32` does not exist (use `_mm256_mul_epi32` + shifts), `ComplexSimd` `a * b` operator in Appendix B vs `.mul()` methods
- [ ] Define `Simd`/`SimdMask` public API surface and naming (ADR)
- [ ] ADRs: backend strategy, dispatch strategy, error/panic policy, unsafe policy
- [ ] Reserve crate names on crates.io (publish 0.0.0 placeholders)

---

## Phase 1 — Core infrastructure (Weeks 1–3)

### tpt-simd-core
- [ ] `SimdElement` trait + impls (i8/i16/i32/i64/u8/u16/u32/f32/f64)
- [ ] `SimdOps`/`SimdVector` traits
- [ ] Runtime feature detection helpers (`std` + `no_std` paths)
- [ ] Vector width constants per target
- [ ] Type aliases: `F32x8`, `I32x8`, `I16x16`, `I8x32`, 128/256/512-bit families
- [ ] Shared `ComplexSimd` type definition (see decisions)
- [ ] Backend selection plumbing (`nightly` → `core::simd`, else `tpt-simd-vector`)
- [ ] Core-done checklist

### tpt-simd-vector (stable backend)
- [ ] `Simd<T, N>` and `SimdMask<T, N>` types (array-backed)
- [ ] Arithmetic/bitwise/comparison ops, splat, from_array/to_array, load/store (aligned + unaligned)
- [ ] `core::arch` fast paths: SSE2/AVX2 (x86_64), NEON (aarch64)
- [ ] Scalar fallback for all other targets
- [ ] Verify codegen: array backend auto-vectorizes; fast paths beat it
- [ ] Layout/ABI tests (size, align, lane order)
- [ ] Tests mirroring `core::simd` behavior so the two backends are interchangeable
- [ ] Vector-crate-done checklist

### tpt-simd-complex (f32 first)
- [ ] `new`, `add`, `sub`, `mul`, `div`, `conj`, `mag_sq`, `mag`, `phase`
- [ ] `twiddle_mul`, `butterfly` (in-place)
- [ ] Operator overloads (`+ - * /`) so Appendix B example compiles
- [ ] FMA (`fmadd/fmsub`) and `addsub` fast path on x86; NEON equivalent
- [ ] Interleaved ↔ split (SoA/AoS) conversions
- [ ] f64 variant (stretch)
- [ ] Hit target: ≥3× complex multiply speedup vs scalar
- [ ] Crate-done checklist

### tpt-simd-fixed (i32 first)
- [ ] `Fixed<T, INT_BITS, FRAC_BITS, N>` with compile-time format checks
- [ ] `add`, `sub`, `mul` (64-bit intermediate, correct rounding), `div`
- [ ] `from_f32`, `to_f32`, `saturate`
- [ ] Format conversion between Q formats (for cookbook)
- [ ] Overflow semantics documented (wrapping vs saturating variants)
- [ ] i16 variant (stretch)
- [ ] Hit target: ≥2× vs manual fixed-point
- [ ] Crate-done checklist

### Phase 1 exit
- [ ] Workspace builds on all CI targets
- [ ] Core traits frozen for 0.1 review
- [ ] Complex + fixed tested vs scalar refs

---

## Phase 2 — Common patterns (Weeks 4–6)

### tpt-simd-horizontal
- [ ] `horizontal_sum_i32`, `_f32`, `_i16` (→ i32)
- [ ] `horizontal_max_i16`, `horizontal_min_f32`, `horizontal_product_f32`
- [ ] Add missing min/max/sum variants for symmetry (decide scope)
- [ ] NaN handling policy documented for min/max
- [ ] Hit target: ≥4–8× vs scalar
- [ ] Crate-done checklist

### tpt-simd-dot
- [ ] `dot_product_i16`, `dot_product_f32`, `dot_product_complex_f32`, `dot_product_saturating_i16`
- [ ] Handle arbitrary lengths / tails; length-mismatch policy
- [ ] Accumulation order/precision documented (f32 reassociation)
- [ ] Hit target: ≥8× vs scalar (256 × i16)
- [ ] Verify FLAC LPC example (Appendix B.3) compiles and passes
- [ ] Crate-done checklist

### tpt-simd-butterfly
- [ ] `butterfly_f32`, `butterfly_i16`, `butterfly_complex_f32`, `butterfly_with_twiddle_f32`
- [ ] Fix spec: `butterfly_with_twiddle_f32` signature (real vectors + complex twiddle is ambiguous)
- [ ] Verify Appendix B.5 example output
- [ ] Hit target: ≥2×
- [ ] Crate-done checklist

### tpt-simd-saturate
- [ ] `saturating_add_i16`, `saturating_sub_i16`, `saturating_add_i8`, `saturating_mul_i16`
- [ ] `pack_i16_to_i8` (home crate), plus unsigned-saturating pack variants
- [ ] Hit target: ≥4×
- [ ] Crate-done checklist

### Phase 2 exit
- [ ] Benchmark suite vs scalar (criterion) published in repo
- [ ] 3–8× speedups demonstrated (table from spec §6.2 reproduced)
- [ ] Docs + examples for all four crates

---

## Phase 3 — Data shuffling (Weeks 7–8)

### tpt-simd-permute
- [ ] `transpose_8x8_i16`, `transpose_4x4_f32`
- [ ] `interleave_stereo_i16`, `deinterleave_stereo_i16`
- [ ] `interleave_yuv420`
- [ ] `unpack_i8_to_i16` (and re-export `pack_i16_to_i8` from saturate)
- [ ] General permutation helper (`permutevar8x32` equivalent)
- [ ] Length-mismatch and tail handling
- [ ] Verify Appendix B.4 example
- [ ] Hit target: ≥5× 8×8 transpose
- [ ] Crate-done checklist

### tpt-simd-gather
- [ ] `gather_i32`, `gather_f32` (unsafe)
- [ ] `gather_checked_i32` (panics on OOB)
- [ ] Software fallback for CPUs without gather
- [ ] Benchmark vs scalar (gather is often slow — document honestly)
- [ ] Crate-done checklist

### tpt-simd-scatter
- [ ] `scatter_i32`, `scatter_f32` (unsafe)
- [ ] `scatter_checked_i32`
- [ ] AVX-512 scatter path + AVX2/scalar fallback
- [ ] Define behavior for duplicate indices
- [ ] Crate-done checklist

### tpt-simd-aligned
- [ ] `Aligned32<T>`, `Aligned64<T>` (+ `Aligned16`)
- [ ] `aligned_vec_f32` (fix spec: `Vec<Aligned32<f32>>` wastes space; consider aligned buffer type)
- [ ] `is_aligned`
- [ ] Safe aligned load/store wrappers
- [ ] `alloc`-gated allocation helpers
- [ ] Crate-done checklist

### Phase 3 exit
- [ ] Integration tests with real codec data patterns (sample YUV/PCM fixtures — check fixture licenses)

---

## Phase 4 — Arithmetic variants (Weeks 9–10)

### tpt-simd-mul
- [ ] `mul_hi_i16`, `mul_lo_i32`, `mul_hi_i32`
- [ ] `mul_add_sub_f32` (`fmaddsub`)
- [ ] `complex_mul_f32` (home for complex multiply; complex crate uses it)
- [ ] Widening multiplies (i16→i32, i32→i64) (decide scope)
- [ ] Crate-done checklist

### tpt-simd-rounding
- [ ] `round_f32`, `floor_f32`, `ceil_f32`, `trunc_f32`
- [ ] `round_to_nearest_even_i32`, `round_with_bias_i32`
- [ ] SSE2 fallback (no `roundps` before SSE4.1); NEON `vrndn`
- [ ] Define tie-breaking (half-away vs half-even) per function
- [ ] Crate-done checklist

### tpt-simd-shift
- [ ] `shift_left_i32`, `shift_right_logical_i32`, `shift_right_arithmetic_i32`
- [ ] `rotate_left_i32`, `shift_with_rounding_i32`
- [ ] Out-of-range shift amount policy (≥32)
- [ ] Per-lane variable shift variants
- [ ] Crate-done checklist

### Phase 4 exit
- [ ] `cargo-show-asm` audit recorded for hot functions in all Phase 1–4 crates
- [ ] Fallback to explicit intrinsics where LLVM codegen is poor

---

## Phase 5 — Comparison & control flow (Weeks 11–12)

### tpt-simd-compare
- [ ] `cmp_gt_i32`, `cmp_lt_i32`, `cmp_eq_i32`, `cmp_ne_i32`, `cmp_gt_f32`
- [ ] Fill out symmetry (`ge`, `le`, f32 `lt/eq/ne/ge/le`, i16/i8) (decide scope)
- [ ] Mask ops: `any`, `all`, `count`, `to_bitmask`
- [ ] NaN comparison semantics documented
- [ ] Crate-done checklist

### tpt-simd-blend
- [ ] `blend_i32`, `blend_f32`, `blend_i8`
- [ ] `select_f32` (home crate)
- [ ] Immediate-blend (const mask) variants
- [ ] Crate-done checklist

### tpt-simd-select
- [ ] `select_i32`, `select_f32` (re-export from blend; this crate's index-based variant is distinct — rename to avoid clash)
- [ ] `select_from_slice_i32` (uses gather; bounds policy)
- [ ] Crate-done checklist

### Phase 5 exit
- [ ] Branch-elimination benchmarks showing benefit vs branchy scalar

---

## Phase 6 — DSP primitives (Weeks 13–14)

### tpt-simd-convolve
- [ ] `convolve_1d_f32` (output length / edge-mode policy)
- [ ] `convolve_2d_separable_f32` (fix spec: `&[[f32]]` is not valid Rust — use `&[f32]` + stride/width or slice of rows)
- [ ] `fir_filter_i16` (saturation/rounding policy)
- [ ] Crate-done checklist

### tpt-simd-interpolate
- [ ] `interpolate_linear_f32` (scalar), `interpolate_linear_simd_f32`
- [ ] `interpolate_cubic_f32`
- [ ] `interpolate_lanczos_f32` (kernel size `a`, `sinc` approximation)
- [ ] Crate-done checklist

### tpt-simd-matrix
- [ ] `mat4x4_mul_f32`, `mat8x8_mul_i16`
- [ ] `mat4x4_transpose_f32`, `mat8x8_transpose_i16` (reuse permute)
- [ ] `mat3x3_inverse_f32` (singular-matrix → `None`, epsilon policy)
- [ ] Crate-done checklist

### tpt-simd-window
- [ ] `hamming`, `hanning`, `blackman`, `kaiser` window generators (`alloc`)
- [ ] `apply_window_f32`, `apply_window_simd_f32`
- [ ] Cosine approximation (polynomial/LUT) with documented max error
- [ ] Bessel I0 for Kaiser
- [ ] Optional `no_alloc` variants writing into caller slices
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
- [ ] Scalar fallback verified on every crate
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
