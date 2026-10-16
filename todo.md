# tpt-simd — Project Todo

Owner: TPT Solutions · License: `MIT OR Apache-2.0` · Spec: [spec.txt](spec.txt)

## Project decisions (locked)

- [x] License: dual MIT OR Apache-2.0, copyright TPT Solutions
- [x] Dependencies must be MIT-compatible (permissive; **no Apache-2.0-only deps**)
- [x] Backend: `core::simd` behind `nightly` feature; stable path via own `tpt-simd-vector` crate
- [x] 22 crates = spec's 21 + new `tpt-simd-vector` (stable backend types); the workspace has since grown to 28 `tpt-simd-*` crates (including blas, reduce, math, rng, sparse and the unpublished `testutil`) plus the `tpt-simd` umbrella
- [x] `no_std` + `alloc` (`std` optional), MSRV = stable-2
- [x] Clippy lint `chunks_exact_to_as_chunks` (new in clippy 1.98, breaks `-D warnings`) allowed workspace-wide in `Cargo.toml`; `chunks_exact` + `zip` reads better in the kernels and benches than `as_chunks`
- [x] Targets in v0.1: x86_64 (SSE/AVX/AVX2/AVX-512), ARM64 (NEON/SVE), RISC-V RVV, scalar fallback
- [x] GitHub + GitHub Actions; tpt-kinetix and tpt-cadence are existing separate repos
- [x] Duplicate fns get one home crate, re-exported elsewhere:
  - `pack_i16_to_i8` → `tpt-simd-saturate` (re-exported by permute)
  - `select_f32` → `tpt-simd-blend` (re-exported by select)
  - `ComplexSimd` type → `tpt-simd-core`/`vector`; `complex_mul` lives in `tpt-simd-mul`; `tpt-simd-complex` builds on it (breaks mul↔complex cycle)

## Per-crate "definition of done" (apply to every crate)

Audit status: see [docs/crate-done-audit.md](docs/crate-done-audit.md). fmt, clippy `-D warnings`, `cargo doc`, tests (incl. `scalar-only`), criterion benches for every non-core/vector crate (aligned, mul, select, matrix added; some show slowdowns being fixed, see docs/benchmarks.md), `missing_docs`, `# Safety` and no_std (thumbv7em, wasm32, riscv32, aarch64 bare-metal) pass workspace-wide. asm audit covers every crate on x86_64 (two passes, see docs/asm-audit.md) and aarch64 codegen for the kernels that had no NEON path (see docs/verification.md). Non-host SIMD is verified only by emulation (arm64 and riscv64 containers, correctness only); SVE and RVV intrinsics are unavailable on stable Rust.

- [ ] Scalar reference implementation
- [ ] SIMD implementation(s) for each target in scope
- [ ] Unit tests vs scalar reference (incl. NaN/inf/overflow/underflow/empty/tail lengths)
- [ ] proptest property tests (SIMD == scalar)
- [ ] Doc comment on every public item (perf notes, example, `# Safety` on unsafe fns)
- [ ] criterion benchmark vs scalar (and vs manual intrinsics / `wide` where relevant)
- [ ] `cargo-show-asm` check of hot functions — done for 14 crates on x86_64 (AVX2/SSE2), see docs/asm-audit.md; aligned/mul/select/matrix/blas and non-x86 targets not audited
- [ ] `no_std` build passes; clippy + rustfmt clean; `cargo doc` clean
- [ ] Dual-license headers/metadata (`license = "MIT OR Apache-2.0"`)

---

## Phase 0 — Project setup (before Week 1)

### Repo & licensing
- [x] `git init`, default branch (`master`), `.gitignore`, `.gitattributes`
- [x] Create GitHub repo under TPT Solutions org (`origin` is github.com/tpt-solutions/tpt-simd)
- [x] Add `LICENSE-MIT` and `LICENSE-APACHE` (copyright TPT Solutions)
- [x] README with dual-license statement ("MIT or Apache-2.0, at your option") and contribution clause
- [x] `CONTRIBUTING.md` (contributions are dual-licensed unless stated), `SECURITY.md`
- [x] `cargo-deny` config: allow-list MIT, BSD-2/3, ISC, Zlib, Unicode, `MIT OR Apache-2.0` duals; deny Apache-2.0-only, GPL/LGPL/AGPL
- [x] Audit candidate deps (`wide`, `proptest`, `criterion`, `bytemuck`, etc.) for license compatibility
- [x] Decide dev-dependency policy (criterion/proptest are dev-only; confirm licenses anyway)
- [x] Add NOTICE/THIRD-PARTY-LICENSES generation (`cargo-about`; `about.toml` + `about.hbs`, output `THIRD-PARTY-LICENSES.md`; dev-deps excluded because criterion → ciborium is Apache-2.0-only and is not distributed)

### Workspace
- [x] Root `Cargo.toml` workspace with `[workspace.package]` (version, edition, license, authors, repository, rust-version)
- [x] Scaffold the original 22 crate dirs with `Cargo.toml` + `src/lib.rs` (`#![no_std]`, `extern crate alloc`)
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
- [x] Feature flags: `std`, `alloc` (blas), `nightly` (reserved, no-op), `scalar-only`. No per-ISA Cargo features (`sse2`/`avx2`/`avx512`/`neon`/`sve`/`rvv`): ADR 0001 decision 3 selects ISA paths at compile time through `RUSTFLAGS`, so those names would be no-ops
- [x] Decide runtime dispatch vs compile-time `target_feature` strategy (document in ADR) — compile-time `cfg(target_feature)`; see ADR 0001
- [x] `rust-toolchain.toml` (stable) + separate nightly job config (nightly CI job in ci.yml)
- [ ] Pin and test MSRV (stable-2) with `cargo-msrv` — `rust-version = "1.95"`; `cargo +1.95 check --workspace --all-targets` passes; `cargo-msrv` itself not run
- [x] Shared test-utils / scalar-reference helper crate (unpublished) for proptest strategies

### CI (GitHub Actions)
- [x] fmt + clippy + doc (`-D warnings`)
- [x] Test matrix: stable + nightly (`nightly` feature) × Linux/Windows/macOS — workflow written, not yet run on GitHub
- [x] `no_std` build check (e.g. `thumbv7em`/`wasm32-unknown-unknown`)
- [x] x86_64 runners with target-feature matrix (SSE2, AVX, AVX2) — written, not yet run
- [x] AVX-512 testing: run natively on this host (i5-1135G7 has AVX-512 F/BW/DQ/VL), not via SDE. Full suite with `-C target-cpu=native` passes; covers the `vpscatterdd` scatter path and the AVX-512BW dot kernel (docs/benchmarks.md)
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
- [x] Fix spec API inconsistencies: `Fixed::saturate`, `_mm256_mulhi_epi32` does not exist (use `_mm256_mul_epi32` + shifts), `ComplexSimd` `a * b` operator in Appendix B vs `.mul()` methods Resolved: `spec.txt` now describes the as-built `Fixed` API (ADR 0002).
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
- [ ] Verify codegen: array backend auto-vectorizes; fast paths beat it — `mul_add`/`round`/`floor` are per-lane libm calls without the `std` feature; with `std` they use std intrinsics (5x on portable complex mul; see docs/benchmarks.md); `Simd::mul_add` never lowers to vector FMA even under `target-cpu=native` (libm/libc `fmaf` per lane), so crates need `core::arch` FMA paths (done in mul; matrix uses unfused mul+add); a fix in vector needs `unsafe`, which conflicts with `forbid(unsafe_code)` (ADR note needed)
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
- [ ] Hit target: ≥3× complex multiply speedup vs scalar — 3.3–4.5× only vs a non-vectorised baseline; parity (~1.0–1.1×) with an auto-vectorised zip loop (load/store bound), so not reachable; value is fused-FMA semantics + portable API; see docs/benchmarks.md (Phase 4 performance-target review)
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
- [x] Workspace builds on all CI targets Checked at build level for the host, all no_std targets, aarch64 and riscv64 Linux; emulated test runs in docs/verification.md.
- [ ] Core traits frozen for 0.1 review
- [x] Complex + fixed tested vs scalar refs

---

## Phase 2 — Common patterns (Weeks 4–6)

### tpt-simd-horizontal
- [x] `horizontal_sum_i32`, `_f32`, `_i16` (→ i32)
- [x] `horizontal_max_i16`, `horizontal_min_f32`, `horizontal_product_f32`
- [x] Add missing min/max/sum variants for symmetry (decide scope) (min/max for i16/i32/f32, u8/i8 sums, generic `reduce_*`)
- [x] NaN handling policy documented for min/max
- [ ] Hit target: ≥4–8× vs scalar — NOT reachable: a single-register reduction is ~1 ns and LLVM already emits the optimal shuffle tree; 1.5–2.6× only vs the strict-order scalar f32 loop, parity for int sum/max; see docs/benchmarks.md
- [ ] Crate-done checklist

### tpt-simd-dot
- [x] `dot_product_i16`, `dot_product_f32`, `dot_product_complex_f32`, `dot_product_saturating_i16`
- [x] Handle arbitrary lengths / tails; length-mismatch policy
- [x] Accumulation order/precision documented (f32 reassociation)
- [ ] Hit target: ≥8× vs scalar (256 × i16) — NOT reachable: 1.4–1.6× over the auto-vectorised scalar loop (both use `vpmaddwd`; L1 load-bound, extra accumulators gave no gain); f32 dot 5.7–12.5× vs strict serial loop
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
- [x] Verify Appendix B.4 example `tpt-simd/tests/appendix_b.rs`.
- [x] Hit target: ≥5× 8×8 transpose — met on AVX2 (7.0× over scalar on independent blocks, 4.6× SSE2; 2.9× for a lone serial block); see docs/benchmarks.md
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
- [x] Integration tests with real codec data patterns (sample YUV/PCM fixtures — check fixture licenses) Done with synthetic fixtures generated in code (`tpt-simd/tests/codec_patterns.rs`), so no fixture licences to check.

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
- [x] `cargo-show-asm` audit recorded for hot functions in all Phase 1–4 crates — x86_64 only (complex, fixed, horizontal, dot, butterfly, saturate, permute, gather, scatter, mul, rounding, shift, vector, core); aligned, NEON/wasm/RVV/AVX-512 not inspected; see docs/asm-audit.md
- [ ] Fallback to explicit intrinsics where LLVM codegen is poor — done for `round_f32`/`round_to_nearest_even_i32`/`round_with_bias_i32` (169/64/66 → 11–13 instrs, verified over all 2^32 f32 patterns) and the dot i16 tail; open: no-std per-lane libm for `mul_add`/`sqrt`/`floor` etc. (needs `unsafe` in forbid-unsafe vector crate; ADR note), float `Simd::min/max` always libm per lane, saturating f32→i32 cast scalar, `shift_with_rounding_var_i32` scalar, `Fixed<i16>` mul via 64-bit lanes, `mul_add_sub_f32` not one `vfmaddsub`

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
- [x] Integration tests: FFT and MDCT built from tpt-simd crates, validated vs rustfft / reference `tpt-simd/tests/fft_mdct.rs`: FFT vs rustfft, MDCT vs f64 direct sum.
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
- [x] mdBook (or similar) site, hosted on GitHub Pages — book written in docs/book (not yet committed); GitHub Pages hosting not set up
- [x] Cookbook:
  - [x] "How to implement an FFT using tpt-simd"
  - [x] "How to do motion compensation in a video codec"
  - [x] "How to convert between fixed-point formats"
  - [x] "How to apply a window function to audio data"
- [x] Migration guide:
  - [x] from manual intrinsics
  - [x] from scalar code
  - [x] from other SIMD crates (`wide`, `simba`, `std::simd`)
- [x] All Appendix B examples compile and run (as an integration test, not doctests) — checked by `tpt-simd/tests/appendix_b.rs` (integration test; the book no longer claims doctests).

### Release
- [ ] Full benchmark report vs scalar / manual intrinsics / `wide` / `simba` / `rustfft`
- [x] Performance-regression guard in CI — `cargo xtask bench-compare <baseline> <new>` (20% default, tested; 221 benchmarks, 0 regressed on the full results). `bench.yml` now stores master results in an actions cache and fails PRs more than 25% slower than the newest master baseline. Not yet run on GitHub Actions
- [x] Per-crate READMEs, keywords, categories, `rust-version`, license fields — all 41 publishable crates have README, CHANGELOG and an example
- [x] CHANGELOG, semver policy, release process — per-crate changelogs written; semver policy and release checklist drafted in `docs/release.md` (maintainer review still pending)
- [ ] `cargo publish --dry-run` for all crates; publish in dependency order
- [ ] Tag v0.1.0, GitHub release, announce (r/rust, users.rust-lang.org, This Week in Rust)

---

## Phase 8 — Cross-architecture hardening (v0.1 scope: everything in spec)

- [x] x86_64: SSE / AVX / AVX2 paths verified on real hardware Verified on this host (AVX2+FMA and AVX-512, i5-1135G7). Other x86 hardware not tested.
- [ ] x86_64: AVX-512 specializations for all crates, tested via SDE/hardware — done so far: `tpt-simd-dot` i16 (masked tail) and f32 (bit-identical AVX-512 main loop), `tpt-simd-scatter` (existing path), `tpt-simd-blas` gemm f32/f64 (32x4 / 16x4 tiles), `tpt-simd-compare` fused counts (mask + popcnt; modest, noisy gain), `tpt-simd-permute` `unpack_i8_to_i16` (explicit AVX-512BW, 2.1x native; docs/benchmarks.md), `tpt-simd-convolve` `convolve_1d_f32` (16-lane FMA axpy, 2.3x native over AVX2 build), `fir_filter_i16` (8-output i64 path, 3.3x native over AVX2 build), `tpt-simd-reduce` `sum_f32` and `sum_f64` (AVX-512F block hooks, bit-identical; 2.5-3.4x and 1.8x over the native portable path), `tpt-simd-blas` level-1 `dot`/`nrm2`/`asum` f32 and f64 (AVX-512F block kernels, bit-identical; 1.6-5.4x over AVX2 build); select and blend are single 8-lane ops with no wider AVX-512 form, so not changed; blas `axpy`/`scal` tried with AVX-512F, slower at 4096 and within noise at 1M, so reverted (docs/benchmarks.md); other crates still AVX2-only
- [ ] ARM64: NEON paths for every crate (Appendix A NEON intrinsics list)
  - Status (2026-10-16): NEON paths exist in 19 crates: core (detect), blend, butterfly, mul, permute, rounding, saturate, dot, fixed, reduce, select, shift, convolve, matrix, blas (gemm), compare, search, text, hash (PMULL CRC, runtime detected). Each one runs its existing tests under QEMU user mode and matches the portable or reference result (bit for bit where the crate promises it). Not done: NEON for `gather` and `scatter`, and the Phase 11 crates, which deny `unsafe_code`. Decision (recommended, applied 2026-10-16): gather and scatter have no NEON instruction, so their portable per-lane code is the aarch64 path. Blas level-1 `dot`/`nrm2`/`asum` f32 and f64 now have NEON block kernels (`neon::l1_*`, same lane mapping and unfused multiply-add as the portable loop, so bit-identical by construction; `neon::tests` checks that against a scalar lane model under QEMU). `axpy`/`scal` stay portable, and LAPACK reaches its inner loops through `axpy`/`dot`/`scal`/`gemm`, so the lapack kernels inherit these paths. Not measured on aarch64 (no emulated timing); LLVM may already vectorise the portable loops. Remaining: the Phase 11 policy (see that section).
- [ ] ARM64: SVE implementation (scalable vector length; test several VLs in QEMU)
  - Status: blocked on the toolchain. QEMU user mode can run SVE (`-cpu max`), but SVE intrinsics are not stable in Rust.
- [ ] RISC-V: RVV implementation (check `core::arch::riscv64` stabilization status; fallback if intrinsics unstable)
  - Status: blocked on the toolchain, as for SVE. QEMU user mode runs RVV, and the CI `qemu-riscv` job checks the build.
- [x] Scalar fallback verified on every crate — `cargo check --workspace --exclude tpt-simd-testutil --no-default-features` passes on thumbv7em-none-eabihf, wasm32-unknown-unknown, riscv32imc-none-elf, aarch64-none-softfloat (compile only, not run)
- [x] Test all vector widths (128/256/512-bit) Host matrix: SSE2, AVX2+FMA, AVX-512 (`target-cpu=native`), scalar-only.
- [ ] Runtime dispatch correctness (right path picked, no UB on unsupported CPUs)
- [x] Byte-exact cross-arch results where documented; document float-differences otherwise — `cross_arch_golden` (10 kernels) passes on aarch64 under QEMU with the same digests as x86_64
- [ ] Unsafe audit + Miri pass
  - Status: CI `miri` job covers the unsafe-bearing crates (commit 1eeb510). Re-run locally for `tpt-simd-dot` (`cargo +nightly miri test --features std,scalar-only`): 4 test results, 0 failed. The other crates are checked by the CI job, not by hand here. Confirm the Actions run is green.
- [x] Fuzzing (`cargo-fuzz`) of slice-based APIs — 4 targets (`fuzz/`) each run 60 s under nightly libFuzzer in WSL: dot_i16 33.5M runs, gather_scatter_i32 26.2M, transpose_8x8_i16 28.7M, gemm_f32 1.04M; no crashes. New `fuzz` CI job runs each target for 60 s (not yet run on GitHub Actions)

---

## Phase 9 — Future roadmap (post-v0.1)

### 6–12 months
- [x] AVX-512 tuning pass beyond baseline specializations — done for the two existing kernels: `dot_i16_avx512bw` masked tail + in-register fold (255-element case 2.2x faster, docs/benchmarks.md); scatter AVX-512 path kept as-is: crossover sweep showed no size with a consistent loss (run-to-run noise up to 2.3x, docs/benchmarks.md). Other crates have no AVX-512 path yet
- [ ] SVE tuning (predication, VL-agnostic loops) — blocked: no SVE hardware or QEMU here; SVE intrinsics unavailable on stable
- [ ] RVV tuning — blocked: no RISC-V hardware or QEMU here; RVV intrinsics unavailable on stable
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
- [x] Clone tpt-math and add baseline criterion benches first (gemm 64/256/1024, dot/norm, LU solve, CG on a Poisson matrix, Monte Carlo 10^7 samples); confirm gains justify the work and how much LLVM already autovectorises — done for gemm/dot/norm/LU solve (docs/benchmarks.md); gemm 4–57x, dot 3–4x, norm 5x. LU solve/inverse and stats mean/variance also benched; CG on Poisson (4096/16384 unknowns: 4.3–4.8 / 32–37 ms) and Monte Carlo 10^7 (`integrate` 10.05 ms, `estimate_mean` 30.1 ms) baselines added
- [x] Decide dispatch for library consumers: ADR 0001 is compile-time `cfg(target_feature)`, so users without `-C target-cpu=native` silently get the slow path. Choose runtime dispatch (`is_x86_feature_detected!`, needs `std`) behind a feature, or document required build flags; write ADR 0003 — ADR 0003 accepted; `runtime-dispatch` implemented for blas (umbrella forwards it)
- [x] Decide float-determinism policy for reordered reductions and polynomial math (ADR 0001 requires bit-identical; relax to documented ULP/tolerance for `tpt-simd-math` and SIMD reductions); check tpt-math tests and formal-verification consumers for exact-value dependence — tiers accepted in ADR 0003; tpt-math suites (71) pass with simd on and off, so no exact-value dependence found
- [x] Add new crates to the workspace; keep `no_std`, MIT/Apache-only deps — tpt-simd-blas/math/rng/reduce/sparse are workspace members (`tpt-simd-*` glob) and pass thumbv7em `no_std`; no new non-MIT/Apache deps

### tpt-simd-blas (highest value)
- [x] f32/f64 `axpy`, `scal`, `dot`, `nrm2`, `asum`
- [x] `gemv` (column-major, matches tpt-math storage)
- [x] Packed, register-blocked `gemm` microkernel (e.g. 8x4 f32 / 4x4 f64, FMA) with cache blocking
- [x] Complex variants via split re/im — free functions `_c32`/`_c64` on split planes and interleaved `[[T; 2]]`: axpy, scal, dotu/dotc, nrm2, asum, gemv (N/T/H), gemm (4 real gemm calls; 3M deliberately not used); no `ComplexSimd` type. cgemm 256: 6–20x vs naive (20–42 GFLOP/s); cgemv N 30x, H 1.9x; dotc 1.3x, axpy parity; no complex getrf/potrf/trsm yet
- [x] Portable reference + AVX2/FMA path; NEON gemm microkernels added 2026-10-16 (aarch64, same tiles and per-lane FMA order as AVX2, bit-identical)
- [x] LAPACK-style `getrf`/`getrs`/`trsm`/`potrf`/`potrs` (f32/f64, blocked on gemm) — LU only 1.2–2x vs naive reference (panel factor/row swaps dominate), Cholesky 2–14.5x; see docs/benchmarks.md
- [x] Tests vs scalar reference (tails, NaN/inf, non-multiple sizes), proptest, criterion bench, `cargo-show-asm` — tests and criterion benches exist (incl. lapack); `cargo xtask asm` covers the gemm microkernel, the blocked `getrf` and `potrf` panels, and the f32 dot (2026-10-16)
- [x] Target: gemm >= 3x scalar f32 on AVX2 — 5.5–32x vs naive strided loop (28–43 GFLOP/s), see docs/benchmarks.md
- [ ] Wire into `tpt-math-linalg-dense` (`DMatrix` mul, `DVector::dot`/`norm`, LU/Cholesky/QR inner loops) and `tpt-math-linalg-complex` behind `simd` — **wired and verified** (tpt-math commit `91e0aaf`; dense tests pass with `simd` and `simd-runtime`, 20 unit tests + doctest each): `tpt-math-linalg-dense` features `simd` / `simd-runtime` route f32/f64 `DMatrix * DMatrix`, `DMatrix * DVector` (move-based, no copy, needs `T: 'static`) and f64 `solve`/`inverse` (getrf/getrs) to tpt-simd-blas. NOT wired: `dot`/`norm` (take `&self`; a safe route needs a copy, and tpt-math forbids `unsafe`); no Cholesky/QR in dense; `tpt-math-linalg-complex` (its `Vec<Complex<T>>` needs an O(n²) copy into planes, and blas has no complex getrf/potrf/trsm). **Correction:** the previous entry claimed this wiring was already done, but tpt-math at the time had no tpt-simd dependency and no `simd` feature anywhere (clean tree, no stash/branch). The previous `tpt-math-stats`/`tpt-math-linalg-sparse` "wired" claims (lines below) are not present in the repo either; re-verify before relying on them.

### tpt-simd-math
- [x] Vector `exp`, `ln`, `sin`, `cos`, `tanh`, `erf` (polynomial approximations, documented max ULP error) — f32 only; max ULP 0.8–3.4 (docs in crate)
- [x] Accuracy tests against `libm` over full range, special values (NaN/inf/subnormal)
- [ ] Wire into `tpt-math-stats`, `tpt-math-prob-dist`, `tpt-math-prob-monte-carlo`, `tpt-math-prob-sampler`, autodiff — blocked: `tpt-simd-math` is f32 only and tpt-math is all f64, so wiring needs f64 transforms or an API change (`tpt-math-stats` `simd` feature uses tpt-simd-reduce for mean/variance only, not these math fns)

### tpt-simd-rng
- [x] Lane-parallel xoshiro / Philox generators (independent streams per lane, reproducible seeding)
- [x] Vectorised uniform -> normal (Box-Muller or ziggurat) — Box-Muller, 5-7.5x vs libm scalar
- [x] Statistical quality checks (e.g. PractRand/TestU01-style smoke tests) — chi-square/moment/bit-balance smoke tests (not PractRand)
- [ ] Wire into `tpt-math-prob-sampler` and `tpt-math-prob-monte-carlo` — not done: both are generic over an `Rng` trait, so vectorising needs an API change, not just a feature flag (bulk `fill_u64`/`fill_f64` on `Rng`, batch `Distribution::sample_into`, f64 transforms; lane streams not bit-identical to `SplitMix64`; ceiling only a few x at ~1 ns/sample)

### tpt-simd-sparse
- [x] CSR/CSC SpMV using `tpt-simd-gather` — implemented with multi-accumulator loads (gather measured, not adopted; ~1.2–2x, memory bound)
- [x] Fused vector updates for CG / BiCGSTAB — ~1.5x vs 3 passes (cache-resident)
- [ ] Wire into `tpt-math-linalg-sparse` (**not present in the tpt-math repo; re-verify**) — `simd` feature routes `conjugate_gradient`/`bicgstab` to tpt-simd-sparse (2D Poisson f64: CG 2.1–2.5x, BiCGSTAB 2.0–2.3x; f32/f64 only); standalone `CsrMatrix::matvec` deliberately not routed (0.83–0.9x, per-call view validation, tpt-math forbids `unsafe`); iteration count near `tol` may differ by one

### tpt-simd-reduce
- [x] Pairwise / compensated sum, mean, variance, covariance over slices — done in tpt-simd-reduce
- [x] min/max/argmin/argmax over slices — done (plain fold; LLVM parity)
- [x] Builds on `tpt-simd-horizontal`; wire into `tpt-math-stats` — crate done; `tpt-math-stats` mean/variance wiring (1.2–1.3x, memory-bound) is **not present in the tpt-math repo**; not re-done yet

### Existing stubs that map directly to tpt-math
- [ ] `tpt-simd-convolve` (FIR) and `tpt-simd-window` -> `tpt-math-signal-filter`
- [ ] `tpt-simd-matrix` (3x3 / 4x4 multiply, transpose, inverse) -> `tpt-math-linalg-fixed`, `tpt-math-geometry`, `tpt-math-spatial`
- [ ] Later: IIR biquad cascades (serial dependency; batch across channels)

### Phase 10 exit
- [ ] End-to-end tpt-math benches with and without `simd` show target speedups — dense only (`linalg_dense_bench`, plain build, criterion 1 s warm-up / 3 s measure, noisy): `DMatrix*DMatrix` 64/256/512 5.9x/7.3x/9.9x (549→93 µs, 36.9→5.05 ms, 498→49.6 ms); `DMatrix*DVector` 64 3.7x; `inverse` 32/64/256 2.5x/2.3x/3.2x; `solve` 64/256 3.0x/4.2x. Not yet benched: LU/Cholesky-based paths in sparse, stats, complex (see above). Targets (3–8x) met for gemm and solve; inverse 64 only 2.3x
- [x] tpt-math test suite passes with `simd` on and off — 71 suites, 679 tests off / 684 on, 0 failures (re-run this pass; the earlier "691/692" figure was not reproduced)
- [ ] `no_std` builds pass in both repos — tpt-simd: `tpt-simd-blas/math/rng/reduce/sparse` pass `--no-default-features` on thumbv7em-none-eabihf; tpt-math: `tpt-math-linalg-dense` (default-off and `simd`) passes on thumbv7em and wasm32. Full tpt-math workspace no_std not run

### Repo housekeeping found while scoping
- [x] Workspace `Cargo.toml` / `rust-toolchain.toml` were rewritten to an older shape (10 members, no `keywords`/`libm`/lints/path deps) and no longer parse; keep reconciled manifest with all real crates as members
- [x] `tpt-simd-dot`: `[[bench]] name = "dot"` declared but `benches/dot.rs` missing
- [x] `tpt-simd-core` docs link `docs/adr/0001-backend-strategy.md`; real file is `0001-backend-and-dispatch.md`
- [x] `tpt-simd` umbrella crate is still a 2-line stub with no re-exports
- [x] `tpt-simd-mul` is missing `mul_hi_i16`, `mul_lo_i32`, `mul_hi_i32`, `mul_add_sub_f32`
- [x] Tick Phase 0 / 1 / 2 checkboxes that are already done in code; commit the uncommitted butterfly/complex/fixed/horizontal/saturate/dot work
- [x] Generated `build.*` / `test.*` logs were tracked in git — untracked and ignored
- [x] CI `avx512` job only echoed a note; it now runs the suite (compiled for Skylake-X) on runners whose CPU reports AVX-512F/BW/VL, and warns otherwise
- [x] `chunks_exact_to_as_chunks` allow warned on clippy 1.97 (unknown lint) in every crate — `unknown_lints` now allowed in the workspace lints
- [x] CONTRIBUTING linked a missing `CODE_OF_CONDUCT.md` — now links Contributor Covenant 2.1; **maintainers still need to add an enforcement contact**
- [x] Adoption tooling: `cargo xtask {ci,no-std,msrv,cpu}` (no dependencies), umbrella examples (`quickstart`, `cpu_report`), bug/feature issue templates, Dependabot for cargo and actions
- [x] `cargo-msrv` run — `cargo msrv verify --rust-version 1.95 -- cargo check --workspace --all-targets` reports Rust 1.95.0 compatible (cargo-msrv 0.19.3); `cargo msrv find` (lowest passing version) not run, as it installs older toolchains

## Phase 12 — Review follow-ups and adoption

### Needs a maintainer decision or account action
- [ ] Add an enforcement contact to the Code of Conduct link in CONTRIBUTING.md (currently Contributor Covenant 2.1 with no contact)
- [x] Review and commit the pending housekeeping: staged deletions of `build.*` / `test.*`, `.cargo/`, `xtask/`, `tpt-simd/examples/`, issue templates, `dependabot.yml`, and the `ci.yml` / `deny.toml` changes
- [ ] Confirm the `avx512` CI job actually executes tests on a hosted runner (first run after the change); adjust the CPU-flag check if the runner image lacks AVX-512 BW/VL

### Runtime dispatch and correctness
- [x] Move runtime CPU dispatch into `tpt-simd-core` (shared helper, cached result) so every kernel crate with an intrinsics path can select it, not only `tpt-simd-blas` (ADR 0003 follow-up) — **partly done:** `tpt-simd-core::dispatch` (`Cache`, `x86_avx2_fma()`) added and used by blas; `hash`, `search` and `text` keep calling `is_x86_feature_detected!` directly: they are dependency-free by design, std already caches each query, and their checks differ (PCLMULQDQ/SSE4.1, AVX2 only), so a shared helper would add a dependency for no gain
- [x] Add `cargo-hack --feature-powerset` to CI to catch feature-combination breakage — `feature-powerset` job in ci.yml runs `cargo hack check --workspace --feature-powerset` (514 combinations locally, all clean with `-D warnings`). The run found dead-code warnings in `tpt-simd-search` (memchr builds) and `tpt-simd-text` (simdutf8 builds); gated those helpers and re-ran the affected crates' tests
- [x] Add a CI check that `cargo xtask no-std` target list matches the `no-std` job in ci.yml (avoid the two drifting apart) — unit test `no_std_targets_match_ci_workflow` in `xtask/src/main.rs`, runs under `cargo test --workspace`
- [x] Remove the `scalar-only` and `nightly` no-op features or document them as reserved in one place — kept both (removing is a breaking change); only `nightly` is a no-op (reserved). `scalar-only` is not a no-op: it forces the portable path. Documented in the book's feature table (`docs/book/src/api.md`)

### Release readiness
- [ ] Run `cargo-semver-checks` against the crate set before the 0.1.0 publish
  - Decision: defer to the first release after 0.1.0. There is no published baseline yet, so the check has nothing to compare against.
- [x] Write the semver policy and release checklist — drafted in `docs/release.md` for maintainer review (the per-crate CHANGELOGs exist)
- [x] Run `cargo-msrv` and record the result in docs (see the housekeeping item above)

### Codegen and performance regression
- [x] Add `cargo-show-asm` snapshots for the hot kernels
  - Done as a floor check, not a text snapshot: `cargo xtask asm` (CI `asm` job) counts vector and FMA instructions in four emitted kernels under `-C target-cpu=x86-64-v3`: f32 gemm microkernel (282 vector / 36 FMA measured), `convolve_1d_f32` (98 / 20), `fir_filter_i16` (5 / 0; the explicit path is AVX-512 only), `dot_i16` (83 / 0). Floors are about half the measured counts. Added 2026-10-16: blocked `getrf` and `potrf` (`fsingle::*_with_workspace`, 100 and 102 vector instructions, their trailing updates call gemm) and the blas f32 dot (`single::dot`, 133 vector). Not covered: `reduce`. Its kernels are `#[inline]` and emit no symbol. Covering them needs `#[inline(never)]` on hot public paths, which costs speed for small slices. Decision (recommended, applied 2026-10-16): leave them uncovered, and rely on the runtime `bench-compare` guard for them. (dot, gemm, convolve, reduce) so a compiler upgrade that drops the SIMD path fails review
- [x] Publish criterion baseline comparisons from `bench.yml` on `master` pushes — master pushes save the baseline to an actions cache (recommended option: cache, not a committed file, since a committed number depends on one runner class). History under `docs/` is still manual: `docs/benchmarks.md` records the Phase 11 baseline

### Adoption
- [x] Add a scalar-to-tpt-simd porting table to the mdBook migration section (`docs/book/src/migration/scalar-code.md`)
- [x] Add a `cargo generate` (or copyable) starter template for a codec or DSP crate that depends on tpt-simd with the recommended feature set — `templates/dsp-starter/` (tested against the local crate; `tpt-simd = "0.1"` needs 0.1.0 on crates.io)
- [x] Add a "which crate do I need?" decision table to the README, mapping common tasks (FIR, FFT, dot, matmul, RNG) to crates — also added the Phase 11 crates to the crate table and corrected the crate count (40 published + testutil)
- [x] Show `cargo xtask cpu` output in the getting-started book page so users can see which path their machine takes (output checked against a `-C target-cpu=native` rebuild)

---

## Phase 11 — New crate candidates

Decision (recommended, applied 2026-10-15): the Phase 11 crates ship in 0.1 on autovectorised code. Explicit SIMD paths and asm audits move to post-0.1 work. Reference property tests and criterion benches were added for all 13 crates (see Definition-of-done gaps). asm audits remain open. Decision (recommended, applied 2026-10-16): explicit SIMD paths are not planned for the Phase 11 crates. Every one of them sets `#![deny(unsafe_code)]`, and intrinsics need `unsafe`, so they would need a policy change first. The slower rows in `docs/benchmarks.md` stay documented as they are. Revisit if that policy changes.

Status as of 2026-10-12. All 13 proposed crates exist, each with unit tests and a doctest, and each passes clippy with `-D warnings` and the `thumbv7em-none-eabihf` no_std check. **No crate has an explicit SIMD path yet**; all rely on autovectorisation. Policy is in `docs/adr/0004-no-runtime-wrapping.md`: own implementation by default, no runtime dependencies. Incumbents are dev-dependency oracles only, or sit behind a default-off feature. The one exception is `tpt-simd-half`, which keeps `half` for f16/bf16 interop.

### Licence check (crates.io API, 2026-10-12, against the `deny.toml` allow-list)

- Allowed to use as oracles or features: all MIT, BSD-2/3, ISC, Zlib, and MIT-OR-Apache duals checked (e.g. `crc32fast`, `simd-adler32`, `simdutf8`, `faster-hex`, `memchr`, `rubato`, `biquad`, `fast_image_resize`, `yuv`, `libblur`, `half`).
- **Do not use:** `stream-vbyte` (Copyfree Open Innovation License), `concrete-ntt` (BSD-3-Clause-Clear), `xxhash-rust` (BSL-1.0). Not on the allow-list; the maintainer has ruled them out.
- **Excluded, Apache-2.0 only:** `nalgebra`, `nalgebra-sparse`, `simba`, `parry3d`, `simsimd`.
- `cargo deny check licenses` passes for the workspace.

### Crates

#### tpt-simd-distance
- [x] Inner product, squared L2, cosine (f32), exact i8/u8 variants, Hamming on `u64` bit-vectors; top-k (`top_k_l2_f32`)
- [ ] Explicit SIMD paths (AVX2, NEON)
- [ ] proptest vs scalar reference
- [ ] Criterion bench vs naive loop, and asm audit
- [ ] Build on `tpt-simd-dot` / `tpt-simd-reduce` (optional)

#### tpt-simd-bits
- [x] Popcount, bitset and/or/xor/andnot, zigzag, delta encode/decode, prefix sums (u32, u64)
- [x] `RankSelect`: O(1) `rank1`, `select1` by binary search, checked against a naive scan
- [ ] Explicit SIMD paths for popcount and prefix sums
- [ ] proptest, criterion bench, asm audit
- [x] Bitpacking / varint codecs — `tpt-simd-bits`: LEB128 varints for u32 (`varint_encode_u32`, `varint_decode_u32`) and fixed-width LSB-first bitpacking for 1..=32 bits (`bitpack_u32`, `bitunpack_u32`). Scalar, no_std, no allocation; known-answer and round-trip tests. The `bitpacking` and `roaring` crates were not needed under ADR 0004

#### tpt-simd-geom
- [x] SoA `distance_sq`, `distance`, `dot`, AABB overlap, ray–AABB slab test
- [x] SoA quaternion `quat_mul` and `quat_rotate`; `mat4_transform_points` (column-major, homogeneous divide)
- [ ] Explicit SIMD paths (4-wide SoA lanes)
- [ ] proptest, criterion bench, asm audit
- [ ] Wire into `tpt-math-geometry` / `tpt-math-spatial` (external repo)

#### tpt-simd-dct
- [x] Orthonormal DCT-II/III, MDCT/IMDCT with sine window; TDAC reconstruction tested
- [x] FFT-based path for large sizes — `tpt-simd-dct::FastDct` (power-of-two N, 2N-point FFT, O(N) memory), checked against the table DCT for every size up to 512 in both directions; 5x faster at N = 256 and 75x at N = 4096 (docs/benchmarks.md)
- [x] Wavelets — orthonormal Haar, multi-level, in place (`haar_forward` / `haar_inverse`); energy preserved and exact reconstruction tested; one-level values checked by hand
- [ ] Explicit SIMD paths, proptest, criterion bench, asm audit

#### tpt-simd-ntt
- [x] Montgomery and Barrett reduction, in-place forward/inverse NTT, cyclic convolution
- [ ] Explicit SIMD paths (vectorised butterflies)
- [ ] Bigint multiply built on the NTT
- [ ] proptest, criterion bench, asm audit
- [x] Optional: a primality check for the modulus — `tpt-simd-ntt::is_prime` (trial division, `u32`); checked against a sieve up to 10 000 and by property test. The contexts still accept any valid modulus and do not call it

#### tpt-simd-transpose
- [x] Blocked out-of-place transpose, in-place square transpose, strided copy (generic over `Copy`)
- [ ] Explicit SIMD tiles (the 4×4 and 8×8 register tiles are in `tpt-simd-permute`)
- [ ] proptest, criterion bench, asm audit

#### tpt-simd-sort
- [x] Introsort with 3-way partition and heapsort fallback; `select_nth`, `top_k_smallest`, `argsort`, `partition`
- [x] Sorting networks / bitonic merge for small fixed blocks — decision: not built for 0.1 (`tpt-simd-sort` README says so), so nothing to ship or drop
- [ ] Benchmark result: the first quick run had this roughly 5× slower than `std::sort_unstable` at 1M elements. Tune or drop it based on a validated full run; do not keep it unless it wins
- [ ] proptest, asm audit

#### tpt-simd-audio
- [x] Mix, interleave/deinterleave, peak/RMS, FIR, RBJ biquad (low-pass, high-pass), windowed-sinc resampler
- [x] Oracles: add `biquad` and `rubato` as dev-dependencies and compare outputs — `tests/oracle_biquad.rs` (lowpass and highpass match `biquad` 0.5 within 1e-3) and `tests/oracle_resample.rs` (a 1 kHz tone resampled 48k to 24k keeps amplitude within 2% of 1.0 and agrees with `rubato`). Note: `biquad` 0.5.0 maps the cutoff to a quarter of the RBJ frequency, so the oracle is given 4 x f0 (see the test module docs).
- [x] Channel-batched biquad over interleaved frames — `tpt-simd-audio::process_interleaved` (one filter per channel, no allocation); equals per-channel processing bit for bit, including split calls
- [ ] Explicit SIMD FIR / biquad batches (uses the `convolve` / `window` / `interpolate` crates where they fit)
- [ ] proptest, criterion bench, asm audit

#### tpt-simd-pixel
- [x] RGB/RGBA conversion, planar/interleaved layout, u8↔unit f32, premultiply, RGB↔HSV, histogram
- [ ] Optional oracles: `yuv` and `libblur` for colour/blur cross-checks (resizing and YUV are out of scope for this crate)
- [ ] Explicit SIMD paths, proptest, criterion bench, asm audit

#### tpt-simd-half
- [x] Wraps `half` 2.7.1 for f16/bf16; length-checked slice conversions; int8 and ggml-layout Q8_0 / Q4_0 block quantisation
- [x] f16/bf16 arithmetic kernels — `tpt-simd-half`: `add_f16`, `mul_f16`, `mul_bf16`, `add_bf16`. f16 results are correctly rounded (computed in f64, exact for two f16 values). bf16 mul is correctly rounded; bf16 add is within 1 ulp (the f32 intermediate can double-round). Scalar; no SIMD path
- [x] Decision (recommended, applied): keep `half` as a required dependency for 0.1. ADR 0004 allows it for f16/bf16 interop, and making it optional would change the API. `half` pulls in `zerocopy` through its dependency tree
- [ ] Explicit SIMD conversion paths, proptest, criterion bench

#### tpt-simd-hash
- [x] Own CRC-32 (slicing-by-8, compile-time tables) and Adler-32 (deferred modulo); FNV-1a 64; rolling hash
- [x] Differential tests against `crc32fast` and `simd-adler32` (dev-dependencies)
- [ ] PCLMULQDQ CRC path (x86_64) and ARM CRC path
- [ ] proptest, criterion bench (see the Phase 11 benchmark status below), asm audit

#### tpt-simd-text
- [x] Own hex encode/decode (validate-then-write); UTF-8 validation via `core`, with `simdutf8` behind a default-off feature; UTF-8↔UTF-16 transcoding
- [x] Differential tests against `faster-hex` and `simdutf8`
- [ ] Explicit SIMD hex and UTF-8 paths
- [ ] proptest, criterion bench, asm audit

#### tpt-simd-search
- [x] Own byte search (SWAR), substring search, ASCII case-insensitive search, `count_lines`; `memchr` behind a default-off feature
- [x] Differential tests against `memchr`
- [x] Substring search: done. `tpt-simd-search::find` is its own Two-Way algorithm, linear in the haystack; `memchr::memmem` is available behind the `memchr` feature
- [ ] Explicit SIMD paths, proptest, criterion bench, asm audit

### Benchmarks (all 13 crates)

- [x] Bench suites exist in `bench-vs/benches/`: `hash_vs`, `text_vs`, `search_vs`, `sort_vs` (head-to-head against incumbents) and `kernels` (absolute timings and scalar baselines for the rest). All compile.
- [x] A first run in `--quick` mode completed. Its numbers are not valid yet: (1) quick mode has too few samples; (2) some output lines were interleaved; (3) the incumbent throughput is implausible (`crc32fast` reports about 65 GiB/s on one core), so the harness must be fixed before results are trusted.
- [x] Fix the harness so inputs cannot be reused across iterations, then rerun in full mode, one suite at a time — full-mode reruns done 2026-10-12 (Round 5, two passes per suite, all passed); no input-reuse bug found (docs/incumbent-benchmarks.md, Round 4 and Round 5)
- [x] Record validated results in `docs/benchmarks.md`, including the crates where we are slower than the incumbent — Phase 11 criterion baseline recorded (2026-10-15, one host, noisy; slower rows listed with reasons). Incumbent head-to-head results are in `docs/incumbent-benchmarks.md` — the head-to-head results are in `docs/incumbent-benchmarks.md` (Rounds 5–6d) and the accepted gaps are in ADR 0005; `docs/benchmarks.md` itself is not yet updated

### Definition-of-done gaps (apply to every crate above)

- [x] proptest property tests (SIMD == scalar) for each crate — `tests/proptest_reference.rs` in all 13 Phase 11 crates (256 cases per property, clippy `-D warnings` clean). Each checks the kernel against a plain reference (a naive loop, `std`, the `half` crate, or a bitwise CRC). Explicit SIMD paths are still not built, so the comparison is against scalar references, not SIMD-vs-scalar.
- [ ] `cargo-show-asm` audit of hot functions
- [ ] Explicit SIMD paths for x86_64 and aarch64 (none built yet)

---

## Phase 12 — Overlap review of existing crates

Rule: existing tpt-simd crates are complete and stay as-is **unless** a MIT-compatible incumbent is both better (API/features/correctness) and faster in a criterion bench on our target set. Only then may the crate body be replaced, and not with an unconditional runtime wrapper: ADR 0004 requires an own implementation, and the incumbent is kept as a dev-dependency oracle or behind a default-off feature. Otherwise keep our implementation. Licences below from the crates.io API (2026-10-12); overlap judgements are provisional (APIs not diffed). Excluded as Apache-2.0-only: nalgebra, nalgebra-sparse, simba. Before wrapping, run `cargo deny` including transitive deps and confirm no_std/stable-Rust support.

### Benchmark candidates (full or near-full overlap; wrap only if the incumbent wins)
- [x] tpt-simd-matrix vs `glam` / `ultraviolet` (3x3/4x4 mul, transpose, inverse) — **keep, mixed result**. Two runs: mat4 mul flips (run 1 tpt 7.2 vs glam 8.2 ns; run 2 glam 6.4 vs tpt 7.4 ns); glam faster on mat4 transpose both runs (2.0 vs 3.6 ns); tpt faster on mat3 inverse in run 2 (16.9 vs 22.4 ns, ultraviolet 17.0), tied in run 1. ultraviolet is slower on mul/transpose. No incumbent is consistently better and faster. Bench: `bench-vs/benches/matrix_vs_glam.rs`. Round 5 decision: keep the array-typed API; the register-typed API is a separate decision (docs/adr/0005-accepted-performance-gaps.md).
- [x] tpt-simd-blas gemm f32/f64 vs `gemm`, `faer`, `matrixmultiply` — **keep; decided (docs/adr/0005-accepted-performance-gaps.md)**. Round 5 (new machine): at 512 f32 and f64 are at parity with the fastest incumbent; at 64 behind by 1.4–1.5×, which is accepted as a small-size gap. Earlier notes below are from the previous machine. Latest run (same idle machine, workspace variant, Gelem/s): f32 64: tpt 84, gemm 126, faer 127, matrixmultiply 97; 256: tpt 124, gemm 134, faer 138, matrixmultiply 107; 512: tpt 130, gemm 136, faer 131, matrixmultiply 114. f64 64: tpt 47, gemm 62, faer 62, matrixmultiply 50; 512: tpt 56, gemm 67, faer 64, matrixmultiply 53 (f64 numbers move a lot between runs, compare within a run only). Changes so far: 16x6 f32 and 8x6 f64 AVX2 tiles (12 accumulators); a full-tile store kernel that accumulates into C, which made ~2–5% difference at 64 and nothing measurable elsewhere (candidate to revert). Measured (timing-only experiment, reverted): skipping B packing at f32 n=64 saves ~0.7 µs of 5.9 µs (~12%), so B packing is part of the small-size gap but not all of it. Pack-A cost not measured. Deferred: optimisation will be revisited when all crates are benched together, before any wrap decision. Bench: `bench-vs/benches/gemm_vs.rs`, `gemm_f64_vs.rs`.
- [x] tpt-simd-fixed scalar types vs `fixed` / `az` (keep SIMD fixed-point vectors regardless) — Round 7: 8-lane `saturating_mul` is ~6× the scalar `fixed` ops; keep the SIMD vector. `az` not benched (cast traits, not fixed-point).
- [x] tpt-simd-window formulas vs `dasp_window` / `apodize` (both scalar, stale; keep SIMD generate/apply unless slower) — Round 7: ~6–12× faster than apodize; `dasp_window` not benched (its API is per-sample, not a buffer fill).
- [~] tpt-simd-aligned vs `aligned-vec` / `aligned` — Round 7: `aligned-vec` only. Ahead at 256 KB, parity at 64 B, 1.4–1.7× behind at 4 KB (open). `aligned` not benched.
- [~] tpt-simd-blend, select, shift vs `wide` / `pulp` / `macerator` vector ops (consider merging blend+select) — Round 7: blend ~2.5× faster than `wide`, shift at parity; `select` (lane gather) not benched; `pulp`/`macerator` not benched.
- [x] tpt-simd-vector vs `wide` fixed-width types — decision: keep the generic `Simd<T,N>`/`SimdMask`. It is the portable reference backend the other crates build on, and the head-to-head shows it ahead on blend (~2.5x) and at parity on shift (docs/incumbent-benchmarks.md)
- [x] tpt-simd-horizontal vs `wide` / `pulp` reductions — decision: keep as a separate crate for 0.1. `tpt-simd-reduce` already depends on it, and folding it in would not reduce the public surface.
- [x] tpt-simd-complex scalar type vs `num-complex` — decision: keep the SIMD batch types. A `num-complex` conversion layer is optional interop and is deferred past 0.1.

### Keep (no equivalent found; no action beyond optional interop/oracles)
- [ ] permute, scatter, gather, butterfly, compare (slice kernels), dot, mul, saturate, rounding, convolve, interpolate, math, rng, sparse, reduce, core
- [x] Use `rand_xoshiro` / `rand_philox` as rng test vectors and implement `rand_core` traits — xoshiro256++ (scalar, jump, and the eight-lane generator) and SplitMix64 match `rand_xoshiro` word for word (`tpt-simd-rng/tests/reference_vectors.rs`); Philox4x32-10 matches the Random123 known-answer vectors (`tpt-simd-rng/tests/philox_kat.rs`); `RngCore` and `SeedableRng` are implemented for `Xoshiro256pp` behind the default-off `rand_core` feature, tested against `rand_xoshiro` (`tests/rand_core_traits.rs`); `libm` / `sleef` as math accuracy oracles; `statrs` as distribution oracle

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
