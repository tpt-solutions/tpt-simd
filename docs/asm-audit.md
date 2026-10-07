# Assembly audit (cargo-show-asm)

Phase 4 exit item: "cargo-show-asm audit recorded for hot functions in all
Phase 1-4 crates". This file records what was inspected, how, what was found
and what was changed.

Date: 2026-10-07. Toolchain: rustc 1.97.1 (LLVM 22.1.6), `cargo-show-asm`
0.2.63, host `x86_64-pc-windows-msvc`, CPU with AVX2, FMA, AVX-VNNI (no
AVX-512).

## Method

* Crates are generic / `#[inline]`, so each hot public function was wrapped in
  a concrete, `#[inline(never)]`, non-generic wrapper in a **scratch crate
  outside the repository** (about 110 wrappers). The scratch crate depends on
  the workspace crates by path and is not committed.
* Asm was emitted (`cargo rustc --release -- --emit asm`, which is what
  `cargo asm` drives) and read with a script that counts mnemonics, calls
  and loop back-edges per function; the interesting ones were read by hand.
  `cargo asm --release --lib <wrapper>` was also used directly (e.g.
  `complex_mul_f32`: `vmulps`, `vfmsub231ps`, `vmulps`, `vfmadd231ps`, 13
  instructions).
* Build configurations:
  1. `-C target-cpu=native` (AVX2 + FMA + AVX-VNNI): main column below;
  2. `-C target-feature=+avx2,+fma`: same results as (1) apart from
     AVX-VNNI use in `dot_product_i16`;
  3. default x86_64 baseline (SSE2): spot checks only;
  4. each of the above with the `std` feature on all crates. Several
     lane operations switch implementation on `std` (see "libm" below), so
     both are reported.
* Profile: release with `lto = "off"`, `codegen-units = 1`.

**Gotcha for whoever repeats this.** With `lto = "thin"` (the workspace
`[profile.bench]` setting) `--emit asm` produced the pre-optimisation
module: every lane loop, even a plain `[f32; 8]` add, came out as 8 scalar
`vaddss`. That is an artefact of the emit, not of the real build, and cost a
full first pass before it was spotted. Audit with `lto` off (plain
`cargo asm --release`).

## Results (x86_64, AVX2 + FMA, `no_std` default unless noted)

Legend: OK = expected vector instructions, no leaks. By design = scalar or
per-lane work is inherent. POOR = see findings. "ins" is the instruction
count of the whole wrapper (a rough size indicator).

### tpt-simd-complex / tpt-simd-mul (complex)

| Function | Key instructions | Verdict |
|---|---|---|
| `complex_mul_f32` | `vmulps` x2, `vfmsub231ps`, `vfmadd231ps` (13 ins) | OK |
| `ComplexSimdF32Ext::twiddle_mul_fma` | same 4-op FMA sequence (13 ins) | OK |
| `mul_conj` | `vfmadd231ps`, `vfnmadd231ps`, `vmulps` x2 (13 ins) | OK |
| `fft_butterfly` | `vfmadd231ps`, `vfmsub231ps`, `vaddps`/`vsubps` (20 ins) | OK |
| `a * b` (operator, unfused) | `vmulps` x4, `vaddps`, `vsubps` (15 ins) | OK |
| `mag_sq`, `scale_scalar`, `i_mul` | `vmulps`/`vaddps`, `vbroadcastss`, `vxorps` (8-9 ins) | OK |
| `norm_sq` | no_std: 8 calls to `libm::fmaf`; with `std`: `vmulps` + `vfmadd231ps` (8 ins) | POOR without `std` |
| `fmadd`, `fmsub` | no_std: 8 calls to `libm::fmaf` (318/332 ins); with `std`: 3 FMA instructions (14 ins) | POOR without `std` |
| `from_polar` | per-lane `libm::sincosf` call | By design (transcendental) |
| `from_interleaved` / `to_interleaved` | `vpermpd`/`vshufps`, `vpunpck*dq`; one `panic_fmt` (length assert) outside the data path | OK |
| `interleaved_to_split` (slice) | `vpermpd` x4, `vshufps` x6, 3 loops (main, 4-wide, scalar tail) | OK |
| `split_to_interleaved` (slice) | `vperm2f128`, `vunpck*`, `vpermilps`, 3 loops | OK |
| `complex_mul_f32_portable` (reference) | no_std: libm `fmaf` x8; `std`: same FMA sequence as `complex_mul_f32` | By design (reference path) |

### tpt-simd-fixed

(`Fixed<i32,16,16,8>`, plus `Fixed<i16,8,8,16>` where noted.)

| Function | Key instructions | Verdict |
|---|---|---|
| `wrapping_add`, `saturating_add` | `vpaddd`; saturating: compare/`vblendvps` (13 ins) | OK |
| `i16` `saturating_add` | `vpaddsw` (6 ins) | OK |
| `wrapping_mul`, `saturating_mul`, `mul_trunc` (i32) | `vpmuldq` x2 + 64-bit shifts/blends (22 / 42 / 13 ins) | OK (intrinsics path) |
| `i16` `wrapping_mul` / `saturating_mul` / `mul_trunc` | vectorised but through 64-bit lanes: `vpmovsxwq` x8, `vpmuldq` x4 (49 / 74 / 15 ins) | Acceptable; improvable (i32 intermediate) |
| `to_f32` | `vcvtdq2ps`, `vmulps` (7 ins) | OK |
| `from_f32` | no_std: 8 `libm::rint` calls (162 ins); `std`: `vroundps` + 8 scalar `vcvttss2si` with NaN/overflow fix-ups (75 ins) | POOR |
| `clamp`, `saturating_abs` | `vpmaxsd`/`vpminsd`; `vpabsd`-style (7 / 12 ins) | OK |
| `convert` | `vpmovsxdq`, `vpsllq`, `vpcmpgtq`, `vblendvpd` (20 ins) | OK |
| `div`, `saturating_div` | 8 scalar `idiv` per call; no SIMD integer divide exists | By design |

### tpt-simd-horizontal

| Function | Key instructions | Verdict |
|---|---|---|
| `horizontal_sum_i32` | `vpaddd` x3, `vpshufd` x2 (8 ins) | OK |
| `horizontal_sum_f32`, `_product_f32` | `vaddps`/`vmulps` tree + final scalar op (10 ins), fixed order | OK |
| `horizontal_sum_i16` | `vpmovsxwd`, `vpaddd`, `vextracti128` (12 ins) | OK |
| `horizontal_sum_u8` | `vpsadbw`, `vpaddq` (9 ins) | OK (best case) |
| `horizontal_sum_i8` | `vpmovsxbw`, `vpaddw` (15 ins) | OK |
| `horizontal_max/min_i16` | `vphminposuw` (7 ins) | OK |
| `horizontal_max/min_i32` | `vpmaxsd`/`vpminsd` x3 (8 ins) | OK |
| `horizontal_max/min_f32` | `vcmpunordps`, `vblendvps`, `vmaxss`/`vminss` (26 ins) | OK (NaN-ignoring semantics need the blends) |

### tpt-simd-dot (slice kernels; the wrapper tail-jumps into the crate)

| Function | Key instructions | Verdict |
|---|---|---|
| `dot_product_i16` | main loop `vpmaddwd` + `vpaddd` (with `-C target-cpu=native` LLVM fuses to `vpdpwssd`), 4x unrolled 32 elems/iter, 128-bit and scalar tails | OK; tail bounds check fixed (below) |
| `dot_product_f32` | portable, 8 lanes: `vmulps` + `vaddps`, deliberately unfused for bit-identical results; one `memcpy` call in the setup; many `vmovaps` are Windows callee-saved xmm6-15 save/restore, not spills in the loop | OK |
| `dot_product_saturating_i16` | `vpmovsxwd`, `vpmulld`, `vpcmpgtd`/`vblendvps` saturating add; `memset`/`memcpy` calls outside the loop | OK (no native saturating 32-bit add) |
| `dot_product_complex_f32` | `vmulps`/`vaddps`/`vsubps`, unfused | OK |

### tpt-simd-butterfly

| Function | Key instructions | Verdict |
|---|---|---|
| `butterfly_f32` | `vaddps`, `vsubps` (8 ins) | OK |
| `butterfly_i16` / `_saturating` | `vpaddw`+`vpsubw` / `vpaddsw`+`vpsubsw` (8 ins) | OK |
| `butterfly_complex_f32` | `vaddps` x2, `vsubps` x2 (14 ins) | OK |
| `butterfly_with_twiddle_f32` / `_complex_f32` | `vmulps`, `vaddps`/`vsubps` (9 / 26 ins); unfused | OK |

### tpt-simd-saturate

| Function | Key instructions | Verdict |
|---|---|---|
| `saturating_add/sub_i16`, `_i8`, `_u8` | `vpaddsw`, `vpsubsw`, `vpaddsb`, `vpaddusb` (6 ins) | OK |
| `saturating_mul_i16` | `vpmovsxwd` x4, `vpmulld` x2, `vpackssdw` (12 ins). Vectorised, but the doc comment claims `pmullw`+`pmulhw`+unpack+pack, which is not what is generated | OK; doc wording stale |
| `pack_i16_to_i8`, `_to_u8`, `pack_i32_to_i16` | `vpacksswb`, `vpackuswb`, `vpackssdw` (5 ins each) | OK |
| `pack_*_slice` | `vpacksswb`/`vpackuswb`/`vpackssdw` main loops + masked/scalar tail | OK |

### tpt-simd-permute

| Function | Key instructions | Verdict |
|---|---|---|
| `transpose_8x8_i16` | `vpunpck{l,h}wd`, `vpunpck{l,h}dq`, `vpermq`, `vinserti128` (26 ins) | OK |
| `transpose_4x4_f32` | `vpunpck{l,h}dq`, `vpunpck{l,h}qdq` (17 ins) | OK |
| `*_portable` transposes | scalar-ish shuffles (113 / 20 ins) | Reference, expected |
| `permute_f32`, `permute_i32` | `vpermps` (6 ins) | OK |
| `permute_f32_portable` | `vgatherdps` x2 (LLVM's choice) | Reference |
| `permute` (generic, `u16` x 8) | ~42 scalar indexed loads/stores, no bounds checks | By design (no hardware path for generic T) |
| `interleave_stereo_i16` / `deinterleave_stereo_i16` | `vpunpck{l,h}wd`, `vperm2i128` / `vpshufb`, `vpermq`, 3 loops | OK |
| `interleave_yuv420` | `vpunpck{l,h}bw`, `vperm2i128`; `memcpy` for the Y plane | OK |
| `unpack_i8_to_i16` | `vpmovsxbw`, 3 loops | OK |

### tpt-simd-gather / tpt-simd-scatter

| Function | Key instructions | Verdict |
|---|---|---|
| `try_gather_i32/f32`, `gather_checked_*` | range check by vector compare + `vmovmskps` (no per-lane bounds checks), then one `vpgatherdd` / `vgatherdps`; `panic_fmt` outlined | OK |
| `try_scatter_*`, `scatter_checked_*` | validation loop, then 8 scalar stores (AVX2 has no scatter; no AVX-512 on the host) | By design; AVX-512 `vpscatterdd` path unverified |

### tpt-simd-mul

| Function | Key instructions | Verdict |
|---|---|---|
| `mul_hi_i16` | `vpmulhw` (6 ins) | OK |
| `mul_lo_i32` | `vpmulld` (6 ins) | OK |
| `mul_hi_i32` | `vpmuldq` x2, `vpshufd`, `vpblendd` (12 ins) | OK |
| `mul_widen_i16` / `_i32` | `vpmovsxwd`+`vpmulld` / `vpmovzxdq`+`vpmuldq` (7 ins) | OK |
| `mul_add_sub_f32` | no_std: 8 `libm::fmaf` calls (112 ins); `std`: `vfmadd231ps` + `vxorps` + `vblendps` (10 ins), not a single `vfmaddsub` | POOR without `std` |

### tpt-simd-rounding (after the fix below)

| Function | Key instructions | Verdict |
|---|---|---|
| `floor_f32`, `ceil_f32`, `trunc_f32`, `round_ties_even_f32` | one `vroundps` (5 ins) | OK |
| `round_f32` (half away) | `vandps`, `vorps`, `vaddps`, `vroundps` (11 ins); was 169 ins of per-lane bit tricks | OK (fixed) |
| `round_to_nearest_even_i32` | `vroundps`, `vcvttps2dq`, `vcmpgeps`, `vxorps`, `vcmpordps`, `vandps` (11 ins); was 64 ins with 8 scalar `vcvttss2si` | OK (fixed) |
| `round_with_bias_i32` | same plus `vaddps` (13 ins); was 66 ins | OK (fixed) |

### tpt-simd-shift

| Function | Key instructions | Verdict |
|---|---|---|
| `shift_left_i32`, `shift_right_logical_i32`, `shift_right_arithmetic_i32` | `vpslld`/`vpsrld`/`vpsrad` with an xmm count, plus range handling (10 ins) | OK |
| `rotate_left_i32` | `vpslld`/`vpsrld` + shuffles (12 ins) | OK |
| `shift_left_var_i32`, `shift_right_logical_var_i32`, `shift_right_arithmetic_var_i32` | `vpsllvd`, `vpsrlvd`, `vpsravd` (6 ins) | OK |
| `rotate_left_var_i32` | `vpsllvd`, `vpsrlvd`, `vpor` (12 ins) | OK |
| `shift_with_rounding_i32` | 64-bit lane arithmetic, `vpsrlq` (34 ins) | OK |
| `shift_with_rounding_var_i32` | `Simd::from_fn` per-lane scalar loop, 7 back-edges, 171 ins | POOR (no AVX2 kernel) |

### tpt-simd-vector / tpt-simd-core (generic lane ops)

| Function | Key instructions | Verdict |
|---|---|---|
| `Simd<f32,8>` `+`, `*`, comparisons | `vaddps`, `vmulps`, `vcmpltps` (6 ins) | OK |
| `abs` (i32) | `vpabsd` | OK |
| `sat_add` (i32) | compare/blend sequence (76 ins) | OK (no native 32-bit saturating add) |
| `from_slice`, `copy_to_slice` | one `vmovups`; length assert outside the data path | OK |
| `mul_add`, `sqrt`, `floor`, `ceil`, `round`, `round_ties_even`, `trunc` | no_std: 8 per-lane `libm` calls (~60-110 ins). With `std`: `vfmadd213ps`, `vsqrtps`, `vroundps` (5-11 ins) | POOR without `std` (see libm) |
| `min`, `max` (f32) | 8 per-lane `libm::fminf`/`fmaxf` calls (85 ins) **with or without `std`** | POOR |
| `cast::<i32>` (f32 -> i32) | 8 scalar `vcvttss2si` + `vucomiss` saturation fix-ups (63 ins) | POOR |
| `atan2` | per-lane `libm::atan2f` | By design |

## Findings

1. **libm per-lane calls without `std`.** `lane_mul_add`, `lane_sqrt`,
   `lane_floor/ceil/round/trunc/round_ties_even` use `libm` unless the `std`
   feature is on. On a target compiled with `+fma` / SSE4.1 / AVX the
   hardware instruction is not used in the no_std build, which turns
   `fmadd`, `fmsub`, `norm_sq`, `mul_add_sub_f32`, `Simd::mul_add`,
   `Simd::sqrt` and `Simd::floor...` into 8 calls each (60-330 instructions
   instead of 1-14). The `std` build is fine. Not fixed: `f32::mul_add` and
   friends are not in `core`, and the only no_std route is
   `core::arch` intrinsics, which needs `unsafe` in `tpt-simd-vector`
   (`forbid(unsafe_code)`, ADR 0001). Candidate for an ADR note, or an
   `unsafe`-gated `cfg(target_feature)` fast path in a different crate.
2. **`Simd::min` / `max` on floats always call `libm::fminf/fmaxf` per
   lane**, with or without `std`. Not fixed: libm's NaN and signed-zero
   behaviour differs subtly from `minnum`, so a swap needs its own
   bit-exactness test over specials, and `Simd` is `forbid(unsafe_code)`.
3. **`f32 -> i32` saturating casts are scalar** (`Simd::cast`,
   `Fixed::from_f32`, `LaneCast`): 8 `vcvttss2si` plus NaN/overflow fix-ups.
   The same fix used in `tpt-simd-rounding` (below) would apply but needs
   `unsafe`, so it cannot live in `tpt-simd-vector`.
4. **`shift_with_rounding_var_i32`** is a scalar per-lane loop. The other four
   variable shifts have `vpsllvd`-style kernels. A vector version needs
   64-bit lanes (as `shift_with_rounding_i32` does) plus the `n == 0` and
   `n >= 32` policy blends. Not done (not clearly small).
5. **`Fixed<i16, ..>` multiplies** are vectorised but widen to 64-bit lanes.
   An `i32` intermediate (products fit: `|p| <= 2^30`, plus the rounding
   term) would be about 3x fewer instructions. Not done: touches the shared
   `mul_lane` semantics code.
6. **`mul_add_sub_f32`** (`tpt-simd-mul`) is FMA + `vxorps` + `vblendps`
   with `std`; a single `vfmaddsub` would be tighter. `tpt-simd-mul` was
   not touched (another agent is editing it); left to its owner.
7. **No bounds-check leaks in hot loops.** The only `panic_bounds_check`
   found was in the scalar tail of the AVX2 `dot_i16` (fixed). Every other
   panic edge (`panic_fmt`, `assert_failed`, `len_mismatch`) is a
   once-per-call precondition check outside the data loops.
8. **Baseline (SSE2, no target features).** Spot checks: `pack_*` are
   `packsswb`/`packuswb`/`packssdw`; `butterfly_*`, `mul_hi_i16` (`pmulhw`),
   `horizontal_*` vectorise to SSE2; `permute_f32`/`permute_i32` are
   scalar (no variable permute before AVX2); `complex_mul_f32` and the
   "fused" complex ops call software `fmaf` per lane (correct, bit-identical
   by design, but slow, ~230-330 instructions); `floor/ceil/trunc_f32` use
   the integer bit tricks. As documented in ADR 0001, SSE2 builds get
   whatever LLVM auto-vectorises.

## Changes made

* `tpt-simd-rounding/src/lib.rs`
  * `round_f32` (ties away from zero) now has an AVX path:
    `trunc(x + copysign(0.49999997, x))` with `vroundps` (169 -> 11
    instructions).
  * `round_to_nearest_even_i32` and `round_with_bias_i32` use
    `vcvttps2dq` plus a two-op fix-up (positive overflow -> `i32::MAX`,
    NaN -> 0) in place of eight scalar `as i32` casts (64/66 -> 11/13
    instructions).
  * Verified bit-for-bit against the previous implementation (and
    `(x + bias).floor() as i32`) over **all 2^32 `f32` bit patterns**,
    NaN compared as NaN: 0 mismatches. Crate tests pass with default
    flags, `target-cpu=native`, `+avx2,+fma`, `--features std` and
    `--features scalar-only`.
* `tpt-simd-dot/src/x86.rs`: the scalar tail of `dot_i16` iterates a zipped
  slice pair instead of indexing `a[i]`/`b[i]`, removing the per-element
  bounds check on `b`. Slightly larger code (the compiler now vectorises
  the tail too); same results.
* Both crates: `cargo fmt --check` and `cargo clippy --all-targets
  -D warnings` pass (default flags, `target-cpu=native`, `scalar-only`).

## What was not inspected

* **Non-x86 targets**: NEON/aarch64, wasm32 `simd128`, RISC-V V. No cross
  targets were audited, so none of the NEON claims in the docs are verified.
* **AVX-512** (host lacks it) and AVX-VNNI beyond what `-C target-cpu=native`
  happened to emit.
* **Other ISA baselines** (SSE4.1, AVX without AVX2) were not compiled
  separately; only SSE2, `avx2+fma` and native.
* **Inlined-into-user-loop behaviour.** Each function was inspected as a
  standalone `inline(never)` wrapper. Wrappers pass `Simd` by pointer (Rust
  ABI), so load/store shuffling at the boundary is an artefact. Codegen of
  the same functions inside a caller's hot loop was only checked for a few
  (`Simd` add/mul-accumulate over slices vectorises: `vaddps`/`vmulps` on
  ymm).
* **Not covered here**: `tpt-simd-aligned`, `-mul` benches, `-select`,
  `-matrix`, `-blas` (other agents editing, and Phase 5+). `tpt-simd-core`'s
  detect/width modules are not hot code.
* `cargo asm` was run on the scratch crate, not per-crate on the repo crates
  directly (generic functions have no symbol to select without a wrapper).
* Instruction counts include prologue/epilogue and, on Windows, saves of
  callee-saved `xmm6`-`xmm15`; they are indicative only. No throughput
  or latency analysis (llvm-mca) was done.

## Reproducing

1. Scratch crate with `#[inline(never)] #[no_mangle]` wrappers over each
   function above, path dependencies on the workspace crates, release
   profile with `lto = "off"`, `codegen-units = 1`.
2. `RUSTFLAGS="-C target-cpu=native" cargo asm --release --lib <wrapper>`
   (or `cargo rustc --release -- --emit asm` and read the `.s`). Add
   `--features std` (forwarded to every crate's `std`) for the std column.
3. For slice kernels that tail-jump (dot, complex slice helpers, pack slice
   helpers), emit the dependency's own asm with
   `cargo rustc --release -p <crate> -- --emit asm`.
