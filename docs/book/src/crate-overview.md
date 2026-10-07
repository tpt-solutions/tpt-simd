# Crate overview

## Workspace layout

| Crate | Purpose | Public API surface |
| --- | --- | --- |
| `tpt-simd-core` | Shared foundation: `Simd`/`SimdMask` backend type aliases, lane traits, `ComplexSimd`, feature detection, width constants | `SimdVector`, `SimdOps`, `DefaultBackend`, `width::`, `detect::Features`, `ComplexSimd<T,N>` |
| `tpt-simd-vector` | Stable backend: array-wrapped `Simd<T,N>`/`SimdMask<T,N>`, `forbid(unsafe_code)` | `Simd`, `SimdMask`, `SimdElement`, `SimdInt`, `SimdFloat`, `LaneCast`, `MaskLane` |
| `tpt-simd-shared` | Shared helpers used across crates | `AlignedBuf` (aligned allocation), `transpose_8x8_i16` (re-exported by `permute`) |
| `tpt-simd-mul` | FMA-accelerated complex multiply, multiply variants | `complex_mul_f32`, `mul_i32`, `mul_i16`, `mul_f32` |
| `tpt-simd-fixed` | Fixed-point vector `Fixed<T, INT_BITS, FRAC_BITS, N>` | `Fixed`, `FixedRepr`, `convert`, `wrapping_*`, `saturating_*`, `from_f32`, `to_f32`, `div`, `checked_div`, `mul_trunc` |
| `tpt-simd-dot` | Integer and float dot products | `dot_product_i16`, `dot_product_f32`, `dot_product_complex_f32`, `dot_product_saturating_i16` |
| `tpt-simd-butterfly` | Radix-2 butterfly (f32, i16, generic, complex) | `butterfly_f32`, `butterfly_i16`, `butterfly_i16_saturating`, `butterfly_generic`, `butterfly_generic_saturating`, `butterfly_complex_f32`, `butterfly_with_twiddle_f32`, `butterfly_with_twiddle_complex_f32` |
| `tpt-simd-convolve` | 1D/2D convolution kernels | `convolve_1d_f32`, `convolve_2d_separable_f32` |
| `tpt-simd-interpolate` | Linear/cubic/lanczos resampling | `linear`, `cubic`, `lanczos3` |
| `tpt-simd-window` | Hamming/Hanning/Blackman/Kaiser + apply | `hamming_window_into_f32`, `hanning_window_into_f32`, `blackman_window_into_f32`, `kaiser_window_into_f32`, `apply_window_f32`, `apply_window_simd_f32`, `cos_approx_f32`, `bessel_i0_f32` |
| `tpt-simd-permute` | Transpose, interleave, permute | `transpose_8x8_i16`, `transpose_4x4_f32`, `interleave_stereo_i16`, `permute`, `permute_f32`, `permute_i32` |
| `tpt-simd-gather` | Non-contiguous loads | `gather_i32`, `gather_f32`, `try_gather_i32`, `try_gather_f32` |
| `tpt-simd-scatter` | Non-contiguous stores | `scatter_i32`, `scatter_f32`, `try_scatter_i32`, `try_scatter_f32` |
| `tpt-simd-aligned` | Aligned allocation and slice helpers | `AlignedBuf`, `aligned_from_slice`, `from_aligned_slice` |
| `tpt-simd-blend` | Mask-based blend/select | `blend_f32`, `blend_i16`, `select_f32`, `select_i16`, `blend_mask8` |
| `tpt-simd-select` | Element selection | `select`, `select_first`, `select_last` |
| `tpt-simd-compare` | Comparison with masks | `cmp_eq`, `cmp_ne`, `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `find_first`, `count` |
| `tpt-simd-horizontal` | Horizontal reductions | `sum_f32`, `sum_i32`, `max_f32`, `min_f32`, `hadd`, `hsub` |
| `tpt-simd-rounding` | Rounding + saturation | `round`, `floor`, `ceil`, `trunc`, `round_even`, `sat_add`, `sat_sub`, `sat_mul` |
| `tpt-simd-shift` | Shift + descale | `shl`, `shr`, `sar`, `rol`, `shift_with_rounding`, `descale` |
| `tpt-simd-saturate` | Saturating arithmetic + packing | `saturating_add`, `saturating_sub`, `saturating_mul`, `pack_i16_to_i8`, `pack_i32_to_i16` |
| `tpt-simd-math` | exp/ln/sin/cos/tanh/erf | `exp_f32`, `ln_f32`, `sin_f32`, `cos_f32`, `tan_f32`, `tanh_f32`, `erf_f32` |
| `tpt-simd-rng` | xoshiro256++/Philox | `random_u64`, `random_f32`, `normal_f32`, `philox4x32_10` |
| `tpt-simd-sparse` | Sparse matrix kernels | `spmv_csr_f32`, `cg_update` |

## Backend and dispatch

See [ADR 0001](crate-overview.md#backend-and-dispatch) (accepted) for the
backend and dispatch strategy, and [ADR 0003](crate-overview.md#runtime-dispatch)
for the `runtime-dispatch` feature on kernel crates.

### Backend

`tpt-simd-vector` provides `Simd<T, N>` / `SimdMask<T, N>` as array wrappers
(`#[repr(transparent)]` over `[T; N]`) with lane-loop operations. It is safe
(`forbid(unsafe_code)`), `no_std`, and stable. `tpt-simd-core` re-exports it.
The `nightly` feature is reserved for a `core::simd`-backed implementation and
currently changes nothing.

## Public API surface

All crates expose a safe, documented public API. The only `unsafe` is confined
to `core::arch` intrinsics (in `tpt-simd-vector` backends) and raw-pointer
gather/scatter (`tpt-simd-gather`, `tpt-simd-scatter`). Every `unsafe` block
carries a `// SAFETY:` comment and every `unsafe fn` carries a `# Safety`
section, per [ADR 0002](crate-overview.md#unsafe-policy).

## Unsafe policy

Per [ADR 0002](crate-overview.md#unsafe-policy):

- `tpt-simd-vector`, `tpt-simd-core` and crates whose portable path is safe
  use `#![forbid(unsafe_code)]` unless they contain an intrinsics path.
- `unsafe` is allowed only for `core::arch` intrinsics and raw-pointer
  gather/scatter. Every `unsafe` block has a `// SAFETY:` comment; every
  `unsafe fn` has a `# Safety` section; `unsafe_op_in_unsafe_fn` is denied.
- Intrinsic paths are gated by `cfg(target_feature)` so the required CPU
  feature is guaranteed by the compiler flags.

## Float determinism tiers

Documented in each crate's docs:

- **Tier 1 (bit-exact vs scalar reference, cross-target):** all Phase 1-6 crates.
- **Tier 2 (documented ulp/relative tolerance, may differ across targets or
  dispatch path):** `blas`, `reduce` (except `min/max/arg*`), `math`,
  `sparse`. Results are deterministic for a fixed build *and* CPU class. With
  `runtime-dispatch` the same binary can give slightly different results on
  different machines (FMA vs non-FMA rounding).
- **Tier 3 (statistical):** `rng` outputs are bit-reproducible for a seed, but
  normal/uniform-float transforms use approximations with documented error.

## Performance

Every crate README documents its own benchmarks. Use
`cargo bench -p <crate>` to reproduce (add the `-C target-cpu=native` RUSTFLAGS
for native builds). Summary numbers and caveats live in
[docs/benchmarks.md](../../benchmarks.md).


### Dispatch

Fast paths are selected with `cfg(all(target_arch = "...", target_feature =
"..."))` inside each crate, always next to a portable scalar/lane-loop
implementation that is the behavioural reference. There are no per-architecture
Cargo features; build with `-C target-cpu=...` / `-C target-feature=...` to
enable them. The `scalar-only` feature forces the portable path everywhere
(used to test the fallback on any machine).
