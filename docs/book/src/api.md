# API reference

The public API is documented in each crate's README (in its `README.md` at the
crate root) and in the crate doc comments (which are rendered to `cargo doc`).
This page is the canonical index linking to every public item and states the
per-crate policies that apply across the workspace.

## Policy applied to every public item

- **Doc comment on every public item.** Every public function, type and trait
  carries a doc comment. The comment must include: a description, `# Panics`
  (where applicable), `# Safety` (on `unsafe fn`), `# Performance`, and an
  `# Example` showing real codec / DSP usage.
- **Performance notes.** Each item states the intrinsic or pattern it lowers
  to and the expected speedup versus a scalar baseline.
- **Safety.** `unsafe` is confined to `core::arch` intrinsics and
  raw-pointer gather/scatter. See [crate-overview.md](crate-overview.md#unsafe-policy).

## Foundation

| Crate | Public API |
| --- | --- |
| `tpt-simd-core` | `SimdVector`, `SimdOps`, `DefaultBackend`, `width::NATIVE_VECTOR_BITS`, `detect::Features`, `ComplexSimd<T,N>` |
| `tpt-simd-vector` | `Simd<T,N>`, `SimdMask<T,N>`, `SimdElement`, `SimdInt`, `SimdFloat`, `LaneCast`, `MaskLane` |
| `tpt-simd-shared` | `AlignedBuf`, `aligned_from_slice`, `from_aligned_slice` |

## Arithmetic

| Crate | Public API |
| --- | --- |
| `tpt-simd-mul` | `complex_mul_f32`, `complex_mul_f64`, `mul_i32`, `mul_i16`, `mul_f32`, `mul_add_f32` |
| `tpt-simd-fixed` | `Fixed<T, INT_BITS, FRAC_BITS, N>`, `FixedRepr`, `convert`, `wrapping_add`, `saturating_mul`, `from_f32`, `to_f32`, `div`, `checked_div`, `mul_trunc`, `abs`, `min`, `max`, `clamp` |
| `tpt-simd-saturate` | `saturating_add_i16`, `saturating_sub_i16`, `saturating_mul_i16`, `pack_i16_to_i8`, `pack_i32_to_i16` |
| `tpt-simd-rounding` | `round_f32`, `floor_f32`, `ceil_f32`, `trunc_f32`, `round_even_f32`, `sat_add_i16`, `sat_sub_i16`, `sat_mul_i16` |
| `tpt-simd-shift` | `shl_i32`, `shr_i32`, `sar_i32`, `rol_i32`, `shift_with_rounding_i32`, `descale_i32` |
| `tpt-simd-math` | `exp_f32`, `ln_f32`, `sin_f32`, `cos_f32`, `tan_f32`, `tanh_f32`, `erf_f32` |
| `tpt-simd-rng` | `random_u64`, `random_f32`, `normal_f32`, `philox4x32_10` |

## Complex / transforms

| Crate | Public API |
| --- | --- |
| `tpt-simd-complex` | `ComplexSimd<T,N>`, `ComplexSimdF32Ext`, `ComplexSimdF64Ext`, `interleaved_to_split`, `split_to_interleaved` |
| `tpt-simd-butterfly` | `butterfly_f32`, `butterfly_i16`, `butterfly_i16_saturating`, `butterfly_generic`, `butterfly_generic_saturating`, `butterfly_complex_f32`, `butterfly_with_twiddle_f32`, `butterfly_with_twiddle_complex_f32` |
| `tpt-simd-window` | `hamming_window_into_f32`, `hanning_window_into_f32`, `blackman_window_into_f32`, `kaiser_window_into_f32`, `apply_window_f32`, `apply_window_simd_f32`, `cos_approx_f32`, `cos_approx_simd_f32`, `bessel_i0_f32`, `bessel_i0_f64` |
| `tpt-simd-convolve` | `convolve_1d_f32`, `convolve_2d_separable_f32` |
| `tpt-simd-interpolate` | `linear_f32`, `cubic_f32`, `lanczos3_f32` |

## Data movement

| Crate | Public API |
| --- | --- |
| `tpt-simd-permute` | `transpose_8x8_i16`, `transpose_4x4_f32`, `interleave_stereo_i16`, `permute`, `permute_f32`, `permute_i32` |
| `tpt-simd-gather` | `gather_i32`, `gather_f32`, `try_gather_i32`, `try_gather_f32` |
| `tpt-simd-scatter` | `scatter_i32`, `scatter_f32`, `try_scatter_i32`, `try_scatter_f32` |
| `tpt-simd-aligned` | `AlignedBuf`, `aligned_from_slice`, `from_aligned_slice` |
| `tpt-simd-blend` | `blend_f32`, `blend_i16`, `select_f32`, `select_i16`, `blend_mask8` |
| `tpt-simd-select` | `select`, `select_first`, `select_last` |
| `tpt-simd-compare` | `cmp_eq`, `cmp_ne`, `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge`, `find_first`, `count` |

## Reductions

| Crate | Public API |
| --- | --- |
| `tpt-simd-horizontal` | `sum_f32`, `sum_i32`, `max_f32`, `min_f32`, `hadd_f32`, `hadd_i32`, `hsub_f32` |
| `tpt-simd-dot` | `dot_product_i16`, `dot_product_f32`, `dot_product_complex_f32`, `dot_product_saturating_i16`, `dot_scaled_f32` |

## Linear algebra

| Crate | Public API |
| --- | --- |
| `tpt-simd-matrix` | `mat4x4_mul_f32`, `mat8x8_mul_i16`, `mat4x4_transpose_f32`, `mat8x8_transpose_i16`, `mat3x3_inverse_f32` |
| `tpt-simd-blas` | `gemm_f32`, `gemm_f64`, `gemv_f32`, `gemv_f64`, `syrk_f32`, `syrk_f64`, `omatmul_f32` |
| `tpt-simd-sparse` | `spmv_csr_f32`, `cg_update` |

## Feature flags

| Crate | Features |
| --- | --- |
| all | `std`, `alloc`, `nightly`, `scalar-only`, `runtime-dispatch` (kernel crates only) |

See each crate's README for the full feature table.
