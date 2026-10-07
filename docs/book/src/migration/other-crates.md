# From other SIMD crates

## `wide`

`wide` provides general-purpose SIMD wrappers but no complex, fixed-point or
codec-specific types. tpt-simd fills that gap.

| `wide` | tpt-simd |
| --- | --- |
| `Simd<f32, N>` (custom type) | `tpt_simd_core::Simd<f32, 8>` (type alias `F32x8`) |
| `Mask` | `SimdMask<T, N>` |
| `Wide::splat` | `F32x8::splat` |

Move to `tpt_simd_core` types and operators.

## `simba`

`simba` is a math/glyph-focused SIMD library with its own vector types.
tpt-simd's `Simd<T, N>` and `ComplexSimd<T, N>` are drop-in replacements for
element-wise DSP kernels.

| `simba` | tpt-simd |
| --- | --- |
| `Scalar`, `Lane`, `Simd<T, N>` | `tpt_simd_core::Simd<T, N>` |
| `Complex<T, N>` | `tpt_simd_core::ComplexSimd<T, N>` |
| `uint`, `sint` ops | operators on `Simd` |

## `rustfft`

`rustfft` is a full FFT implementation. tpt-simd does **not** replace it — it
provides the SIMD kernels (butterflies, complex multiply, transposes, windows)
that `rustfft` does not expose. You can use both: `rustfft` for the FFT
shell, tpt-simd for your own kernels or for replacing internal SIMD blocks.

## `std::simd`

Rust's portable SIMD (`core::simd`) is **nightly-only**. tpt-simd provides a
stable, `no_std` backend via `tpt-simd-vector`, with an optional `core::simd`
backend under the `nightly` feature. Use tpt-simd for stable code; the API is
designed to be on the same plane as `core::simd`.

## Migration steps

1. Replace your SIMD type with `tpt_simd_core::Simd<T, N>` (alias `F32x8`,
   `I16x16`, etc.).
2. Replace `wide`/`simba` operators with `+ - * /`, `mul_add`, `min`, `max`.
3. Replace complex multiply with `tpt_simd_mul::complex_mul_f32` / `tpt_simd_complex`.
4. Replace custom FFT code with `tpt_simd_butterfly` + `tpt_simd_complex`.
5. Re-run tests; the workspace's proptest property tests (SIMD == scalar)
   are the target.

## See also

- [From manual intrinsics](manual-intrinsics.md)
- [From scalar code](scalar-code.md)
