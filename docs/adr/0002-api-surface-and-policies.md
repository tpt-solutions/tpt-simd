# ADR 0002: Public API surface, naming, error/panic and unsafe policy

Status: accepted

## Types

* `Simd<T, N>` and `SimdMask<T, N>` (tpt-simd-vector). Element types:
  `i8 i16 i32 i64 u8 u16 u32 u64 f32 f64`.
* `ComplexSimd<T, N>` (tpt-simd-core): split real/imag vectors, with
  `new/add/sub/mul/div/conj/mag_sq/mag/phase/twiddle_mul/butterfly` and the
  `+ - * /` operators. Generic, unfused arithmetic. `tpt-simd-mul` owns the
  FMA-accelerated `complex_mul_f32`; `tpt-simd-complex` builds on it.
* Spec deviation: `Fixed::saturate` is replaced by explicit
  `*_saturating` / `*_wrapping` operation variants (see tpt-simd-fixed).

## Naming

* Free functions: `<op>_<type>` (`saturating_add_i16`, `dot_product_f32`).
* Methods on `Simd`: short names (`min`, `abs`, `sat_add`, `shl_scalar`).
* Lane-count suffixes are not repeated in names; the vector shape is in the
  type (`Simd<i16, 16>`).
* One home crate per function; other crates re-export (`pack_i16_to_i8` →
  saturate, `select_f32` → blend, `complex_mul_f32` → mul).

## Integer / float semantics (all crates)

* Integer `+ - *`, negate: wrapping. Saturation is always an explicit
  `sat_*`/`saturating_*` operation.
* Shifts `>= BITS`: 0 (arithmetic right shift: sign fill).
* Float min/max ignore NaN operands; ordered comparisons with NaN are false,
  `!=` is true.
* Rounding functions state their tie rule in their docs (`round_*` = ties
  away from zero, `*_even` = ties to even).

## Error / panic policy

* No `Result` in hot paths. Length-mismatched slices **panic** with a clear
  message (documented under `# Panics`); out-of-bounds in `*_checked`
  variants panics. Empty input is valid and returns the identity/empty
  result.
* `Option` is used only where failure is mathematically possible
  (`mat3x3_inverse_f32` on singular input).

## Unsafe policy

* `tpt-simd-vector`, `tpt-simd-core` and crates whose portable path is safe
  use `#![forbid(unsafe_code)]` unless they contain an intrinsics path.
* `unsafe` is allowed only for `core::arch` intrinsics and raw-pointer
  gather/scatter. Every `unsafe` block has a `// SAFETY:` comment; every
  `unsafe fn` has a `# Safety` section; `unsafe_op_in_unsafe_fn` is denied.
* Intrinsic paths are gated by `cfg(target_feature)` so the required CPU
  feature is guaranteed by the compiler flags.
