# How to convert between fixed-point formats

`Fixed<T, INT_BITS, FRAC_BITS, N>` is type-safe: the Q-format is checked at
compile time, and conversions between formats are explicit. Conversions are
saturating (or wrapping, depending on the operation) and round to nearest
with ties away from zero for `convert`.

## The type parameters

- `T`: raw storage type (`i32` or `i16`).
- `INT_BITS`: integer bits **including** the sign bit.
- `FRAC_BITS`: fractional bits.
- `N`: lane count.

`Fixed<i32, 16, 16, 8>` is the classic Q16.16. `INT_BITS + FRAC_BITS <= T::BITS`
is enforced at compile time.

## Converting between formats

Use `convert::<I2, F2>()` to convert to another Q-format of the same storage
type. This is saturating and rounds ties away from zero.

```rust
use tpt_simd_core::F32x8;
use tpt_simd_fixed::Fixed;

// Q16.16
let a: Fixed<i32, 16, 16, 8> = Fixed::from_f32(F32x8::splat(1.5));

// Convert to Q8.8 (narrow format: stored in wider lanes, saturated on ops)
let b: Fixed<i32, 8, 8, 8> = a.convert();

// Convert back to Q16.16
let a2: Fixed<i32, 16, 16, 8> = b.convert();
assert_eq!(a2.to_f32(), F32x8::splat(1.5));
```

## Construction and rounding rules

| Operation | Tie rule | Overflow behavior |
| --- | --- | --- |
| `from_f32` | ties to even | saturates, NaN -> 0 |
| `to_f32` | round to nearest | exact when raw fits in 24 bits, else rounded |
| `convert` | ties away from zero | saturating |
| operators `+ - * /` | n/a (integer) | wrap at storage width |

```rust
use tpt_simd_core::F32x8;
use tpt_simd_fixed::Fixed;

let a: Fixed<i32, 16, 16, 8> = Fixed::from_f32(F32x8::splat(1.5));
assert_eq!(a.to_f32(), F32x8::splat(1.5));

// Wrap semantics of operators (ADR 0002)

let big = Fixed::<i32, 16, 16, 8>::from_raw(F32x8::splat(1 << 30)); // 16384.0
assert_eq!(big.wrapping_mul(big), Fixed::<i32, 16, 16, 8>::max_value());
```

## Narrow formats

When `INT_BITS + FRAC_BITS < T::BITS` (e.g. `Q8.8` in `i32`), the format is a
narrow format stored in wider lanes. Saturating operations clamp to the format
range; wrapping operations wrap at the **storage** width (`i32`).

```rust
use tpt_simd_core::F32x8;
use tpt_simd_fixed::Fixed;

// Q8.8 stored in i32 lanes
let a: Fixed<i32, 8, 8, 8> = Fixed::from_f32(F32x8::splat(127.99));
// clamping at the Q8.8 range
let b = a.saturating_add(a);
assert_eq!(b.to_f32(), F32x8::splat(255.98));
```

## Performance notes

- Conversions are integer-only; `to_f32`/`from_f32` are exact when the raw
  value fits in 24 bits and otherwise use `f32` conversion.
- `convert` is saturating and ties-away-from-zero; it does not use
  `mul_add` or other reassociating reductions.
- All conversions are bit-identical across backends and targets.
- For resamplers that need fractional bit shifts, use
  `tpt_simd_shift::shift_with_rounding_i32` / `descale_i32` instead of raw
  shifts to get the correct fixed-point rounding.

## See also

- [How to do motion compensation in a video codec](motion-compensation.md)
- [How to implement an FFT using tpt-simd](fft.md)
