# Appendix B: Example usage (spec)

These examples are from the design spec `spec.txt`. They are **verified to
compile and run as doctests** (see [known issues](known-issues.md)).

## B.1 Complex multiplication (FFT)

```rust
use tpt_simd_complex::ComplexSimd;
use tpt_simd_core::F32x8; // stable backend; core::simd is nightly-only

let a = ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(2.0));
let b = ComplexSimd::new(F32x8::splat(3.0), F32x8::splat(4.0));
let result = a * b;  // (1+2i)(3+4i) = (3-8) + (4+6)i = -5 + 10i
```

## B.2 Fixed-point motion compensation

```rust
use tpt_simd_fixed::Fixed;
use tpt_simd_core::I32x8;

let a: Fixed<i32, 16, 16, 8> = Fixed::from_f32(F32x8::splat(1.5));
let b: Fixed<i32, 16, 16, 8> = Fixed::from_f32(F32x8::splat(2.5));
let result = a * b;  // 1.5 * 2.5 = 3.75 with correct rounding
```

## B.3 Dot product (FLAC LPC)

```rust
use tpt_simd_dot::dot_product_i16;

let a: Vec<i16> = (0..256).collect();
let b: Vec<i16> = (0..256).collect();
let result = dot_product_i16(&a, &b);
```

## B.4 8x8 transpose (Video DCT)

```rust
use tpt_simd_permute::transpose_8x8_i16;

let mut block: [[i16; 8]; 8] = [[0; 8]; 8];
// ... fill block ...
transpose_8x8_i16(&mut block);
```

## B.5 FFT Butterfly

```rust
use tpt_simd_butterfly::butterfly_f32;
use tpt_simd_core::F32x8;

let mut a = F32x8::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
let mut b = F32x8::from_array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
butterfly_f32(&mut a, &mut b);
// a = [9, 9, 9, 9, 9, 9, 9, 9]
// b = [-7, -5, -3, -1, 1, 3, 5, 7]
```
