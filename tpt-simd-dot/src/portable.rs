//! Portable reference implementations (always available, no `unsafe`).
//!
//! These define the exact behaviour (including float accumulation order) that
//! the intrinsic fast paths must reproduce bit-for-bit. They are public so
//! tests and benchmarks can compare against them. Length equality is checked
//! by the top-level functions; these functions only `debug_assert!` it and
//! otherwise process `min(a.len(), b.len())` elements.

use tpt_simd_core::{ComplexSimd, F32x8, I16x16, I32x8, I32x16, Simd};
use tpt_simd_horizontal::{horizontal_sum_f32, reduce_sum};

/// Portable [`dot_product_i16`](crate::dot_product_i16).
///
/// # Examples
/// ```
/// assert_eq!(tpt_simd_dot::portable::dot_product_i16(&[2, 3], &[4, 5]), 23);
/// ```
pub fn dot_product_i16(a: &[i16], b: &[i16]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [I32x16::zero(); 2];
    let mut ca = a.chunks_exact(32);
    let mut cb = b.chunks_exact(32);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        for k in 0..2 {
            let xv = I16x16::from_slice(&x[16 * k..]);
            let yv = I16x16::from_slice(&y[16 * k..]);
            acc[k] += xv.cast::<i32>() * yv.cast::<i32>();
        }
    }
    let mut sum = reduce_sum(acc[0] + acc[1]);
    for (&x, &y) in ca.remainder().iter().zip(cb.remainder()) {
        sum = sum.wrapping_add(x as i32 * y as i32);
    }
    sum
}

/// Portable [`dot_product_saturating_i16`](crate::dot_product_saturating_i16).
///
/// # Examples
/// ```
/// assert_eq!(tpt_simd_dot::portable::dot_product_saturating_i16(&[2, 3], &[4, 5]), 23);
/// ```
pub fn dot_product_saturating_i16(a: &[i16], b: &[i16]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = I32x8::zero();
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        let p = Simd::<i16, 8>::from_slice(x).cast::<i32>()
            * Simd::<i16, 8>::from_slice(y).cast::<i32>();
        acc = acc.sat_add(p);
    }
    // Remaining < 8 elements: element j goes to lane j (zero padding adds 0).
    let (rx, ry) = (ca.remainder(), cb.remainder());
    if !rx.is_empty() {
        let xv = Simd::<i16, 8>::from_slice_or(rx, 0);
        let yv = Simd::<i16, 8>::from_slice_or(ry, 0);
        acc = acc.sat_add(xv.cast::<i32>() * yv.cast::<i32>());
    }
    // Halving tree with saturating adds.
    let l = acc.to_array();
    let t = [
        l[0].saturating_add(l[4]),
        l[1].saturating_add(l[5]),
        l[2].saturating_add(l[6]),
        l[3].saturating_add(l[7]),
    ];
    let u = [t[0].saturating_add(t[2]), t[1].saturating_add(t[3])];
    u[0].saturating_add(u[1])
}

/// Portable [`dot_product_f32`](crate::dot_product_f32).
///
/// # Examples
/// ```
/// assert_eq!(tpt_simd_dot::portable::dot_product_f32(&[2.0, 3.0], &[4.0, 5.0]), 23.0);
/// ```
pub fn dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [F32x8::zero(); 4];
    let mut ca = a.chunks_exact(32);
    let mut cb = b.chunks_exact(32);
    for (x, y) in ca.by_ref().zip(cb.by_ref()) {
        for k in 0..4 {
            let xv = F32x8::from_slice(&x[8 * k..]);
            let yv = F32x8::from_slice(&y[8 * k..]);
            acc[k] += xv * yv;
        }
    }
    // Remainder (< 32 elements) continues the same element -> accumulator map;
    // the last partial 8-chunk is zero padded (adds +0 to untouched lanes).
    for (k, (x, y)) in ca
        .remainder()
        .chunks(8)
        .zip(cb.remainder().chunks(8))
        .enumerate()
    {
        let xv = F32x8::from_slice_or(x, 0.0);
        let yv = F32x8::from_slice_or(y, 0.0);
        acc[k] += xv * yv;
    }
    horizontal_sum_f32((acc[0] + acc[1]) + (acc[2] + acc[3]))
}

/// Portable [`dot_product_complex_f32`](crate::dot_product_complex_f32).
///
/// # Examples
/// ```
/// let r = tpt_simd_dot::portable::dot_product_complex_f32(&[], &[]);
/// assert_eq!(r.real.to_array(), [0.0; 8]);
/// ```
pub fn dot_product_complex_f32(
    a: &[ComplexSimd<f32, 8>],
    b: &[ComplexSimd<f32, 8>],
) -> ComplexSimd<f32, 8> {
    debug_assert_eq!(a.len(), b.len());
    // One accumulator pair, products added in increasing index order. The
    // multiply-add chain is throughput-bound (4 muls + 4 adds per product),
    // so extra accumulators gain nothing measurable. Plain arrays keep LLVM
    // vectorising across the 8 lanes; per lane the arithmetic is exactly
    // `ComplexSimd::mul` followed by `add`.
    let mut re = [0.0f32; 8];
    let mut im = [0.0f32; 8];
    for (x, y) in a.iter().zip(b) {
        madd(&mut re, &mut im, x, y);
    }
    ComplexSimd::new(F32x8::from_array(re), F32x8::from_array(im))
}

/// `acc += x * y` per lane, with the unfused arithmetic of `ComplexSimd::mul`.
#[inline(always)]
fn madd(re: &mut [f32; 8], im: &mut [f32; 8], x: &ComplexSimd<f32, 8>, y: &ComplexSimd<f32, 8>) {
    let (xr, xi, yr, yi) = (
        x.real.as_array(),
        x.imag.as_array(),
        y.real.as_array(),
        y.imag.as_array(),
    );
    for j in 0..8 {
        re[j] += xr[j] * yr[j] - xi[j] * yi[j];
        im[j] += xr[j] * yi[j] + xi[j] * yr[j];
    }
}
