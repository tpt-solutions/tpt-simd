//! Window function generation and application.
//!
//! * Generators: Hamming, Hanning (Hann), Blackman, Kaiser. Each has an
//!   allocation-free `*_window_into_f32(&mut [f32])` form that fills a caller
//!   slice, and (with the `alloc` or `std` feature) a `*_window_f32(size)`
//!   form returning a `Vec<f32>`.
//! * Application: [`apply_window_f32`] (slices), [`apply_window_simd_f32`]
//!   (slices of 8-lane vectors) and the scalar reference
//!   [`apply_window_scalar_f32`].
//! * Math helpers: [`cos_approx_f32`] / [`cos_approx_simd_f32`] (polynomial
//!   cosine, documented error) and [`bessel_i0_f64`] / [`bessel_i0_f32`].
//!
//! ## Window convention
//!
//! All windows are *symmetric* (denominator `N - 1`, the form used for FIR
//! design and analysis): `w[n]` for `n in 0..N`, `w[0] == w[N-1]`. Size `0`
//! gives an empty window and size `1` gives `[1.0]`.
//!
//! ## Accuracy
//!
//! The cosine-sum windows use [`cos_approx_simd_f32`] (max error `2e-6`), so
//! they are accurate to about `3e-6` absolute versus an `f64` evaluation
//! (checked by the crate's tests). Kaiser is evaluated in `f64` and rounded.
#![no_std]
#![deny(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(any(feature = "std", test))]
extern crate std;

#[cfg(feature = "alloc")]
use alloc::{vec, vec::Vec};
use tpt_simd_core::Simd;

type F32x8 = Simd<f32, 8>;

// ---- cosine approximation ---------------------------------------------

const PI_HI: f32 = 3.140_625; // exactly representable, 9 significant bits
const PI_LO: f32 = 9.676_536e-4; // PI - PI_HI

/// 8-lane polynomial cosine of `x` radians.
///
/// Range reduction `n = round(x / pi)`, `r = x - n*pi` (two-constant
/// Cody-Waite, `|r| <= pi/2`), `cos(x) = (-1)^n cos(r)`, with `cos(r)` an
/// even degree-12 Taylor polynomial in `r` (truncation error `< 7e-9`).
///
/// **Documented maximum absolute error: `2e-6`** for `|x| <= 1000` (the
/// observed worst case in the crate's tests is well under `1e-6` for
/// `|x| <= 100`); the error grows roughly linearly with `|x|` beyond that
/// because `x` itself is only an `f32`. Not intended for huge arguments.
///
/// ```
/// use tpt_simd_core::F32x8;
/// use tpt_simd_window::cos_approx_simd_f32;
/// let r = cos_approx_simd_f32(F32x8::splat(0.0));
/// assert!((r[0] - 1.0).abs() < 2e-6);
/// ```
#[inline]
pub fn cos_approx_simd_f32(x: F32x8) -> F32x8 {
    const C: [f32; 7] = [
        1.0,
        -0.5,
        0.041_666_668,
        -0.001_388_889,
        2.480_158_7e-5,
        -2.755_732e-7,
        2.087_675_7e-9,
    ];
    let s = F32x8::splat;
    let n = (x * s(core::f32::consts::FRAC_1_PI)).round();
    let r = (-n).mul_add(s(PI_HI), x);
    let r = (-n).mul_add(s(PI_LO), r);
    let r2 = r * r;
    let mut p = s(C[6]);
    for &c in C[..6].iter().rev() {
        p = p.mul_add(r2, s(c));
    }
    let parity = n - s(2.0) * (n * s(0.5)).floor();
    p * (s(1.0) - s(2.0) * parity)
}

/// Scalar convenience wrapper over [`cos_approx_simd_f32`] (same error
/// bound: `2e-6` for `|x| <= 1000`).
///
/// ```
/// use tpt_simd_window::cos_approx_f32;
/// assert!((cos_approx_f32(core::f32::consts::PI) + 1.0).abs() < 2e-6);
/// ```
#[inline]
pub fn cos_approx_f32(x: f32) -> f32 {
    cos_approx_simd_f32(F32x8::splat(x))[0]
}

// ---- Bessel I0 ----------------------------------------------------------

/// Modified Bessel function of the first kind, order 0, via its power series
/// `sum_k (x^2/4)^k / (k!)^2` summed in `f64` until the terms fall below
/// `1e-17` of the sum. Relative error is a few `f64` ulps for `|x| <= 700`;
/// larger arguments overflow to `inf`.
///
/// ```
/// use tpt_simd_window::bessel_i0_f64;
/// assert_eq!(bessel_i0_f64(0.0), 1.0);
/// assert!((bessel_i0_f64(1.0) - 1.266_065_877_752_008_4).abs() < 1e-14);
/// ```
pub fn bessel_i0_f64(x: f64) -> f64 {
    let q = x * x * 0.25;
    let (mut term, mut sum) = (1.0f64, 1.0f64);
    for k in 1..2000u32 {
        let kf = f64::from(k);
        term *= q / (kf * kf);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// `f32` Bessel I0 (computed in `f64`, rounded once; relative error
/// `<= 6e-8`).
///
/// ```
/// use tpt_simd_window::bessel_i0_f32;
/// assert!((bessel_i0_f32(2.0) - 2.279_585_3).abs() < 1e-6);
/// ```
#[inline]
pub fn bessel_i0_f32(x: f32) -> f32 {
    bessel_i0_f64(f64::from(x)) as f32
}

// ---- generators ---------------------------------------------------------

/// `w[n] = a0 - a1 cos(2 pi n/(N-1)) + a2 cos(4 pi n/(N-1))`.
fn cosine_sum_into(out: &mut [f32], a0: f64, a1: f64, a2: f64) {
    let n = out.len();
    match n {
        0 => return,
        1 => {
            out[0] = 1.0;
            return;
        }
        _ => {}
    }
    let denom = (n - 1) as f64;
    let two_pi = 2.0 * core::f64::consts::PI;
    let mut base = 0usize;
    for chunk in out.chunks_mut(8) {
        // Phases are formed in f64 then rounded once to f32.
        let ph1 = F32x8::from_fn(|i| (two_pi * ((base + i) as f64) / denom) as f32);
        let ph2 = F32x8::from_fn(|i| (2.0 * two_pi * ((base + i) as f64) / denom) as f32);
        let w = F32x8::splat(a0 as f32) - F32x8::splat(a1 as f32) * cos_approx_simd_f32(ph1)
            + F32x8::splat(a2 as f32) * cos_approx_simd_f32(ph2);
        w.store_partial(chunk);
        base += 8;
    }
}

/// Fill `out` with a symmetric Hamming window
/// (`0.54 - 0.46 cos(2 pi n / (N-1))`, `N = out.len()`). No allocation.
///
/// ```
/// let mut w = [0.0f32; 5];
/// tpt_simd_window::hamming_window_into_f32(&mut w);
/// assert!((w[0] - 0.08).abs() < 1e-5 && (w[2] - 1.0).abs() < 1e-5);
/// ```
pub fn hamming_window_into_f32(out: &mut [f32]) {
    cosine_sum_into(out, 0.54, 0.46, 0.0);
}

/// Fill `out` with a symmetric Hanning/Hann window
/// (`0.5 - 0.5 cos(2 pi n / (N-1))`). No allocation.
///
/// ```
/// let mut w = [0.0f32; 5];
/// tpt_simd_window::hanning_window_into_f32(&mut w);
/// assert!(w[0].abs() < 1e-5 && (w[2] - 1.0).abs() < 1e-5);
/// ```
pub fn hanning_window_into_f32(out: &mut [f32]) {
    cosine_sum_into(out, 0.5, 0.5, 0.0);
}

/// Fill `out` with a symmetric (classic, a = 0.16) Blackman window
/// (`0.42 - 0.5 cos(2 pi n/(N-1)) + 0.08 cos(4 pi n/(N-1))`). No allocation.
///
/// ```
/// let mut w = [0.0f32; 5];
/// tpt_simd_window::blackman_window_into_f32(&mut w);
/// assert!(w[0].abs() < 1e-5 && (w[2] - 1.0).abs() < 1e-5);
/// ```
pub fn blackman_window_into_f32(out: &mut [f32]) {
    cosine_sum_into(out, 0.42, 0.5, 0.08);
}

/// Fill `out` with a symmetric Kaiser window with shape parameter `beta`:
/// `I0(beta * sqrt(1 - (2n/(N-1) - 1)^2)) / I0(beta)`. No allocation.
///
/// Larger `beta` gives lower side lobes and a wider main lobe
/// (`beta = 0` is rectangular, about `5` resembles Hamming, about `8.6`
/// resembles Blackman).
///
/// # Panics
/// If `beta` is negative, NaN or infinite.
///
/// ```
/// let mut w = [0.0f32; 7];
/// tpt_simd_window::kaiser_window_into_f32(&mut w, 8.0);
/// assert!((w[3] - 1.0).abs() < 1e-6);
/// assert!(w[0] < 0.01);
/// ```
pub fn kaiser_window_into_f32(out: &mut [f32], beta: f32) {
    assert!(
        beta.is_finite() && beta >= 0.0,
        "kaiser beta must be finite and >= 0"
    );
    let n = out.len();
    match n {
        0 => return,
        1 => {
            out[0] = 1.0;
            return;
        }
        _ => {}
    }
    let beta = f64::from(beta);
    let norm = 1.0 / bessel_i0_f64(beta);
    let denom = (n - 1) as f64;
    for (i, o) in out.iter_mut().enumerate() {
        let r = 2.0 * (i as f64) / denom - 1.0;
        let arg = libm::sqrt((1.0 - r * r).max(0.0));
        *o = (bessel_i0_f64(beta * arg) * norm) as f32;
    }
}

/// Symmetric Hamming window of `size` samples (see
/// [`hamming_window_into_f32`]).
///
/// ```
/// let w = tpt_simd_window::hamming_window_f32(8);
/// assert_eq!(w.len(), 8);
/// assert!((w[0] - w[7]).abs() < 1e-6);
/// ```
#[cfg(feature = "alloc")]
pub fn hamming_window_f32(size: usize) -> Vec<f32> {
    let mut v = vec![0.0; size];
    hamming_window_into_f32(&mut v);
    v
}

/// Symmetric Hanning window of `size` samples (see
/// [`hanning_window_into_f32`]).
///
/// ```
/// assert_eq!(tpt_simd_window::hanning_window_f32(3).len(), 3);
/// ```
#[cfg(feature = "alloc")]
pub fn hanning_window_f32(size: usize) -> Vec<f32> {
    let mut v = vec![0.0; size];
    hanning_window_into_f32(&mut v);
    v
}

/// Symmetric Blackman window of `size` samples (see
/// [`blackman_window_into_f32`]).
///
/// ```
/// assert_eq!(tpt_simd_window::blackman_window_f32(16).len(), 16);
/// ```
#[cfg(feature = "alloc")]
pub fn blackman_window_f32(size: usize) -> Vec<f32> {
    let mut v = vec![0.0; size];
    blackman_window_into_f32(&mut v);
    v
}

/// Symmetric Kaiser window of `size` samples (see
/// [`kaiser_window_into_f32`]).
///
/// # Panics
/// If `beta` is negative, NaN or infinite.
///
/// ```
/// assert_eq!(tpt_simd_window::kaiser_window_f32(9, 5.0).len(), 9);
/// ```
#[cfg(feature = "alloc")]
pub fn kaiser_window_f32(size: usize, beta: f32) -> Vec<f32> {
    let mut v = vec![0.0; size];
    kaiser_window_into_f32(&mut v, beta);
    v
}

// ---- application --------------------------------------------------------

/// Scalar reference: `data[i] *= window[i]`.
///
/// # Panics
/// If the lengths differ.
pub fn apply_window_scalar_f32(data: &mut [f32], window: &[f32]) {
    assert_eq!(
        data.len(),
        window.len(),
        "data and window lengths must match"
    );
    for (d, w) in data.iter_mut().zip(window) {
        *d *= *w;
    }
}

/// `data[i] *= window[i]`, processed 8 lanes at a time (scalar tail).
/// Bit-identical to [`apply_window_scalar_f32`].
///
/// # Panics
/// If the lengths differ.
///
/// ```
/// let mut d = [1.0f32, 2.0, 3.0];
/// tpt_simd_window::apply_window_f32(&mut d, &[0.5, 0.5, 2.0]);
/// assert_eq!(d, [0.5, 1.0, 6.0]);
/// ```
pub fn apply_window_f32(data: &mut [f32], window: &[f32]) {
    assert_eq!(
        data.len(),
        window.len(),
        "data and window lengths must match"
    );
    let mut d = data.chunks_exact_mut(8);
    let mut w = window.chunks_exact(8);
    for (dc, wc) in (&mut d).zip(&mut w) {
        (F32x8::from_slice(dc) * F32x8::from_slice(wc)).copy_to_slice(dc);
    }
    apply_window_scalar_f32(d.into_remainder(), w.remainder());
}

/// Multiply a slice of 8-lane vectors by the matching window vectors.
///
/// # Panics
/// If the lengths differ.
///
/// ```
/// use tpt_simd_core::F32x8;
/// let mut d = [F32x8::splat(2.0); 2];
/// tpt_simd_window::apply_window_simd_f32(&mut d, &[F32x8::splat(0.25); 2]);
/// assert_eq!(d[1], F32x8::splat(0.5));
/// ```
pub fn apply_window_simd_f32(data: &mut [F32x8], window: &[F32x8]) {
    assert_eq!(
        data.len(),
        window.len(),
        "data and window lengths must match"
    );
    for (d, w) in data.iter_mut().zip(window) {
        *d *= *w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const PI: f64 = core::f64::consts::PI;

    #[test]
    fn cos_approx_error_bound() {
        let mut worst = 0.0f64;
        let mut x = -1000.0f32;
        while x <= 1000.0 {
            let e = (f64::from(cos_approx_f32(x)) - f64::from(x).cos()).abs();
            worst = worst.max(e);
            x += 0.0613;
        }
        assert!(worst <= 2e-6, "worst cos error {worst}");
        let mut worst = 0.0f64;
        let mut x = -100.0f32;
        while x <= 100.0 {
            worst = worst.max((f64::from(cos_approx_f32(x)) - f64::from(x).cos()).abs());
            x += 0.0071;
        }
        assert!(worst <= 1e-6, "worst cos error (|x|<=100) {worst}");
    }

    #[test]
    fn bessel_values() {
        // Reference values from tables / high-precision evaluation.
        assert!((bessel_i0_f64(1.0) - 1.2660658777520082).abs() < 1e-15);
        assert!((bessel_i0_f64(5.0) / 27.239871823604442 - 1.0).abs() < 1e-14);
        assert!((bessel_i0_f64(10.0) / 2815.716628466254 - 1.0).abs() < 1e-14);
        assert_eq!(bessel_i0_f64(-3.0), bessel_i0_f64(3.0));
    }

    fn ref_cos_sum(n: usize, a0: f64, a1: f64, a2: f64) -> std::vec::Vec<f64> {
        (0..n)
            .map(|i| {
                let p = 2.0 * PI * i as f64 / (n - 1) as f64;
                a0 - a1 * p.cos() + a2 * (2.0 * p).cos()
            })
            .collect()
    }

    fn close(a: &[f32], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((f64::from(*x) - y).abs() <= tol, "{x} vs {y}");
        }
    }

    #[test]
    fn cosine_windows_match_f64() {
        for n in [2usize, 3, 7, 8, 9, 64, 1000, 4097] {
            let mut w = std::vec![0.0f32; n];
            hamming_window_into_f32(&mut w);
            close(&w, &ref_cos_sum(n, 0.54, 0.46, 0.0), 3e-6);
            hanning_window_into_f32(&mut w);
            close(&w, &ref_cos_sum(n, 0.5, 0.5, 0.0), 3e-6);
            blackman_window_into_f32(&mut w);
            close(&w, &ref_cos_sum(n, 0.42, 0.5, 0.08), 3e-6);
        }
    }

    #[test]
    fn kaiser_matches_f64_and_symmetric() {
        for (n, beta) in [(2usize, 5.0f32), (9, 0.0), (64, 8.6), (201, 14.0)] {
            let mut w = std::vec![0.0f32; n];
            kaiser_window_into_f32(&mut w, beta);
            let b = f64::from(beta);
            for (i, x) in w.iter().enumerate() {
                let r = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
                let e = bessel_i0_f64(b * (1.0 - r * r).max(0.0).sqrt()) / bessel_i0_f64(b);
                assert!((f64::from(*x) - e).abs() < 1e-6);
                assert!((x - w[n - 1 - i]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn degenerate_sizes() {
        let mut e: [f32; 0] = [];
        hamming_window_into_f32(&mut e);
        kaiser_window_into_f32(&mut e, 3.0);
        let mut one = [0.0f32];
        blackman_window_into_f32(&mut one);
        assert_eq!(one, [1.0]);
        kaiser_window_into_f32(&mut one, 3.0);
        assert_eq!(one, [1.0]);
    }

    #[test]
    #[should_panic]
    fn kaiser_negative_beta() {
        kaiser_window_into_f32(&mut [0.0; 4], -1.0);
    }

    #[cfg(feature = "alloc")]
    #[test]
    fn alloc_variants_match_into() {
        let mut w = std::vec![0.0f32; 33];
        hanning_window_into_f32(&mut w);
        assert_eq!(hanning_window_f32(33), w);
        kaiser_window_into_f32(&mut w, 6.0);
        assert_eq!(kaiser_window_f32(33, 6.0), w);
        assert!(hamming_window_f32(0).is_empty());
        assert_eq!(blackman_window_f32(10).len(), 10);
    }

    proptest! {
        #[test]
        fn apply_matches_scalar(
            v in prop::collection::vec((-1000.0f32..1000.0, -2.0f32..2.0), 0..100),
        ) {
            let (mut a, w): (std::vec::Vec<f32>, std::vec::Vec<f32>) = v.into_iter().unzip();
            let mut b = a.clone();
            apply_window_f32(&mut a, &w);
            apply_window_scalar_f32(&mut b, &w);
            prop_assert_eq!(a, b);
        }

        #[test]
        fn simd_apply_matches_scalar(
            v in prop::collection::vec(-100.0f32..100.0, 32),
        ) {
            let mut d = [F32x8::from_slice(&v[0..8]), F32x8::from_slice(&v[8..16])];
            let w = [F32x8::from_slice(&v[16..24]), F32x8::from_slice(&v[24..32])];
            let mut flat: std::vec::Vec<f32> = v[0..16].to_vec();
            apply_window_simd_f32(&mut d, &w);
            apply_window_scalar_f32(&mut flat, &v[16..32]);
            for i in 0..16 {
                prop_assert_eq!(d[i / 8][i % 8].to_bits(), flat[i].to_bits());
            }
        }
    }

    #[test]
    #[should_panic]
    fn apply_len_mismatch() {
        apply_window_f32(&mut [1.0; 3], &[1.0; 2]);
    }
}
