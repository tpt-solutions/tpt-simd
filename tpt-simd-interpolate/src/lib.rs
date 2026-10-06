//! SIMD interpolation kernels.
//!
//! Implemented:
//! * [`interpolate_linear_f32`] / [`interpolate_linear_simd_f32`]
//! * [`interpolate_cubic_f32`] (Catmull-Rom, 8 lanes) and its scalar
//!   reference [`interpolate_cubic_scalar_f32`]
//! * [`interpolate_lanczos_f32`] (Lanczos-`a` windowed sinc, 8 lanes) and its
//!   scalar reference [`interpolate_lanczos_scalar_f32`]
//! * [`sinc_f32`] / [`sinc_simd_f32`] — normalised sinc,
//!   `sin(pi x) / (pi x)`; the SIMD version uses a polynomial approximation
//!   with a documented error bound.
//!
//! Linear and cubic interpolation use FMA in a fixed order and are
//! bit-identical between the SIMD and scalar versions. Lanczos uses the
//! `sinc` approximation, so the SIMD result matches the scalar (`libm`)
//! reference to a small tolerance rather than bit-for-bit.
#![no_std]
#![deny(missing_docs)]

#[cfg(any(feature = "std", test))]
extern crate std;

use tpt_simd_core::Simd;

type F32x8 = Simd<f32, 8>;

/// Scalar linear interpolation `a + t * (b - a)` (one fused multiply-add).
///
/// Exactly `a` at `t == 0`; at `t == 1` the result is `b` up to one rounding
/// of `b - a`. `t` outside `[0, 1]` extrapolates.
///
/// ```
/// use tpt_simd_interpolate::interpolate_linear_f32;
/// assert_eq!(interpolate_linear_f32(2.0, 4.0, 0.5), 3.0);
/// ```
#[inline]
pub fn interpolate_linear_f32(a: f32, b: f32, t: f32) -> f32 {
    libm::fmaf(t, b - a, a)
}

/// 8-lane linear interpolation; lane `i` equals
/// [`interpolate_linear_f32`]`(a[i], b[i], t[i])` bit-for-bit.
///
/// ```
/// use tpt_simd_core::F32x8;
/// use tpt_simd_interpolate::interpolate_linear_simd_f32;
/// let r = interpolate_linear_simd_f32(F32x8::splat(0.0), F32x8::splat(10.0), F32x8::splat(0.25));
/// assert_eq!(r, F32x8::splat(2.5));
/// ```
#[inline]
pub fn interpolate_linear_simd_f32(a: F32x8, b: F32x8, t: F32x8) -> F32x8 {
    t.mul_add(b - a, a)
}

/// Scalar Catmull-Rom cubic: interpolates between `b` (at `t = 0`) and `c`
/// (at `t = 1`) using `a` and `d` as the outer neighbours.
///
/// Polynomial `((c3 t + c2) t + c1) t + c0` (Horner, FMA) with
/// `c0 = b`, `c1 = (c - a)/2`, `c2 = a - 5b/2 + 2c - d/2`,
/// `c3 = (d - a)/2 + 3(b - c)/2`.
///
/// ```
/// use tpt_simd_interpolate::interpolate_cubic_scalar_f32;
/// // Collinear points interpolate linearly.
/// assert_eq!(interpolate_cubic_scalar_f32(0.0, 1.0, 2.0, 3.0, 0.5), 1.5);
/// ```
#[inline]
pub fn interpolate_cubic_scalar_f32(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    let c1 = 0.5 * (c - a);
    let c2 = ((a - 2.5 * b) + 2.0 * c) - 0.5 * d;
    let c3 = 0.5 * (d - a) + 1.5 * (b - c);
    let r = libm::fmaf(c3, t, c2);
    let r = libm::fmaf(r, t, c1);
    libm::fmaf(r, t, b)
}

/// 8-lane Catmull-Rom cubic interpolation (see
/// [`interpolate_cubic_scalar_f32`]); bit-identical to it lane by lane.
///
/// ```
/// use tpt_simd_core::F32x8;
/// use tpt_simd_interpolate::interpolate_cubic_f32;
/// let s = |x| F32x8::splat(x);
/// assert_eq!(interpolate_cubic_f32(s(0.0), s(1.0), s(2.0), s(3.0), s(0.5)), s(1.5));
/// ```
#[inline]
pub fn interpolate_cubic_f32(a: F32x8, b: F32x8, c: F32x8, d: F32x8, t: F32x8) -> F32x8 {
    let s = F32x8::splat;
    let c1 = s(0.5) * (c - a);
    let c2 = ((a - s(2.5) * b) + s(2.0) * c) - s(0.5) * d;
    let c3 = s(0.5) * (d - a) + s(1.5) * (b - c);
    let r = c3.mul_add(t, c2);
    let r = r.mul_add(t, c1);
    r.mul_add(t, b)
}

/// Normalised sinc `sin(pi x) / (pi x)` (`1` at `x == 0`), computed with
/// `libm::sinf`.
///
/// ```
/// use tpt_simd_interpolate::sinc_f32;
/// assert_eq!(sinc_f32(0.0), 1.0);
/// assert!(sinc_f32(2.0).abs() < 1e-6);
/// ```
#[inline]
pub fn sinc_f32(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        return 1.0;
    }
    let px = core::f32::consts::PI * x;
    libm::sinf(px) / px
}

/// 8-lane normalised sinc using a polynomial `sin(pi r)` approximation.
///
/// Range reduction: `n = round(x)` (magic-number rounding), `r = x - n` (`|r| <= 1/2`),
/// `sin(pi x) = (-1)^n sin(pi r)`, and `sin(pi r)` is an odd degree-11 Taylor
/// polynomial in `r` (truncation error `< 6e-8`).
///
/// Maximum absolute error versus the exact sinc: `<= 2e-6` for `|x| <= 64`
/// (verified by the crate's tests); it grows slowly for larger `|x|` because
/// of `f32` range-reduction rounding. Lanes with `|x| < 1e-6` return `1`.
///
/// ```
/// use tpt_simd_core::F32x8;
/// use tpt_simd_interpolate::sinc_simd_f32;
/// let r = sinc_simd_f32(F32x8::from_array([0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0, 3.0]));
/// assert_eq!(r[0], 1.0);
/// assert!((r[1] - 2.0 / core::f32::consts::PI).abs() < 2e-6);
/// ```
#[inline]
pub fn sinc_simd_f32(x: F32x8) -> F32x8 {
    F32x8::from_array(sinc_lanes(x.to_array()))
}

/// Plain-array sinc kernel: only `+ - * /`, comparisons and bit casts in
/// fixed-trip-count lane loops (no `round`/`floor`/`fma` library calls), so
/// each loop compiles to one vector instruction per step.
#[inline(always)]
fn sinc_lanes(x: [f32; 8]) -> [f32; 8] {
    const C1: f32 = core::f32::consts::PI;
    const C3: f32 = -5.167_712_8;
    const C5: f32 = 2.550_164;
    const C7: f32 = -0.599_264_5;
    const C9: f32 = 0.082_145_89;
    const C11: f32 = -0.007_370_431;
    // 1.5 * 2^23: adding it rounds to nearest integer (ties to even) and
    // leaves the integer in the low mantissa bits (parity = bit 0).
    const MAGIC: f32 = 12_582_912.0;
    let mut out = [0.0f32; 8];
    for i in 0..8 {
        let t = x[i] + MAGIC;
        let n = t - MAGIC;
        let r = x[i] - n;
        let r2 = r * r;
        let mut p = C11;
        for c in [C9, C7, C5, C3, C1] {
            p = p * r2 + c;
        }
        let flip = (t.to_bits() & 1) << 31;
        let px = C1 * x[i];
        let tiny = x[i].abs() < 1e-6;
        // Avoid 0/0 in the unused lane.
        let denom = if tiny { 1.0 } else { px };
        let v = f32::from_bits((p * r).to_bits() ^ flip) / denom;
        out[i] = if tiny { 1.0 } else { v };
    }
    out
}

/// Scalar Lanczos interpolation with kernel size `a`.
///
/// `samples` holds `2a` consecutive equally spaced samples; `samples[i]` is
/// located at position `i - (a - 1)` relative to the sample just left of the
/// interpolation point, and `t in [0, 1)` is the fractional distance of the
/// point past that sample (so `t = 0` returns `samples[a - 1]`). See
/// [`interpolate_lanczos_f32`] for the kernel and normalisation.
///
/// # Panics
/// If `a == 0` or `samples.len() != 2 * a`.
pub fn interpolate_lanczos_scalar_f32(samples: &[f32], t: f32, a: usize) -> f32 {
    assert!(a > 0 && samples.len() == 2 * a, "need 2*a samples, a > 0");
    let af = a as f32;
    let (mut acc, mut wsum) = (0.0f32, 0.0f32);
    for (i, &s) in samples.iter().enumerate() {
        let x = t - (i as f32 - (af - 1.0));
        let w = if x.abs() < af {
            sinc_f32(x) * sinc_f32(x / af)
        } else {
            0.0
        };
        acc = libm::fmaf(w, s, acc);
        wsum += w;
    }
    acc / wsum
}

/// 8-lane Lanczos-`a` interpolation.
///
/// Kernel: `L(x) = sinc(x) * sinc(x / a)` for `|x| < a`, else `0`.
/// `samples` must contain exactly `2a` vectors (`a >= 1`); `samples[i]` sits
/// at position `i - (a - 1)` relative to the sample left of the point and
/// `t` (per lane, normally in `[0, 1)`) is the fractional offset, so the
/// weight of tap `i` is `L(t - (i - (a - 1)))`.
///
/// The weights are normalised to sum to one, which preserves constant
/// signals (DC) exactly up to rounding and compensates the truncated kernel.
/// Sinc is evaluated with [`sinc_simd_f32`] (error `<= 2e-6` per factor), so
/// results agree with [`interpolate_lanczos_scalar_f32`] to about `1e-5`
/// relative to the sample magnitude.
///
/// # Panics
/// If `a == 0` or `samples.len() != 2 * a`.
///
/// ```
/// use tpt_simd_core::F32x8;
/// use tpt_simd_interpolate::interpolate_lanczos_f32;
/// // a = 2: four samples of a constant signal stay constant.
/// let s = [F32x8::splat(3.0); 4];
/// let r = interpolate_lanczos_f32(&s, F32x8::splat(0.3), 2);
/// assert!((r[0] - 3.0).abs() < 1e-5);
/// // t = 0 returns samples[a - 1].
/// let v = [1.0, 2.0, 5.0, 7.0].map(F32x8::splat);
/// assert!((interpolate_lanczos_f32(&v, F32x8::splat(0.0), 2)[0] - 2.0).abs() < 1e-5);
/// ```
pub fn interpolate_lanczos_f32(samples: &[F32x8], t: F32x8, a: usize) -> F32x8 {
    assert!(a > 0 && samples.len() == 2 * a, "need 2*a samples, a > 0");
    let af = a as f32;
    let t = t.to_array();
    let (mut acc, mut wsum) = ([0.0f32; 8], [0.0f32; 8]);
    for (i, s) in samples.iter().enumerate() {
        let off = i as f32 - (af - 1.0);
        let mut x = [0.0f32; 8];
        let mut xa = [0.0f32; 8];
        for l in 0..8 {
            x[l] = t[l] - off;
            xa[l] = x[l] / af;
        }
        let (s1, s2) = (sinc_lanes(x), sinc_lanes(xa));
        let s = s.to_array();
        for l in 0..8 {
            let w = if x[l].abs() < af { s1[l] * s2[l] } else { 0.0 };
            acc[l] += w * s[l];
            wsum[l] += w;
        }
    }
    let mut out = [0.0f32; 8];
    for l in 0..8 {
        out[l] = acc[l] / wsum[l];
    }
    F32x8::from_array(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn v8(a: [f32; 8]) -> F32x8 {
        F32x8::from_array(a)
    }

    #[test]
    fn sinc_error_bound() {
        let mut worst = 0.0f32;
        let mut x = -64.0f32;
        while x <= 64.0 {
            let r = sinc_simd_f32(F32x8::splat(x))[0];
            let e = (r as f64 - exact_sinc(x as f64)).abs() as f32;
            worst = worst.max(e);
            x += 0.00731;
        }
        assert!(worst <= 2e-6, "worst sinc error {worst}");
    }

    fn exact_sinc(x: f64) -> f64 {
        if x == 0.0 {
            1.0
        } else {
            (core::f64::consts::PI * x).sin() / (core::f64::consts::PI * x)
        }
    }

    #[test]
    fn linear_endpoints() {
        assert_eq!(interpolate_linear_f32(1.0, 5.0, 0.0), 1.0);
        assert_eq!(interpolate_linear_f32(1.0, 5.0, 1.0), 5.0);
    }

    #[test]
    fn cubic_endpoints() {
        assert_eq!(interpolate_cubic_scalar_f32(9.0, 1.0, 2.0, -4.0, 0.0), 1.0);
        let r = interpolate_cubic_scalar_f32(9.0, 1.0, 2.0, -4.0, 1.0);
        assert!((r - 2.0).abs() < 1e-6);
    }

    #[test]
    fn lanczos_cases() {
        for a in 1..=4usize {
            let s: std::vec::Vec<F32x8> = (0..2 * a).map(|_| F32x8::splat(-7.5)).collect();
            let r = interpolate_lanczos_f32(&s, F32x8::splat(0.37), a);
            assert!((r[0] + 7.5).abs() < 1e-4);
        }
        // Linear ramp is reproduced closely by a = 3.
        let s: std::vec::Vec<F32x8> = (0..6).map(|i| F32x8::splat(i as f32)).collect();
        let r = interpolate_lanczos_f32(&s, F32x8::splat(0.5), 3);
        assert!((r[0] - 2.5).abs() < 0.05);
    }

    #[test]
    #[should_panic]
    fn lanczos_bad_len() {
        interpolate_lanczos_f32(&[F32x8::zero(); 3], F32x8::zero(), 2);
    }

    proptest! {
        #[test]
        fn linear_matches_scalar(v in prop::collection::vec(-1000.0f32..1000.0, 24)) {
            let g = |k: usize| -> [f32; 8] { v[k * 8..k * 8 + 8].try_into().unwrap() };
            let r = interpolate_linear_simd_f32(v8(g(0)), v8(g(1)), v8(g(2)));
            for l in 0..8 {
                prop_assert_eq!(r[l].to_bits(), interpolate_linear_f32(g(0)[l], g(1)[l], g(2)[l]).to_bits());
            }
        }

        #[test]
        fn cubic_matches_scalar(v in prop::collection::vec(-100.0f32..100.0, 40)) {
            let g = |k: usize| -> [f32; 8] { v[k * 8..k * 8 + 8].try_into().unwrap() };
            let r = interpolate_cubic_f32(v8(g(0)), v8(g(1)), v8(g(2)), v8(g(3)), v8(g(4)));
            for l in 0..8 {
                let e = interpolate_cubic_scalar_f32(g(0)[l], g(1)[l], g(2)[l], g(3)[l], g(4)[l]);
                prop_assert_eq!(r[l].to_bits(), e.to_bits());
            }
        }

        #[test]
        fn lanczos_matches_scalar(
            a in 1usize..5,
            t in prop::collection::vec(0.0f32..1.0, 8),
            vals in prop::collection::vec(-1.0f32..1.0, 64),
        ) {
            let samples: std::vec::Vec<F32x8> = (0..2 * a)
                .map(|i| v8(vals[i * 8..i * 8 + 8].try_into().unwrap()))
                .collect();
            let tv = v8(t.clone().try_into().unwrap());
            let r = interpolate_lanczos_f32(&samples, tv, a);
            for l in 0..8 {
                let col: std::vec::Vec<f32> = samples.iter().map(|s| s[l]).collect();
                let e = interpolate_lanczos_scalar_f32(&col, t[l], a);
                prop_assert!((r[l] - e).abs() <= 1e-4 + 1e-4 * e.abs(), "lane {} {} vs {}", l, r[l], e);
            }
        }
    }
}
