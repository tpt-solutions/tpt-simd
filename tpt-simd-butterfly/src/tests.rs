use super::*;
use proptest::prelude::*;
use tpt_simd_core::{F32x8, I16x16};

fn simd_of<T: tpt_simd_core::SimdElement + core::fmt::Debug, const N: usize>(
    elem: impl Strategy<Value = T>,
) -> impl Strategy<Value = Simd<T, N>> {
    proptest::collection::vec(elem, N).prop_map(|v| Simd::from_slice(&v))
}
use tpt_simd_testutil::{approx_eq_f32, f32_with_specials, i16_edgy};

fn same_bits(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

#[test]
fn appendix_b5() {
    let mut a = F32x8::from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let mut b = F32x8::from_array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    butterfly_f32(&mut a, &mut b);
    assert_eq!(a.to_array(), [9.0; 8]);
    assert_eq!(b.to_array(), [-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0]);
}

#[test]
fn f32_specials() {
    let mut a = F32x8::from_array([
        f32::NAN,
        f32::INFINITY,
        f32::INFINITY,
        f32::MAX,
        0.0,
        -0.0,
        1.0,
        2.0,
    ]);
    let mut b = F32x8::from_array([
        1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        -0.0,
        -0.0,
        f32::NAN,
        2.0,
    ]);
    let (ea, eb) = (a, b);
    butterfly_f32(&mut a, &mut b);
    for i in 0..8 {
        assert!(same_bits(a[i], ea[i] + eb[i]));
        assert!(same_bits(b[i], ea[i] - eb[i]));
    }
    assert!(a[0].is_nan() && b[1].is_nan() && a[3].is_infinite());
}

#[test]
fn i16_wrap_and_sat() {
    let mut a = I16x16::from_fn(|i| if i % 2 == 0 { i16::MAX } else { i16::MIN });
    let mut b = I16x16::splat(1);
    let (mut a2, mut b2) = (a, b);
    butterfly_i16(&mut a, &mut b);
    butterfly_i16_saturating(&mut a2, &mut b2);
    assert_eq!(a[0], i16::MIN);
    assert_eq!(a2[0], i16::MAX);
    assert_eq!(b[1], i16::MAX); // MIN - 1 wraps
    assert_eq!(b2[1], i16::MIN);
}

proptest! {
    #[test]
    fn prop_f32(a in simd_of::<f32, 8>(f32_with_specials()), b in simd_of::<f32, 8>(f32_with_specials())) {
        let (mut x, mut y) = (a, b);
        butterfly_f32(&mut x, &mut y);
        for i in 0..8 {
            prop_assert!(same_bits(x[i], a[i] + b[i]));
            prop_assert!(same_bits(y[i], a[i] - b[i]));
        }
    }

    #[test]
    fn prop_i16(a in simd_of::<i16, 16>(i16_edgy()), b in simd_of::<i16, 16>(i16_edgy())) {
        let (mut x, mut y) = (a, b);
        let (mut xs, mut ys) = (a, b);
        butterfly_i16(&mut x, &mut y);
        butterfly_i16_saturating(&mut xs, &mut ys);
        for i in 0..16 {
            prop_assert_eq!(x[i], a[i].wrapping_add(b[i]));
            prop_assert_eq!(y[i], a[i].wrapping_sub(b[i]));
            prop_assert_eq!(xs[i], a[i].saturating_add(b[i]));
            prop_assert_eq!(ys[i], a[i].saturating_sub(b[i]));
        }
    }

    #[test]
    fn prop_twiddle(
        ar in simd_of::<f32, 8>(f32_with_specials()), ai in simd_of::<f32, 8>(f32_with_specials()),
        br in simd_of::<f32, 8>(f32_with_specials()), bi in simd_of::<f32, 8>(f32_with_specials()),
        wr in simd_of::<f32, 8>(f32_with_specials()), wi in simd_of::<f32, 8>(f32_with_specials()),
    ) {
        let mut a = ComplexSimd::new(ar, ai);
        let mut b = ComplexSimd::new(br, bi);
        butterfly_with_twiddle_complex_f32(&mut a, &mut b, ComplexSimd::new(wr, wi));
        for i in 0..8 {
            let tr = br[i] * wr[i] - bi[i] * wi[i];
            let ti = br[i] * wi[i] + bi[i] * wr[i];
            prop_assert!(same_bits(a.real[i], ar[i] + tr));
            prop_assert!(same_bits(a.imag[i], ai[i] + ti));
            prop_assert!(same_bits(b.real[i], ar[i] - tr));
            prop_assert!(same_bits(b.imag[i], ai[i] - ti));
        }
        let (mut x, mut y) = (ar, br);
        butterfly_with_twiddle_f32(&mut x, &mut y, wr);
        for i in 0..8 {
            prop_assert!(same_bits(x[i], ar[i] + br[i] * wr[i]));
            prop_assert!(same_bits(y[i], ar[i] - br[i] * wr[i]));
        }
    }
}

/// 8-point radix-2 DIT FFT built from 12 butterflies, one SIMD butterfly per
/// stage (each stage processes the 4 butterflies of one stage as lanes; the
/// 8-lane vectors are padded with a duplicate half).
#[test]
fn fft8_vs_dft() {
    let x: [(f32, f32); 8] = [
        (1.0, 0.0),
        (2.0, -1.0),
        (0.5, 0.25),
        (-3.0, 2.0),
        (4.0, 4.0),
        (0.0, -2.0),
        (1.5, 1.5),
        (-1.0, 0.75),
    ];
    // Bit-reversed input order.
    let rev = [0usize, 4, 2, 6, 1, 5, 3, 7];
    let mut re: [f32; 8] = core::array::from_fn(|i| x[rev[i]].0);
    let mut im: [f32; 8] = core::array::from_fn(|i| x[rev[i]].1);
    let mut len = 2;
    while len <= 8 {
        let half = len / 2;
        // Gather the 4 butterflies of this stage, duplicated to fill 8 lanes.
        let mut idx_a = [0usize; 4];
        let mut idx_b = [0usize; 4];
        let mut tw = [(0.0f32, 0.0f32); 4];
        let mut k = 0;
        for start in (0..8).step_by(len) {
            for j in 0..half {
                idx_a[k] = start + j;
                idx_b[k] = start + j + half;
                let ang = -2.0 * core::f64::consts::PI * j as f64 / len as f64;
                tw[k] = (ang.cos() as f32, ang.sin() as f32);
                k += 1;
            }
        }
        let g = |arr: &[f32; 8], idx: &[usize; 4]| Simd::<f32, 8>::from_fn(|l| arr[idx[l % 4]]);
        let mut a = ComplexSimd::new(g(&re, &idx_a), g(&im, &idx_a));
        let mut b = ComplexSimd::new(g(&re, &idx_b), g(&im, &idx_b));
        let w = ComplexSimd::new(
            Simd::from_fn(|l| tw[l % 4].0),
            Simd::from_fn(|l| tw[l % 4].1),
        );
        butterfly_with_twiddle_complex_f32(&mut a, &mut b, w);
        for l in 0..4 {
            re[idx_a[l]] = a.real[l];
            im[idx_a[l]] = a.imag[l];
            re[idx_b[l]] = b.real[l];
            im[idx_b[l]] = b.imag[l];
        }
        len *= 2;
    }
    for k in 0..8 {
        let (mut sr, mut si) = (0.0f64, 0.0f64);
        for (n, &(xr, xi)) in x.iter().enumerate() {
            let ang = -2.0 * core::f64::consts::PI * (k * n) as f64 / 8.0;
            let (c, s) = (ang.cos(), ang.sin());
            sr += xr as f64 * c - xi as f64 * s;
            si += xr as f64 * s + xi as f64 * c;
        }
        assert!(
            approx_eq_f32(re[k], sr as f32, 1e-5, 1e-5),
            "re[{k}] {} vs {}",
            re[k],
            sr
        );
        assert!(
            approx_eq_f32(im[k], si as f32, 1e-5, 1e-5),
            "im[{k}] {} vs {}",
            im[k],
            si
        );
    }
}

/// Same FFT using plain (non-twiddle) butterflies for the trivial stages:
/// a length-2 DFT of whole vectors is exactly `butterfly_f32`.
#[test]
fn butterfly_is_dft2() {
    let mut a = F32x8::from_fn(|i| i as f32);
    let mut b = F32x8::from_fn(|i| 10.0 + i as f32);
    butterfly_f32(&mut a, &mut b);
    for i in 0..8 {
        assert_eq!(a[i], 10.0 + 2.0 * i as f32);
        assert_eq!(b[i], -10.0);
    }
}
