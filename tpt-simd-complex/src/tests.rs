extern crate std;
use super::*;
use proptest::prelude::*;
use std::vec::Vec;
use tpt_simd_testutil::f32_with_specials;

fn same(x: f32, y: f32) -> bool {
    x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())
}

fn sfma(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

fn cv<const N: usize>(r: [f32; N], i: [f32; N]) -> ComplexSimd<f32, N> {
    ComplexSimd::new(Simd::from_array(r), Simd::from_array(i))
}

#[test]
fn appendix_b1_operator() {
    let a = ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(2.0));
    let b = ComplexSimd::new(F32x8::splat(3.0), F32x8::splat(4.0));
    let r = a * b;
    assert_eq!(r.real.to_array(), [-5.0; 8]);
    assert_eq!(r.imag.to_array(), [10.0; 8]);
}

#[test]
fn i_mul_exact() {
    let z = cv([1.0, -0.0, f32::NAN, f32::INFINITY], [2.0, 3.0, 1.0, -1.0]);
    let r = z.i_mul();
    assert_eq!(r.real.to_array()[..2], [-2.0, -3.0]);
    assert_eq!(r.imag.to_array()[..2], [1.0, -0.0]);
    let q = z.neg_i_mul().i_mul();
    assert_eq!(q.real.to_array()[..2], [1.0, -0.0]);
}

#[test]
fn from_polar_basic() {
    let z = ComplexSimd::<f32, 4>::from_polar(
        Simd::splat(2.0),
        Simd::from_array([
            0.0,
            core::f32::consts::FRAC_PI_2,
            core::f32::consts::PI,
            1.0,
        ]),
    );
    assert!((z.real[1]).abs() < 1e-6 && (z.imag[1] - 2.0).abs() < 1e-6);
    assert!((z.real[2] + 2.0).abs() < 1e-6);
    assert!((z.imag[3] - 2.0 * libm::sinf(1.0)).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "from_interleaved")]
fn from_interleaved_len_panics() {
    let _ = ComplexSimd::<f32, 4>::from_interleaved(&[0.0; 7]);
}

#[test]
#[should_panic(expected = "interleaved_to_split")]
fn deinterleave_len_panics() {
    interleaved_to_split(&[0.0; 4], &mut [0.0; 2], &mut [0.0; 1]);
}

#[test]
fn empty_slices_ok() {
    interleaved_to_split(&[], &mut [], &mut []);
    split_to_interleaved(&[], &[], &mut []);
}

#[test]
fn mul_specials_vs_scalar() {
    let (n, inf) = (f32::NAN, f32::INFINITY);
    let a = cv(
        [n, inf, 0.0, f32::MAX, 1.0, -0.0, 2.0, 1e30],
        [0.0, 1.0, inf, f32::MAX, 1.0, 0.0, -3.0, 1e30],
    );
    let b = cv(
        [1.0, 0.0, 1.0, 2.0, inf, -0.0, 0.5, 1e30],
        [1.0, 1.0, 0.0, 2.0, 0.0, 1.0, 4.0, 1e30],
    );
    let r = a.twiddle_mul_fma(b);
    for l in 0..8 {
        let (ar, ai, br, bi) = (a.real[l], a.imag[l], b.real[l], b.imag[l]);
        assert!(same(r.real[l], sfma(ar, br, -(ai * bi))));
        assert!(same(r.imag[l], sfma(ar, bi, ai * br)));
    }
}

proptest! {
    #[test]
    fn fused_ops_match_scalar(
        v in prop::collection::vec(f32_with_specials(), 48),
    ) {
        let g = |k: usize| -> [f32; 8] { v[k * 8..k * 8 + 8].try_into().unwrap() };
        let (ar, ai, br, bi, cr, ci) = (g(0), g(1), g(2), g(3), g(4), g(5));
        let (a, b, c) = (cv(ar, ai), cv(br, bi), cv(cr, ci));
        let fm = a.fmadd(b, c);
        let fs = a.fmsub(b, c);
        let mc = a.mul_conj(b);
        let ns = a.norm_sq();
        for l in 0..8 {
            prop_assert!(same(fm.real[l], sfma(ar[l], br[l], sfma(-ai[l], bi[l], cr[l]))));
            prop_assert!(same(fm.imag[l], sfma(ar[l], bi[l], sfma(ai[l], br[l], ci[l]))));
            prop_assert!(same(fs.real[l], sfma(ar[l], br[l], sfma(-ai[l], bi[l], -cr[l]))));
            prop_assert!(same(fs.imag[l], sfma(ar[l], bi[l], sfma(ai[l], br[l], -ci[l]))));
            prop_assert!(same(mc.real[l], sfma(ar[l], br[l], -(ai[l] * -bi[l]))));
            prop_assert!(same(mc.imag[l], sfma(ar[l], -bi[l], ai[l] * br[l])));
            prop_assert!(same(ns[l], sfma(ar[l], ar[l], ai[l] * ai[l])));
        }
    }

    #[test]
    fn n4_and_n8_twiddle_agree(
        v in prop::collection::vec(f32_with_specials(), 32),
    ) {
        let g = |k: usize| -> [f32; 8] { v[k * 8..k * 8 + 8].try_into().unwrap() };
        let (ar, ai, br, bi) = (g(0), g(1), g(2), g(3));
        let r8 = cv(ar, ai).twiddle_mul_fma(cv(br, bi));
        for h in 0..2 {
            let s = |x: [f32; 8]| -> [f32; 4] { x[h * 4..h * 4 + 4].try_into().unwrap() };
            let r4 = cv(s(ar), s(ai)).twiddle_mul_fma(cv(s(br), s(bi)));
            for l in 0..4 {
                prop_assert!(same(r4.real[l], r8.real[h * 4 + l]));
                prop_assert!(same(r4.imag[l], r8.imag[h * 4 + l]));
            }
        }
    }

    #[test]
    fn interleave_roundtrip(v in prop::collection::vec(f32_with_specials(), 0..40)) {
        let n = v.len();
        let aos: Vec<f32> = v.iter().flat_map(|&x| [x, -x]).collect();
        let (mut re, mut im) = (std::vec![0.0; n], std::vec![0.0; n]);
        interleaved_to_split(&aos, &mut re, &mut im);
        for i in 0..n {
            prop_assert!(same(re[i], v[i]) && same(im[i], -v[i]));
        }
        let mut back = std::vec![0.0; 2 * n];
        split_to_interleaved(&re, &im, &mut back);
        for i in 0..2 * n {
            prop_assert!(same(back[i], aos[i]));
        }
        if n >= 8 {
            let z = ComplexSimd::<f32, 8>::from_interleaved(&aos[..16]);
            let mut o = [0.0; 16];
            z.to_interleaved(&mut o);
            for i in 0..16 {
                prop_assert!(same(o[i], aos[i]));
            }
        }
    }
}

/// Radix-2 DIT FFT of size 16 where each of the 8 SIMD lanes is an
/// independent transform (lane `l` holds sample `k` of signal `l`); checked
/// against a naive O(n^2) DFT in f64.
#[test]
fn fft16_lanes_vs_naive_dft() {
    const LEN: usize = 16;
    // signals[l][k]
    let mut signals = [[(0.0f32, 0.0f32); LEN]; 8];
    for (l, sig) in signals.iter_mut().enumerate() {
        for (k, s) in sig.iter_mut().enumerate() {
            let t = (l * 31 + k * 7) as f32;
            *s = (libm::sinf(t * 0.37) * 3.0, libm::cosf(t * 0.11) - 0.5);
        }
    }
    // Bit-reversed load; element k is a ComplexSimd across 8 signals.
    let rev = |k: usize| k.reverse_bits() >> (usize::BITS - 4);
    let mut x: Vec<ComplexSimd<f32, 8>> = (0..LEN)
        .map(|k| {
            let kk = rev(k);
            cv(
                core::array::from_fn(|l| signals[l][kk].0),
                core::array::from_fn(|l| signals[l][kk].1),
            )
        })
        .collect();
    let mut len = 2;
    while len <= LEN {
        let half = len / 2;
        for start in (0..LEN).step_by(len) {
            for j in 0..half {
                let ang = -2.0 * core::f32::consts::PI * j as f32 / len as f32;
                let w = ComplexSimd::<f32, 8>::from_polar(Simd::splat(1.0), Simd::splat(ang));
                let (lo, hi) = x.split_at_mut(start + half);
                ComplexSimd::fft_butterfly(&mut lo[start + j], &mut hi[j], w);
            }
        }
        len *= 2;
    }
    for l in 0..8 {
        for m in 0..LEN {
            let (mut er, mut ei) = (0.0f64, 0.0f64);
            for (k, s) in signals[l].iter().enumerate() {
                let a = -2.0 * core::f64::consts::PI * (m * k) as f64 / LEN as f64;
                let (sn, cs) = (libm::sin(a), libm::cos(a));
                er += s.0 as f64 * cs - s.1 as f64 * sn;
                ei += s.0 as f64 * sn + s.1 as f64 * cs;
            }
            assert!((x[m].real[l] as f64 - er).abs() < 1e-3, "re l{l} m{m}");
            assert!((x[m].imag[l] as f64 - ei).abs() < 1e-3, "im l{l} m{m}");
        }
    }
}
