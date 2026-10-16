//! Phase 6 exit: an FFT and an MDCT built from tpt-simd kernels, validated
//! against `rustfft` and a direct f64 reference.
//!
//! * FFT: iterative radix-2 decimation-in-time transform. Each stage gathers
//!   eight butterflies into `ComplexSimd` lanes, runs
//!   `tpt_simd::butterfly::butterfly_with_twiddle_complex_f32` (twiddle
//!   multiply plus butterfly), and scatters the results back. Bit reversal is
//!   scalar.
//! * MDCT: each output is one `tpt_simd::dot::dot_product_f32` over a row of
//!   the cosine kernel.

use rustfft::{FftPlanner, num_complex::Complex};
use tpt_simd::butterfly::butterfly_with_twiddle_complex_f32;
use tpt_simd::dot::dot_product_f32;
use tpt_simd::{ComplexSimd, F32x8, Simd};

/// In-place forward FFT of `re + i*im`, length a power of two, at least 16.
fn fft_simd(re: &mut [f32], im: &mut [f32]) {
    let n = re.len();
    assert!(n.is_power_of_two() && n >= 16 && im.len() == n);

    // Bit-reversal permutation.
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        // Butterfly p in 0..n/2: block = p / half, j = p % half.
        let mut p = 0;
        while p < n / 2 {
            let mut top = [0usize; 8];
            let mut bot = [0usize; 8];
            let mut wr = [0f32; 8];
            let mut wi = [0f32; 8];
            for lane in 0..8 {
                let (block, j) = ((p + lane) / half, (p + lane) % half);
                top[lane] = block * len + j;
                bot[lane] = top[lane] + half;
                let angle = -2.0 * core::f64::consts::PI * j as f64 / len as f64;
                wr[lane] = angle.cos() as f32;
                wi[lane] = angle.sin() as f32;
            }
            let gather = |v: &[f32], idx: &[usize; 8]| -> F32x8 {
                Simd::from_array(core::array::from_fn(|l| v[idx[l]]))
            };
            let mut a = ComplexSimd::new(gather(re, &top), gather(im, &top));
            let mut b = ComplexSimd::new(gather(re, &bot), gather(im, &bot));
            let w = ComplexSimd::new(Simd::from_array(wr), Simd::from_array(wi));
            butterfly_with_twiddle_complex_f32(&mut a, &mut b, w);
            for lane in 0..8 {
                re[top[lane]] = a.real[lane];
                im[top[lane]] = a.imag[lane];
                re[bot[lane]] = b.real[lane];
                im[bot[lane]] = b.imag[lane];
            }
            p += 8;
        }
        len *= 2;
    }
}

/// MDCT with `m` outputs from `2m` inputs, using one dot product per output.
/// Window-free; kernel `cos(pi/m * (n + 1/2 + m/2) * (k + 1/2))`.
fn mdct_simd(x: &[f32]) -> Vec<f32> {
    let m = x.len() / 2;
    (0..m)
        .map(|k| {
            let row: Vec<f32> = (0..2 * m)
                .map(|n| {
                    let arg = core::f64::consts::PI / m as f64
                        * (n as f64 + 0.5 + m as f64 / 2.0)
                        * (k as f64 + 0.5);
                    arg.cos() as f32
                })
                .collect();
            dot_product_f32(&row, x)
        })
        .collect()
}

/// Deterministic test signal in [-1, 1).
fn signal(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
fn fft_matches_rustfft() {
    // Under Miri, stop at 256: the 1024-point transform dominates the test's runtime.
    let sizes: &[usize] = if cfg!(miri) {
        &[16, 32, 64, 256]
    } else {
        &[16, 32, 64, 256, 1024]
    };
    for &n in sizes {
        let x = signal(n, n as u64);
        let y = signal(n, n as u64 + 1);
        let mut re = x.clone();
        let mut im = y.clone();
        fft_simd(&mut re, &mut im);

        let mut buf: Vec<Complex<f32>> = x
            .iter()
            .zip(&y)
            .map(|(&a, &b)| Complex::new(a, b))
            .collect();
        FftPlanner::<f32>::new()
            .plan_fft_forward(n)
            .process(&mut buf);

        let scale = buf.iter().map(|c| c.norm()).fold(1.0f32, f32::max);
        for k in 0..n {
            let err = ((re[k] - buf[k].re).powi(2) + (im[k] - buf[k].im).powi(2)).sqrt();
            assert!(err <= 1e-5 * n as f32 * scale, "n={n} bin {k}: err {err}");
        }
    }
}

#[test]
fn fft_impulse_and_dc() {
    // Impulse at 0 -> flat spectrum of ones; constant -> single DC bin of n.
    let n = 64;
    let mut re = vec![0f32; n];
    let mut im = vec![0f32; n];
    re[0] = 1.0;
    fft_simd(&mut re, &mut im);
    assert!(re.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    assert!(im.iter().all(|&v| v.abs() < 1e-6));

    let mut re = vec![1f32; n];
    let mut im = vec![0f32; n];
    fft_simd(&mut re, &mut im);
    assert!((re[0] - n as f32).abs() < 1e-4);
    assert!(re[1..].iter().all(|v| v.abs() < 1e-4));
}

#[test]
fn mdct_matches_direct_f64_sum() {
    let m = 64;
    let x = signal(2 * m, 99);
    let got = mdct_simd(&x);
    for k in 0..m {
        let want: f64 = (0..2 * m)
            .map(|n| {
                let arg = core::f64::consts::PI / m as f64
                    * (n as f64 + 0.5 + m as f64 / 2.0)
                    * (k as f64 + 0.5);
                f64::from(x[n]) * arg.cos()
            })
            .sum();
        let err = (f64::from(got[k]) - want).abs();
        assert!(
            err <= 1e-4 * (2 * m) as f64,
            "k={k}: got {} want {want}",
            got[k]
        );
    }
}
