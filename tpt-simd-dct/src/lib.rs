//! Table-based DCT and MDCT for codec-sized blocks.
//!
//! | type | transform | input | output |
//! |---|---|---|---|
//! | [`Dct`] | orthonormal DCT-II (`forward`) | N | N |
//! | [`Dct`] | orthonormal DCT-III, the inverse (`inverse`) | N | N |
//! | [`Mdct`] | MDCT (`forward`) | 2N | N |
//! | [`Mdct`] | IMDCT (`inverse`) | N | 2N |
//!
//! ## Cost
//!
//! Each transform is a matrix-vector product with a precomputed cosine table,
//! so it costs O(N²) per block. The inner loops run over contiguous table rows
//! and vectorise well, but this is not an FFT. Use it for block sizes up to a
//! few thousand.
//!
//! ## Perfect reconstruction (MDCT)
//!
//! Apply the same window (for example [`sine_window`]) at analysis and at
//! synthesis, hop by N samples, and overlap-add the windowed IMDCT outputs.
//! Samples covered by two frames are then reconstructed exactly up to rounding.
//!
//! ## Length policy
//!
//! Buffer lengths must match the transform size; otherwise the function
//! **panics**.
//!
//! ## Example
//!
//! ```
//! use tpt_simd_dct::Dct;
//!
//! let dct = Dct::new(4);
//! let mut coeffs = [0.0f32; 4];
//! dct.forward(&[1.0; 4], &mut coeffs);
//! assert!((coeffs[0] - 2.0).abs() < 1e-6);
//! assert!(coeffs[1..].iter().all(|c| c.abs() < 1e-6));
//! ```
#![no_std]
#![deny(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::PI;

#[cold]
#[inline(never)]
fn size_mismatch(func: &str, what: &str, expected: usize, got: usize) -> ! {
    panic!("{func}: {what} length {got}, expected {expected}");
}

/// Sine window of length `len`: `w[i] = sin(pi (i + 0.5) / len)`.
///
/// For an MDCT with N coefficients, use `len = 2N`. It satisfies the
/// Princen-Bradley condition `w[i]^2 + w[i + N]^2 = 1`.
#[must_use]
pub fn sine_window(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| libm::sin(PI * (i as f64 + 0.5) / len as f64) as f32)
        .collect()
}

/// Orthonormal DCT-II of size `n`, with its inverse DCT-III.
#[derive(Clone, Debug)]
pub struct Dct {
    n: usize,
    /// `table[k * n + i] = s_k * cos(pi / n * (i + 1/2) * k)`, with
    /// `s_0 = sqrt(1/n)` and `s_k = sqrt(2/n)` for `k > 0`.
    table: Vec<f32>,
}

impl Dct {
    /// Creates a transform of size `n`. Panics if `n` is zero.
    #[must_use]
    pub fn new(n: usize) -> Self {
        assert!(n != 0, "Dct::new: size must be non-zero");
        let mut table = vec![0.0f32; n * n];
        for k in 0..n {
            let s = if k == 0 {
                libm::sqrt(1.0 / n as f64)
            } else {
                libm::sqrt(2.0 / n as f64)
            };
            for i in 0..n {
                let angle = PI / n as f64 * (i as f64 + 0.5) * k as f64;
                table[k * n + i] = (s * libm::cos(angle)) as f32;
            }
        }
        Self { n, table }
    }

    /// Transform size.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the size is zero (never, by construction).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Orthonormal DCT-II: `out = C * input`.
    ///
    /// # Panics
    ///
    /// Panics if either slice does not have length [`Dct::len`].
    pub fn forward(&self, input: &[f32], out: &mut [f32]) {
        self.check("Dct::forward", input, out);
        for (k, o) in out.iter_mut().enumerate() {
            let row = &self.table[k * self.n..(k + 1) * self.n];
            *o = row.iter().zip(input).map(|(&c, &x)| c * x).sum();
        }
    }

    /// Inverse of [`Dct::forward`] (DCT-III): `out = C^T * input`.
    ///
    /// # Panics
    ///
    /// Panics if either slice does not have length [`Dct::len`].
    pub fn inverse(&self, input: &[f32], out: &mut [f32]) {
        self.check("Dct::inverse", input, out);
        out.fill(0.0);
        for (k, &x) in input.iter().enumerate() {
            let row = &self.table[k * self.n..(k + 1) * self.n];
            for (o, &c) in out.iter_mut().zip(row) {
                *o += c * x;
            }
        }
    }

    fn check(&self, func: &str, input: &[f32], out: &[f32]) {
        if input.len() != self.n {
            size_mismatch(func, "input", self.n, input.len());
        }
        if out.len() != self.n {
            size_mismatch(func, "output", self.n, out.len());
        }
    }
}

/// MDCT with N coefficients per frame and a window length of 2N.
#[derive(Clone, Debug)]
pub struct Mdct {
    n: usize,
    /// `table[k * 2n + i] = cos(pi / n * (i + 1/2 + n/2) * (k + 1/2))`.
    table: Vec<f32>,
}

impl Mdct {
    /// Creates an MDCT with `n` coefficients. Panics if `n` is zero.
    #[must_use]
    pub fn new(n: usize) -> Self {
        assert!(n != 0, "Mdct::new: size must be non-zero");
        let len = 2 * n;
        let mut table = vec![0.0f32; n * len];
        let half = n as f64 / 2.0;
        for k in 0..n {
            for i in 0..len {
                let angle = PI / n as f64 * (i as f64 + 0.5 + half) * (k as f64 + 0.5);
                table[k * len + i] = libm::cos(angle) as f32;
            }
        }
        Self { n, table }
    }

    /// Number of coefficients per frame (N). The frame has 2N samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if N is zero (never, by construction).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Forward MDCT: `out[k] = sum_i input[i] * cos(...)`, with `input` of length
    /// 2N (window it first) and `out` of length N.
    ///
    /// # Panics
    ///
    /// Panics if `input.len() != 2 * len()` or `out.len() != len()`.
    pub fn forward(&self, input: &[f32], out: &mut [f32]) {
        let len = 2 * self.n;
        if input.len() != len {
            size_mismatch("Mdct::forward", "input", len, input.len());
        }
        if out.len() != self.n {
            size_mismatch("Mdct::forward", "output", self.n, out.len());
        }
        for (k, o) in out.iter_mut().enumerate() {
            let row = &self.table[k * len..(k + 1) * len];
            *o = row.iter().zip(input).map(|(&c, &x)| c * x).sum();
        }
    }

    /// Inverse MDCT: `out[i] = (2 / N) * sum_k coeffs[k] * cos(...)`, producing
    /// 2N aliased samples. Window them and overlap-add with the neighbouring
    /// frames to cancel the aliasing.
    ///
    /// # Panics
    ///
    /// Panics if `coeffs.len() != len()` or `out.len() != 2 * len()`.
    pub fn inverse(&self, coeffs: &[f32], out: &mut [f32]) {
        let len = 2 * self.n;
        if coeffs.len() != self.n {
            size_mismatch("Mdct::inverse", "coefficients", self.n, coeffs.len());
        }
        if out.len() != len {
            size_mismatch("Mdct::inverse", "output", len, out.len());
        }
        out.fill(0.0);
        let scale = 2.0 / self.n as f32;
        for (k, &x) in coeffs.iter().enumerate() {
            let row = &self.table[k * len..(k + 1) * len];
            for (o, &c) in out.iter_mut().zip(row) {
                *o += c * x * scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec;

    fn signal(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| libm::sinf(i as f32 * 0.37) + 0.1 * (i % 7) as f32)
            .collect()
    }

    #[test]
    #[cfg_attr(miri, ignore = "libm x86 routine uses inline asm, which Miri does not support")]
    fn dct_constant_input_is_pure_dc() {
        let dct = Dct::new(4);
        let mut c = [0.0f32; 4];
        dct.forward(&[1.0; 4], &mut c);
        assert!((c[0] - 2.0).abs() < 1e-6);
        assert!(c[1..].iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn dct_round_trip_and_energy_preserved() {
        for n in [1usize, 2, 5, 8, 16, 33] {
            let dct = Dct::new(n);
            let x = signal(n);
            let mut c = vec![0.0; n];
            dct.forward(&x, &mut c);
            let mut y = vec![0.0; n];
            dct.inverse(&c, &mut y);
            for i in 0..n {
                assert!((x[i] - y[i]).abs() < 1e-4, "n={n} i={i}");
            }
            let ex: f32 = x.iter().map(|v| v * v).sum();
            let ec: f32 = c.iter().map(|v| v * v).sum();
            assert!((ex - ec).abs() <= 1e-3 * (1.0 + ex), "n={n}");
        }
    }

    #[test]
    fn mdct_tdac_reconstructs_overlap_add() {
        let n = 16;
        let mdct = Mdct::new(n);
        let w = sine_window(2 * n);
        let frames = 6;
        let total = (frames + 1) * n;
        let x = signal(total);
        let mut y = vec![0.0f32; total];
        let mut buf = vec![0.0f32; 2 * n];
        let mut coeffs = vec![0.0f32; n];
        let mut back = vec![0.0f32; 2 * n];
        for f in 0..frames {
            let start = f * n;
            for i in 0..2 * n {
                buf[i] = x[start + i] * w[i];
            }
            mdct.forward(&buf, &mut coeffs);
            mdct.inverse(&coeffs, &mut back);
            for i in 0..2 * n {
                y[start + i] += back[i] * w[i];
            }
        }
        // Samples [n, frames * n) are covered by two frames each.
        for i in n..frames * n {
            assert!((x[i] - y[i]).abs() < 1e-4, "i={i}: {} vs {}", x[i], y[i]);
        }
    }

    #[test]
    #[should_panic(expected = "Dct::forward: input length 3, expected 4")]
    fn dct_length_mismatch_panics() {
        let dct = Dct::new(4);
        let mut out = [0.0f32; 4];
        dct.forward(&[0.0; 3], &mut out);
    }

    #[test]
    #[should_panic(expected = "Mdct::forward: input length 7, expected 8")]
    fn mdct_length_mismatch_panics() {
        let m = Mdct::new(4);
        let mut out = [0.0f32; 4];
        m.forward(&[0.0; 7], &mut out);
    }
}
