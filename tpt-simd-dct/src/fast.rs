//! FFT-based orthonormal DCT-II and DCT-III for power-of-two sizes.
//!
//! The transform is computed with one complex FFT of length `2N` and no `N x N` table, so
//! the cost is `O(N log N)` and the memory is `O(N)`. The FFT is zero-padded, which is simpler
//! than the packed `N`-point algorithm and costs about twice as many operations. Measured
//! forward times: 4.2 µs against 21.5 µs for the table at N = 256, and 123 µs against 9.2 ms at
//! N = 4096 (see `docs/benchmarks.md`).
//!
//! Derivation, with `M = 2N` and `s_0 = sqrt(1/N)`, `s_k = sqrt(2/N)`:
//!
//! * forward: `X_k = s_k * Re(e^{-i pi k / M} * F_k)` where `F` is the `M`-point DFT of the
//!   input zero-padded to length `M`;
//! * inverse: `y_n = Re(sum_k B_k e^{+2 pi i n k / M})` with `B_k = s_k X_k e^{i pi k / M}`
//!   for `k < N` and zero above, which is evaluated as `Re(DFT(conj(B)))`.

use alloc::vec;
use alloc::vec::Vec;
use core::f64::consts::PI;

/// Orthonormal DCT-II and DCT-III of a power-of-two size `N`, computed with an FFT.
///
/// Results agree with [`Dct`](crate::Dct) to f32 rounding. Use `Dct` for sizes that are not
/// powers of two.
#[derive(Clone, Debug)]
pub struct FastDct {
    n: usize,
    /// `cos(pi k / M)` and `sin(pi k / M)` for `k < N`.
    phase_cos: Vec<f64>,
    phase_sin: Vec<f64>,
    /// `cos(2 pi j / M)` and `sin(2 pi j / M)` for `j < M / 2`.
    twiddle_cos: Vec<f64>,
    twiddle_sin: Vec<f64>,
    /// Orthonormal scale: `s_0` at index 0, `s_k` elsewhere.
    scale: Vec<f64>,
}

impl FastDct {
    /// Creates a transform of size `n`. Panics unless `n` is a non-zero power of two.
    #[must_use]
    pub fn new(n: usize) -> Self {
        assert!(
            n.is_power_of_two(),
            "FastDct::new: size must be a power of two"
        );
        let m = 2 * n;
        let phase_cos = (0..n)
            .map(|k| libm::cos(PI * k as f64 / m as f64))
            .collect();
        let phase_sin = (0..n)
            .map(|k| libm::sin(PI * k as f64 / m as f64))
            .collect();
        let twiddle_cos = (0..m / 2)
            .map(|j| libm::cos(2.0 * PI * j as f64 / m as f64))
            .collect();
        let twiddle_sin = (0..m / 2)
            .map(|j| libm::sin(2.0 * PI * j as f64 / m as f64))
            .collect();
        let scale = (0..n)
            .map(|k| {
                if k == 0 {
                    libm::sqrt(1.0 / n as f64)
                } else {
                    libm::sqrt(2.0 / n as f64)
                }
            })
            .collect();
        Self {
            n,
            phase_cos,
            phase_sin,
            twiddle_cos,
            twiddle_sin,
            scale,
        }
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
    /// Panics if either slice does not have length [`FastDct::len`].
    pub fn forward(&self, input: &[f32], out: &mut [f32]) {
        self.check("FastDct::forward", input, out);
        let m = 2 * self.n;
        let mut re = vec![0.0f64; m];
        let mut im = vec![0.0f64; m];
        for (r, &x) in re.iter_mut().zip(input) {
            *r = f64::from(x);
        }
        self.fft(&mut re, &mut im);
        for k in 0..self.n {
            // Re(e^{-i theta} F) with theta = pi k / M is cos * Re(F) + sin * Im(F).
            let value = self.phase_cos[k] * re[k] + self.phase_sin[k] * im[k];
            out[k] = (self.scale[k] * value) as f32;
        }
    }

    /// Orthonormal DCT-III, the inverse of [`forward`](Self::forward): `out = C^T * input`.
    ///
    /// # Panics
    ///
    /// Panics if either slice does not have length [`FastDct::len`].
    pub fn inverse(&self, input: &[f32], out: &mut [f32]) {
        self.check("FastDct::inverse", input, out);
        let m = 2 * self.n;
        // Build conj(B) with B_k = a_k * e^{i theta}, a_k = s_k * X_k.
        let mut re = vec![0.0f64; m];
        let mut im = vec![0.0f64; m];
        for k in 0..self.n {
            let a = self.scale[k] * f64::from(input[k]);
            re[k] = a * self.phase_cos[k];
            im[k] = -a * self.phase_sin[k];
        }
        self.fft(&mut re, &mut im);
        for (o, &r) in out.iter_mut().zip(&re) {
            *o = r as f32;
        }
    }

    fn check(&self, what: &str, input: &[f32], out: &[f32]) {
        assert!(
            input.len() == self.n && out.len() == self.n,
            "{what}: slices must have length {}",
            self.n
        );
    }

    /// In-place forward complex FFT of length `2N` (radix-2, decimation in time).
    fn fft(&self, re: &mut [f64], im: &mut [f64]) {
        let m = re.len();
        // Bit-reversal permutation.
        let mut j = 0usize;
        for i in 1..m {
            let mut bit = m >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= m {
            let half = len / 2;
            let step = m / len;
            for start in (0..m).step_by(len) {
                for k in 0..half {
                    // Twiddle e^{-2 pi i k * step / M}.
                    let wr = self.twiddle_cos[k * step];
                    let wi = -self.twiddle_sin[k * step];
                    let (a, b) = (start + k, start + k + half);
                    let tr = wr * re[b] - wi * im[b];
                    let ti = wr * im[b] + wi * re[b];
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            len <<= 1;
        }
    }
}
