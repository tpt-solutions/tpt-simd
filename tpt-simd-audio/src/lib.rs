//! Audio kernels on `f32` samples.
//!
//! | function / type | notes |
//! |---|---|
//! | [`mix_into`] | `dst += gain * src` |
//! | [`interleave`], [`deinterleave`] | planar <-> interleaved |
//! | [`peak`], [`rms`] | level of a buffer |
//! | [`fir`] | causal direct-form FIR, zero history |
//! | [`Biquad`] | RBJ low-pass / high-pass, direct form II transposed, `f64` state |
//! | [`resample_sinc`] | windowed-sinc (Blackman) resampling |
//!
//! Length rules are checked up front and cause a **panic** on mismatch.
//!
//! ## Example
//!
//! ```
//! use tpt_simd_audio::{Biquad, peak};
//!
//! let mut x = [1.0f32, 0.0, 0.0, 0.0];
//! let mut f = Biquad::lowpass(48_000.0, 1_000.0, core::f64::consts::FRAC_1_SQRT_2);
//! f.process(&mut x);
//! assert!(peak(&x) > 0.0);
//! ```
#![no_std]
#![deny(unsafe_code)]
#![warn(missing_docs)]

use core::f64::consts::PI;

fn check(func: &str, a: usize, b: usize) {
    if a != b {
        panic!("{func}: lengths differ ({a} vs {b})");
    }
}

/// `dst[i] += gain * src[i]`.
///
/// # Panics
///
/// Panics if the slices differ in length.
pub fn mix_into(dst: &mut [f32], src: &[f32], gain: f32) {
    check("mix_into", dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d += gain * s;
    }
}

/// Interleaves equal-length channels into `out`, frame by frame.
///
/// `out.len()` must be `channels.len() * frames`.
///
/// # Panics
///
/// Panics if the channels differ in length or `out` has the wrong length.
pub fn interleave(channels: &[&[f32]], out: &mut [f32]) {
    let frames = channels.first().map_or(0, |c| c.len());
    for c in channels {
        check("interleave (channel lengths)", c.len(), frames);
    }
    check("interleave (output)", out.len(), channels.len() * frames);
    let nch = channels.len();
    for (f, frame) in out.chunks_exact_mut(nch.max(1)).take(frames).enumerate() {
        for (slot, ch) in frame.iter_mut().zip(channels) {
            *slot = ch[f];
        }
    }
}

/// Splits interleaved `input` into equal-length channels.
///
/// `input.len()` must be `channels.len() * frames`, where every channel slice
/// has length `frames`.
///
/// # Panics
///
/// Panics if the channels differ in length or `input` has the wrong length.
pub fn deinterleave(input: &[f32], channels: &mut [&mut [f32]]) {
    let frames = channels.first().map_or(0, |c| c.len());
    for c in channels.iter() {
        check("deinterleave (channel lengths)", c.len(), frames);
    }
    check("deinterleave (input)", input.len(), channels.len() * frames);
    let nch = channels.len();
    if nch == 0 {
        return;
    }
    for (f, frame) in input.chunks_exact(nch).enumerate() {
        for (ch, &s) in channels.iter_mut().zip(frame) {
            ch[f] = s;
        }
    }
}

/// Largest absolute sample value. Zero for an empty buffer.
#[must_use]
pub fn peak(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |m, &s| m.max(s.abs()))
}

/// Root mean square of the buffer. Zero for an empty buffer.
#[must_use]
pub fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let sum: f64 = x.iter().map(|&s| f64::from(s) * f64::from(s)).sum();
    libm::sqrt(sum / x.len() as f64) as f32
}

/// Causal FIR filter with zero history: `y[n] = sum_k taps[k] * x[n - k]`.
///
/// `y` must have the same length as `x`. Samples before the start of `x` are
/// treated as zero.
///
/// # Panics
///
/// Panics if `x.len() != y.len()`.
pub fn fir(taps: &[f32], x: &[f32], y: &mut [f32]) {
    check("fir", x.len(), y.len());
    for (n, out) in y.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (k, &t) in taps.iter().enumerate().take(n + 1) {
            acc += t * x[n - k];
        }
        *out = acc;
    }
}

/// Second-order IIR section (RBJ cookbook), direct form II transposed.
///
/// Coefficients are normalised so that `a0 = 1`. State is kept in `f64`.
#[derive(Clone, Copy, Debug)]
pub struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    fn from_rbj(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Low-pass with corner frequency `f0` Hz and quality `q`.
    ///
    /// # Panics
    ///
    /// Panics unless `0 < f0 < sample_rate / 2` and `q > 0`.
    #[must_use]
    pub fn lowpass(sample_rate: f64, f0: f64, q: f64) -> Self {
        let (cw, alpha, _) = rbj_params(sample_rate, f0, q);
        Self::from_rbj(
            (1.0 - cw) / 2.0,
            1.0 - cw,
            (1.0 - cw) / 2.0,
            1.0 + alpha,
            -2.0 * cw,
            1.0 - alpha,
        )
    }

    /// High-pass with corner frequency `f0` Hz and quality `q`.
    ///
    /// # Panics
    ///
    /// Panics unless `0 < f0 < sample_rate / 2` and `q > 0`.
    #[must_use]
    pub fn highpass(sample_rate: f64, f0: f64, q: f64) -> Self {
        let (cw, alpha, _) = rbj_params(sample_rate, f0, q);
        Self::from_rbj(
            (1.0 + cw) / 2.0,
            -(1.0 + cw),
            (1.0 + cw) / 2.0,
            1.0 + alpha,
            -2.0 * cw,
            1.0 - alpha,
        )
    }

    /// Clears the filter state.
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Filters one sample.
    #[inline]
    pub fn tick(&mut self, x: f32) -> f32 {
        let x = f64::from(x);
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y as f32
    }

    /// Filters a buffer in place.
    pub fn process(&mut self, x: &mut [f32]) {
        for s in x.iter_mut() {
            *s = self.tick(*s);
        }
    }
}

fn rbj_params(sample_rate: f64, f0: f64, q: f64) -> (f64, f64, f64) {
    assert!(
        sample_rate > 0.0 && f0 > 0.0 && f0 < sample_rate / 2.0,
        "Biquad: need 0 < f0 < sample_rate / 2"
    );
    assert!(q > 0.0, "Biquad: q must be positive");
    let w0 = 2.0 * PI * f0 / sample_rate;
    let alpha = libm::sin(w0) / (2.0 * q);
    (libm::cos(w0), alpha, w0)
}

/// Filters interleaved audio with one biquad per channel.
///
/// `samples` holds frames of `filters.len()` samples (`[c0, c1, ..., c0, c1, ...]`), and
/// channel `c` is processed by `filters[c]`. Each filter keeps its own state, so calls can
/// be split at frame boundaries and the result equals processing each channel on its own.
///
/// # Panics
///
/// Panics if `filters` is empty or `samples.len()` is not a multiple of `filters.len()`.
pub fn process_interleaved(filters: &mut [Biquad], samples: &mut [f32]) {
    assert!(!filters.is_empty(), "process_interleaved: no filters");
    assert!(
        samples.len().is_multiple_of(filters.len()),
        "process_interleaved: length is not a whole number of frames"
    );
    for frame in samples.chunks_exact_mut(filters.len()) {
        for (filter, sample) in filters.iter_mut().zip(frame.iter_mut()) {
            *sample = filter.tick(*sample);
        }
    }
}

/// Blackman window value at position `t` in `[0, 1]`.
fn blackman(t: f64) -> f64 {
    0.42 - 0.5 * libm::cos(2.0 * PI * t) + 0.08 * libm::cos(4.0 * PI * t)
}

/// Resamples `input` from `in_rate` to `out_rate` with a windowed-sinc kernel.
///
/// `half_width` is the number of input samples on each side of the kernel
/// centre (typically 16 to 64); larger values give a sharper cutoff at more
/// cost. When downsampling, the cutoff is lowered to the new Nyquist frequency
/// to limit aliasing. Samples outside `input` are treated as zero.
///
/// Returns the number of output samples written, which is
/// `floor(input.len() * out_rate / in_rate)`, capped at `out.len()`.
///
/// # Panics
///
/// Panics if a rate is not positive, or `half_width` is zero.
pub fn resample_sinc(
    input: &[f32],
    in_rate: f64,
    out_rate: f64,
    half_width: usize,
    out: &mut [f32],
) -> usize {
    assert!(
        in_rate > 0.0 && out_rate > 0.0,
        "resample_sinc: rates must be positive"
    );
    assert!(half_width > 0, "resample_sinc: half_width must be non-zero");
    let ratio = out_rate / in_rate;
    let total = (input.len() as f64 * ratio) as usize;
    let count = total.min(out.len());
    // Cutoff relative to the input Nyquist: lower when downsampling.
    let cutoff = ratio.min(1.0);
    let hw = half_width as f64;
    for (n, slot) in out.iter_mut().enumerate().take(count) {
        // Position of this output sample in input coordinates.
        let pos = n as f64 / ratio;
        let centre = libm::floor(pos) as isize;
        let mut acc = 0.0f64;
        let lo = centre - half_width as isize + 1;
        let hi = centre + half_width as isize;
        for i in lo..=hi {
            if i < 0 || i as usize >= input.len() {
                continue;
            }
            let d = pos - i as f64;
            let x = PI * cutoff * d;
            let sinc = if x.abs() < 1e-12 {
                1.0
            } else {
                libm::sin(x) / x
            };
            let win_t = (d / hw + 1.0) / 2.0;
            let win = if (0.0..=1.0).contains(&win_t) {
                blackman(win_t)
            } else {
                0.0
            };
            acc += f64::from(input[i as usize]) * cutoff * sinc * win;
        }
        *slot = acc as f32;
    }
    count
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec;
    use std::vec::Vec;

    #[test]
    fn mix_adds_scaled_source() {
        let mut d = [1.0f32, 2.0, 3.0];
        mix_into(&mut d, &[1.0, 1.0, 1.0], 0.5);
        assert_eq!(d, [1.5, 2.5, 3.5]);
    }

    #[test]
    fn interleave_round_trip() {
        let l = [1.0f32, 2.0, 3.0];
        let r = [-1.0f32, -2.0, -3.0];
        let mut out = [0.0f32; 6];
        interleave(&[&l, &r], &mut out);
        assert_eq!(out, [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]);
        let mut l2 = [0.0f32; 3];
        let mut r2 = [0.0f32; 3];
        deinterleave(&out, &mut [&mut l2, &mut r2]);
        assert_eq!(l2, l);
        assert_eq!(r2, r);
    }

    #[test]
    fn peak_and_rms() {
        assert_eq!(peak(&[]), 0.0);
        assert_eq!(peak(&[0.5, -2.0, 1.0]), 2.0);
    }

    // libm::sqrt uses inline asm on x86, which Miri does not support.
    #[test]
    #[cfg_attr(miri, ignore = "libm::sqrt uses inline asm on x86")]
    fn rms_matches_expected() {
        assert_eq!(rms(&[]), 0.0);
        assert!((rms(&[3.0, -3.0, 3.0, -3.0]) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn fir_impulse_returns_taps() {
        let taps = [0.5f32, 0.25, 0.125];
        let mut x = [0.0f32; 5];
        x[0] = 1.0;
        let mut y = [0.0f32; 5];
        fir(&taps, &x, &mut y);
        assert_eq!(y, [0.5, 0.25, 0.125, 0.0, 0.0]);
    }

    #[test]
    fn lowpass_passes_dc_and_attenuates_nyquist() {
        let fs = 48_000.0;
        let mut lp = Biquad::lowpass(fs, 1_000.0, core::f64::consts::FRAC_1_SQRT_2);
        let mut dc = vec![1.0f32; 4000];
        lp.process(&mut dc);
        assert!((dc[3999] - 1.0).abs() < 1e-3);

        lp.reset();
        let mut nyq: Vec<f32> = (0..4000)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        lp.process(&mut nyq);
        assert!(peak(&nyq[2000..]) < 0.01);
    }

    #[test]
    fn highpass_blocks_dc() {
        let mut hp = Biquad::highpass(48_000.0, 1_000.0, core::f64::consts::FRAC_1_SQRT_2);
        let mut dc = vec![1.0f32; 8000];
        hp.process(&mut dc);
        assert!(peak(&dc[6000..]) < 1e-3);
    }

    #[test]
    #[cfg_attr(
        miri,
        ignore = "4800-sample sinc sweep takes ~75s under Miri; numeric check, not UB-relevant"
    )]
    fn resample_preserves_sine_amplitude_and_frequency() {
        let fs_in = 48_000.0;
        let fs_out = 24_000.0;
        let f = 1_000.0;
        let input: Vec<f32> = (0..4800)
            .map(|i| libm::sin(2.0 * PI * f * i as f64 / fs_in) as f32)
            .collect();
        let mut out = vec![0.0f32; 2400];
        let n = resample_sinc(&input, fs_in, fs_out, 32, &mut out);
        assert_eq!(n, 2400);
        // Away from the edges, the amplitude should be close to 1.
        let mid = &out[200..2200];
        assert!((peak(mid) - 1.0).abs() < 0.02, "peak {}", peak(mid));
        // Zero crossings should be spaced by half a period (12 output samples at 24 kHz / 1 kHz).
        let crossings = mid.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count();
        let expected = (2000.0 * f / fs_out) as usize;
        assert!(
            (crossings as isize - expected as isize).abs() <= 1,
            "crossings {crossings}"
        );
    }

    #[test]
    #[should_panic(expected = "interleave (output): lengths differ (5 vs 6)")]
    fn interleave_wrong_output_length_panics() {
        let l = [1.0f32, 2.0, 3.0];
        let r = [1.0f32, 2.0, 3.0];
        let mut out = [0.0f32; 5];
        interleave(&[&l, &r], &mut out);
    }
}
