//! SIMD convolution and FIR primitives.
//!
//! Implemented:
//! * [`convolve_1d_f32`] — full 1-D convolution (`len = input + kernel - 1`),
//!   zero extension at both edges.
//! * [`convolve_2d_separable_f32`] — "same"-size separable 2-D convolution on
//!   a row-major `&[f32]` image (`width` x `height`), zero padding.
//! * [`fir_filter_i16`] — causal Q15 FIR filter with round-half-up and
//!   saturation to `i16`.
//!
//! Each has a scalar reference (`*_scalar_*`) that performs the same
//! floating-point operations in the same order, so results are bit-identical.
//!
//! ## Accumulation order
//!
//! Every output sample is `fma(kernel[k], input[n-k], acc)` accumulated over
//! `k` in ascending order, starting from `0.0`. The SIMD paths vectorise
//! across *outputs* (an "axpy" per kernel tap), which keeps that order.
#![no_std]
#![deny(missing_docs)]

#[cfg(any(feature = "std", test))]
extern crate std;

use tpt_simd_core::Simd;

type F32x8 = Simd<f32, 8>;

/// Number of output samples [`convolve_1d_f32`] produces:
/// `input_len + kernel_len - 1`, or `0` when either operand is empty.
///
/// ```
/// use tpt_simd_convolve::convolve_1d_output_len;
/// assert_eq!(convolve_1d_output_len(10, 3), 12);
/// assert_eq!(convolve_1d_output_len(0, 3), 0);
/// ```
#[inline]
pub const fn convolve_1d_output_len(input_len: usize, kernel_len: usize) -> usize {
    if input_len == 0 || kernel_len == 0 {
        0
    } else {
        input_len + kernel_len - 1
    }
}

/// `dst[i] = fma(src[i], k, dst[i])` for all `i` (equal lengths).
#[inline]
fn axpy(dst: &mut [f32], src: &[f32], k: f32) {
    debug_assert_eq!(dst.len(), src.len());
    let kv = F32x8::splat(k);
    let mut d = dst.chunks_exact_mut(8);
    let mut s = src.chunks_exact(8);
    for (dc, sc) in (&mut d).zip(&mut s) {
        let r = F32x8::from_slice(sc).mul_add(kv, F32x8::from_slice(dc));
        r.copy_to_slice(dc);
    }
    for (x, y) in d.into_remainder().iter_mut().zip(s.remainder()) {
        *x = libm::fmaf(*y, k, *x);
    }
}

/// Full 1-D convolution: `output[n] = sum_k kernel[k] * input[n - k]`, where
/// `input` is treated as zero outside `0..input.len()`.
///
/// Output length policy: `output.len()` must equal
/// [`convolve_1d_output_len`]`(input.len(), kernel.len())`, i.e.
/// `input.len() + kernel.len() - 1` ("full" mode), or `0` if either input is
/// empty. To obtain a "same"-size result take
/// `output[(kernel.len() - 1) / 2..][..input.len()]`.
///
/// Performance: one fused multiply-add stream of 8 lanes per kernel tap;
/// best for kernels much shorter than the input.
///
/// # Panics
/// If `output.len()` is not the length given above.
///
/// ```
/// use tpt_simd_convolve::convolve_1d_f32;
/// let mut out = [0.0; 4];
/// convolve_1d_f32(&[1.0, 2.0, 3.0], &[1.0, 1.0], &mut out);
/// assert_eq!(out, [1.0, 3.0, 5.0, 3.0]);
/// ```
pub fn convolve_1d_f32(input: &[f32], kernel: &[f32], output: &mut [f32]) {
    assert_eq!(
        output.len(),
        convolve_1d_output_len(input.len(), kernel.len()),
        "convolve_1d_f32: output length must be input + kernel - 1"
    );
    output.fill(0.0);
    if input.is_empty() {
        return;
    }
    for (k, &kv) in kernel.iter().enumerate() {
        axpy(&mut output[k..k + input.len()], input, kv);
    }
}

/// Scalar reference for [`convolve_1d_f32`] (bit-identical results).
///
/// # Panics
/// If `output.len()` is not `input.len() + kernel.len() - 1` (or `0`).
pub fn convolve_1d_scalar_f32(input: &[f32], kernel: &[f32], output: &mut [f32]) {
    assert_eq!(
        output.len(),
        convolve_1d_output_len(input.len(), kernel.len()),
        "convolve_1d_scalar_f32: output length must be input + kernel - 1"
    );
    for (n, o) in output.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (k, &kv) in kernel.iter().enumerate() {
            if n >= k && n - k < input.len() {
                acc = libm::fmaf(kv, input[n - k], acc);
            }
        }
        *o = acc;
    }
}

/// Index range `lo..hi` of `0..len` for which `x + shift` is also in `0..len`.
#[inline]
fn tap_range(len: usize, shift: isize) -> Option<(usize, usize)> {
    let len_i = len as isize;
    let lo = (-shift).max(0);
    let hi = (len_i - shift).min(len_i);
    if lo >= hi {
        None
    } else {
        Some((lo as usize, hi as usize))
    }
}

/// Separable 2-D convolution of a row-major image, "same" output size.
///
/// * `input` and `output` are `width * height` samples, row-major
///   (`index = y * width + x`).
/// * `kernel_h` is applied along each row, `kernel_v` down each column.
///   Each kernel's centre is index `len / 2` (for even lengths the upper of
///   the two middle taps), so
///   `out[y][x] = sum_j kv[j] * (sum_i kh[i] * in[y + cv - j][x + ch - i])`.
/// * Samples outside the image are zero (zero-padding edge policy).
/// * `scratch` holds the horizontal-pass result and must have at least
///   `width * height` elements (this keeps the crate allocation-free).
///
/// The horizontal pass runs first (into `scratch`), then the vertical pass
/// (into `output`).
///
/// # Panics
/// If `input.len()`, `output.len()` is not `width * height`, if
/// `scratch.len() < width * height`, or if either kernel is empty.
///
/// ```
/// use tpt_simd_convolve::convolve_2d_separable_f32;
/// // 3x3 image, identity kernels leave it unchanged.
/// let img = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
/// let (mut out, mut tmp) = ([0.0; 9], [0.0; 9]);
/// convolve_2d_separable_f32(&img, 3, 3, &[1.0], &[1.0], &mut out, &mut tmp);
/// assert_eq!(out, img);
/// ```
pub fn convolve_2d_separable_f32(
    input: &[f32],
    width: usize,
    height: usize,
    kernel_h: &[f32],
    kernel_v: &[f32],
    output: &mut [f32],
    scratch: &mut [f32],
) {
    let n = check_2d(input, width, height, kernel_h, kernel_v, output, scratch);
    let tmp = &mut scratch[..n];
    tmp.fill(0.0);
    output.fill(0.0);
    if n == 0 {
        return;
    }
    let ch = (kernel_h.len() / 2) as isize;
    for (src_row, dst_row) in input.chunks_exact(width).zip(tmp.chunks_exact_mut(width)) {
        for (i, &kv) in kernel_h.iter().enumerate() {
            let shift = ch - i as isize;
            if let Some((lo, hi)) = tap_range(width, shift) {
                let s = (lo as isize + shift) as usize;
                axpy(&mut dst_row[lo..hi], &src_row[s..s + (hi - lo)], kv);
            }
        }
    }
    let cv = (kernel_v.len() / 2) as isize;
    for (j, &kv) in kernel_v.iter().enumerate() {
        let shift = cv - j as isize;
        if let Some((lo, hi)) = tap_range(height, shift) {
            let s = (lo as isize + shift) as usize;
            axpy(
                &mut output[lo * width..hi * width],
                &tmp[s * width..(s + hi - lo) * width],
                kv,
            );
        }
    }
}

fn check_2d(
    input: &[f32],
    width: usize,
    height: usize,
    kernel_h: &[f32],
    kernel_v: &[f32],
    output: &[f32],
    scratch: &[f32],
) -> usize {
    let n = width.checked_mul(height).expect("image size overflow");
    assert_eq!(input.len(), n, "input must be width * height");
    assert_eq!(output.len(), n, "output must be width * height");
    assert!(
        scratch.len() >= n,
        "scratch must be at least width * height"
    );
    assert!(
        !kernel_h.is_empty() && !kernel_v.is_empty(),
        "kernels must be non-empty"
    );
    n
}

/// Scalar reference for [`convolve_2d_separable_f32`] (bit-identical
/// results; same panics).
pub fn convolve_2d_separable_scalar_f32(
    input: &[f32],
    width: usize,
    height: usize,
    kernel_h: &[f32],
    kernel_v: &[f32],
    output: &mut [f32],
    scratch: &mut [f32],
) {
    let n = check_2d(input, width, height, kernel_h, kernel_v, output, scratch);
    let ch = (kernel_h.len() / 2) as isize;
    let cv = (kernel_v.len() / 2) as isize;
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f32;
            for (i, &kv) in kernel_h.iter().enumerate() {
                let sx = x as isize + ch - i as isize;
                if (0..width as isize).contains(&sx) {
                    acc = libm::fmaf(kv, input[y * width + sx as usize], acc);
                }
            }
            scratch[y * width + x] = acc;
        }
    }
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f32;
            for (j, &kv) in kernel_v.iter().enumerate() {
                let sy = y as isize + cv - j as isize;
                if (0..height as isize).contains(&sy) {
                    acc = libm::fmaf(kv, scratch[sy as usize * width + x], acc);
                }
            }
            output[y * width + x] = acc;
        }
    }
    let _ = n;
}

/// Round-half-up Q15 rescale of a 64-bit accumulator, saturated to `i16`.
#[inline]
fn q15_round_sat(acc: i64) -> i16 {
    ((acc + (1 << 14)) >> 15).clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16
}

/// Causal Q15 FIR filter.
///
/// `output[n] = sat_i16((sum_k coefficients[k] * input[n - k] + 2^14) >> 15)`
/// with `input[m] = 0` for `m < 0`.
///
/// * Coefficients are Q15 fixed point (`32768 == 1.0`, so `0x7FFF` is just
///   under 1.0 and `i16::MIN` is -1.0).
/// * The accumulator is 64-bit, so it cannot overflow for any realistic tap
///   count (`> 2^32` taps would be needed).
/// * Rounding: round half up (add `2^14`, arithmetic shift right by 15, i.e.
///   ties toward +infinity).
/// * Saturation: the rescaled value is clamped to `i16::MIN..=i16::MAX`.
///
/// `output.len()` must equal `input.len()`; with no coefficients the output
/// is all zeros.
///
/// # Panics
/// If `output.len() != input.len()`.
///
/// ```
/// use tpt_simd_convolve::fir_filter_i16;
/// let mut out = [0i16; 4];
/// // 0.5 * x[n] + 0.5 * x[n-1]
/// fir_filter_i16(&[100, 200, 300, 400], &[16384, 16384], &mut out);
/// assert_eq!(out, [50, 150, 250, 350]);
/// ```
pub fn fir_filter_i16(input: &[i16], coefficients: &[i16], output: &mut [i16]) {
    assert_eq!(
        input.len(),
        output.len(),
        "fir_filter_i16: output length must equal input length"
    );
    type I64x4 = Simd<i64, 4>;
    let taps = coefficients.len();
    // Edge region (history not fully available) handled by the scalar path.
    let head = taps.saturating_sub(1).min(input.len());
    fir_scalar_range(input, coefficients, output, 0..head);
    let mut n = head;
    // Interior: vectorise across 4 outputs; every tap read is in range.
    while n + 4 <= input.len() && taps > 0 {
        let mut acc = I64x4::zero();
        for (k, &c) in coefficients.iter().enumerate() {
            let x = Simd::<i16, 4>::from_slice(&input[n - k..]).map(i64::from);
            acc += x * I64x4::splat(i64::from(c));
        }
        let r = acc.map(q15_round_sat);
        r.copy_to_slice(&mut output[n..]);
        n += 4;
    }
    fir_scalar_range(input, coefficients, output, n..input.len());
}

fn fir_scalar_range(
    input: &[i16],
    coefficients: &[i16],
    output: &mut [i16],
    r: core::ops::Range<usize>,
) {
    for n in r {
        let mut acc = 0i64;
        for (k, &c) in coefficients.iter().enumerate() {
            if n >= k {
                acc += i64::from(c) * i64::from(input[n - k]);
            }
        }
        output[n] = q15_round_sat(acc);
    }
}

/// Scalar reference for [`fir_filter_i16`] (identical results).
///
/// # Panics
/// If `output.len() != input.len()`.
pub fn fir_filter_scalar_i16(input: &[i16], coefficients: &[i16], output: &mut [i16]) {
    assert_eq!(
        input.len(),
        output.len(),
        "output length must equal input length"
    );
    fir_scalar_range(input, coefficients, output, 0..input.len());
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::vec;
    use std::vec::Vec;

    #[test]
    fn conv1d_known() {
        let mut o = [0.0; 4];
        convolve_1d_f32(&[1.0, 2.0, 3.0], &[1.0, 1.0], &mut o);
        assert_eq!(o, [1.0, 3.0, 5.0, 3.0]);
    }

    #[test]
    fn conv1d_empty() {
        convolve_1d_f32(&[], &[1.0], &mut []);
        convolve_1d_f32(&[1.0], &[], &mut []);
    }

    #[test]
    #[should_panic]
    fn conv1d_bad_len() {
        convolve_1d_f32(&[1.0], &[1.0], &mut [0.0; 2]);
    }

    proptest! {
        #[test]
        fn conv1d_matches_scalar(
            x in prop::collection::vec(-100.0f32..100.0, 0..70),
            k in prop::collection::vec(-4.0f32..4.0, 0..20),
        ) {
            let n = convolve_1d_output_len(x.len(), k.len());
            let (mut a, mut b) = (vec![0.0; n], vec![0.0; n]);
            convolve_1d_f32(&x, &k, &mut a);
            convolve_1d_scalar_f32(&x, &k, &mut b);
            prop_assert_eq!(a, b);
        }

        #[test]
        fn conv2d_matches_scalar(
            (w, h) in (1usize..14, 1usize..14),
            kh in prop::collection::vec(-2.0f32..2.0, 1..7),
            kv in prop::collection::vec(-2.0f32..2.0, 1..7),
            seed in prop::collection::vec(-50.0f32..50.0, 196),
        ) {
            let img: Vec<f32> = seed[..w * h].to_vec();
            let (mut a, mut b) = (vec![0.0; w * h], vec![0.0; w * h]);
            let mut t = vec![0.0; w * h];
            convolve_2d_separable_f32(&img, w, h, &kh, &kv, &mut a, &mut t);
            convolve_2d_separable_scalar_f32(&img, w, h, &kh, &kv, &mut b, &mut t);
            prop_assert_eq!(a, b);
        }

        #[test]
        fn fir_matches_scalar(
            x in prop::collection::vec(any::<i16>(), 0..60),
            c in prop::collection::vec(any::<i16>(), 0..12),
        ) {
            let (mut a, mut b) = (vec![0i16; x.len()], vec![0i16; x.len()]);
            fir_filter_i16(&x, &c, &mut a);
            fir_filter_scalar_i16(&x, &c, &mut b);
            prop_assert_eq!(a, b);
        }
    }

    #[test]
    fn conv2d_box_blur_values() {
        // 3x3 box blur of a single impulse in a 3x3 image.
        let mut img = [0.0f32; 9];
        img[4] = 9.0;
        let (mut o, mut t) = ([0.0; 9], [0.0; 9]);
        convolve_2d_separable_f32(&img, 3, 3, &[1.0; 3], &[1.0; 3], &mut o, &mut t);
        assert_eq!(o, [9.0; 9]);
    }

    #[test]
    fn conv2d_zero_size() {
        convolve_2d_separable_f32(&[], 0, 5, &[1.0], &[1.0], &mut [], &mut []);
    }

    #[test]
    fn fir_rounding_and_saturation() {
        let mut o = [0i16; 3];
        // 0.5 * 1 = 0.5 rounds half up to 1; 0.5 * -1 = -0.5 rounds to 0.
        fir_filter_i16(&[1, -1, 0], &[16384], &mut o);
        assert_eq!(o, [1, 0, 0]);
        // -1.0 * -32768 = +32768 saturates to 32767.
        let mut o = [0i16; 1];
        fir_filter_i16(&[i16::MIN], &[i16::MIN], &mut o);
        assert_eq!(o, [i16::MAX]);
        // Sum of two large taps saturates negative.
        let mut o = [0i16; 2];
        fir_filter_i16(&[i16::MAX, i16::MAX], &[i16::MIN, i16::MIN], &mut o);
        assert_eq!(o, [-32767, i16::MIN]);
    }

    #[test]
    fn fir_identity() {
        let x: Vec<i16> = (0..37).map(|i| (i * 311 - 5000) as i16).collect();
        let mut o = vec![0i16; x.len()];
        // 32767/32768 is not exact identity; use delay-free 0x7FFF and check closeness.
        fir_filter_i16(&x, &[i16::MAX], &mut o);
        for (a, b) in x.iter().zip(&o) {
            assert!((i32::from(*a) - i32::from(*b)).abs() <= 1);
        }
    }
}
