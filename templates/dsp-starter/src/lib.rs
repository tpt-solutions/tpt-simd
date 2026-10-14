//! Starter kernels for a DSP or codec crate on tpt-simd.
//!
//! Every SIMD kernel here has a scalar reference next to it, and the tests
//! require the two to agree. Keep that pattern when you add kernels: the
//! scalar version is what you check the fast path against.

use tpt_simd::convolve::{fir_filter_i16, fir_filter_scalar_i16};
use tpt_simd::dot::dot_product_f32;

/// Dot product of two equal-length slices, SIMD path with a scalar reference.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    dot_product_f32(a, b)
}

/// Dot product, scalar reference (strict left-to-right order).
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot_scalar: length mismatch");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Causal Q15 FIR filter. `out` must be the same length as `samples`.
pub fn fir(samples: &[i16], taps: &[i16], out: &mut [i16]) {
    assert_eq!(samples.len(), out.len(), "fir: output length mismatch");
    fir_filter_i16(samples, taps, out);
}

/// Causal Q15 FIR filter, scalar reference.
pub fn fir_scalar(samples: &[i16], taps: &[i16], out: &mut [i16]) {
    assert_eq!(samples.len(), out.len(), "fir_scalar: output length mismatch");
    fir_filter_scalar_i16(samples, taps, out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_matches_scalar_for_tail_lengths() {
        // Lengths around the vector width exercise the tail handling.
        for n in [0, 1, 7, 8, 9, 31, 64, 1001] {
            let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.25).collect();
            let b: Vec<f32> = (0..n).map(|i| 1.0 - i as f32 * 0.5).collect();
            let fast = dot(&a, &b);
            let reference = dot_scalar(&a, &b);
            let tol = 1e-4 * reference.abs().max(1.0);
            assert!((fast - reference).abs() <= tol, "n = {n}: {fast} vs {reference}");
        }
    }

    #[test]
    fn fir_is_bit_exact_against_scalar() {
        let samples: Vec<i16> = (0..257).map(|i| (i * 37 % 200) as i16).collect();
        let taps: [i16; 5] = [4096, 8192, 12288, 8192, 4096];
        let mut fast = vec![0i16; samples.len()];
        let mut reference = vec![0i16; samples.len()];
        fir(&samples, &taps, &mut fast);
        fir_scalar(&samples, &taps, &mut reference);
        assert_eq!(fast, reference);
    }
}
