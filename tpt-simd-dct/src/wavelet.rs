//! Orthonormal Haar wavelet transform, in place and multi-level.
//!
//! One level maps each pair `(x[2i], x[2i+1])` to the approximation `(x0 + x1) / sqrt(2)`
//! and the detail `(x0 - x1) / sqrt(2)`. Level `l` writes its approximations to the front of
//! the transformed range and its details after them, then recurses on the approximations.
//! The transform is orthonormal, so it preserves energy, and its inverse is exact.

use alloc::vec;

const FRAC_1_SQRT_2: f32 = core::f32::consts::FRAC_1_SQRT_2;

/// Applies `levels` levels of the orthonormal Haar transform to `data` in place.
///
/// After the transform, the first `len >> levels` values are the coarsest approximations and
/// the rest are detail coefficients, finest last.
///
/// # Panics
///
/// Panics unless `data.len()` is divisible by `2^levels`.
pub fn haar_forward(data: &mut [f32], levels: u32) {
    check(data.len(), levels, "haar_forward");
    let mut scratch = vec![0.0f32; data.len()];
    let mut len = data.len();
    for _ in 0..levels {
        let half = len / 2;
        for i in 0..half {
            let (a, b) = (data[2 * i], data[2 * i + 1]);
            scratch[i] = (a + b) * FRAC_1_SQRT_2;
            scratch[half + i] = (a - b) * FRAC_1_SQRT_2;
        }
        data[..len].copy_from_slice(&scratch[..len]);
        len = half;
    }
}

/// Inverse of [`haar_forward`] with the same `levels`, in place.
///
/// # Panics
///
/// Panics unless `data.len()` is divisible by `2^levels`.
pub fn haar_inverse(data: &mut [f32], levels: u32) {
    check(data.len(), levels, "haar_inverse");
    let mut scratch = vec![0.0f32; data.len()];
    let mut len = data.len() >> levels;
    for _ in 0..levels {
        let next_len = len * 2;
        for i in 0..len {
            let (approx, detail) = (data[i], data[len + i]);
            scratch[2 * i] = (approx + detail) * FRAC_1_SQRT_2;
            scratch[2 * i + 1] = (approx - detail) * FRAC_1_SQRT_2;
        }
        data[..next_len].copy_from_slice(&scratch[..next_len]);
        len = next_len;
    }
}

fn check(len: usize, levels: u32, what: &str) {
    assert!(
        levels < usize::BITS && len.is_multiple_of(1usize << levels),
        "{what}: length {len} is not divisible by 2^{levels}"
    );
}
