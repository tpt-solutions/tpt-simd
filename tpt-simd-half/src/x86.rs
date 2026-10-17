//! F16C conversions, eight values per instruction. Used only when the CPU
//! has F16C and AVX, either enabled at compile time or detected at run time
//! (`runtime-dispatch`). Round to nearest even, the same rounding as
//! `half::f16::from_f32`.

use core::arch::x86_64::*;

/// Converts `src` to `f16` bits in `dst`, eight at a time. Returns the number
/// of elements handled; the caller converts the rest.
///
/// # Safety
/// The CPU must support F16C and AVX, and `src.len() == dst.len()` (checked by
/// the caller).
#[target_feature(enable = "f16c,avx")]
pub(crate) unsafe fn f32_to_f16(src: &[f32], dst: &mut [crate::f16]) -> usize {
    let mut i = 0;
    while i + 8 <= src.len() {
        // SAFETY: i + 8 <= len for both slices, so the 8-float load and the
        // 8-half (16-byte) store are in bounds. `f16` is a 16-bit value.
        unsafe {
            let v = _mm256_loadu_ps(src.as_ptr().add(i));
            let h = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(v);
            _mm_storeu_si128(dst.as_mut_ptr().add(i) as *mut __m128i, h);
        }
        i += 8;
    }
    i
}

/// Converts `f16` values in `src` to `f32` in `dst`, eight at a time. Exact.
/// Returns the number of elements handled.
///
/// # Safety
/// As for [`f32_to_f16`].
#[target_feature(enable = "f16c,avx")]
pub(crate) unsafe fn f16_to_f32(src: &[crate::f16], dst: &mut [f32]) -> usize {
    let mut i = 0;
    while i + 8 <= src.len() {
        // SAFETY: i + 8 <= len for both slices, so the 16-byte load and the
        // 8-float store are in bounds.
        unsafe {
            let h = _mm_loadu_si128(src.as_ptr().add(i) as *const __m128i);
            _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
        }
        i += 8;
    }
    i
}
