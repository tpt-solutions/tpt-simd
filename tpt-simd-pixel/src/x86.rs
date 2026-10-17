//! SSE2 kernels. SSE2 is part of the x86_64 baseline, so these are always
//! compiled on x86_64 (unless `scalar-only`). Each one matches the scalar code
//! in `lib.rs` bit for bit.

use core::arch::x86_64::*;

/// Premultiplies whole RGBA8 pixels in `buf`, two pixels per vector. Returns
/// the number of bytes handled; the caller finishes the rest with the scalar
/// code.
///
/// # Safety
/// `buf` is only read and written inside its own bounds; the CPU must support
/// SSE2 (always true on x86_64).
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn premultiply_rgba8(buf: &mut [u8]) -> usize {
    let zero = _mm_setzero_si128();
    let round = _mm_set1_epi16(128);
    // Lanes 3 and 7 are the two alpha bytes; they pass through unchanged.
    let keep_alpha = _mm_setr_epi16(0, 0, 0, -1, 0, 0, 0, -1);
    let mut i = 0;
    while i + 8 <= buf.len() {
        // SAFETY: i + 8 <= buf.len(), so the 8-byte load and store are in bounds.
        unsafe {
            let p = buf.as_mut_ptr().add(i) as *mut __m128i;
            // Eight u16 lanes: r0 g0 b0 a0 r1 g1 b1 a1.
            let w = _mm_unpacklo_epi8(_mm_loadl_epi64(p), zero);
            // Alpha of pixel 0 into lanes 0..3, alpha of pixel 1 into lanes 4..7.
            let alpha =
                _mm_shufflehi_epi16::<0b11_11_11_11>(_mm_shufflelo_epi16::<0b11_11_11_11>(w));
            // c * a <= 255 * 255 and + 128 both fit in u16, so the low 16 bits are exact.
            let t = _mm_add_epi16(_mm_mullo_epi16(w, alpha), round);
            let r = _mm_srli_epi16::<8>(_mm_add_epi16(t, _mm_srli_epi16::<8>(t)));
            let out = _mm_or_si128(
                _mm_and_si128(keep_alpha, w),
                _mm_andnot_si128(keep_alpha, r),
            );
            _mm_storel_epi64(p, _mm_packus_epi16(out, out));
        }
        i += 8;
    }
    i
}
