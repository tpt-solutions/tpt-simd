//! Phase 3 exit: codec-shaped data patterns, using synthetic fixtures generated
//! in code (no external sample files, so no fixture licensing question).
//!
//! * I420 to NV12 as the permute crate documents: the Y plane unchanged, then U and V
//!   interleaved as u0, v0, u1, v1, ...
//! * Interleaved 16-bit stereo PCM, round-tripped through deinterleave.
//! * Saturating i16 -> u8 packing of a synthetic residual, checked against a
//!   plain clamp.

use tpt_simd::I16x16;
use tpt_simd::permute::{
    deinterleave_stereo_i16, interleave_stereo_i16, interleave_yuv420, unpack_i8_to_i16,
};
use tpt_simd::saturate::pack_i16_to_u8;

fn frame(w: usize, h: usize, seed: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        (s >> 24) as u8
    };
    // A gradient plus noise, so the planes are not trivially compressible.
    let y: Vec<u8> = (0..w * h)
        .map(|i| ((i % w) as u8).wrapping_add(next() / 8))
        .collect();
    let u: Vec<u8> = (0..w * h / 4).map(|_| next()).collect();
    let v: Vec<u8> = (0..w * h / 4).map(|_| next()).collect();
    (y, u, v)
}

#[test]
fn yuv420_interleave_matches_documented_layout() {
    // CIF (352x288) is ~100k pixels; skip it under Miri, the smaller frames cover the layout.
    let sizes: &[(usize, usize)] = if cfg!(miri) {
        &[(16, 8), (64, 48)]
    } else {
        &[(16, 8), (64, 48), (352, 288)]
    };
    for &(w, h) in sizes {
        let (y, u, v) = frame(w, h, (w * h) as u32);
        let mut out = vec![0u8; y.len() + 2 * u.len()];
        interleave_yuv420(&y, &u, &v, &mut out);

        // NV12: Y plane unchanged, then chroma interleaved u0, v0, u1, v1, ...
        let mut want = y.clone();
        for (&uu, &vv) in u.iter().zip(&v) {
            want.push(uu);
            want.push(vv);
        }
        assert_eq!(out, want, "{w}x{h}");
    }
}

#[test]
fn stereo_pcm_round_trip() {
    let n = 4096;
    let left: Vec<i16> = (0..n)
        .map(|i| ((i as i32 * 977) % 65536 - 32768) as i16)
        .collect();
    let right: Vec<i16> = (0..n)
        .map(|i| ((i as i32 * -331) % 65536 - 32768) as i16)
        .collect();

    let mut packed = vec![0i16; 2 * n];
    interleave_stereo_i16(&left, &right, &mut packed);
    for i in 0..n {
        assert_eq!(packed[2 * i], left[i], "L{i}");
        assert_eq!(packed[2 * i + 1], right[i], "R{i}");
    }

    let (mut l2, mut r2) = (vec![0i16; n], vec![0i16; n]);
    deinterleave_stereo_i16(&packed, &mut l2, &mut r2);
    assert_eq!(l2, left);
    assert_eq!(r2, right);
}

#[test]
fn residual_pack_saturates_like_clamp() {
    // Residuals from -600 to 600 in steps, reconstructed to pixel range.
    let res: Vec<i16> = (0..256).map(|i| (i as i16 - 128) * 5).collect();
    for chunk in res.chunks_exact(16) {
        let packed = pack_i16_to_u8(I16x16::from_slice(chunk));
        for (lane, &r) in chunk.iter().enumerate() {
            assert_eq!(packed[lane], r.clamp(0, 255) as u8, "value {r}");
        }
    }
}

#[test]
fn unpack_i8_is_sign_extension() {
    let src: Vec<i8> = (-128i16..128).map(|v| v as i8).collect();
    let mut dst = vec![0i16; src.len()];
    unpack_i8_to_i16(&src, &mut dst);
    for (&s, &d) in src.iter().zip(&dst) {
        assert_eq!(d, i16::from(s));
    }
}
