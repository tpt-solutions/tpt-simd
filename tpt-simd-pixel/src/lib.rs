//! Pixel kernels.
//!
//! | function | notes |
//! |---|---|
//! | [`rgb8_to_rgba8`], [`rgba8_to_rgb8`] | channel expansion and drop |
//! | [`planar_to_interleaved_rgb8`], [`interleaved_to_planar_rgb8`] | three 8-bit planes |
//! | [`u8_to_unit_f32`], [`unit_f32_to_u8`] | `[0, 255]` <-> `[0, 1]` |
//! | [`premultiply_rgba8`] | colour times alpha, rounded |
//! | [`rgb_to_hsv`], [`hsv_to_rgb`] | `f32`, hue in degrees `[0, 360)` |
//! | [`histogram_u8`] | 256-bin count |
//!
//! Length rules are checked up front and cause a **panic** on mismatch.
//!
//! ## Example
//!
//! ```
//! use tpt_simd_pixel::rgb8_to_rgba8;
//!
//! let rgb = [10u8, 20, 30, 40, 50, 60];
//! let mut rgba = [0u8; 8];
//! rgb8_to_rgba8(&rgb, &mut rgba, 255);
//! assert_eq!(rgba, [10, 20, 30, 255, 40, 50, 60, 255]);
//! ```
#![no_std]
// `deny` rather than `forbid`: the entry points carry `#[allow(unsafe_code)]`
// for the SSE2 kernels, and builds without them contain no `unsafe` at all.
#![deny(unsafe_code)]
#![warn(missing_docs)]

#[cfg(all(not(feature = "scalar-only"), target_arch = "x86_64"))]
#[allow(unsafe_code)]
mod x86;

fn check(func: &str, a: usize, b: usize) {
    if a != b {
        panic!("{func}: lengths differ ({a} vs {b})");
    }
}

/// Expands RGB8 pixels to RGBA8, setting alpha to `alpha`.
///
/// # Panics
///
/// Panics unless `src.len() / 3 * 4 == dst.len()` and `src.len()` is a multiple of 3.
pub fn rgb8_to_rgba8(src: &[u8], dst: &mut [u8], alpha: u8) {
    check("rgb8_to_rgba8 (multiple of 3)", src.len() % 3, 0);
    check("rgb8_to_rgba8", dst.len(), src.len() / 3 * 4);
    for (out, px) in dst.chunks_exact_mut(4).zip(src.chunks_exact(3)) {
        out[0] = px[0];
        out[1] = px[1];
        out[2] = px[2];
        out[3] = alpha;
    }
}

/// Drops the alpha channel of RGBA8 pixels.
///
/// # Panics
///
/// Panics unless `src.len()` is a multiple of 4 and `dst.len() == src.len() / 4 * 3`.
pub fn rgba8_to_rgb8(src: &[u8], dst: &mut [u8]) {
    check("rgba8_to_rgb8 (multiple of 4)", src.len() % 4, 0);
    check("rgba8_to_rgb8", dst.len(), src.len() / 4 * 3);
    for (out, px) in dst.chunks_exact_mut(3).zip(src.chunks_exact(4)) {
        out[0] = px[0];
        out[1] = px[1];
        out[2] = px[2];
    }
}

/// Interleaves three planes of equal length into RGB8.
///
/// # Panics
///
/// Panics if the planes differ in length, or `out.len() != 3 * plane length`.
pub fn planar_to_interleaved_rgb8(r: &[u8], g: &[u8], b: &[u8], out: &mut [u8]) {
    check("planar_to_interleaved_rgb8 (g)", g.len(), r.len());
    check("planar_to_interleaved_rgb8 (b)", b.len(), r.len());
    check("planar_to_interleaved_rgb8", out.len(), 3 * r.len());
    for (i, px) in out.chunks_exact_mut(3).enumerate() {
        px[0] = r[i];
        px[1] = g[i];
        px[2] = b[i];
    }
}

/// Splits RGB8 pixels into three planes of equal length.
///
/// # Panics
///
/// Panics unless `input.len()` is a multiple of 3 and each plane has length
/// `input.len() / 3`.
pub fn interleaved_to_planar_rgb8(input: &[u8], r: &mut [u8], g: &mut [u8], b: &mut [u8]) {
    check(
        "interleaved_to_planar_rgb8 (multiple of 3)",
        input.len() % 3,
        0,
    );
    let n = input.len() / 3;
    check("interleaved_to_planar_rgb8 (r)", r.len(), n);
    check("interleaved_to_planar_rgb8 (g)", g.len(), n);
    check("interleaved_to_planar_rgb8 (b)", b.len(), n);
    for (i, px) in input.chunks_exact(3).enumerate() {
        r[i] = px[0];
        g[i] = px[1];
        b[i] = px[2];
    }
}

/// Converts `u8` samples to `f32` in `[0, 1]` (divides by 255).
///
/// # Panics
///
/// Panics if the slices differ in length.
pub fn u8_to_unit_f32(src: &[u8], dst: &mut [f32]) {
    check("u8_to_unit_f32", src.len(), dst.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = f32::from(s) / 255.0;
    }
}

/// Converts `f32` samples in `[0, 1]` to `u8`, rounding and clamping.
///
/// # Panics
///
/// Panics if the slices differ in length.
pub fn unit_f32_to_u8(src: &[f32], dst: &mut [u8]) {
    check("unit_f32_to_u8", src.len(), dst.len());
    for (d, &s) in dst.iter_mut().zip(src) {
        // NaN compares false, so it clamps to 0.
        let v = s * 255.0;
        *d = if v >= 255.0 {
            255
        } else if v > 0.0 {
            // Round half up; `v` is in (0, 255), so the cast cannot overflow.
            (v + 0.5) as u8
        } else {
            0
        };
    }
}

/// Multiplies RGB by alpha for each RGBA8 pixel, in place, rounding to nearest.
///
/// Alpha itself is unchanged. The result for channel `c` and alpha `a` is
/// `round(c * a / 255)`.
///
/// # Panics
///
/// Panics unless `buf.len()` is a multiple of 4.
#[allow(unsafe_code)]
pub fn premultiply_rgba8(buf: &mut [u8]) {
    check("premultiply_rgba8 (multiple of 4)", buf.len() % 4, 0);
    #[cfg(all(not(feature = "scalar-only"), target_arch = "x86_64"))]
    // SAFETY: SSE2 is part of the x86_64 baseline; the kernel stays in bounds.
    let done = unsafe { x86::premultiply_rgba8(buf) };
    #[cfg(any(feature = "scalar-only", not(target_arch = "x86_64")))]
    let done = 0;
    // The SIMD part ends on a pixel boundary, so the tail is whole pixels.
    for px in buf[done..].chunks_exact_mut(4) {
        let a = u32::from(px[3]);
        for c in &mut px[..3] {
            // Exact integer rounding: t = c*a + 128, then (t + (t >> 8)) >> 8.
            let t = u32::from(*c) * a + 128;
            *c = ((t + (t >> 8)) >> 8) as u8;
        }
    }
}

/// `x` modulo `m`, in `[0, m)` for positive `m`. (`f32::rem_euclid` needs `std`.)
#[inline]
fn wrap_positive(x: f32, m: f32) -> f32 {
    let r = x % m;
    if r < 0.0 { r + m } else { r }
}

/// Converts RGB in `[0, 1]` to HSV. Returns `(h, s, v)` with `h` in degrees
/// `[0, 360)`, and `s`, `v` in `[0, 1]`. Grey inputs give hue 0.
#[must_use]
pub fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;
    let v = max;
    let s = if max == 0.0 { 0.0 } else { delta / max };
    let h = if delta == 0.0 {
        0.0
    } else if max == r {
        60.0 * wrap_positive((g - b) / delta, 6.0)
    } else if max == g {
        60.0 * ((b - r) / delta + 2.0)
    } else {
        60.0 * ((r - g) / delta + 4.0)
    };
    (h, s, v)
}

/// Converts HSV (hue in degrees, `s` and `v` in `[0, 1]`) to RGB in `[0, 1]`.
#[must_use]
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h = wrap_positive(h, 360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - (wrap_positive(h, 2.0) - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (r + m, g + m, b + m)
}

/// Counts occurrences of each byte value.
pub fn histogram_u8(src: &[u8], bins: &mut [u32; 256]) {
    for &s in src {
        bins[usize::from(s)] += 1;
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec;
    use std::vec::Vec;

    #[test]
    fn rgb_rgba_round_trip() {
        let rgb: Vec<u8> = (0..30u8).collect();
        let mut rgba = vec![0u8; 40];
        rgb8_to_rgba8(&rgb, &mut rgba, 9);
        assert_eq!(rgba[3], 9);
        let mut back = vec![0u8; 30];
        rgba8_to_rgb8(&rgba, &mut back);
        assert_eq!(back, rgb);
    }

    #[test]
    fn planar_round_trip() {
        let r: Vec<u8> = (0..7).collect();
        let g: Vec<u8> = (10..17).collect();
        let b: Vec<u8> = (20..27).collect();
        let mut inter = vec![0u8; 21];
        planar_to_interleaved_rgb8(&r, &g, &b, &mut inter);
        assert_eq!(&inter[..6], &[0, 10, 20, 1, 11, 21]);
        let (mut r2, mut g2, mut b2) = (vec![0u8; 7], vec![0u8; 7], vec![0u8; 7]);
        interleaved_to_planar_rgb8(&inter, &mut r2, &mut g2, &mut b2);
        assert_eq!((r2, g2, b2), (r, g, b));
    }

    #[test]
    fn unit_conversion_round_trip() {
        let src: Vec<u8> = (0..=255).collect();
        let mut f = vec![0.0f32; 256];
        u8_to_unit_f32(&src, &mut f);
        let mut back = vec![0u8; 256];
        unit_f32_to_u8(&f, &mut back);
        assert_eq!(back, src);
        let mut clamp = [0u8; 3];
        unit_f32_to_u8(&[-1.0, 2.0, f32::NAN], &mut clamp);
        assert_eq!(clamp, [0, 255, 0]);
    }

    #[test]
    fn premultiply_matches_reference_rounding() {
        for c in 0..=255u32 {
            for a in 0..=255u32 {
                let mut px = [c as u8, c as u8, c as u8, a as u8];
                premultiply_rgba8(&mut px);
                let want = (c as f64 * a as f64 / 255.0).round() as i32;
                assert!((i32::from(px[0]) - want).abs() <= 1, "c={c} a={a}");
                assert_eq!(px[3], a as u8);
            }
        }
        let mut opaque = [10u8, 20, 30, 255];
        premultiply_rgba8(&mut opaque);
        assert_eq!(opaque, [10, 20, 30, 255]);
        let mut clear = [10u8, 20, 30, 0];
        premultiply_rgba8(&mut clear);
        assert_eq!(clear, [0, 0, 0, 0]);
    }

    #[test]
    fn hsv_known_colours() {
        assert_eq!(rgb_to_hsv(1.0, 0.0, 0.0), (0.0, 1.0, 1.0));
        let (h, _, _) = rgb_to_hsv(0.0, 1.0, 0.0);
        assert!((h - 120.0).abs() < 1e-4);
        let (h, _, _) = rgb_to_hsv(0.0, 0.0, 1.0);
        assert!((h - 240.0).abs() < 1e-4);
        assert_eq!(rgb_to_hsv(0.5, 0.5, 0.5), (0.0, 0.0, 0.5));
        let (h, _, _) = rgb_to_hsv(1.0, 0.0, 0.5);
        assert!((h - 330.0).abs() < 1e-3, "h={h}");
    }

    #[test]
    fn hsv_round_trip_on_grid() {
        for ri in 0..=4 {
            for gi in 0..=4 {
                for bi in 0..=4 {
                    let (r, g, b) = (ri as f32 / 4.0, gi as f32 / 4.0, bi as f32 / 4.0);
                    let (h, s, v) = rgb_to_hsv(r, g, b);
                    let (r2, g2, b2) = hsv_to_rgb(h, s, v);
                    assert!(
                        (r - r2).abs() < 1e-5 && (g - g2).abs() < 1e-5 && (b - b2).abs() < 1e-5
                    );
                }
            }
        }
    }

    #[test]
    fn histogram_counts() {
        let mut bins = [0u32; 256];
        histogram_u8(&[0, 0, 255, 7], &mut bins);
        assert_eq!(bins[0], 2);
        assert_eq!(bins[255], 1);
        assert_eq!(bins[7], 1);
        assert_eq!(bins.iter().sum::<u32>(), 4);
    }

    #[test]
    #[should_panic(expected = "rgb8_to_rgba8: lengths differ (12 vs 8)")]
    fn rgb_to_rgba_wrong_size_panics() {
        let mut dst = [0u8; 12];
        rgb8_to_rgba8(&[0u8; 6], &mut dst, 0);
    }
}
