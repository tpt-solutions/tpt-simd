//! Property tests: pixel conversions against the per-pixel definitions.

use proptest::prelude::*;
use tpt_simd_pixel::{
    histogram_u8, hsv_to_rgb, interleaved_to_planar_rgb8, planar_to_interleaved_rgb8,
    premultiply_rgba8, rgb_to_hsv, rgb8_to_rgba8, rgba8_to_rgb8, u8_to_unit_f32, unit_f32_to_u8,
};

proptest! {
    #[test]
    fn rgb_to_rgba_and_back(px in prop::collection::vec(any::<[u8; 3]>(), 0..200), alpha in any::<u8>()) {
        let src: Vec<u8> = px.iter().flatten().copied().collect();
        let mut rgba = vec![0u8; px.len() * 4];
        rgb8_to_rgba8(&src, &mut rgba, alpha);
        for (i, p) in px.iter().enumerate() {
            prop_assert_eq!(&rgba[i * 4..i * 4 + 4], &[p[0], p[1], p[2], alpha][..]);
        }
        let mut back = vec![0u8; src.len()];
        rgba8_to_rgb8(&rgba, &mut back);
        prop_assert_eq!(back, src);
    }

    #[test]
    fn planar_and_interleaved_round_trip(n in 0usize..200, seed in any::<u8>()) {
        let r: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_mul(3) ^ seed).collect();
        let g: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_mul(5).wrapping_add(seed)).collect();
        let b: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_mul(7)).collect();
        let mut inter = vec![0u8; n * 3];
        planar_to_interleaved_rgb8(&r, &g, &b, &mut inter);
        for i in 0..n {
            prop_assert_eq!(&inter[i * 3..i * 3 + 3], &[r[i], g[i], b[i]][..]);
        }
        let (mut r2, mut g2, mut b2) = (vec![0u8; n], vec![0u8; n], vec![0u8; n]);
        interleaved_to_planar_rgb8(&inter, &mut r2, &mut g2, &mut b2);
        prop_assert_eq!((r2, g2, b2), (r, g, b));
    }

    #[test]
    fn u8_unit_f32_round_trips(src in prop::collection::vec(any::<u8>(), 0..300)) {
        let mut unit = vec![0f32; src.len()];
        u8_to_unit_f32(&src, &mut unit);
        for (&s, &u) in src.iter().zip(&unit) {
            prop_assert!((u - f32::from(s) / 255.0).abs() <= 1e-6);
        }
        let mut back = vec![0u8; src.len()];
        unit_f32_to_u8(&unit, &mut back);
        prop_assert_eq!(back, src);
    }

    #[test]
    fn premultiply_is_within_one_of_the_formula(px in prop::collection::vec(any::<[u8; 4]>(), 0..200)) {
        let mut buf: Vec<u8> = px.iter().flatten().copied().collect();
        premultiply_rgba8(&mut buf);
        for (i, p) in px.iter().enumerate() {
            let a = u32::from(p[3]);
            for c in 0..3 {
                let want = (u32::from(p[c]) * a) as f64 / 255.0;
                prop_assert!((f64::from(buf[i * 4 + c]) - want).abs() <= 1.0);
            }
            prop_assert_eq!(buf[i * 4 + 3], p[3]);
        }
    }

    #[test]
    fn hsv_round_trips_within_float_error(r in 0.0f32..=1.0, g in 0.0f32..=1.0, b in 0.0f32..=1.0) {
        let (h, s, v) = rgb_to_hsv(r, g, b);
        let (r2, g2, b2) = hsv_to_rgb(h, s, v);
        prop_assert!((r - r2).abs() < 1e-4 && (g - g2).abs() < 1e-4 && (b - b2).abs() < 1e-4);
    }

    #[test]
    fn histogram_counts_every_byte(src in prop::collection::vec(any::<u8>(), 0..500)) {
        let mut bins = [0u32; 256];
        histogram_u8(&src, &mut bins);
        let mut want = [0u32; 256];
        for &s in &src {
            want[usize::from(s)] += 1;
        }
        prop_assert_eq!(bins, want);
    }
}
