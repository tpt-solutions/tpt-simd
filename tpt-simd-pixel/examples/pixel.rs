//! Pixel kernels: channel layout, colour conversion, premultiply and histograms.
//!
//! Run with: `cargo run -p tpt-simd-pixel --example pixel`

use tpt_simd_pixel::{
    histogram_u8, hsv_to_rgb, premultiply_rgba8, rgb_to_hsv, rgb8_to_rgba8, u8_to_unit_f32,
};

fn main() {
    // Add an opaque alpha channel to RGB pixels.
    let rgb = [10u8, 20, 30, 40, 50, 60];
    let mut rgba = [0u8; 8];
    rgb8_to_rgba8(&rgb, &mut rgba, 255);
    assert_eq!(rgba, [10, 20, 30, 255, 40, 50, 60, 255]);

    // Premultiply colour by alpha in place: 200 * 128/255 is about 100.
    let mut px = [200u8, 200, 200, 128];
    premultiply_rgba8(&mut px);
    assert_eq!(px[3], 128);
    assert!((px[0] as i32 - 100).abs() <= 1);

    // Normalise bytes to [0, 1] for float pipelines.
    let mut unit = [0.0f32; 2];
    u8_to_unit_f32(&[0, 255], &mut unit);
    assert_eq!(unit, [0.0, 1.0]);

    // Colour space: pure red is hue 0, full saturation, full value.
    let (h, s, v) = rgb_to_hsv(1.0, 0.0, 0.0);
    assert_eq!((h, s, v), (0.0, 1.0, 1.0));
    let (r, g, b) = hsv_to_rgb(120.0, 1.0, 1.0);
    assert_eq!((r, g, b), (0.0, 1.0, 0.0));

    // Luminance histogram of a tiny image.
    let mut bins = [0u32; 256];
    histogram_u8(&[0, 0, 255, 7], &mut bins);
    assert_eq!((bins[0], bins[7], bins[255]), (2, 1, 1));

    println!("pixel: all checks passed");
}
