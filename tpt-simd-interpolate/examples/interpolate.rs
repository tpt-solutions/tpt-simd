//! Interpolation: linear and Catmull-Rom cubic, scalar and 8-lane.
//!
//! Run with: `cargo run -p tpt-simd-interpolate --example interpolate`

use tpt_simd_core::Simd;
use tpt_simd_interpolate::{
    interpolate_cubic_f32, interpolate_linear_f32, interpolate_linear_simd_f32,
};

fn main() {
    assert_eq!(interpolate_linear_f32(0.0, 10.0, 0.25), 2.5);

    // Eight independent interpolations at once.
    let a = Simd::<f32, 8>::splat(0.0);
    let b = Simd::<f32, 8>::splat(10.0);
    let t = Simd::<f32, 8>::splat(0.5);
    assert_eq!(interpolate_linear_simd_f32(a, b, t).to_array(), [5.0; 8]);

    // Catmull-Rom through the points 0, 1, 2, 3 (collinear), evaluated between the middle two.
    let p = |v: f32| Simd::<f32, 8>::splat(v);
    let r = interpolate_cubic_f32(p(0.0), p(1.0), p(2.0), p(3.0), p(0.5));
    assert_eq!(r.to_array(), [1.5; 8]);

    println!("interpolate: all checks passed");
}
