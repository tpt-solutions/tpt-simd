//! The portable `Simd` and `SimdMask` types that the other crates build on.
//!
//! Run with: `cargo run -p tpt-simd-vector --example vector`

use tpt_simd_vector::Simd;

fn main() {
    // Arithmetic is lane-wise.
    let a = Simd::<f32, 4>::from_array([1.0, 2.0, 3.0, 4.0]);
    let c = a * Simd::splat(2.0) + Simd::splat(1.0);
    assert_eq!(c.to_array(), [3.0, 5.0, 7.0, 9.0]);

    // Comparisons produce a SimdMask; select picks per lane.
    let m = c.simd_gt(Simd::splat(4.0));
    assert_eq!(m.count(), 3);
    let s = m.select(c, Simd::zero());
    assert_eq!(s.to_array(), [0.0, 5.0, 7.0, 9.0]);

    // Integer arithmetic wraps; sat_* saturates.
    let x = Simd::<i16, 8>::splat(i16::MAX);
    assert_eq!((x + Simd::splat(1))[0], i16::MIN);
    assert_eq!(x.sat_add(Simd::splat(1))[0], i16::MAX);

    println!("vector: all checks passed");
}
