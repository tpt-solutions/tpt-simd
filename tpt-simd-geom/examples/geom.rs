//! Batched 3D geometry over structure-of-arrays data: distances, AABB overlap and ray casts.
//!
//! Run with: `cargo run -p tpt-simd-geom --example geom`

use tpt_simd_geom::{BoxesSoa, Soa3, aabb_overlap, distance_sq};

fn main() {
    // Points stored as three separate axis arrays (structure-of-arrays).
    let x = [0.0f32, 3.0, 1.0];
    let y = [0.0f32, 4.0, 0.0];
    let z = [0.0f32, 0.0, 0.0];
    let pts = Soa3::new(&x, &y, &z);

    // Squared distance from every point to the origin.
    let mut d2 = [0.0f32; 3];
    distance_sq(pts, [0.0, 0.0, 0.0], &mut d2);
    assert_eq!(d2, [0.0, 25.0, 1.0]);

    // Which boxes overlap the query box?
    let (bmin_x, bmin_y, bmin_z) = ([0.0f32, 10.0], [0.0f32, 10.0], [0.0f32, 10.0]);
    let (bmax_x, bmax_y, bmax_z) = ([1.0f32, 11.0], [1.0f32, 11.0], [1.0f32, 11.0]);
    let boxes = BoxesSoa::new(
        Soa3::new(&bmin_x, &bmin_y, &bmin_z),
        Soa3::new(&bmax_x, &bmax_y, &bmax_z),
    );
    let mut hit = [false; 2];
    aabb_overlap(boxes, [0.5, 0.5, 0.5], [2.0, 2.0, 2.0], &mut hit);
    assert_eq!(hit, [true, false]);

    println!("geom: all checks passed");
}
