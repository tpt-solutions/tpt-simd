//! Batched 3D geometry over structure-of-arrays `f32` data.
//!
//! | function | input | output |
//! |---|---|---|
//! | [`distance_sq`] / [`distance`] | points, query point | squared / Euclidean distance per point |
//! | [`dot`] | points, direction | dot product per point |
//! | [`aabb_overlap`] | boxes, query box | `bool` per box |
//! | [`ray_aabb`] | boxes, ray origin and direction, `t_max` | entry distance per box, or `+inf` on a miss |
//!
//! ## Layout
//!
//! Coordinates are split into three slices, one per axis ([`Soa3`]). Boxes use
//! two of them, minimum and maximum corners ([`BoxesSoa`]).
//!
//! ## Length policy
//!
//! Slices that describe the same set of items must have the same length, and
//! output slices must match the item count. Otherwise the function **panics**.
//! Empty input is valid.
//!
//! ## Numerics
//!
//! Plain `f32` arithmetic with separate multiply and add (no FMA), so results
//! are the same on every target. NaN coordinates are not supported: a NaN in a
//! box or ray gives an unspecified hit or miss.
//!
//! ## Example
//!
//! ```
//! use tpt_simd_geom::{Soa3, distance_sq};
//!
//! let x = [0.0, 3.0];
//! let y = [0.0, 4.0];
//! let z = [0.0, 0.0];
//! let pts = Soa3::new(&x, &y, &z);
//! let mut out = [0.0f32; 2];
//! distance_sq(pts, [0.0, 0.0, 0.0], &mut out);
//! assert_eq!(out, [0.0, 25.0]);
//! ```
#![no_std]
#![deny(unsafe_code)]
#![warn(missing_docs)]

#[cold]
#[inline(never)]
fn len_mismatch(func: &str, a: usize, b: usize) -> ! {
    panic!("{func}: lengths differ ({a} vs {b})");
}

/// Borrowed structure-of-arrays view of a set of 3D vectors or points.
#[derive(Clone, Copy, Debug)]
pub struct Soa3<'a> {
    /// X coordinates.
    pub x: &'a [f32],
    /// Y coordinates.
    pub y: &'a [f32],
    /// Z coordinates.
    pub z: &'a [f32],
}

impl<'a> Soa3<'a> {
    /// Creates a view. Panics if the three slices differ in length.
    #[must_use]
    pub fn new(x: &'a [f32], y: &'a [f32], z: &'a [f32]) -> Self {
        if x.len() != y.len() {
            len_mismatch("Soa3::new (x, y)", x.len(), y.len());
        }
        if x.len() != z.len() {
            len_mismatch("Soa3::new (x, z)", x.len(), z.len());
        }
        Self { x, y, z }
    }

    /// Number of vectors in the view.
    #[must_use]
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Returns `true` if the view holds no vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

/// Axis-aligned boxes stored as SoA minimum and maximum corners.
#[derive(Clone, Copy, Debug)]
pub struct BoxesSoa<'a> {
    /// Minimum corners.
    pub min: Soa3<'a>,
    /// Maximum corners.
    pub max: Soa3<'a>,
}

impl<'a> BoxesSoa<'a> {
    /// Creates a view. Panics if the corner sets differ in length.
    #[must_use]
    pub fn new(min: Soa3<'a>, max: Soa3<'a>) -> Self {
        if min.len() != max.len() {
            len_mismatch("BoxesSoa::new", min.len(), max.len());
        }
        Self { min, max }
    }

    /// Number of boxes in the view.
    #[must_use]
    pub fn len(&self) -> usize {
        self.min.len()
    }

    /// Returns `true` if the view holds no boxes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.min.is_empty()
    }
}

fn check_out(func: &str, n: usize, out: usize) {
    if n != out {
        len_mismatch(func, n, out);
    }
}

/// Squared Euclidean distance from `q` to each point, written to `out`.
///
/// # Panics
///
/// Panics if `out.len()` differs from the point count.
pub fn distance_sq(points: Soa3<'_>, q: [f32; 3], out: &mut [f32]) {
    check_out("distance_sq", points.len(), out.len());
    let [qx, qy, qz] = q;
    for (((o, &x), &y), &z) in out.iter_mut().zip(points.x).zip(points.y).zip(points.z) {
        let dx = x - qx;
        let dy = y - qy;
        let dz = z - qz;
        *o = dx * dx + dy * dy + dz * dz;
    }
}

/// Euclidean distance from `q` to each point, written to `out`.
///
/// # Panics
///
/// Panics if `out.len()` differs from the point count.
pub fn distance(points: Soa3<'_>, q: [f32; 3], out: &mut [f32]) {
    distance_sq(points, q, out);
    for o in out.iter_mut() {
        *o = libm::sqrtf(*o);
    }
}

/// Dot product of each vector with `d`, written to `out`.
///
/// # Panics
///
/// Panics if `out.len()` differs from the vector count.
pub fn dot(vectors: Soa3<'_>, d: [f32; 3], out: &mut [f32]) {
    check_out("dot", vectors.len(), out.len());
    let [dx, dy, dz] = d;
    for (((o, &x), &y), &z) in out.iter_mut().zip(vectors.x).zip(vectors.y).zip(vectors.z) {
        *o = x * dx + y * dy + z * dz;
    }
}

/// Writes `true` for each box that overlaps the query box `[qmin, qmax]`.
///
/// Boxes touching the query box on a face or edge count as overlapping.
///
/// # Panics
///
/// Panics if `out.len()` differs from the box count.
pub fn aabb_overlap(boxes: BoxesSoa<'_>, qmin: [f32; 3], qmax: [f32; 3], out: &mut [bool]) {
    check_out("aabb_overlap", boxes.len(), out.len());
    let [qx0, qy0, qz0] = qmin;
    let [qx1, qy1, qz1] = qmax;
    let b = boxes;
    for (i, o) in out.iter_mut().enumerate() {
        *o = b.min.x[i] <= qx1
            && b.max.x[i] >= qx0
            && b.min.y[i] <= qy1
            && b.max.y[i] >= qy0
            && b.min.z[i] <= qz1
            && b.max.z[i] >= qz0;
    }
}

/// Slab test of a ray against each box.
///
/// For each box, `out_t[i]` is the distance along the ray at which it is first
/// entered, clamped to `0` when the origin is already inside. If the ray misses
/// the box, or the entry point is beyond `t_max`, the value is `f32::INFINITY`.
/// A ray with a zero direction component is treated as parallel to that axis:
/// it hits only if its origin lies within that slab.
///
/// # Panics
///
/// Panics if `out_t.len()` differs from the box count.
pub fn ray_aabb(
    boxes: BoxesSoa<'_>,
    origin: [f32; 3],
    dir: [f32; 3],
    t_max: f32,
    out_t: &mut [f32],
) {
    check_out("ray_aabb", boxes.len(), out_t.len());
    let mut inv = [0.0f32; 3];
    let mut parallel = [false; 3];
    for a in 0..3 {
        if dir[a] == 0.0 {
            parallel[a] = true;
        } else {
            inv[a] = 1.0 / dir[a];
        }
    }
    let b = boxes;
    for (i, o) in out_t.iter_mut().enumerate() {
        let mins = [b.min.x[i], b.min.y[i], b.min.z[i]];
        let maxs = [b.max.x[i], b.max.y[i], b.max.z[i]];
        let mut tnear = f32::NEG_INFINITY;
        let mut tfar = f32::INFINITY;
        let mut hit = true;
        for a in 0..3 {
            if parallel[a] {
                if origin[a] < mins[a] || origin[a] > maxs[a] {
                    hit = false;
                    break;
                }
            } else {
                let t1 = (mins[a] - origin[a]) * inv[a];
                let t2 = (maxs[a] - origin[a]) * inv[a];
                let (lo, hi) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
                if lo > tnear {
                    tnear = lo;
                }
                if hi < tfar {
                    tfar = hi;
                }
            }
        }
        *o = if hit && tnear <= tfar && tfar >= 0.0 && tnear <= t_max {
            if tnear < 0.0 { 0.0 } else { tnear }
        } else {
            f32::INFINITY
        };
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec;
    use std::vec::Vec;

    fn pts(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let x = (0..n).map(|i| (i as f32 * 0.37) - 5.0).collect();
        let y = (0..n).map(|i| (i as f32 * 0.11) + 1.0).collect();
        let z = (0..n).map(|i| ((i * 7 % 13) as f32) - 6.0).collect();
        (x, y, z)
    }

    #[test]
    fn distance_matches_scalar_for_all_tail_lengths() {
        for n in [0usize, 1, 3, 8, 9, 17] {
            let (x, y, z) = pts(n);
            let p = Soa3::new(&x, &y, &z);
            let q = [0.5, -1.0, 2.0];
            let mut d2 = vec![0.0; n];
            distance_sq(p, q, &mut d2);
            let mut d = vec![0.0; n];
            distance(p, q, &mut d);
            for i in 0..n {
                let e = (x[i] - q[0]).powi(2) + (y[i] - q[1]).powi(2) + (z[i] - q[2]).powi(2);
                assert_eq!(d2[i], e);
                assert!((d[i] - e.sqrt()).abs() <= 1e-6 * (1.0 + e.sqrt()));
            }
        }
    }

    #[test]
    fn dot_is_componentwise() {
        let x = [1.0, 0.0];
        let y = [0.0, 2.0];
        let z = [0.0, 0.0];
        let mut out = [0.0; 2];
        dot(Soa3::new(&x, &y, &z), [3.0, 4.0, 5.0], &mut out);
        assert_eq!(out, [3.0, 8.0]);
    }

    #[test]
    fn aabb_overlap_including_touching() {
        let mn = (vec![0.0f32, 10.0, 0.5], vec![0.0f32; 3], vec![0.0f32; 3]);
        let mx = (vec![1.0f32, 11.0, 2.0], vec![1.0f32; 3], vec![1.0f32; 3]);
        let boxes = BoxesSoa::new(
            Soa3::new(&mn.0, &mn.1, &mn.2),
            Soa3::new(&mx.0, &mx.1, &mx.2),
        );
        let mut out = [false; 3];
        aabb_overlap(boxes, [1.0, 0.0, 0.0], [2.0, 1.0, 1.0], &mut out);
        // box 0 touches the query at x = 1; box 1 is far away; box 2 overlaps.
        assert_eq!(out, [true, false, true]);
    }

    #[test]
    fn ray_hits_entry_distance_and_misses() {
        let mn = (vec![2.0f32, 2.0], vec![-1.0f32, 5.0], vec![-1.0f32, -1.0]);
        let mx = (vec![3.0f32, 3.0], vec![1.0f32, 6.0], vec![1.0f32, 1.0]);
        let boxes = BoxesSoa::new(
            Soa3::new(&mn.0, &mn.1, &mn.2),
            Soa3::new(&mx.0, &mx.1, &mx.2),
        );
        let mut t = [0.0; 2];
        ray_aabb(boxes, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 100.0, &mut t);
        assert_eq!(t[0], 2.0);
        assert_eq!(t[1], f32::INFINITY);
    }

    #[test]
    fn ray_origin_inside_clamps_to_zero_and_respects_t_max() {
        let mn = (vec![-1.0f32], vec![-1.0f32], vec![-1.0f32]);
        let mx = (vec![1.0f32], vec![1.0f32], vec![1.0f32]);
        let boxes = BoxesSoa::new(
            Soa3::new(&mn.0, &mn.1, &mn.2),
            Soa3::new(&mx.0, &mx.1, &mx.2),
        );
        let mut t = [0.0; 1];
        ray_aabb(boxes, [0.0; 3], [0.0, 0.0, 1.0], 10.0, &mut t);
        assert_eq!(t[0], 0.0);

        let far = (vec![5.0f32], vec![-1.0f32], vec![-1.0f32]);
        let far_max = (vec![6.0f32], vec![1.0f32], vec![1.0f32]);
        let far_boxes = BoxesSoa::new(
            Soa3::new(&far.0, &far.1, &far.2),
            Soa3::new(&far_max.0, &far_max.1, &far_max.2),
        );
        ray_aabb(far_boxes, [0.0; 3], [1.0, 0.0, 0.0], 4.0, &mut t);
        assert_eq!(t[0], f32::INFINITY);
        ray_aabb(far_boxes, [0.0; 3], [1.0, 0.0, 0.0], 5.0, &mut t);
        assert_eq!(t[0], 5.0);
    }

    #[test]
    fn ray_parallel_axis_uses_slab_membership() {
        let mn = (vec![2.0f32], vec![-1.0f32], vec![-1.0f32]);
        let mx = (vec![3.0f32], vec![1.0f32], vec![1.0f32]);
        let boxes = BoxesSoa::new(
            Soa3::new(&mn.0, &mn.1, &mn.2),
            Soa3::new(&mx.0, &mx.1, &mx.2),
        );
        let mut t = [0.0; 1];
        ray_aabb(boxes, [0.0, 0.5, 0.0], [1.0, 0.0, 0.0], 100.0, &mut t);
        assert_eq!(t[0], 2.0);
        ray_aabb(boxes, [0.0, 1.5, 0.0], [1.0, 0.0, 0.0], 100.0, &mut t);
        assert_eq!(t[0], f32::INFINITY);
    }

    #[test]
    fn ray_matches_brute_force_sampling() {
        // Compare the slab result against dense sampling along the ray.
        let n = 40usize;
        let mut mn = (Vec::new(), Vec::new(), Vec::new());
        let mut mx = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let c = [
                (i % 5) as f32 - 2.0,
                (i * 3 % 7) as f32 - 3.0,
                (i * 2 % 5) as f32 - 2.0,
            ];
            mn.0.push(c[0]);
            mn.1.push(c[1]);
            mn.2.push(c[2]);
            mx.0.push(c[0] + 1.0);
            mx.1.push(c[1] + 0.5);
            mx.2.push(c[2] + 1.5);
        }
        let boxes = BoxesSoa::new(
            Soa3::new(&mn.0, &mn.1, &mn.2),
            Soa3::new(&mx.0, &mx.1, &mx.2),
        );
        let origin = [-6.0, 0.25, 0.75];
        let dir = [1.0, 0.1, -0.05];
        let mut t = vec![0.0; n];
        ray_aabb(boxes, origin, dir, 1000.0, &mut t);
        for i in 0..n {
            let inside = |p: [f32; 3]| {
                (0..3).all(|a| {
                    let (lo, hi) = match a {
                        0 => (mn.0[i], mx.0[i]),
                        1 => (mn.1[i], mx.1[i]),
                        _ => (mn.2[i], mx.2[i]),
                    };
                    p[a] >= lo && p[a] <= hi
                })
            };
            let mut first = None;
            let mut s = 0.0f32;
            while s < 20.0 {
                let p = [
                    origin[0] + dir[0] * s,
                    origin[1] + dir[1] * s,
                    origin[2] + dir[2] * s,
                ];
                if inside(p) {
                    first = Some(s);
                    break;
                }
                s += 0.001;
            }
            match first {
                Some(s) => assert!((t[i] - s).abs() < 0.01, "box {i}: got {} want ~{s}", t[i]),
                None => assert_eq!(t[i], f32::INFINITY, "box {i}"),
            }
        }
    }

    #[test]
    #[should_panic(expected = "distance_sq: lengths differ (2 vs 1)")]
    fn output_length_mismatch_panics() {
        let x = [0.0, 1.0];
        let y = [0.0, 1.0];
        let z = [0.0, 1.0];
        let mut out = [0.0; 1];
        distance_sq(Soa3::new(&x, &y, &z), [0.0; 3], &mut out);
    }

    #[test]
    #[should_panic(expected = "Soa3::new (x, y): lengths differ (1 vs 2)")]
    fn soa_length_mismatch_panics() {
        let _ = Soa3::new(&[0.0], &[0.0, 1.0], &[0.0]);
    }
}
