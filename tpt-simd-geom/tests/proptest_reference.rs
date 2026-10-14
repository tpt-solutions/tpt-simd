//! Property tests: SoA geometry kernels against per-point formulas.

use proptest::prelude::*;
use tpt_simd_geom::{BoxesSoa, Soa3, aabb_overlap, distance, distance_sq, dot};

/// Small integer-valued coordinates, so that boxes touch often.
fn coords(n: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-4i8..=4, n).prop_map(|v| v.into_iter().map(f32::from).collect())
}

proptest! {
    #[test]
    fn distance_sq_and_distance_match_formula(
        (xs, ys, zs) in (0usize..200).prop_flat_map(|n| (coords(n), coords(n), coords(n))),
        q in (-4i8..=4, -4i8..=4, -4i8..=4),
    ) {
        let q = [f32::from(q.0), f32::from(q.1), f32::from(q.2)];
        let points = Soa3::new(&xs, &ys, &zs);
        let mut sq = vec![0f32; xs.len()];
        let mut d = vec![0f32; xs.len()];
        distance_sq(points, q, &mut sq);
        distance(points, q, &mut d);
        for i in 0..xs.len() {
            let want = (xs[i] - q[0]).powi(2) + (ys[i] - q[1]).powi(2) + (zs[i] - q[2]).powi(2);
            prop_assert!((sq[i] - want).abs() <= 1e-3 * want.max(1.0));
            prop_assert!((d[i] - want.sqrt()).abs() <= 1e-3 * want.sqrt().max(1.0));
        }
    }

    #[test]
    fn dot_matches_formula(
        (xs, ys, zs) in (0usize..200).prop_flat_map(|n| (coords(n), coords(n), coords(n))),
        d in (-4i8..=4, -4i8..=4, -4i8..=4),
    ) {
        let d = [f32::from(d.0), f32::from(d.1), f32::from(d.2)];
        let mut out = vec![0f32; xs.len()];
        dot(Soa3::new(&xs, &ys, &zs), d, &mut out);
        for i in 0..xs.len() {
            let want = xs[i] * d[0] + ys[i] * d[1] + zs[i] * d[2];
            prop_assert!((out[i] - want).abs() <= 1e-3 * want.abs().max(1.0));
        }
    }

    #[test]
    fn aabb_overlap_matches_per_axis_test(
        (min_x, min_y, min_z) in (0usize..120).prop_flat_map(|n| (coords(n), coords(n), coords(n))),
        qmin in (-4i8..=4, -4i8..=4, -4i8..=4),
        qext in 0i8..=3,
    ) {
        // Boxes with max >= min: the extent per axis is a small non-negative value.
        let n = min_x.len();
        let max_x: Vec<f32> = min_x.iter().enumerate().map(|(i, v)| v + f32::from((i % 4) as i8)).collect();
        let max_y: Vec<f32> = min_y.iter().enumerate().map(|(i, v)| v + f32::from((i % 3) as i8)).collect();
        let max_z: Vec<f32> = min_z.iter().enumerate().map(|(i, v)| v + f32::from((i % 2) as i8)).collect();

        let boxes = BoxesSoa::new(
            Soa3::new(&min_x, &min_y, &min_z),
            Soa3::new(&max_x, &max_y, &max_z),
        );
        let qmin = [f32::from(qmin.0), f32::from(qmin.1), f32::from(qmin.2)];
        let qmax = [qmin[0] + f32::from(qext), qmin[1] + f32::from(qext), qmin[2] + f32::from(qext)];
        let mut out = vec![false; n];
        aabb_overlap(boxes, qmin, qmax, &mut out);
        for i in 0..n {
            let overlaps = min_x[i] <= qmax[0] && max_x[i] >= qmin[0]
                && min_y[i] <= qmax[1] && max_y[i] >= qmin[1]
                && min_z[i] <= qmax[2] && max_z[i] >= qmin[2];
            prop_assert_eq!(out[i], overlaps);
        }
    }
}
