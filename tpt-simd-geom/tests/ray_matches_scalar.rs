//! `ray_aabb` must return the same bits as a plain per-box loop, for every box.
//! The reference below is the scalar algorithm written out in full. Inputs
//! include ties, zero direction components (parallel slabs), boxes that touch
//! the origin, and signed zeros, which the blend-based SIMD path must reproduce.

use tpt_simd_geom::{BoxesSoa, Soa3, ray_aabb};

fn reference(
    mins: &[[f32; 3]],
    maxs: &[[f32; 3]],
    origin: [f32; 3],
    dir: [f32; 3],
    t_max: f32,
) -> Vec<f32> {
    let mut inv = [0.0f32; 3];
    let mut parallel = [false; 3];
    for a in 0..3 {
        if dir[a] == 0.0 {
            parallel[a] = true;
        } else {
            inv[a] = 1.0 / dir[a];
        }
    }
    mins.iter()
        .zip(maxs)
        .map(|(mn, mx)| {
            let mut tnear = f32::NEG_INFINITY;
            let mut tfar = f32::INFINITY;
            let mut hit = true;
            for a in 0..3 {
                if parallel[a] {
                    if origin[a] < mn[a] || origin[a] > mx[a] {
                        hit = false;
                        break;
                    }
                } else {
                    let t1 = (mn[a] - origin[a]) * inv[a];
                    let t2 = (mx[a] - origin[a]) * inv[a];
                    let (lo, hi) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
                    if lo > tnear {
                        tnear = lo;
                    }
                    if hi < tfar {
                        tfar = hi;
                    }
                }
            }
            if hit && tnear <= tfar && tfar >= 0.0 && tnear <= t_max {
                if tnear < 0.0 { 0.0 } else { tnear }
            } else {
                f32::INFINITY
            }
        })
        .collect()
}

fn run(mins: &[[f32; 3]], maxs: &[[f32; 3]], origin: [f32; 3], dir: [f32; 3], t_max: f32) {
    let n = mins.len();
    let cols = |f: fn(&[f32; 3]) -> f32, v: &[[f32; 3]]| v.iter().map(f).collect::<Vec<f32>>();
    let (mx, my, mz) = (
        cols(|p| p[0], mins),
        cols(|p| p[1], mins),
        cols(|p| p[2], mins),
    );
    let (xx, xy, xz) = (
        cols(|p| p[0], maxs),
        cols(|p| p[1], maxs),
        cols(|p| p[2], maxs),
    );
    let boxes = BoxesSoa::new(Soa3::new(&mx, &my, &mz), Soa3::new(&xx, &xy, &xz));
    let mut got = vec![0.0f32; n];
    ray_aabb(boxes, origin, dir, t_max, &mut got);
    let want = reference(mins, maxs, origin, dir, t_max);
    for i in 0..n {
        assert_eq!(
            got[i].to_bits(),
            want[i].to_bits(),
            "box {i} origin {origin:?} dir {dir:?} t_max {t_max}"
        );
    }
}

fn boxes(n: usize, seed: &mut u64) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    let mut next = || {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        // Small integers and halves, so ties and exact zeros come up often.
        ((*seed % 17) as f32 - 8.0) * 0.5
    };
    let mut mins = Vec::with_capacity(n);
    let mut maxs = Vec::with_capacity(n);
    for _ in 0..n {
        let mn = [next(), next(), next()];
        let ext = [next().abs(), next().abs(), next().abs()];
        mins.push(mn);
        maxs.push([mn[0] + ext[0], mn[1] + ext[1], mn[2] + ext[2]]);
    }
    (mins, maxs)
}

#[test]
fn ray_matches_reference_over_lengths_and_directions() {
    let dirs = [
        [0.6f32, 0.8, -0.3],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [-0.5, 0.0, 0.25],
        [0.0, 0.0, -2.0],
    ];
    let origins = [[0.5f32, 0.25, -0.75], [0.0, 0.0, 0.0], [-3.0, 2.0, 1.0]];
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for n in 0..=40 {
        let (mins, maxs) = boxes(n, &mut seed);
        for dir in dirs {
            for origin in origins {
                for t_max in [100.0f32, 1.5, 0.0, -1.0] {
                    run(&mins, &maxs, origin, dir, t_max);
                }
            }
        }
    }
}

#[test]
fn signed_zero_ties_match_reference() {
    // Origin at zero and the slab edges at zero: t1 and t2 are +0 and -0 in turn.
    let mins = [
        [0.0f32, -0.0, 0.0],
        [-0.0, 0.0, -0.0],
        [-1.0, -1.0, -1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ];
    let maxs = [
        [1.0f32, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [-0.0, -0.0, -0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ];
    for dir in [[1.0f32, 1.0, 1.0], [-1.0, 1.0, -1.0], [0.0, 1.0, 0.0]] {
        run(&mins, &maxs, [0.0, 0.0, 0.0], dir, 10.0);
        run(&mins, &maxs, [-0.0, 0.0, -0.0], dir, 10.0);
    }
}
