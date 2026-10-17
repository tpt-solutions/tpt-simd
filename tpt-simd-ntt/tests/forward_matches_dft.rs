//! `NttPlan::forward` must equal the direct sum `X_k = sum_j a_j * w^(j*k) mod q`
//! for every length and modulus here. The reference uses plain `u64` arithmetic,
//! so it does not share any code with the Montgomery path. Lengths from 1 to 256
//! cover the scalar stages (half-width < 8) and the AVX2 stages (>= 8). The
//! modulus 2013265921 is close to 2^31, where the lane arithmetic is tightest.

use tpt_simd_ntt::NttPlan;

fn pow_mod(mut b: u64, mut e: u64, q: u64) -> u64 {
    let mut r = 1u64;
    b %= q;
    while e > 0 {
        if e & 1 == 1 {
            r = r * b % q;
        }
        b = b * b % q;
        e >>= 1;
    }
    r
}

fn direct_dft(a: &[u32], q: u32, w: u64) -> Vec<u32> {
    let q64 = u64::from(q);
    let n = a.len();
    (0..n)
        .map(|k| {
            let wk = pow_mod(w, k as u64, q64);
            let mut acc = 0u64;
            let mut wjk = 1u64;
            for &x in a {
                acc = (acc + u64::from(x) * wjk) % q64;
                wjk = wjk * wk % q64;
            }
            acc as u32
        })
        .collect()
}

fn check(q: u32, g: u32, n: usize, seed: &mut u64) {
    let mut a: Vec<u32> = (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            (*seed % u64::from(q)) as u32
        })
        .collect();
    let w = pow_mod(u64::from(g), (u64::from(q) - 1) / n as u64, u64::from(q));
    let want = direct_dft(&a, q, w);
    let plan = NttPlan::new(q, g, n);
    plan.forward(&mut a);
    assert_eq!(a, want, "q={q} n={n}");
}

#[test]
fn forward_matches_direct_dft() {
    let mut seed = 0x243F_6A88_85A3_08D3u64;
    for &(q, g) in &[
        (998_244_353u32, 3u32),
        (469_762_049, 3),
        (2_013_265_921, 31),
    ] {
        let mut n = 1usize;
        while n <= 256 {
            check(q, g, n, &mut seed);
            n *= 2;
        }
    }
}

#[test]
fn inverse_round_trip_for_large_lengths() {
    let q = 2_013_265_921u32;
    let plan = NttPlan::new(q, 31, 4096);
    let mut seed = 0x1357_9BDF_2468_ACE0u64;
    let orig: Vec<u32> = (0..4096)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % u64::from(q)) as u32
        })
        .collect();
    let mut a = orig.clone();
    plan.forward(&mut a);
    plan.inverse(&mut a);
    assert_eq!(a, orig);
}
