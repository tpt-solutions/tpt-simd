extern crate std;

use crate::math::*;
use crate::*;
use proptest::prelude::*;
use std::vec;
use std::vec::Vec;

// ---------------------------------------------------------------- references

/// Independent scalar SplitMix64 (straight from the reference C code).
fn ref_splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn rotl(x: u64, k: u32) -> u64 {
    x.rotate_left(k)
}

/// Independent scalar xoshiro256++ (straight from the reference C code).
fn ref_xoshiro(s: &mut [u64; 4]) -> u64 {
    let result = rotl(s[0].wrapping_add(s[3]), 23).wrapping_add(s[0]);
    let t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    result
}

/// Independent scalar Philox4x32-10 written round-by-round.
fn ref_philox(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    fn mulhilo(a: u32, b: u32) -> (u32, u32) {
        let p = u64::from(a) * u64::from(b);
        ((p >> 32) as u32, p as u32)
    }
    let (mut c, mut k) = (ctr, key);
    for r in 0..10 {
        let (hi0, lo0) = mulhilo(0xD251_1F53, c[0]);
        let (hi1, lo1) = mulhilo(0xCD9E_8D57, c[2]);
        c = [hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0];
        if r < 9 {
            k[0] = k[0].wrapping_add(0x9E37_79B9);
            k[1] = k[1].wrapping_add(0xBB67_AE85);
        }
    }
    c
}

// ------------------------------------------------------------ known answers

#[test]
fn splitmix64_known_answers() {
    // Published reference outputs for seed 1234567.
    let mut g = SplitMix64::new(1_234_567);
    let kat = [
        6_457_827_717_110_365_317u64,
        3_203_168_211_198_807_973,
        9_817_491_932_198_370_423,
        4_593_380_528_125_082_431,
        16_408_922_859_458_223_821,
    ];
    for k in kat {
        assert_eq!(g.next_u64(), k);
    }
    // And against the independent scalar reference, including seed 0.
    for seed in [0u64, 1, u64::MAX, 0xDEAD_BEEF] {
        let (mut a, mut st) = (SplitMix64::new(seed), seed);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), ref_splitmix(&mut st));
        }
    }
}

#[test]
fn xoshiro256pp_known_answers() {
    // State {1,2,3,4}: published outputs of the reference implementation.
    let mut g = Xoshiro256pp::from_state([1, 2, 3, 4]);
    let kat = [
        41_943_041u64,
        58_720_359,
        3_588_806_011_781_223,
        3_591_011_842_654_386,
        9_228_616_714_210_784_205,
        9_973_669_472_204_895_162,
        14_011_001_112_246_962_877,
        12_406_186_145_184_390_807,
        15_849_039_046_786_891_736,
        10_450_023_813_501_588_000,
    ];
    for k in kat {
        assert_eq!(g.next_u64(), k);
    }
    // Seeding via SplitMix64 matches the reference procedure.
    let mut sm = 987_654_321u64;
    let mut st = [
        ref_splitmix(&mut sm),
        ref_splitmix(&mut sm),
        ref_splitmix(&mut sm),
        ref_splitmix(&mut sm),
    ];
    let mut g = Xoshiro256pp::from_seed(987_654_321);
    assert_eq!(g.state(), st);
    for _ in 0..1000 {
        assert_eq!(g.next_u64(), ref_xoshiro(&mut st));
    }
}

#[test]
#[should_panic]
fn xoshiro_zero_state_rejected() {
    let _ = Xoshiro256pp::from_state([0; 4]);
}

#[test]
fn x8_lanes_match_scalar_streams() {
    for seed in [0u64, 7, 0xFFFF_FFFF_FFFF_FFFF] {
        let mut v = Xoshiro256ppX8::from_seed(seed);
        let mut lanes: Vec<Xoshiro256pp> = Vec::new();
        let mut s = Xoshiro256pp::from_seed(seed);
        for _ in 0..8 {
            lanes.push(s.clone());
            s.jump();
        }
        for _ in 0..500 {
            let out = v.next_u64x8();
            for i in 0..8 {
                assert_eq!(out[i], lanes[i].next_u64());
            }
        }
        for i in 0..8 {
            assert_eq!(v.lane(i), lanes[i]);
        }
    }
}

#[test]
fn blocks_are_consecutive_streams() {
    let mut a = Xoshiro256ppX8::from_seed_block(5, 2);
    let mut b = Xoshiro256ppX8::from_seed(5);
    b.jump_blocks(2);
    assert_eq!(a, b);
    // Lane 0 of block 1 is lane 8 of the unbounded stream sequence.
    let mut s = Xoshiro256pp::from_seed(5);
    for _ in 0..8 {
        s.jump();
    }
    assert_eq!(Xoshiro256ppX8::from_seed_block(5, 1).lane(0), s);
    assert_ne!(a.next_u64x8(), Xoshiro256ppX8::from_seed(5).next_u64x8());
}

// GF(2) check that `jump` / `long_jump` really advance by 2^128 / 2^192.

type Col = [u64; 4];

fn apply(m: &[Col], v: Col) -> Col {
    let mut r = [0u64; 4];
    for (bit, col) in m.iter().enumerate() {
        if (v[bit / 64] >> (bit % 64)) & 1 != 0 {
            for k in 0..4 {
                r[k] ^= col[k];
            }
        }
    }
    r
}

fn step_matrix() -> Vec<Col> {
    (0..256)
        .map(|bit| {
            let mut s = [0u64; 4];
            s[bit / 64] = 1 << (bit % 64);
            let mut x = Xoshiro256pp::from_state(s);
            x.next_u64();
            x.state()
        })
        .collect()
}

fn pow2_matrix(squarings: usize) -> Vec<Col> {
    let mut m = step_matrix();
    for _ in 0..squarings {
        let sq: Vec<Col> = m.iter().map(|&c| apply(&m, c)).collect();
        m = sq;
    }
    m
}

#[test]
fn jump_advances_exactly_2_pow_128_and_192() {
    let start = Xoshiro256pp::from_seed(2024).state();
    let m128 = pow2_matrix(128);
    let mut g = Xoshiro256pp::from_state(start);
    g.jump();
    assert_eq!(g.state(), apply(&m128, start));
    let m192 = pow2_matrix(192);
    let mut g = Xoshiro256pp::from_state(start);
    g.long_jump();
    assert_eq!(g.state(), apply(&m192, start));
}

#[test]
fn philox_known_answers() {
    // Random123 kat_vectors for philox4x32-10.
    assert_eq!(
        philox4x32_10([0; 4], [0; 2]),
        [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]
    );
    assert_eq!(
        philox4x32_10([u32::MAX; 4], [u32::MAX; 2]),
        [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd]
    );
    assert_eq!(
        philox4x32_10(
            [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
            [0xa4093822, 0x299f31d0]
        ),
        [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1]
    );
    // And the independent round-by-round reference.
    for i in 0..200u32 {
        let c = [i, i.wrapping_mul(31), !i, i ^ 0xABCD];
        let k = [i.wrapping_mul(7), 99 + i];
        assert_eq!(philox4x32_10(c, k), ref_philox(c, k));
    }
}

#[test]
fn philox_x8_matches_scalar_blocks() {
    let seed = 0x1234_5678_9ABC_DEF0u64;
    let stream = 0xFEED_0000_0000_0003u64;
    let mut g = Philox4x32X8::new(seed, stream);
    let key = [seed as u32, (seed >> 32) as u32];
    for step in 0..50u64 {
        let first = g.next_u64x8();
        let second = g.next_u64x8();
        for b in 0..8u64 {
            let n = step * 8 + b;
            let w = ref_philox(
                [
                    n as u32,
                    (n >> 32) as u32,
                    stream as u32,
                    (stream >> 32) as u32,
                ],
                key,
            );
            assert_eq!(first[b as usize], u64::from(w[0]) | u64::from(w[1]) << 32);
            assert_eq!(second[b as usize], u64::from(w[2]) | u64::from(w[3]) << 32);
        }
    }
    // Random access.
    let mut a = Philox4x32X8::new(seed, stream);
    a.seek(8 * 37);
    let mut b = Philox4x32X8::new(seed, stream);
    for _ in 0..37 * 2 {
        b.next_u64x8();
    }
    assert_eq!(a.next_u64x8(), b.next_u64x8());
    // Counter carries into the high word.
    let mut c = Philox4x32X8::new(1, 0);
    c.seek(u64::from(u32::MAX) - 3);
    let out = c.next_u64x8();
    let n = u64::from(u32::MAX) + 2;
    let w = ref_philox([n as u32, (n >> 32) as u32, 0, 0], [1, 0]);
    assert_eq!(out[5], u64::from(w[0]) | u64::from(w[1]) << 32);
}

// ------------------------------------------------------ independence / repro

#[test]
fn reproducible_and_seed_sensitive() {
    let mut a = vec![0u64; 1000];
    let mut b = vec![0u64; 1000];
    let mut c = vec![0u64; 1000];
    Xoshiro256ppX8::from_seed(11).fill_u64(&mut a);
    Xoshiro256ppX8::from_seed(11).fill_u64(&mut b);
    Xoshiro256ppX8::from_seed(12).fill_u64(&mut c);
    assert_eq!(a, b);
    assert!(a.iter().zip(&c).all(|(x, y)| x != y));
    let mut p = vec![0u32; 1000];
    let mut q = vec![0u32; 1000];
    Philox4x32X8::from_seed(3).fill_u32(&mut p);
    Philox4x32X8::from_seed(3).fill_u32(&mut q);
    assert_eq!(p, q);
    Philox4x32X8::new(3, 1).fill_u32(&mut q);
    assert_ne!(p, q);
}

#[test]
fn lanes_are_independent() {
    let n = 1 << 14;
    let mut buf = vec![0u64; 8 * n];
    Xoshiro256ppX8::from_seed(99).fill_u64(&mut buf);
    // De-interleave and check no two lanes share any output.
    let lane = |i: usize| -> Vec<u64> { (0..n).map(|k| buf[k * 8 + i]).collect() };
    let mut all: Vec<u64> = buf.clone();
    all.sort_unstable();
    all.dedup();
    assert_eq!(
        all.len(),
        buf.len(),
        "duplicate 64-bit outputs across lanes"
    );
    // Pairwise correlation of uniform lane outputs is ~N(0, 1/n).
    let u: Vec<Vec<f64>> = (0..8)
        .map(|i| lane(i).iter().map(|&x| u64_to_unit_f64(x) - 0.5).collect())
        .collect();
    for i in 0..8 {
        for j in i + 1..8 {
            let dot: f64 = u[i].iter().zip(&u[j]).map(|(a, b)| a * b).sum();
            let corr = dot / (n as f64 / 12.0);
            assert!(
                corr.abs() < 6.0 / (n as f64).sqrt(),
                "lanes {i},{j}: {corr}"
            );
        }
    }
}

// ------------------------------------------------------------------- ranges

#[test]
fn unit_conversions_never_reach_one() {
    assert_eq!(u32_to_unit_f32(0), 0.0);
    assert_eq!(u32_to_unit_f32(u32::MAX), 1.0 - 1.0 / 16_777_216.0);
    assert!(u32_to_unit_f32(u32::MAX) < 1.0);
    assert_eq!(u64_to_unit_f64(0), 0.0);
    assert_eq!(u64_to_unit_f64(u64::MAX), 1.0 - (-52f64).exp2());
    assert!(u64_to_unit_f64(u64::MAX) < 1.0);
}

#[test]
fn uniform_in_unit_interval() {
    let mut g = Xoshiro256ppX8::from_seed(1);
    let mut f = vec![0f32; 1 << 20];
    g.fill_f32(&mut f);
    assert!(f.iter().all(|&x| (0.0..1.0).contains(&x)));
    let mut d = vec![0f64; 1 << 20];
    g.fill_f64(&mut d);
    assert!(d.iter().all(|&x| (0.0..1.0).contains(&x)));
}

// -------------------------------------------------------------- statistics

fn chi_square(counts: &[u64], expected: &[f64]) -> f64 {
    counts
        .iter()
        .zip(expected)
        .map(|(&c, &e)| (c as f64 - e) * (c as f64 - e) / e)
        .sum()
}

fn uniform_checks<R: Rng8>(mut g: R) {
    let n = 1usize << 20;
    let mut f = vec![0f32; n];
    g.fill_f32(&mut f);
    let mean = f.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
    let var = f
        .iter()
        .map(|&x| (f64::from(x) - mean).powi(2))
        .sum::<f64>()
        / n as f64;
    assert!((mean - 0.5).abs() < 1.5e-3, "mean {mean}");
    assert!((var - 1.0 / 12.0).abs() < 1e-3, "var {var}");
    let mut counts = [0u64; 256];
    for &x in &f {
        counts[(x * 256.0) as usize] += 1;
    }
    let chi = chi_square(&counts, &[n as f64 / 256.0; 256]);
    // df = 255: mean 255, sd ~22.6; +-6 sd.
    assert!((119.0..391.0).contains(&chi), "chi2 {chi}");

    let mut d = vec![0f64; n];
    g.fill_f64(&mut d);
    let mean = d.iter().sum::<f64>() / n as f64;
    assert!((mean - 0.5).abs() < 1.5e-3, "mean64 {mean}");
    let mut counts = [0u64; 256];
    for &x in &d {
        counts[(x * 256.0) as usize] += 1;
    }
    let chi = chi_square(&counts, &[n as f64 / 256.0; 256]);
    assert!((119.0..391.0).contains(&chi), "chi2 64 {chi}");

    // Raw bits: every bit position set ~50% of the time.
    let mut w = vec![0u64; n];
    g.fill_u64(&mut w);
    for bit in 0..64 {
        let ones = w.iter().filter(|&&x| (x >> bit) & 1 != 0).count() as f64;
        assert!((ones / n as f64 - 0.5).abs() < 4e-3, "bit {bit}: {ones}");
    }
    let mut w32 = vec![0u32; n];
    g.fill_u32(&mut w32);
    for bit in 0..32 {
        let ones = w32.iter().filter(|&&x| (x >> bit) & 1 != 0).count() as f64;
        assert!((ones / n as f64 - 0.5).abs() < 4e-3, "bit32 {bit}: {ones}");
    }
}

#[test]
fn uniform_statistics_xoshiro() {
    uniform_checks(Xoshiro256ppX8::from_seed(20240607));
}

#[test]
fn uniform_statistics_philox() {
    uniform_checks(Philox4x32X8::from_seed(20240607));
}

struct Moments {
    mean: f64,
    var: f64,
    skew: f64,
    kurt: f64,
    max_abs: f64,
}

fn moments(z: &[f64]) -> Moments {
    let n = z.len() as f64;
    let mean = z.iter().sum::<f64>() / n;
    let m2 = z.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let m3 = z.iter().map(|&x| (x - mean).powi(3)).sum::<f64>() / n;
    let m4 = z.iter().map(|&x| (x - mean).powi(4)).sum::<f64>() / n;
    Moments {
        mean,
        var: m2,
        skew: m3 / m2.powf(1.5),
        kurt: m4 / (m2 * m2) - 3.0,
        max_abs: z.iter().fold(0.0, |m, &x| f64::max(m, x.abs())),
    }
}

fn normal_checks(z: &[f64], max_cut: f64) {
    let n = z.len() as f64;
    let m = moments(z);
    assert!(m.mean.abs() < 6.0 / n.sqrt(), "mean {}", m.mean);
    assert!(
        (m.var - 1.0).abs() < 6.0 * (2.0 / n).sqrt(),
        "var {}",
        m.var
    );
    assert!(m.skew.abs() < 6.0 * (6.0 / n).sqrt(), "skew {}", m.skew);
    assert!(m.kurt.abs() < 6.0 * (24.0 / n).sqrt(), "kurt {}", m.kurt);
    // Tails: P(|z|>3) = 0.0026998, P(|z|>4) = 6.334e-5.
    let t3 = z.iter().filter(|&&x| x.abs() > 3.0).count() as f64 / n;
    assert!(
        (t3 - 0.0026998).abs() < 6.0 * (0.0027 / n).sqrt(),
        "t3 {t3}"
    );
    let t4 = z.iter().filter(|&&x| x.abs() > 4.0).count() as f64;
    let e4 = 6.334e-5 * n;
    assert!((t4 - e4).abs() < 6.0 * e4.sqrt() + 1.0, "t4 {t4} vs {e4}");
    assert!(m.max_abs > 4.0 && m.max_abs <= max_cut, "max {}", m.max_abs);
    // Chi-square on the bins cut at -2,-1,0,1,2.
    let cdf = [
        0.0,
        0.022_750_13,
        0.158_655_25,
        0.5,
        0.841_344_75,
        0.977_249_87,
        1.0,
    ];
    let cuts = [-2.0, -1.0, 0.0, 1.0, 2.0];
    let mut counts = [0u64; 6];
    for &x in z {
        counts[cuts.iter().filter(|&&c| x >= c).count()] += 1;
    }
    let expected: Vec<f64> = (0..6).map(|i| (cdf[i + 1] - cdf[i]) * n).collect();
    let chi = chi_square(&counts, &expected);
    assert!(chi < 30.0, "chi2 {chi} (df 5)"); // p ~ 1e-5
}

#[test]
fn normal_f32_statistics() {
    let mut g = Xoshiro256ppX8::from_seed(31337);
    let mut z = vec![0f32; 1 << 22];
    g.fill_normal_f32(&mut z);
    assert!(z.iter().all(|x| x.is_finite()));
    let zd: Vec<f64> = z.iter().map(|&x| f64::from(x)).collect();
    normal_checks(&zd, 5.78);
}

#[test]
fn normal_f64_statistics() {
    let mut g = Xoshiro256ppX8::from_seed(4242);
    let mut z = vec![0f64; 1 << 22];
    g.fill_normal_f64(&mut z);
    assert!(z.iter().all(|x| x.is_finite()));
    normal_checks(&z, 8.5);
}

#[test]
fn normal_philox_statistics() {
    let mut g = Philox4x32X8::from_seed(555);
    let mut z = vec![0f32; 1 << 21];
    g.fill_normal_f32(&mut z);
    let zd: Vec<f64> = z.iter().map(|&x| f64::from(x)).collect();
    normal_checks(&zd, 5.78);
}

#[test]
fn box_muller_pair_is_uncorrelated_and_radius_exponential() {
    let n = 1usize << 20;
    let mut g = Xoshiro256ppX8::from_seed(8);
    let mut z = vec![0f32; 2 * n];
    g.fill_normal_f32(&mut z);
    // Within a step z0 occupies [0,8) and z1 [8,16) of each 16-chunk.
    let (mut sxy, mut r2) = (0f64, 0f64);
    for c in z.chunks_exact(16) {
        for i in 0..8 {
            sxy += f64::from(c[i]) * f64::from(c[8 + i]);
            r2 += f64::from(c[i]).powi(2) + f64::from(c[8 + i]).powi(2);
        }
    }
    let pairs = n as f64;
    assert!(
        (sxy / pairs).abs() < 6.0 / pairs.sqrt(),
        "cov {}",
        sxy / pairs
    );
    // r^2 = -2 ln u1 ~ Exp(mean 2): E[r^2] = 2, sd of the mean = 2/sqrt(n).
    assert!((r2 / pairs - 2.0).abs() < 6.0 * 2.0 / pairs.sqrt());
}

// --------------------------------------------------------- math accuracy

fn ulp32(exact: f64) -> f64 {
    let e = f64::from(exact.abs().max(1e-30) as f32);
    let bits = (e as f32).to_bits();
    f64::from(f32::from_bits(bits + 1)) - f64::from(f32::from_bits(bits))
}

fn ulp64(exact: f64) -> f64 {
    let bits = exact.abs().max(1e-300).to_bits();
    f64::from_bits(bits + 1) - f64::from_bits(bits)
}

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 11
}

#[test]
fn ln_f32_accuracy() {
    let mut s = 1u64;
    let mut worst = 0.0f64;
    for _ in 0..200_000 {
        let mut x = [0.0f32; 8];
        for v in &mut x {
            // Mix of (0,1], and the full normal range through random exponents.
            *v = match lcg(&mut s) % 3 {
                0 => 1.0 - (lcg(&mut s) % (1 << 24)) as f32 / 16_777_216.0,
                1 => 1.0 + (lcg(&mut s) % (1 << 23)) as f32 / 8_388_608.0 * 1e-3,
                _ => f32::from_bits((lcg(&mut s) as u32 & 0x7F7F_FFFF).max(0x0080_0000)),
            };
        }
        let got = ln_f32(x);
        for i in 0..8 {
            let exact = libm::log(f64::from(x[i]));
            let err = (f64::from(got[i]) - exact).abs() / ulp32(exact);
            worst = worst.max(err);
        }
    }
    std::println!("ln_f32 worst ulp: {worst}");
    assert!(worst <= 1.0, "{worst}");
    assert_eq!(ln_f32([1.0; 8]), [0.0; 8]);
}

#[test]
fn sincos_f32_accuracy() {
    let mut s = 2u64;
    let mut worst = 0.0f64;
    for k in 0..200_000 {
        let mut t = [0.0f32; 8];
        for v in &mut t {
            *v = if k % 4 == 0 {
                (lcg(&mut s) % (1 << 24)) as f32 / 16_777_216.0
            } else if k % 4 == 1 {
                ((lcg(&mut s) % (1 << 24)) as f32 / 16_777_216.0 - 0.5) * 2_000_000.0
            } else {
                (lcg(&mut s) % 4097) as f32 / 4096.0 // lands on quadrant edges
            };
        }
        let (sn, cs) = sincos_turns_f32(t);
        for i in 0..8 {
            let a = f64::from(t[i]) * core::f64::consts::TAU;
            // Reduce in f64 turns first so the reference stays exact.
            let frac = f64::from(t[i]) - libm::floor(f64::from(t[i]));
            let _ = a;
            let a = frac * core::f64::consts::TAU;
            worst = worst
                .max((f64::from(sn[i]) - libm::sin(a)).abs())
                .max((f64::from(cs[i]) - libm::cos(a)).abs());
        }
    }
    std::println!("sincos_f32 worst abs: {worst}");
    assert!(worst <= 2e-7, "{worst}");
    let (sn, cs) = sincos_turns_f32([0.0, 0.25, 0.5, 0.75, 1.0, -0.25, 0.125, 0.375]);
    assert_eq!(sn[0], 0.0);
    assert_eq!(cs[0], 1.0);
    assert_eq!(sn[1], 1.0);
    assert_eq!(cs[2], -1.0);
    assert_eq!(sn[3], -1.0);
    assert_eq!(sn[5], -1.0);
}

#[test]
fn sqrt_f32_accuracy() {
    let mut s = 3u64;
    let mut worst = 0.0f64;
    for _ in 0..200_000 {
        let mut x = [0.0f32; 8];
        for v in &mut x {
            *v = if lcg(&mut s).is_multiple_of(2) {
                (lcg(&mut s) % (1 << 24)) as f32 / 16_777_216.0 * 40.0
            } else {
                f32::from_bits((lcg(&mut s) as u32 & 0x7F7F_FFFF).max(0x0080_0000))
            };
        }
        let got = sqrt_f32(x);
        for i in 0..8 {
            let exact = libm::sqrt(f64::from(x[i]));
            worst = worst.max((f64::from(got[i]) - exact).abs() / ulp32(exact));
        }
    }
    std::println!("sqrt_f32 worst ulp: {worst}");
    assert!(worst <= 1.0, "{worst}");
    assert_eq!(sqrt_f32([0.0; 8]), [0.0; 8]);
}

#[test]
fn ln_f64_accuracy() {
    let mut s = 4u64;
    let mut worst = 0.0f64;
    for _ in 0..100_000 {
        let mut x = [0.0f64; 8];
        for v in &mut x {
            *v = match lcg(&mut s) % 3 {
                0 => 1.0 - (lcg(&mut s) % (1 << 52)) as f64 / 4_503_599_627_370_496.0,
                1 => 1.0 + (lcg(&mut s) as f64) * 1e-19,
                _ => f64::from_bits(
                    ((lcg(&mut s) << 11) ^ lcg(&mut s)) & 0x7FEF_FFFF_FFFF_FFFF
                        | 0x0010_0000_0000_0000,
                ),
            };
        }
        let got = ln_f64(x);
        for i in 0..8 {
            let exact = libm::log(x[i]);
            worst = worst.max((got[i] - exact).abs() / ulp64(exact));
        }
    }
    std::println!("ln_f64 worst ulp (vs libm): {worst}");
    assert!(worst <= 2.0, "{worst}");
    assert_eq!(ln_f64([1.0; 8]), [0.0; 8]);
}

#[test]
fn sincos_f64_accuracy() {
    let mut s = 5u64;
    let mut worst = 0.0f64;
    for k in 0..100_000 {
        let mut t = [0.0f64; 8];
        for v in &mut t {
            *v = if k % 2 == 0 {
                (lcg(&mut s) % (1 << 53)) as f64 / 9_007_199_254_740_992.0
            } else {
                ((lcg(&mut s) % (1 << 40)) as f64) / 1024.0 + (lcg(&mut s) % 1024) as f64 / 1024.0
            };
        }
        let (sn, cs) = sincos_turns_f64(t);
        for i in 0..8 {
            let frac = t[i] - libm::floor(t[i]);
            let a = frac * core::f64::consts::TAU;
            worst = worst
                .max((sn[i] - libm::sin(a)).abs())
                .max((cs[i] - libm::cos(a)).abs());
        }
    }
    std::println!("sincos_f64 worst abs: {worst}");
    assert!(worst <= 8e-16, "{worst}");
}

#[test]
fn sqrt_f64_accuracy() {
    let mut s = 6u64;
    let mut worst = 0.0f64;
    for _ in 0..100_000 {
        let mut x = [0.0f64; 8];
        for v in &mut x {
            *v = if lcg(&mut s).is_multiple_of(2) {
                (lcg(&mut s) % (1 << 53)) as f64 / 9_007_199_254_740_992.0 * 80.0
            } else {
                f64::from_bits(
                    ((lcg(&mut s) << 11) ^ lcg(&mut s)) & 0x7FEF_FFFF_FFFF_FFFF
                        | 0x0010_0000_0000_0000,
                )
            };
        }
        let got = sqrt_f64(x);
        for i in 0..8 {
            let exact = libm::sqrt(x[i]);
            worst = worst.max((got[i] - exact).abs() / ulp64(exact));
        }
    }
    std::println!("sqrt_f64 worst ulp: {worst}");
    assert!(worst <= 1.0, "{worst}");
}

// ---------------------------------------------------------------- proptest

proptest! {
    #[test]
    fn fills_any_length(len in 0usize..300, seed in any::<u64>()) {
        let mut g = Xoshiro256ppX8::from_seed(seed);
        let mut u64s = vec![0u64; len];
        g.fill_u64(&mut u64s);
        let mut u32s = vec![0u32; len];
        g.fill_u32(&mut u32s);
        let mut f32s = vec![9.0f32; len];
        g.fill_f32(&mut f32s);
        prop_assert!(f32s.iter().all(|&x| (0.0..1.0).contains(&x)));
        let mut f64s = vec![9.0f64; len];
        g.fill_f64(&mut f64s);
        prop_assert!(f64s.iter().all(|&x| (0.0..1.0).contains(&x)));
        let mut n32 = vec![f32::NAN; len];
        g.fill_normal_f32(&mut n32);
        prop_assert!(n32.iter().all(|x| x.is_finite() && x.abs() < 5.8));
        let mut n64 = vec![f64::NAN; len];
        g.fill_normal_f64(&mut n64);
        prop_assert!(n64.iter().all(|x| x.is_finite() && x.abs() < 8.6));
    }

    #[test]
    fn fill_is_prefix_of_longer_fill(len in 0usize..200, seed in any::<u64>()) {
        const LONG: usize = 256;
        macro_rules! check {
            ($t:ty, $zero:expr, $f:ident, $mk:expr) => {{
                let mut a = vec![$zero; len];
                let mut b = vec![$zero; LONG];
                let mut ga = $mk;
                let mut gb = $mk;
                ga.$f(&mut a);
                gb.$f(&mut b);
                prop_assert_eq!(&a[..], &b[..len]);
            }};
        }
        check!(u64, 0u64, fill_u64, Xoshiro256ppX8::from_seed(seed));
        check!(u32, 0u32, fill_u32, Xoshiro256ppX8::from_seed(seed));
        check!(f32, 0f32, fill_f32, Xoshiro256ppX8::from_seed(seed));
        check!(f64, 0f64, fill_f64, Xoshiro256ppX8::from_seed(seed));
        check!(f32, 0f32, fill_normal_f32, Xoshiro256ppX8::from_seed(seed));
        check!(f64, 0f64, fill_normal_f64, Xoshiro256ppX8::from_seed(seed));
        check!(u64, 0u64, fill_u64, Philox4x32X8::from_seed(seed));
        check!(u32, 0u32, fill_u32, Philox4x32X8::from_seed(seed));
    }

    #[test]
    fn fill_u64_matches_scalar_lanes(len in 0usize..200, seed in any::<u64>()) {
        let mut out = vec![0u64; len];
        Xoshiro256ppX8::from_seed(seed).fill_u64(&mut out);
        let mut g = Xoshiro256ppX8::from_seed(seed);
        let mut lanes: Vec<Xoshiro256pp> = (0..8).map(|i| g.lane(i)).collect();
        for (k, &x) in out.iter().enumerate() {
            prop_assert_eq!(x, lanes[k % 8].next_u64());
        }
        let _ = g.next_u64x8();
    }
}
