//! Accuracy tests against `f64` `libm` references (error measured in `f32`
//! ULPs of the exact result), special values, entry-point consistency and
//! proptests. Each accuracy test prints its measured maximum
//! (`cargo test -p tpt-simd-math -- --nocapture`) and asserts the documented
//! bound.

use super::*;
use proptest::prelude::*;
use std::vec::Vec;

// ---------------------------------------------------------------------------
// Documented bounds (ULP). Keep in sync with the table in `lib.rs`.
// ---------------------------------------------------------------------------
const EXP_NORMAL_B: f64 = 1.1;
const EXP_SUB_B: f64 = 1.0;
const LN_B: f64 = 1.0;
const TRIG_B: f64 = 3.0;
const TRIG_BIG_B: f64 = 3.5;
const TANH_B: f64 = 1.5;
const ERF_B: f64 = 3.0;

/// Spacing of `f32` values at `r` (the ULP of the exact result).
fn ulp_of(r: f64) -> f64 {
    let m = (r.abs() as f32).to_bits() & 0x7F80_0000;
    let spacing = if m == 0 {
        f64::from_bits(0x36A0_0000_0000_0000) // 2^-149
    } else {
        f32::from_bits(m) as f64 * f64::from_bits(0x3E80_0000_0000_0000) // 2^-23
    };
    spacing.max(f64::from_bits(0x36A0_0000_0000_0000))
}

fn ulp_err(y: f32, r: f64) -> f64 {
    if y.is_nan() || y.is_infinite() {
        return f64::INFINITY;
    }
    (y as f64 - r).abs() / ulp_of(r)
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
}

/// Inputs in `[lo, hi]`: a linear grid, a stride over the bit patterns
/// (dense in every binade, both signs) and random bit patterns.
fn inputs(lo: f32, hi: f32, n: usize) -> Vec<f32> {
    let mut v = Vec::with_capacity(3 * n);
    for i in 0..=n {
        v.push(lo + (hi - lo) * (i as f32 / n as f32));
    }
    // Bit-pattern stride over each same-sign part of the range.
    let mut parts: Vec<(f32, f32, f32)> = Vec::new();
    if hi > 0.0 {
        parts.push((lo.max(0.0), hi, 1.0));
    }
    if lo < 0.0 {
        parts.push(((-hi).max(0.0), -lo, -1.0));
    }
    for (a, b, sign) in parts {
        let (ba, bb) = (a.to_bits(), b.to_bits());
        let step = ((bb - ba) / n as u32).max(1);
        let mut k = ba;
        while k <= bb {
            v.push(sign * f32::from_bits(k));
            k += step;
        }
    }
    // Random bit patterns inside the range.
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut got = 0;
    while got < n {
        let x = f32::from_bits(rng.next());
        if x.is_finite() && x >= lo && x <= hi {
            v.push(x);
            got += 1;
        }
    }
    v
}

fn measure(
    name: &str,
    f: fn(f32) -> f32,
    reference: fn(f64) -> f64,
    xs: &[f32],
    bound: f64,
) -> f64 {
    let (mut max, mut at) = (0.0f64, 0.0f32);
    for &x in xs {
        let e = ulp_err(f(x), reference(x as f64));
        if e > max {
            (max, at) = (e, x);
        }
    }
    std::eprintln!(
        "{name}: max {max:.3} ulp at x = {at:e} ({} points)",
        xs.len()
    );
    assert!(max <= bound, "{name}: {max} ulp at {at:e} exceeds {bound}");
    max
}

const N: usize = 200_000;

#[test]
fn exp_accuracy() {
    // Largest x with a finite result is ln(f32::MAX) = 88.72283...
    measure(
        "exp normal",
        kernels::exp,
        libm::exp,
        &inputs(-87.3, 88.72, N),
        EXP_NORMAL_B,
    );
    measure(
        "exp subnormal",
        kernels::exp,
        libm::exp,
        &inputs(-103.97, -87.3, N),
        EXP_SUB_B,
    );
}

#[test]
fn exp_near_overflow_threshold() {
    let thr = 88.7228_f32; // just below ln(f32::MAX) = 88.72284
    assert!(exp_scalar(thr).is_finite());
    assert_eq!(exp_scalar(88.73), f32::INFINITY);
    assert_eq!(exp_scalar(89.0), f32::INFINITY);
    assert_eq!(exp_scalar(1.0e30), f32::INFINITY);
}

#[test]
fn ln_accuracy() {
    let mut xs: Vec<f32> = Vec::new();
    // All positive finite: stride over every bit pattern, plus every
    // subnormal exponent decade and a log-spaced grid.
    let mut k = 1u32;
    while k < 0x7F80_0000 {
        xs.push(f32::from_bits(k));
        k += 4_099;
    }
    let mut rng = Rng(12345);
    for _ in 0..N {
        let x = f32::from_bits(rng.next() & 0x7FFF_FFFF);
        if x.is_finite() && x > 0.0 {
            xs.push(x);
        }
    }
    // Dense around 1 where cancellation matters, and sqrt(2)/2 boundary.
    for i in 0..=N {
        xs.push(0.5 + 1.5 * (i as f32 / N as f32));
    }
    let b = 1.0f32.to_bits();
    for d in 0..2000u32 {
        xs.push(f32::from_bits(b + d));
        xs.push(f32::from_bits(b - d));
    }
    measure("ln", kernels::ln, libm::log, &xs, LN_B);
}

#[test]
fn sin_cos_accuracy() {
    measure(
        "sin <=8192",
        kernels::sin,
        libm::sin,
        &inputs(-8192.0, 8192.0, N),
        TRIG_B,
    );
    measure(
        "cos <=8192",
        kernels::cos,
        libm::cos,
        &inputs(-8192.0, 8192.0, N),
        TRIG_B,
    );
    let mut big = inputs(8192.0, 100_000.0, N / 2);
    big.extend(
        inputs(-100_000.0, -8192.0, N / 2)
            .into_iter()
            .filter(|x| x.abs() > 8192.0),
    );
    let big: Vec<f32> = big.into_iter().filter(|x| x.abs() > 8192.0).collect();
    measure("sin >8192", kernels::sin, libm::sin, &big, TRIG_BIG_B);
    measure("cos >8192", kernels::cos, libm::cos, &big, TRIG_BIG_B);
}

#[test]
fn sin_cos_absolute_error() {
    let mut worst = 0.0f64;
    for x in inputs(-8192.0, 8192.0, N) {
        worst = worst.max((kernels::sin(x) as f64 - libm::sin(x as f64)).abs());
        worst = worst.max((kernels::cos(x) as f64 - libm::cos(x as f64)).abs());
    }
    std::eprintln!("sin/cos absolute error <=8192: {worst:e}");
    assert!(worst < 2.0e-7, "{worst}");
}

#[test]
fn tanh_accuracy() {
    measure(
        "tanh",
        kernels::tanh,
        libm::tanh,
        &inputs(-20.0, 20.0, N),
        TANH_B,
    );
}

#[test]
fn erf_accuracy() {
    measure("erf", kernels::erf, libm::erf, &inputs(-6.0, 6.0, N), ERF_B);
}

/// Stride over every finite `f32` bit pattern (both signs) and check the
/// documented bound or the documented saturation. Slow; run with
/// `cargo test --release -p tpt-simd-math -- --ignored --nocapture`.
#[test]
#[ignore = "slow: ~1e8 reference evaluations"]
fn full_range_bit_stride() {
    const STRIDE: u32 = 41;
    let mut max = [0.0f64; 6];
    let mut k = 0u32;
    while k < 0x7F80_0000 {
        for sign in [1.0f32, -1.0] {
            let x = sign * f32::from_bits(k);
            let xd = x as f64;
            // exp: bound inside the finite range, saturation outside.
            let y = exp_scalar(x);
            // Overflow when the exact result rounds beyond f32::MAX.
            if x > 88.0 && libm::exp(xd) >= f32::MAX as f64 + 2f64.powi(103) {
                assert_eq!(y, f32::INFINITY, "exp({x:e})");
            } else if x < -103.98 {
                assert_eq!(y, 0.0, "exp({x:e})");
            } else if x >= -87.3 {
                max[0] = max[0].max(ulp_err(y, libm::exp(xd)));
            } else {
                max[1] = max[1].max(ulp_err(y, libm::exp(xd)));
            }
            if sign > 0.0 && k > 0 {
                max[2] = max[2].max(ulp_err(ln_scalar(x), libm::log(xd)));
            }
            if x.abs() <= 8192.0 {
                max[3] = max[3].max(ulp_err(sin_scalar(x), libm::sin(xd)));
                max[3] = max[3].max(ulp_err(cos_scalar(x), libm::cos(xd)));
            }
            max[4] = max[4].max(ulp_err(tanh_scalar(x), libm::tanh(xd)));
            max[5] = max[5].max(ulp_err(erf_scalar(x), libm::erf(xd)));
        }
        k += STRIDE;
    }
    std::eprintln!(
        "full range stride {STRIDE}: exp {:.3} exp-sub {:.3} ln {:.3} trig {:.3} tanh {:.3} erf {:.3}",
        max[0],
        max[1],
        max[2],
        max[3],
        max[4],
        max[5]
    );
    assert!(max[0] <= EXP_NORMAL_B && max[1] <= EXP_SUB_B && max[2] <= LN_B);
    assert!(max[3] <= TRIG_B && max[4] <= TANH_B && max[5] <= ERF_B);
}

// ---------------------------------------------------------------------------
// Special values
// ---------------------------------------------------------------------------

fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

macro_rules! check {
    ($f:ident($x:expr) == $want:expr) => {{
        let (x, want): (f32, f32) = ($x, $want);
        let got = $f(x);
        assert!(
            same(got, want),
            "{}({x:e}) = {got:e} ({:#x}), want {want:e} ({:#x})",
            stringify!($f),
            got.to_bits(),
            want.to_bits()
        );
    }};
}

const INF: f32 = f32::INFINITY;
const NAN: f32 = f32::NAN;
const SUB: f32 = 1.0e-40; // subnormal

#[test]
fn exp_special() {
    check!(exp_scalar(NAN) == NAN);
    check!(exp_scalar(INF) == INF);
    check!(exp_scalar(-INF) == 0.0);
    check!(exp_scalar(0.0) == 1.0);
    check!(exp_scalar(-0.0) == 1.0);
    check!(exp_scalar(SUB) == 1.0);
    check!(exp_scalar(-SUB) == 1.0);
    check!(exp_scalar(89.0) == INF);
    check!(exp_scalar(f32::MAX) == INF);
    check!(exp_scalar(-104.0) == 0.0);
    check!(exp_scalar(-1000.0) == 0.0);
    check!(exp_scalar(f32::MIN) == 0.0);
    // Gradual underflow produces subnormals, not a jump to zero.
    let y = exp_scalar(-100.0);
    assert!(y > 0.0 && y < f32::MIN_POSITIVE);
    // NaN payload / sign preserved.
    let n = f32::from_bits(0xFFC1_2345);
    assert_eq!(exp_scalar(n).to_bits(), n.to_bits());
}

#[test]
fn ln_special() {
    check!(ln_scalar(NAN) == NAN);
    check!(ln_scalar(INF) == INF);
    check!(ln_scalar(-INF) == NAN);
    check!(ln_scalar(0.0) == -INF);
    check!(ln_scalar(-0.0) == -INF);
    check!(ln_scalar(-1.0) == NAN);
    check!(ln_scalar(-SUB) == NAN);
    check!(ln_scalar(f32::MIN) == NAN);
    check!(ln_scalar(1.0) == 0.0);
    // Subnormals: smallest subnormal and largest.
    let tiny = f32::from_bits(1);
    assert!((ln_scalar(tiny) - (-103.278_93)).abs() < 1e-3);
    assert!((ln_scalar(SUB) as f64 - libm::log(SUB as f64)).abs() < 1e-5);
    assert!((ln_scalar(f32::MAX) - 88.722_84).abs() < 1e-4);
}

#[test]
fn trig_special() {
    for f in [sin_scalar, cos_scalar] {
        assert!(f(NAN).is_nan());
        assert!(f(INF).is_nan());
        assert!(f(-INF).is_nan());
        assert!(f(100_001.0).is_nan());
        assert!(f(f32::MAX).is_nan());
        assert!(f(-1.0e9).is_nan());
    }
    check!(sin_scalar(0.0) == 0.0);
    check!(sin_scalar(-0.0) == -0.0);
    check!(cos_scalar(0.0) == 1.0);
    check!(cos_scalar(-0.0) == 1.0);
    check!(sin_scalar(SUB) == SUB);
    check!(sin_scalar(-SUB) == -SUB);
    check!(cos_scalar(SUB) == 1.0);
    // Exact-ish landmark values.
    assert!((sin_scalar(core::f32::consts::FRAC_PI_2) - 1.0).abs() < 1e-7);
    assert!((cos_scalar(core::f32::consts::PI) + 1.0).abs() < 1e-7);
}

#[test]
fn tanh_special() {
    check!(tanh_scalar(NAN) == NAN);
    check!(tanh_scalar(INF) == 1.0);
    check!(tanh_scalar(-INF) == -1.0);
    check!(tanh_scalar(0.0) == 0.0);
    check!(tanh_scalar(-0.0) == -0.0);
    check!(tanh_scalar(SUB) == SUB);
    check!(tanh_scalar(-SUB) == -SUB);
    check!(tanh_scalar(100.0) == 1.0);
    check!(tanh_scalar(-100.0) == -1.0);
    check!(tanh_scalar(f32::MAX) == 1.0);
    check!(tanh_scalar(f32::MIN) == -1.0);
}

#[test]
fn erf_special() {
    check!(erf_scalar(NAN) == NAN);
    check!(erf_scalar(INF) == 1.0);
    check!(erf_scalar(-INF) == -1.0);
    check!(erf_scalar(0.0) == 0.0);
    check!(erf_scalar(-0.0) == -0.0);
    check!(erf_scalar(10.0) == 1.0);
    check!(erf_scalar(-10.0) == -1.0);
    check!(erf_scalar(f32::MAX) == 1.0);
    check!(erf_scalar(f32::MIN) == -1.0);
    // Subnormal: erf(x) ~ 2/sqrt(pi) x.
    let y = erf_scalar(SUB) as f64;
    assert!((y / SUB as f64 - core::f64::consts::FRAC_2_SQRT_PI).abs() < 1e-3);
}

// ---------------------------------------------------------------------------
// Entry-point consistency
// ---------------------------------------------------------------------------

type Scalar = fn(f32) -> f32;
type Vector = fn(Simd<f32, 8>) -> Simd<f32, 8>;
type Slice = fn(&[f32], &mut [f32]);
type InPlace = fn(&mut [f32]);

fn table() -> [(&'static str, Scalar, Vector, Slice, InPlace); 6] {
    [
        ("exp", exp_scalar, exp_f32x8, exp_f32, exp_f32_inplace),
        ("ln", ln_scalar, ln_f32x8, ln_f32, ln_f32_inplace),
        ("sin", sin_scalar, sin_f32x8, sin_f32, sin_f32_inplace),
        ("cos", cos_scalar, cos_f32x8, cos_f32, cos_f32_inplace),
        ("tanh", tanh_scalar, tanh_f32x8, tanh_f32, tanh_f32_inplace),
        ("erf", erf_scalar, erf_f32x8, erf_f32, erf_f32_inplace),
    ]
}

fn sample(n: usize) -> Vec<f32> {
    let specials = [
        NAN,
        INF,
        -INF,
        0.0,
        -0.0,
        SUB,
        -SUB,
        1.0,
        -1.0,
        88.0,
        -88.0,
        1.0e6,
        f32::MAX,
        f32::MIN,
    ];
    let mut rng = Rng(777);
    (0..n)
        .map(|i| {
            if i % 5 == 0 {
                specials[(rng.next() as usize) % specials.len()]
            } else {
                (rng.next() as f32 / u32::MAX as f32 - 0.5) * 40.0
            }
        })
        .collect()
}

#[test]
fn entry_points_agree_for_all_tail_lengths() {
    for (name, s, v, sl, ip) in table() {
        for n in 0..=41 {
            let x = sample(n);
            let want: Vec<f32> = x.iter().map(|&a| s(a)).collect();
            let mut out = std::vec![0.0f32; n];
            sl(&x, &mut out);
            assert!(
                out.iter().zip(&want).all(|(a, b)| same(*a, *b)),
                "{name} slice n={n}"
            );
            let mut buf = x.clone();
            ip(&mut buf);
            assert!(
                buf.iter().zip(&want).all(|(a, b)| same(*a, *b)),
                "{name} inplace n={n}"
            );
        }
        let x = sample(8);
        let arr: [f32; 8] = x.clone().try_into().unwrap();
        let got = v(Simd::from_array(arr)).to_array();
        assert!(
            got.iter().zip(&x).all(|(g, &a)| same(*g, s(a))),
            "{name} simd"
        );
    }
}

#[test]
#[should_panic(expected = "lengths must match")]
fn slice_length_mismatch_panics() {
    let mut out = [0.0f32; 3];
    exp_f32(&[1.0; 4], &mut out);
}

// ---------------------------------------------------------------------------
// proptest
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn prop_exp(x in -87.3f32..88.72) {
        prop_assert!(ulp_err(exp_scalar(x), libm::exp(x as f64)) <= EXP_NORMAL_B);
    }

    #[test]
    fn prop_ln(bits in 1u32..0x7F80_0000) {
        let x = f32::from_bits(bits);
        prop_assert!(ulp_err(ln_scalar(x), libm::log(x as f64)) <= LN_B);
    }

    #[test]
    fn prop_sin_cos(x in -8192.0f32..8192.0) {
        prop_assert!(ulp_err(sin_scalar(x), libm::sin(x as f64)) <= TRIG_B);
        prop_assert!(ulp_err(cos_scalar(x), libm::cos(x as f64)) <= TRIG_B);
        let (s, c) = (sin_scalar(x), cos_scalar(x));
        prop_assert!((s * s + c * c - 1.0).abs() < 1e-6);
    }

    #[test]
    fn prop_tanh_erf(x in -12.0f32..12.0) {
        prop_assert!(ulp_err(tanh_scalar(x), libm::tanh(x as f64)) <= TANH_B);
        prop_assert!(ulp_err(erf_scalar(x), libm::erf(x as f64)) <= ERF_B);
        // Odd symmetry is exact.
        prop_assert_eq!(tanh_scalar(-x).to_bits(), (-tanh_scalar(x)).to_bits());
        prop_assert_eq!(erf_scalar(-x).to_bits(), (-erf_scalar(x)).to_bits());
        prop_assert!(tanh_scalar(x).abs() <= 1.0 && erf_scalar(x).abs() <= 1.0);
    }

    #[test]
    fn prop_slice_matches_scalar(xs in proptest::collection::vec(any::<f32>(), 0..70)) {
        for (name, s, _, sl, _) in table() {
            let mut out = std::vec![0.0f32; xs.len()];
            sl(&xs, &mut out);
            for (o, &x) in out.iter().zip(&xs) {
                prop_assert!(same(*o, s(x)), "{} at {:e}", name, x);
            }
        }
    }

    #[test]
    fn prop_exp_ln_roundtrip(x in 1.0e-30f32..1.0e30) {
        let r = exp_scalar(ln_scalar(x));
        prop_assert!(((r - x) / x).abs() < 1e-5, "{x:e} -> {r:e}");
    }
}
