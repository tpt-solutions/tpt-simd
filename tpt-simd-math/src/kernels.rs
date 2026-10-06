//! Branch-free scalar kernels. Every `if` below is a value select (no side
//! effects, no calls), so LLVM if-converts it and the lane loops in
//! `lib.rs` become straight vector code. Only plain `*`, `+`, `-`, `/` are
//! used (no `mul_add`, no libm), so results are bit-identical on every
//! target and feature set.

// Coefficients are quoted verbatim from their published sources / fits.
#![allow(clippy::excessive_precision)]

/// `1.5 * 2^23`: adding then subtracting rounds to nearest integer for
/// `|t| < 2^22`, and leaves that integer in the low mantissa bits.
const MAGIC: f32 = 12_582_912.0;
const MAGIC_BITS: i32 = 0x4B40_0000;

const LOG2_E: f32 = core::f32::consts::LOG2_E;
// ln 2 split so that `n * LN2_HI` is exact for |n| < 2^15.
const LN2_HI: f32 = 0.693_359_375;
const LN2_LO: f32 = -2.121_944_4e-4;

/// `2^e` for `e` in the normal exponent range.
#[inline(always)]
fn pow2(e: i32) -> f32 {
    f32::from_bits((e.wrapping_add(127) as u32) << 23)
}

#[inline(always)]
pub(crate) fn exp(x: f32) -> f32 {
    // Saturate; NaN fails both comparisons and is restored at the end.
    let xc = x.clamp(-104.0, 89.0);
    let t = xc * LOG2_E + MAGIC;
    let n = t - MAGIC;
    let ni = (t.to_bits() as i32).wrapping_sub(MAGIC_BITS);
    // Cody-Waite reduction: r = x - n ln2, |r| <= ln2 / 2.
    let r = (xc - n * LN2_HI) - n * LN2_LO;
    // exp(r) = 1 + r + r^2 * P(r) (Cephes expf coefficients).
    let mut p = 1.987_569_15e-4_f32;
    p = p * r + 1.398_199_950_7e-3;
    p = p * r + 8.333_451_907_3e-3;
    p = p * r + 4.166_579_589_4e-2;
    p = p * r + 1.666_666_545_9e-1;
    p = p * r + 5.000_000_120_1e-1;
    let y = p * (r * r) + r + 1.0;
    // 2^n in two steps so n = 128 and n = -150 stay representable.
    let e1 = ni >> 1;
    let e2 = ni.wrapping_sub(e1);
    let res = (y * pow2(e1)) * pow2(e2);
    if x.is_nan() { x } else { res }
}

#[inline(always)]
pub(crate) fn ln(x: f32) -> f32 {
    // Scale subnormals (and 0, negatives) up by 2^23 so the exponent
    // extraction below is valid; the exponent is corrected by -23.
    let sub = x < f32::MIN_POSITIVE;
    let xs = if sub { x * 8_388_608.0 } else { x };
    let adj: i32 = if sub { -23 } else { 0 };
    // Split x = 2^e * (1 + f) with 1 + f in [sqrt(2)/2, sqrt(2)).
    let b = (xs.to_bits() as i32).wrapping_sub(0x3F35_04F3);
    let e = (b >> 23) + adj;
    let m = f32::from_bits(((b & 0x007F_FFFF).wrapping_add(0x3F35_04F3)) as u32);
    let f = m - 1.0;
    let ef = e as f32;
    let z = f * f;
    // Cephes logf polynomial.
    let mut y = 7.037_683_629_2e-2_f32;
    y = y * f - 1.151_461_031_0e-1;
    y = y * f + 1.167_699_874_0e-1;
    y = y * f - 1.242_014_084_6e-1;
    y = y * f + 1.424_932_278_7e-1;
    y = y * f - 1.666_805_766_5e-1;
    y = y * f + 2.000_071_476_5e-1;
    y = y * f - 2.499_999_399_3e-1;
    y = y * f + 3.333_333_117_4e-1;
    y = (y * f) * z;
    y += ef * -2.121_944_40e-4;
    y -= 0.5 * z;
    let res = (f + y) + ef * 0.693_359_375;
    let res = if x == f32::INFINITY { x } else { res };
    let res = if x < 0.0 { f32::NAN } else { res };
    let res = if x == 0.0 { f32::NEG_INFINITY } else { res };
    if x.is_nan() { x } else { res }
}

/// Reduce `x` modulo pi/2: returns `(r, quadrant)` with `|r| <= pi/4`.
#[inline(always)]
fn reduce_pio2(x: f32) -> (f32, i32) {
    const TWO_OVER_PI: f32 = core::f32::consts::FRAC_2_PI;
    // pi/2 = A + B + C + D + E with A, B, C having 8 significant bits each, so
    // `q * A`, `q * B`, `q * C` are exact for |q| < 2^16 (|x| < ~1e5) and
    // the subtractions below only round at the scale of the result.
    const A: f32 = 1.570_312_5;
    const B: f32 = 4.844_665_527_343_75e-4;
    const C: f32 = -6.407_499_313_354_492e-7;
    const D: f32 = 9.920_936_294_705_03e-10;
    const E: f32 = -4.978_996_314_197_971e-17;
    let t = x * TWO_OVER_PI + MAGIC;
    let q = t - MAGIC;
    let qi = (t.to_bits() as i32).wrapping_sub(MAGIC_BITS);
    (((((x - q * A) - q * B) - q * C) - q * D) - q * E, qi)
}

/// Cephes sinf kernel, `|r| <= pi/4`.
#[inline(always)]
fn sin_poly(r: f32) -> f32 {
    let z = r * r;
    let mut p = -1.951_529_589_1e-4_f32;
    p = p * z + 8.332_160_873_6e-3;
    p = p * z - 1.666_665_461_1e-1;
    (p * z) * r + r
}

/// Cephes cosf kernel, `|r| <= pi/4`.
#[inline(always)]
fn cos_poly(r: f32) -> f32 {
    let z = r * r;
    let mut p = 2.443_315_711_809_948e-5_f32;
    p = p * z - 1.388_731_625_493_765e-3;
    p = p * z + 4.166_664_568_298_827e-2;
    p * (z * z) - 0.5 * z + 1.0
}

/// Inputs beyond this magnitude are outside the supported range and give NaN.
pub(crate) const TRIG_LIMIT: f32 = 100_000.0;

#[inline(always)]
pub(crate) fn sin(x: f32) -> f32 {
    let (r, q) = reduce_pio2(x);
    let s = sin_poly(r);
    let c = cos_poly(r);
    let v = if q & 1 != 0 { c } else { s };
    let v = if q & 2 != 0 { -v } else { v };
    // Keep the sign of zero (the reduction and polynomial lose it).
    let v = if x == 0.0 { x } else { v };
    // The comparison is false for NaN and +-inf, which both give NaN.
    if x.abs() <= TRIG_LIMIT { v } else { f32::NAN }
}

#[inline(always)]
pub(crate) fn cos(x: f32) -> f32 {
    let (r, q) = reduce_pio2(x);
    let s = sin_poly(r);
    let c = cos_poly(r);
    let v = if q & 1 != 0 { s } else { c };
    let v = if (q.wrapping_add(1)) & 2 != 0 { -v } else { v };
    if x.abs() <= TRIG_LIMIT { v } else { f32::NAN }
}

#[inline(always)]
pub(crate) fn tanh(x: f32) -> f32 {
    let a = x.abs();
    let z = x * x;
    // |x| < 0.625: odd polynomial (Cephes tanhf).
    let mut p = -5.704_988_727_45e-3_f32;
    p = p * z + 2.063_908_879_54e-2;
    p = p * z - 5.373_971_555_31e-2;
    p = p * z + 1.333_144_220_36e-1;
    p = p * z - 3.333_328_194_22e-1;
    let small = if x == 0.0 { x } else { (p * z) * x + x };
    // Otherwise 1 - 2 / (exp(2|x|) + 1); exp overflow to inf gives exactly 1.
    let big = (1.0 - 2.0 / (exp(a + a) + 1.0)).copysign(x);
    let res = if a < 0.625 { small } else { big };
    if x.is_nan() { x } else { res }
}

#[inline(always)]
pub(crate) fn erf(x: f32) -> f32 {
    let a = x.abs();
    // |x| < 1: erf(x) = x * P(x^2), degree-7 Chebyshev fit of erf(x)/x.
    let u = x * x;
    let mut p = -9.673_591_2e-6_f32;
    p = p * u + 1.126_825_4e-4;
    p = p * u - 8.484_396_4e-4;
    p = p * u + 5.221_053_5e-3;
    p = p * u - 2.686_543_8e-2;
    p = p * u + 1.128_378_3e-1;
    p = p * u - 3.761_263_8e-1;
    p = p * u + core::f32::consts::FRAC_2_SQRT_PI;
    let small = p * x;
    // 1 <= |x| < 4: erfc(a) = t * exp(-a^2 + h(t)), t = 1 / (1 + a / 2).
    let t = 1.0 / (1.0 + 0.5 * a);
    let mut h = 3.186_313_8e-1_f32;
    h = h * t - 1.477_073_2;
    h = h * t + 2.741_067_4;
    h = h * t - 2.487_147_8;
    h = h * t + 1.183_518_6;
    h = h * t - 5.745_427e-1;
    h = h * t + 2.035_827_8e-1;
    h = h * t + 3.558_948e-1;
    h = h * t + 1.001_777_5;
    h = h * t - 1.265_585_8;
    let erfc = t * exp(h - a * a);
    let big = (1.0 - erfc).copysign(x);
    let res = if a < 1.0 { small } else { big };
    // erf(4) = 1 - 1.5e-8 rounds to 1 in f32.
    let res = if a >= 4.0 { 1.0_f32.copysign(x) } else { res };
    if x.is_nan() { x } else { res }
}
