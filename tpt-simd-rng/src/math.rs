//! Branch-free lane-wise `ln`, `sin`/`cos` (of a multiple of a full turn),
//! `sqrt` and the Box-Muller transform, on 8-lane arrays.
//!
//! Everything is plain arithmetic on `[T; 8]` (no libm, no FMA, no
//! data-dependent branches), so LLVM vectorises it and the results are
//! bit-identical on every target and feature set. Inputs outside the
//! documented domains give unspecified (but memory-safe) results.
//!
//! Accuracy (measured in this crate's tests against libm over the stated
//! domains; `ulp` = unit in the last place of the exact result):
//!
//! | function | domain | max error |
//! |---|---|---|
//! | [`ln_f32`] | normal positive `f32` | 1 ulp |
//! | [`sincos_turns_f32`] | `abs(t) < 2^20` | 2e-7 absolute |
//! | [`sqrt_f32`] | `0 <= x`, normal | 1 ulp |
//! | [`ln_f64`] | normal positive `f64` | 2 ulp |
//! | [`sincos_turns_f64`] | `abs(t) < 2^40` | 8e-16 absolute (dominated by the `2*pi*t` rounding) |
//! | [`sqrt_f64`] | `0 <= x`, normal | 1 ulp |

const L: usize = 8;

/// Natural logarithm of eight positive normal `f32` values (cephes `logf`
/// polynomial after range reduction to `[sqrt(1/2), sqrt(2)]`).
///
/// Zero, subnormals, negatives, infinities and NaN give unspecified results.
#[inline(always)]
pub fn ln_f32(x: [f32; L]) -> [f32; L] {
    let mut out = [0.0f32; L];
    for i in 0..L {
        let bits = x[i].to_bits();
        let mut e = ((bits >> 23) as i32) - 127;
        let mut m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);
        if m > core::f32::consts::SQRT_2 {
            m *= 0.5;
            e += 1;
        }
        let f = m - 1.0;
        let z = f * f;
        let mut y = 7.037_683_6e-2_f32;
        y = y * f - 1.151_461e-1;
        y = y * f + 1.167_699_9e-1;
        y = y * f - 1.242_014_1e-1;
        y = y * f + 1.424_932_3e-1;
        y = y * f - 1.666_805_8e-1;
        y = y * f + 2.000_071_5e-1;
        y = y * f - 2.499_999_4e-1;
        y = y * f + 3.333_333_4e-1;
        y = y * f * z;
        let ef = e as f32;
        y += ef * -2.121_944_4e-4;
        y -= 0.5 * z;
        out[i] = (f + y) + ef * 0.693_359_4;
    }
    out
}

/// `(sin(2*pi*t), cos(2*pi*t))` for eight `f32` values of `t` (in turns).
///
/// `t` is reduced to the nearest quarter turn exactly, then cephes `sinf` /
/// `cosf` polynomials are evaluated on `[-pi/4, pi/4]` and the quadrant is
/// applied with integer swaps and sign-bit XORs. Valid for `abs(t) < 2^20`.
#[inline(always)]
pub fn sincos_turns_f32(t: [f32; L]) -> ([f32; L], [f32; L]) {
    const MAGIC: f32 = 12_582_912.0; // 1.5 * 2^23: round-to-nearest-even trick
    const TAU: f32 = core::f32::consts::TAU;
    let mut sn = [0.0f32; L];
    let mut cs = [0.0f32; L];
    for i in 0..L {
        let q = (4.0 * t[i] + MAGIC) - MAGIC;
        let r = t[i] - q * 0.25;
        let th = r * TAU;
        let z = th * th;
        let s = th + th * z * (((-1.951_529_6e-4 * z) + 8.332_161e-3) * z - 1.666_665_5e-1);
        let c =
            1.0 - 0.5 * z + z * z * (((2.443_315_7e-5 * z) - 1.388_731_6e-3) * z + 4.166_664_6e-2);
        let qi = (q as i32) as u32;
        let swap = qi & 1 != 0;
        let (so, co) = if swap { (c, s) } else { (s, c) };
        let sneg = (qi >> 1) & 1;
        let cneg = (qi.wrapping_add(1) >> 1) & 1;
        sn[i] = f32::from_bits(so.to_bits() ^ (sneg << 31));
        cs[i] = f32::from_bits(co.to_bits() ^ (cneg << 31));
    }
    (sn, cs)
}

/// Square root of eight non-negative normal `f32` values: bit-hack reciprocal
/// square root, two Newton steps, then one residual correction. No sqrt
/// instruction and no libm.
#[inline(always)]
pub fn sqrt_f32(x: [f32; L]) -> [f32; L] {
    let mut out = [0.0f32; L];
    for i in 0..L {
        let v = x[i];
        let mut y = f32::from_bits(0x5F37_5A86u32.wrapping_sub(v.to_bits() >> 1));
        y *= 1.5 - 0.5 * v * y * y;
        y *= 1.5 - 0.5 * v * y * y;
        let s = v * y;
        out[i] = s + 0.5 * y * (v - s * s);
    }
    out
}

/// Natural logarithm of eight positive normal `f64` values (`atanh` series
/// `2s(1 + s^2/3 + ... + s^22/23)` with `s = f/(2+f)` after reduction to
/// `[sqrt(1/2), sqrt(2)]`).
///
/// Zero, subnormals, negatives, infinities and NaN give unspecified results.
#[inline(always)]
pub fn ln_f64(x: [f64; L]) -> [f64; L] {
    const LN2_HI: f64 = 6.931_471_803_691_238e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    let mut out = [0.0f64; L];
    for i in 0..L {
        let bits = x[i].to_bits();
        let eb = bits >> 52;
        let mut m = f64::from_bits((bits & 0x000F_FFFF_FFFF_FFFF) | 0x3FF0_0000_0000_0000);
        // eb as f64 via the 2^52 magic-number trick (vectorises on AVX2).
        let mut e = f64::from_bits(0x4330_0000_0000_0000 | eb) - (4_503_599_627_370_496.0 + 1023.0);
        if m > core::f64::consts::SQRT_2 {
            m *= 0.5;
            e += 1.0;
        }
        let f = m - 1.0;
        let s = f / (2.0 + f);
        let z = s * s;
        let mut p = 1.0 / 23.0;
        p = p * z + 1.0 / 21.0;
        p = p * z + 1.0 / 19.0;
        p = p * z + 1.0 / 17.0;
        p = p * z + 1.0 / 15.0;
        p = p * z + 1.0 / 13.0;
        p = p * z + 1.0 / 11.0;
        p = p * z + 1.0 / 9.0;
        p = p * z + 1.0 / 7.0;
        p = p * z + 1.0 / 5.0;
        p = p * z + 1.0 / 3.0;
        let r = 2.0 * s + 2.0 * s * z * p;
        out[i] = e * LN2_HI + (r + e * LN2_LO);
    }
    out
}

/// `(sin(2*pi*t), cos(2*pi*t))` for eight `f64` values of `t` (in turns),
/// valid for `abs(t) < 2^40`. Quarter-turn reduction as in
/// [`sincos_turns_f32`], then Taylor polynomials on `[-pi/4, pi/4]`.
#[inline(always)]
pub fn sincos_turns_f64(t: [f64; L]) -> ([f64; L], [f64; L]) {
    const MAGIC: f64 = 6_755_399_441_055_744.0; // 1.5 * 2^52
    const TAU: f64 = core::f64::consts::TAU;
    let mut sn = [0.0f64; L];
    let mut cs = [0.0f64; L];
    for i in 0..L {
        let q = (4.0 * t[i] + MAGIC) - MAGIC;
        let r = t[i] - q * 0.25;
        let th = r * TAU;
        let z = th * th;
        // sin: th * (1 - z/3! + z^2/5! - ... + z^8/17!)
        let mut ps = 1.0 / 355_687_428_096_000.0; // 1/17!
        ps = ps * z - 1.0 / 1_307_674_368_000.0; // 1/15!
        ps = ps * z + 1.0 / 6_227_020_800.0; // 1/13!
        ps = ps * z - 1.0 / 39_916_800.0; // 1/11!
        ps = ps * z + 1.0 / 362_880.0; // 1/9!
        ps = ps * z - 1.0 / 5_040.0;
        ps = ps * z + 1.0 / 120.0;
        ps = ps * z - 1.0 / 6.0;
        ps = ps * z + 1.0;
        let s = th * ps;
        // cos: 1 - z/2! + z^2/4! - ... + z^8/16!
        let mut pc = 1.0 / 20_922_789_888_000.0; // 1/16!
        pc = pc * z - 1.0 / 87_178_291_200.0; // 1/14!
        pc = pc * z + 1.0 / 479_001_600.0; // 1/12!
        pc = pc * z - 1.0 / 3_628_800.0; // 1/10!
        pc = pc * z + 1.0 / 40_320.0;
        pc = pc * z - 1.0 / 720.0;
        pc = pc * z + 1.0 / 24.0;
        pc = pc * z - 0.5;
        let c = pc * z + 1.0;
        let qi = (q as i64) as u64;
        let swap = qi & 1 != 0;
        let (so, co) = if swap { (c, s) } else { (s, c) };
        let sneg = (qi >> 1) & 1;
        let cneg = (qi.wrapping_add(1) >> 1) & 1;
        sn[i] = f64::from_bits(so.to_bits() ^ (sneg << 63));
        cs[i] = f64::from_bits(co.to_bits() ^ (cneg << 63));
    }
    (sn, cs)
}

/// Square root of eight non-negative normal `f64` values (bit-hack rsqrt,
/// three Newton steps, one residual correction).
#[inline(always)]
pub fn sqrt_f64(x: [f64; L]) -> [f64; L] {
    let mut out = [0.0f64; L];
    for i in 0..L {
        let v = x[i];
        let mut y = f64::from_bits(0x5FE6_EB50_C7B5_37A9u64.wrapping_sub(v.to_bits() >> 1));
        y *= 1.5 - 0.5 * v * y * y;
        y *= 1.5 - 0.5 * v * y * y;
        y *= 1.5 - 0.5 * v * y * y;
        let s = v * y;
        out[i] = s + 0.5 * y * (v - s * s);
    }
    out
}

/// Box-Muller transform: `u1` in `(0, 1]`, `u2` in `[0, 1)` give two
/// independent standard normals per lane,
/// `(r cos(2 pi u2), r sin(2 pi u2))` with `r = sqrt(-2 ln u1)`.
///
/// With 24-bit `u1` the largest possible `|z|` is `sqrt(2 ln 2^24) ~ 5.77`.
#[inline(always)]
pub fn box_muller_f32(u1: [f32; L], u2: [f32; L]) -> ([f32; L], [f32; L]) {
    let l = ln_f32(u1);
    let mut a = [0.0f32; L];
    for i in 0..L {
        a[i] = l[i] * -2.0;
    }
    let r = sqrt_f32(a);
    let (s, c) = sincos_turns_f32(u2);
    let mut z0 = [0.0f32; L];
    let mut z1 = [0.0f32; L];
    for i in 0..L {
        z0[i] = r[i] * c[i];
        z1[i] = r[i] * s[i];
    }
    (z0, z1)
}

/// `f64` Box-Muller; see [`box_muller_f32`]. With 52-bit `u1` the largest
/// possible `|z|` is `sqrt(2 ln 2^52) ~ 8.5`.
#[inline(always)]
pub fn box_muller_f64(u1: [f64; L], u2: [f64; L]) -> ([f64; L], [f64; L]) {
    let l = ln_f64(u1);
    let mut a = [0.0f64; L];
    for i in 0..L {
        a[i] = l[i] * -2.0;
    }
    let r = sqrt_f64(a);
    let (s, c) = sincos_turns_f64(u2);
    let mut z0 = [0.0f64; L];
    let mut z1 = [0.0f64; L];
    for i in 0..L {
        z0[i] = r[i] * c[i];
        z1[i] = r[i] * s[i];
    }
    (z0, z1)
}
