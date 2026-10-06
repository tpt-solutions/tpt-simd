use crate::Fixed;
use proptest::prelude::*;
use tpt_simd_core::{F32x8, Simd};
use tpt_simd_testutil::{i32_edgy, simd_of};

const fn rmin(i: u32, f: u32) -> i128 {
    -(1i128 << (i + f - 1))
}
const fn rmax(i: u32, f: u32) -> i128 {
    (1i128 << (i + f - 1)) - 1
}

/// Independent reference: round-half-away via magnitude.
fn ref_mul(a: i32, b: i32, f: u32) -> i128 {
    let p = a as i128 * b as i128;
    let half = if f == 0 { 0 } else { 1i128 << (f - 1) };
    let m = (p.abs() + half) >> f;
    if p < 0 { -m } else { m }
}

fn ref_div(a: i32, b: i32, f: u32) -> i128 {
    let n = (a as i128) << f;
    let d = b as i128;
    let (na, da) = (n.abs(), d.abs());
    let mut q = na / da;
    if 2 * (na % da) >= da {
        q += 1;
    }
    if (n < 0) != (d < 0) { -q } else { q }
}

fn ref_shr_round(w: i128, d: u32) -> i128 {
    let half = 1i128 << (d - 1);
    let m = (w.abs() + half) >> d;
    if w < 0 { -m } else { m }
}

fn ref_conv(w: i128, f: u32, f2: u32) -> i128 {
    if f2 >= f {
        w << (f2 - f)
    } else {
        ref_shr_round(w, f - f2)
    }
}

macro_rules! format_tests {
    ($($modname:ident: $i:literal, $f:literal;)*) => {$(
        mod $modname {
            use super::*;
            type Q = Fixed<i32, $i, $f, 11>; // 11 lanes: exercises the AVX2 tail
            type Q8 = Fixed<i32, $i, $f, 8>;
            const MIN: i128 = rmin($i, $f);
            const MAX: i128 = rmax($i, $f);
            fn raw_strategy() -> BoxedStrategy<i32> {
                // raw values inside the format range
                prop_oneof![
                    i32_edgy().prop_map(|v| (v as i128).clamp(MIN, MAX) as i32),
                    (MIN as i32..=MAX as i32),
                ].boxed()
            }

            proptest! {
                #[test]
                fn mul_matches_scalar(a in simd_of::<i32,_,11>(raw_strategy()), b in simd_of::<i32,_,11>(raw_strategy())) {
                    let (qa, qb) = (Q::from_raw(a), Q::from_raw(b));
                    let sat = qa.saturating_mul(qb).raw();
                    let wrap = qa.wrapping_mul(qb).raw();
                    let tr = qa.mul_trunc(qb).raw();
                    for i in 0..11 {
                        let r = ref_mul(a[i], b[i], $f);
                        prop_assert_eq!(sat[i] as i128, r.clamp(MIN, MAX));
                        prop_assert_eq!(wrap[i], r as i32);
                        let t = (a[i] as i128 * b[i] as i128) >> $f;
                        prop_assert_eq!(tr[i], t as i32);
                    }
                    prop_assert_eq!((qa * qb).raw(), wrap);
                }

                #[test]
                fn mul_8_lane(a in simd_of::<i32,_,8>(raw_strategy()), b in simd_of::<i32,_,8>(raw_strategy())) {
                    let sat = Q8::from_raw(a).saturating_mul(Q8::from_raw(b)).raw();
                    for i in 0..8 {
                        prop_assert_eq!(sat[i] as i128, ref_mul(a[i], b[i], $f).clamp(MIN, MAX));
                    }
                }

                #[test]
                fn add_sub_match(a in simd_of::<i32,_,11>(raw_strategy()), b in simd_of::<i32,_,11>(raw_strategy())) {
                    let (qa, qb) = (Q::from_raw(a), Q::from_raw(b));
                    let (sa, ss) = (qa.saturating_add(qb).raw(), qa.saturating_sub(qb).raw());
                    let (wa, ws) = ((qa + qb).raw(), (qa - qb).raw());
                    for i in 0..11 {
                        prop_assert_eq!(sa[i] as i128, (a[i] as i128 + b[i] as i128).clamp(MIN, MAX));
                        prop_assert_eq!(ss[i] as i128, (a[i] as i128 - b[i] as i128).clamp(MIN, MAX));
                        prop_assert_eq!(wa[i], a[i].wrapping_add(b[i]));
                        prop_assert_eq!(ws[i], a[i].wrapping_sub(b[i]));
                    }
                }

                #[test]
                fn div_matches(a in simd_of::<i32,_,11>(raw_strategy()), b in simd_of::<i32,_,11>(raw_strategy())) {
                    let b = b.map(|v| if v == 0 { 1 } else { v });
                    let (qa, qb) = (Q::from_raw(a), Q::from_raw(b));
                    let w = qa.div(qb).raw();
                    let s = qa.saturating_div(qb).raw();
                    for i in 0..11 {
                        let r = ref_div(a[i], b[i], $f);
                        prop_assert_eq!(w[i], r as i32);
                        prop_assert_eq!(s[i] as i128, r.clamp(MIN, MAX));
                    }
                    prop_assert_eq!(qa.checked_div(qb), Some(qa.div(qb)));
                }

                #[test]
                fn neg_abs(a in simd_of::<i32,_,11>(raw_strategy())) {
                    let q = Q::from_raw(a);
                    for i in 0..11 {
                        prop_assert_eq!((-q).raw()[i], a[i].wrapping_neg());
                        prop_assert_eq!(q.abs().raw()[i], a[i].wrapping_abs());
                        prop_assert_eq!(q.saturating_neg().raw()[i] as i128, (-(a[i] as i128)).clamp(MIN, MAX));
                        prop_assert_eq!(q.saturating_abs().raw()[i] as i128, (a[i] as i128).abs().clamp(MIN, MAX));
                    }
                }

                #[test]
                fn f32_roundtrip(a in simd_of::<i32,_,11>(raw_strategy())) {
                    let q = Q::from_raw(a);
                    let back = Q::from_f32(q.to_f32());
                    for i in 0..11 {
                        // f32 has 24 bits of precision: exact when the raw value fits.
                        if (a[i] as i64).abs() < (1 << 24) {
                            prop_assert_eq!(back.raw()[i], a[i]);
                        } else {
                            prop_assert!((back.raw()[i] as i64 - a[i] as i64).abs() <= 1 << 8);
                        }
                    }
                }

                #[test]
                fn convert_matches(a in simd_of::<i32,_,11>(raw_strategy())) {
                    let q = Q::from_raw(a);
                    let c: Fixed<i32, 12, 8, 11> = q.convert();
                    let d: Fixed<i32, 4, 20, 11> = q.convert();
                    for i in 0..11 {
                        let w = a[i] as i128;
                        let e = ref_conv(w, $f, 8);
                        prop_assert_eq!(c.raw()[i] as i128, e.clamp(rmin(12, 8), rmax(12, 8)));
                        let e = ref_conv(w, $f, 20);
                        prop_assert_eq!(d.raw()[i] as i128, e.clamp(rmin(4, 20), rmax(4, 20)));
                    }
                }
            }
        }
    )*};
}

format_tests! {
    q16_16: 16, 16;
    q8_24: 8, 24;
    q1_31: 1, 31;
    q4_12: 4, 12;
    q2_30: 2, 30;
    q32_0: 32, 0;
    q20_8: 20, 8;
}

#[test]
fn spec_example() {
    let a: Fixed<i32, 16, 16, 8> = Fixed::from_f32(F32x8::splat(1.5));
    let b: Fixed<i32, 16, 16, 8> = Fixed::from_f32(F32x8::splat(2.5));
    assert_eq!((a * b).to_f32(), F32x8::splat(3.75));
}

#[test]
fn mul_ties_away_from_zero() {
    type Q = Fixed<i32, 16, 16, 8>;
    let half = Q::splat_raw(1 << 15);
    for (x, want) in [(1, 1), (-1, -1), (3, 2), (-3, -2), (2, 1), (-2, -1), (0, 0)] {
        assert_eq!(Q::splat_raw(x).saturating_mul(half), Q::splat_raw(want));
        assert_eq!(Q::splat_raw(x).wrapping_mul(half), Q::splat_raw(want));
    }
}

#[test]
fn mul_overflow_extremes() {
    type Q = Fixed<i32, 16, 16, 8>;
    let (mn, mx) = (Q::min_value(), Q::max_value());
    assert_eq!(mn.saturating_mul(mn), mx);
    assert_eq!(mn.saturating_mul(mx), mn);
    assert_eq!(mx.saturating_mul(mx), mx);
    // MIN * MIN = 2^30 (in value) wraps to 0 in 16.16
    assert_eq!(mn.wrapping_mul(mn), Q::splat_raw(0));
}

#[test]
fn from_f32_specials() {
    type Q = Fixed<i32, 8, 24, 8>;
    let v = Simd::from_array([
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        200.0,
        -200.0,
        0.0,
        -0.0,
        1.0,
    ]);
    let q = Q::from_f32(v);
    let r = q.raw().to_array();
    assert_eq!(
        r,
        [i32::MAX, i32::MIN, 0, i32::MAX, i32::MIN, 0, 0, 1 << 24]
    );
    // narrow format clamps to the format range
    let n = Fixed::<i32, 4, 12, 2>::from_f32(Simd::from_array([100.0, -100.0]));
    assert_eq!(n.raw().to_array(), [(1 << 15) - 1, -(1 << 15)]);
}

#[test]
fn division() {
    type Q = Fixed<i32, 16, 16, 4>;
    let three = Q::splat_f32(3.0);
    assert_eq!(Q::splat_f32(1.0).div(three).raw().to_array(), [21845; 4]);
    assert_eq!((Q::splat_f32(-1.0) / three).raw().to_array(), [-21845; 4]);
    // 2 / 3 -> 43690.67 rounds up
    assert_eq!(Q::splat_f32(2.0).div(three).raw().to_array(), [43691; 4]);
    // tie: raw 1 / raw 2 in Q16.16 = 0.5 exactly; (1<<16)/ (1<<17)
    assert_eq!(
        Q::splat_raw(1).div(Q::splat_f32(2.0)).raw().to_array(),
        [1; 4]
    );
    assert_eq!(
        Q::splat_raw(-1).div(Q::splat_f32(2.0)).raw().to_array(),
        [-1; 4]
    );
    // MIN / -1 wraps, saturating variant saturates
    assert_eq!(Q::min_value().div(Q::splat_f32(-1.0)), Q::min_value());
    assert_eq!(
        Q::min_value().saturating_div(Q::splat_f32(-1.0)),
        Q::max_value()
    );
}

#[test]
#[should_panic(expected = "division by zero")]
fn div_by_zero_panics() {
    type Q = Fixed<i32, 16, 16, 4>;
    let _ = Q::one().div(Q::splat_raw(0));
}

#[test]
fn saturating_div_zero() {
    type Q = Fixed<i32, 16, 16, 3>;
    let a = Q::from_raw(Simd::from_array([5, -5, 0]));
    assert_eq!(
        a.saturating_div(Q::splat_raw(0)).raw().to_array(),
        [i32::MAX, i32::MIN, 0]
    );
    assert!(a.checked_div(Q::splat_raw(0)).is_none());
}

#[test]
fn constants_and_empty() {
    type Q = Fixed<i32, 16, 16, 4>;
    assert_eq!(Q::one().raw().to_array(), [65536; 4]);
    assert_eq!(Q::epsilon().raw().to_array(), [1; 4]);
    // zero lanes is a valid (degenerate) vector
    let e = Fixed::<i32, 16, 16, 0>::from_raw(Simd::from_array([]));
    assert_eq!(e.wrapping_mul(e), e);
    assert_eq!(e.saturating_mul(e), e);
}

#[test]
fn i16_format() {
    type Q = Fixed<i16, 4, 12, 8>;
    let a = Q::splat_f32(1.5);
    let b = Q::splat_f32(2.5);
    assert_eq!((a * b).to_f32(), Simd::splat(3.75));
    assert_eq!(
        Q::splat_f32(7.0).saturating_mul(Q::splat_f32(7.0)),
        Q::max_value()
    );
    assert_eq!(
        Q::splat_f32(1.0).div(Q::splat_f32(4.0)).to_f32(),
        Simd::splat(0.25)
    );
    let c: Fixed<i16, 8, 8, 8> = a.convert();
    assert_eq!(c.to_f32(), Simd::splat(1.5));
}
