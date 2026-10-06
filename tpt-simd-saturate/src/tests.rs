use super::*;
use proptest::prelude::*;
use std::vec::Vec;

fn simd_of<T: tpt_simd_core::SimdElement + core::fmt::Debug, const N: usize>(
    elem: impl Strategy<Value = T>,
) -> impl Strategy<Value = Simd<T, N>> {
    proptest::collection::vec(elem, N).prop_map(|v| Simd::from_slice(&v))
}
use tpt_simd_testutil::{i16_edgy, i32_edgy, vec_of};

fn clamp_i8(x: i16) -> i8 {
    x.clamp(-128, 127) as i8
}
fn clamp_u8(x: i16) -> u8 {
    x.clamp(0, 255) as u8
}
fn clamp_i16(x: i32) -> i16 {
    x.clamp(-32768, 32767) as i16
}

#[test]
fn add_sub_edges() {
    let m = Simd::<i16, 16>::splat(i16::MAX);
    let n = Simd::<i16, 16>::splat(i16::MIN);
    assert_eq!(saturating_add_i16(m, m), m);
    assert_eq!(saturating_add_i16(n, n), n);
    assert_eq!(saturating_sub_i16(n, m), n);
    assert_eq!(saturating_sub_i16(m, n), m);
    assert_eq!(
        saturating_add_i8(Simd::splat(-128), Simd::splat(-1)),
        Simd::splat(-128)
    );
    assert_eq!(
        saturating_sub_i8(Simd::splat(127), Simd::splat(-1)),
        Simd::splat(127)
    );
    assert_eq!(
        saturating_add_u8(Simd::splat(255), Simd::splat(1)),
        Simd::splat(255)
    );
    assert_eq!(
        saturating_sub_u8(Simd::splat(0), Simd::splat(1)),
        Simd::splat(0)
    );
    assert_eq!(
        saturating_add_u16(Simd::splat(65535), Simd::splat(1)),
        Simd::splat(65535)
    );
    assert_eq!(
        saturating_sub_u16(Simd::splat(0), Simd::splat(1)),
        Simd::splat(0)
    );
}

#[test]
fn mul_edges() {
    let r = saturating_mul_i16(Simd::splat(i16::MIN), Simd::splat(i16::MIN));
    assert_eq!(r, Simd::splat(i16::MAX));
    let r = saturating_mul_i16(Simd::splat(i16::MIN), Simd::splat(i16::MAX));
    assert_eq!(r, Simd::splat(i16::MIN));
    let r = saturating_mul_i16(Simd::splat(181), Simd::splat(181)); // 32761
    assert_eq!(r, Simd::splat(32761));
    let r = saturating_mul_i16(Simd::splat(182), Simd::splat(182)); // 33124
    assert_eq!(r, Simd::splat(i16::MAX));
}

#[test]
fn pack_preserves_lane_order() {
    // Distinct values per lane so any 128-bit half interleave is detected.
    let v = Simd::<i16, 16>::from_fn(|i| i as i16 * 3 - 20);
    let p = pack_i16_to_i8(v);
    for i in 0..16 {
        assert_eq!(p[i], (i as i16 * 3 - 20) as i8);
    }
    let v = Simd::<i16, 16>::from_fn(|i| i as i16 * 40);
    let p = pack_i16_to_u8(v);
    for i in 0..16 {
        assert_eq!(p[i], clamp_u8(i as i16 * 40));
    }
    let v = Simd::<i32, 8>::from_fn(|i| (i as i32 - 4) * 20000);
    let p = pack_i32_to_i16(v);
    for i in 0..8 {
        assert_eq!(p[i], clamp_i16((i as i32 - 4) * 20000));
    }
}

#[test]
fn slice_tails_and_empty() {
    for len in 0..70usize {
        let src: Vec<i16> = (0..len).map(|i| (i as i16 - 30) * 17).collect();
        let mut d8 = std::vec![0i8; len];
        let mut du = std::vec![0u8; len];
        pack_i16_to_i8_slice(&src, &mut d8);
        pack_i16_to_u8_slice(&src, &mut du);
        for i in 0..len {
            assert_eq!(d8[i], clamp_i8(src[i]));
            assert_eq!(du[i], clamp_u8(src[i]));
        }
        let s32: Vec<i32> = (0..len).map(|i| (i as i32 - 30) * 4000).collect();
        let mut d16 = std::vec![0i16; len];
        pack_i32_to_i16_slice(&s32, &mut d16);
        for i in 0..len {
            assert_eq!(d16[i], clamp_i16(s32[i]));
        }
    }
}

#[test]
#[should_panic]
fn slice_len_mismatch_panics() {
    let mut d = [0i8; 3];
    pack_i16_to_i8_slice(&[1, 2], &mut d);
}

proptest! {
    #[test]
    fn prop_add_sub_i16(a in simd_of::<i16, 16>(i16_edgy()), b in simd_of::<i16, 16>(i16_edgy())) {
        let (s, d) = (saturating_add_i16(a, b), saturating_sub_i16(a, b));
        for i in 0..16 {
            prop_assert_eq!(s[i], (a[i] as i32 + b[i] as i32).clamp(-32768, 32767) as i16);
            prop_assert_eq!(d[i], (a[i] as i32 - b[i] as i32).clamp(-32768, 32767) as i16);
        }
    }

    #[test]
    fn prop_mul_i16(a in simd_of::<i16, 16>(i16_edgy()), b in simd_of::<i16, 16>(i16_edgy())) {
        let m = saturating_mul_i16(a, b);
        for i in 0..16 {
            prop_assert_eq!(m[i], clamp_i16(a[i] as i32 * b[i] as i32));
        }
    }

    #[test]
    fn prop_i8_u8(a in simd_of::<i8, 32>(any::<i8>()), b in simd_of::<i8, 32>(any::<i8>())) {
        let (s, d) = (saturating_add_i8(a, b), saturating_sub_i8(a, b));
        let (ua, ub) = (a.cast::<u8>(), b.cast::<u8>());
        let (us, ud) = (saturating_add_u8(ua, ub), saturating_sub_u8(ua, ub));
        for i in 0..32 {
            prop_assert_eq!(s[i], (a[i] as i16 + b[i] as i16).clamp(-128, 127) as i8);
            prop_assert_eq!(d[i], (a[i] as i16 - b[i] as i16).clamp(-128, 127) as i8);
            prop_assert_eq!(us[i], (ua[i] as u16 + ub[i] as u16).min(255) as u8);
            prop_assert_eq!(ud[i], ua[i].saturating_sub(ub[i]));
        }
    }

    #[test]
    fn prop_u16(a in simd_of::<u16, 16>(any::<u16>()), b in simd_of::<u16, 16>(any::<u16>())) {
        let (s, d) = (saturating_add_u16(a, b), saturating_sub_u16(a, b));
        for i in 0..16 {
            prop_assert_eq!(s[i], (a[i] as u32 + b[i] as u32).min(65535) as u16);
            prop_assert_eq!(d[i], a[i].saturating_sub(b[i]));
        }
    }

    #[test]
    fn prop_pack_slices(s in vec_of(i16_edgy(), 80)) {
        let mut d8 = std::vec![0i8; s.len()];
        let mut du = std::vec![0u8; s.len()];
        pack_i16_to_i8_slice(&s, &mut d8);
        pack_i16_to_u8_slice(&s, &mut du);
        for i in 0..s.len() {
            prop_assert_eq!(d8[i], clamp_i8(s[i]));
            prop_assert_eq!(du[i], clamp_u8(s[i]));
        }
    }

    #[test]
    fn prop_pack_i32(s in vec_of(i32_edgy(), 40)) {
        let mut d = std::vec![0i16; s.len()];
        pack_i32_to_i16_slice(&s, &mut d);
        for i in 0..s.len() {
            prop_assert_eq!(d[i], clamp_i16(s[i]));
        }
    }
}
