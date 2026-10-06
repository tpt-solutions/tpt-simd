use crate::*;
use core::mem::{align_of, size_of};
use proptest::prelude::*;

#[test]
fn layout_matches_array() {
    assert_eq!(size_of::<Simd<f32, 8>>(), 32);
    assert_eq!(size_of::<Simd<i16, 16>>(), 32);
    assert_eq!(align_of::<Simd<i32, 8>>(), align_of::<i32>());
    let v = Simd::<i32, 4>::from_array([1, 2, 3, 4]);
    // Lane i is element i of the backing array (transparent layout).
    let p = &v as *const Simd<i32, 4> as usize;
    for i in 0..4 {
        assert_eq!(&v[i] as *const i32 as usize, p + 4 * i);
    }
}

#[test]
fn integer_wrapping_and_saturating() {
    let a = Simd::<i16, 4>::from_array([i16::MAX, i16::MIN, 5, -5]);
    let b = Simd::splat(1);
    assert_eq!((a + b).to_array(), [i16::MIN, i16::MIN + 1, 6, -4]);
    assert_eq!(a.sat_add(b).to_array(), [i16::MAX, i16::MIN + 1, 6, -4]);
    assert_eq!(a.sat_sub(b).to_array(), [i16::MAX - 1, i16::MIN, 4, -6]);
    assert_eq!((-a).to_array(), [-i16::MAX, i16::MIN, -5, 5]);
    assert_eq!(a.abs().to_array(), [i16::MAX, i16::MIN, 5, 5]);
}

#[test]
fn shift_policy() {
    let v = Simd::<i32, 2>::from_array([-8, 8]);
    assert_eq!((v >> 1).to_array(), [-4, 4]);
    assert_eq!((v >> 40).to_array(), [-1, 0]);
    assert_eq!((v << 40).to_array(), [0, 0]);
    assert_eq!(v.shr_logical(1).to_array(), [0x7FFF_FFFC, 4]);
    let u = Simd::<u8, 2>::from_array([0x81, 1]);
    assert_eq!(u.rotl(1).to_array(), [0x03, 2]);
    assert_eq!(u.rotl(9).to_array(), [0x03, 2]);
}

#[test]
fn float_ops_and_nan_policy() {
    let a = Simd::<f32, 4>::from_array([1.0, f32::NAN, -2.5, 2.5]);
    let b = Simd::<f32, 4>::from_array([2.0, 1.0, f32::NAN, 2.5]);
    assert_eq!(a.min(b).to_array()[..2], [1.0, 1.0]);
    assert_eq!(a.max(b).to_array()[2], -2.5);
    assert!(!a.simd_gt(b).test(1));
    assert!(a.simd_ne(b).test(1));
    assert!(a.is_nan().test(1));
    let r = Simd::<f32, 4>::from_array([0.5, 1.5, 2.5, -0.5]);
    assert_eq!(r.round().to_array(), [1.0, 2.0, 3.0, -1.0]);
    assert_eq!(r.round_ties_even().to_array(), [0.0, 2.0, 2.0, -0.0]);
    assert_eq!(r.trunc().to_array(), [0.0, 1.0, 2.0, -0.0]);
    assert_eq!(Simd::<f32, 2>::from_array([4.0, 9.0]).sqrt().to_array(), [2.0, 3.0]);
}

#[test]
fn casts() {
    let f = Simd::<f32, 4>::from_array([1.9, -1.9, f32::NAN, 1e20]);
    assert_eq!(f.cast::<i32>().to_array(), [1, -1, 0, i32::MAX]);
    let i = Simd::<i32, 2>::from_array([300, -1]);
    assert_eq!(i.cast::<u8>().to_array(), [44, 255]);
    assert_eq!(i.cast::<f64>().to_array(), [300.0, -1.0]);
}

#[test]
fn mask_ops() {
    let a = Simd::<i32, 8>::from_array([1, 2, 3, 4, 5, 6, 7, 8]);
    let m = a.simd_gt(Simd::splat(4));
    assert_eq!(m.count(), 4);
    assert_eq!(m.to_bitmask(), 0b1111_0000);
    assert_eq!(SimdMask::<i32, 8>::from_bitmask(0b1111_0000), m);
    assert!(m.any() && !m.all() && (!m).any());
    let r = m.select(Simd::splat(1), Simd::splat(0));
    assert_eq!(r.to_array(), [0, 0, 0, 0, 1, 1, 1, 1]);
}

#[test]
fn slices() {
    let data = [1i16, 2, 3];
    assert_eq!(Simd::<i16, 4>::from_slice_or(&data, -1).to_array(), [1, 2, 3, -1]);
    let mut out = [0i16; 2];
    Simd::<i16, 4>::from_array([9, 8, 7, 6]).store_partial(&mut out);
    assert_eq!(out, [9, 8]);
}

#[test]
#[should_panic]
fn from_slice_short_panics() {
    let _ = Simd::<i32, 4>::from_slice(&[1, 2, 3]);
}

#[test]
fn swizzle_reverse() {
    let v = Simd::<u8, 4>::from_array([10, 20, 30, 40]);
    assert_eq!(v.reverse().to_array(), [40, 30, 20, 10]);
    assert_eq!(v.swizzle([3, 3, 0, 1]).to_array(), [40, 40, 10, 20]);
}

proptest! {
    #[test]
    fn i16_ops_match_scalar(a in prop::array::uniform16(any::<i16>()), b in prop::array::uniform16(any::<i16>())) {
        let (va, vb) = (Simd::from_array(a), Simd::from_array(b));
        for i in 0..16 {
            prop_assert_eq!((va + vb)[i], a[i].wrapping_add(b[i]));
            prop_assert_eq!((va - vb)[i], a[i].wrapping_sub(b[i]));
            prop_assert_eq!((va * vb)[i], a[i].wrapping_mul(b[i]));
            prop_assert_eq!(va.sat_add(vb)[i], a[i].saturating_add(b[i]));
            prop_assert_eq!(va.sat_sub(vb)[i], a[i].saturating_sub(b[i]));
            prop_assert_eq!(va.min(vb)[i], a[i].min(b[i]));
            prop_assert_eq!((va ^ vb)[i], a[i] ^ b[i]);
            prop_assert_eq!(va.simd_lt(vb).test(i), a[i] < b[i]);
        }
    }

    #[test]
    fn f32_ops_match_scalar(a in prop::array::uniform8(-1e6f32..1e6), b in prop::array::uniform8(-1e6f32..1e6)) {
        let (va, vb) = (Simd::from_array(a), Simd::from_array(b));
        for i in 0..8 {
            prop_assert_eq!((va + vb)[i], a[i] + b[i]);
            prop_assert_eq!((va * vb)[i], a[i] * b[i]);
            prop_assert_eq!(va.mul_add(vb, va)[i], a[i].mul_add(b[i], a[i]));
            prop_assert_eq!(va.max(vb)[i], a[i].max(b[i]));
        }
    }

    #[test]
    fn shifts_match_scalar(a in prop::array::uniform8(any::<i32>()), n in 0u32..40) {
        let v = Simd::from_array(a);
        for i in 0..8 {
            let l = if n >= 32 { 0 } else { a[i] << n };
            let r = if n >= 32 { a[i] >> 31 } else { a[i] >> n };
            prop_assert_eq!((v << n)[i], l);
            prop_assert_eq!((v >> n)[i], r);
        }
    }
}
