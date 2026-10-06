extern crate std;
use super::*;
use proptest::prelude::*;
use std::vec;
use std::vec::Vec;
use tpt_simd_core::{F32x8, I32x8};

fn ref_t8(m: [[i16; 8]; 8]) -> [[i16; 8]; 8] {
    core::array::from_fn(|i| core::array::from_fn(|j| m[j][i]))
}

#[test]
fn transpose8_known() {
    let mut m: [[i16; 8]; 8] =
        core::array::from_fn(|r| core::array::from_fn(|c| (r * 8 + c) as i16));
    let want = ref_t8(m);
    transpose_8x8_i16(&mut m);
    assert_eq!(m, want);
    assert_eq!(m[1][0], 1);
    transpose_8x8_i16(&mut m);
    assert_eq!(m[1][0], 8);
}

#[test]
fn transpose4_nan_and_zero_bits() {
    let mut m = [[0.0f32; 4]; 4];
    m[0][1] = f32::NAN;
    m[2][3] = -0.0;
    m[1][0] = f32::INFINITY;
    transpose_4x4_f32(&mut m);
    assert!(m[1][0].is_nan());
    assert_eq!(m[3][2].to_bits(), (-0.0f32).to_bits());
    assert_eq!(m[0][1], f32::INFINITY);
}

#[test]
fn stereo_empty_and_tails() {
    let mut out: [i16; 0] = [];
    interleave_stereo_i16(&[], &[], &mut out);
    for n in [1usize, 7, 8, 9, 31, 33] {
        let l: Vec<i16> = (0..n as i16).collect();
        let r: Vec<i16> = l.iter().map(|x| -x - 1).collect();
        let mut o = vec![0i16; 2 * n];
        interleave_stereo_i16(&l, &r, &mut o);
        for i in 0..n {
            assert_eq!((o[2 * i], o[2 * i + 1]), (l[i], r[i]));
        }
    }
}

#[test]
#[should_panic(expected = "output length")]
fn stereo_bad_output() {
    interleave_stereo_i16(&[1, 2], &[1, 2], &mut [0; 3]);
}

#[test]
#[should_panic(expected = "length mismatch")]
fn stereo_bad_lr() {
    interleave_stereo_i16(&[1, 2], &[1], &mut [0; 3]);
}

#[test]
#[should_panic(expected = "even")]
fn deinterleave_odd() {
    deinterleave_stereo_i16(&[1, 2, 3], &mut [0; 1], &mut [0; 1]);
}

#[test]
#[should_panic(expected = "4 * u.len()")]
fn yuv_bad_y() {
    interleave_yuv420(&[0; 3], &[0], &[0], &mut [0; 5]);
}

#[test]
#[should_panic(expected = "u and v")]
fn yuv_bad_uv() {
    interleave_yuv420(&[0; 4], &[0], &[], &mut [0; 5]);
}

#[test]
fn yuv_empty() {
    interleave_yuv420(&[], &[], &[], &mut []);
}

#[test]
#[should_panic(expected = "length mismatch")]
fn unpack_bad() {
    unpack_i8_to_i16(&[1], &mut [0; 2]);
}

#[test]
fn reexport_pack() {
    let src = [-500i16, 500];
    let mut dst = [0i8; 2];
    pack_i16_to_i8_slice(&src, &mut dst);
    assert_eq!(dst, [-128, 127]);
    assert_eq!(pack_i16_to_i8(Simd::splat(1000))[0], 127);
}

#[test]
fn permute_basic() {
    let v = F32x8::from_fn(|i| i as f32);
    let r = permute_f32(v, Simd::from_array([8, 9, 10, 11, 12, 13, 14, 15]));
    assert_eq!(r, v);
    let w = I32x8::from_fn(|i| i as i32);
    assert_eq!(permute_i32(w, Simd::splat(u32::MAX)).to_array(), [7; 8]);
}

proptest! {
    #[test]
    fn t8_matches(m in proptest::array::uniform8(proptest::array::uniform8(any::<i16>()))) {
        let mut a = m;
        let mut b = m;
        transpose_8x8_i16(&mut a);
        transpose_8x8_i16_portable(&mut b);
        prop_assert_eq!(a, b);
        prop_assert_eq!(a, ref_t8(m));
    }

    #[test]
    fn t4_matches_bits(m in proptest::array::uniform4(proptest::array::uniform4(any::<u32>()))) {
        let f: [[f32; 4]; 4] = m.map(|r| r.map(f32::from_bits));
        let (mut a, mut b) = (f, f);
        transpose_4x4_f32(&mut a);
        transpose_4x4_f32_portable(&mut b);
        for i in 0..4 { for j in 0..4 {
            prop_assert_eq!(a[i][j].to_bits(), b[i][j].to_bits());
            prop_assert_eq!(a[i][j].to_bits(), f[j][i].to_bits());
        } }
    }

    #[test]
    fn stereo_roundtrip(l in proptest::collection::vec(any::<i16>(), 0..200), seed in any::<i16>()) {
        let r: Vec<i16> = l.iter().map(|x| x.wrapping_add(seed)).collect();
        let mut o = vec![0i16; l.len() * 2];
        interleave_stereo_i16(&l, &r, &mut o);
        let (mut l2, mut r2) = (vec![0i16; l.len()], vec![0i16; l.len()]);
        deinterleave_stereo_i16(&o, &mut l2, &mut r2);
        prop_assert_eq!(l, l2);
        prop_assert_eq!(r, r2);
    }

    #[test]
    fn yuv_matches(u in proptest::collection::vec(any::<u8>(), 0..64), s in any::<u8>()) {
        let v: Vec<u8> = u.iter().map(|x| x ^ s).collect();
        let y: Vec<u8> = (0..u.len() * 4).map(|i| (i * 7) as u8).collect();
        let mut o = vec![0u8; y.len() + 2 * u.len()];
        interleave_yuv420(&y, &u, &v, &mut o);
        prop_assert_eq!(&o[..y.len()], &y[..]);
        for i in 0..u.len() {
            prop_assert_eq!(o[y.len() + 2 * i], u[i]);
            prop_assert_eq!(o[y.len() + 2 * i + 1], v[i]);
        }
    }

    #[test]
    fn unpack_matches(d in proptest::collection::vec(any::<i8>(), 0..200)) {
        let mut o = vec![0i16; d.len()];
        unpack_i8_to_i16(&d, &mut o);
        for (a, b) in d.iter().zip(&o) { prop_assert_eq!(i16::from(*a), *b); }
    }

    #[test]
    fn permute_matches(
        v in proptest::array::uniform8(any::<u32>()),
        idx in proptest::array::uniform8(any::<u32>()),
    ) {
        let f = F32x8::from_array(v.map(f32::from_bits));
        let a = permute_f32(f, Simd::from_array(idx));
        let b = permute_f32_portable(f, Simd::from_array(idx));
        let g = permute(f, Simd::from_array(idx.map(|x| x & 7)));
        for i in 0..8 {
            prop_assert_eq!(a[i].to_bits(), b[i].to_bits());
            prop_assert_eq!(a[i].to_bits(), g[i].to_bits());
        }
        let w = I32x8::from_array(v.map(|x| x as i32));
        prop_assert_eq!(
            permute_i32(w, Simd::from_array(idx)),
            permute_i32_portable(w, Simd::from_array(idx))
        );
    }
}
