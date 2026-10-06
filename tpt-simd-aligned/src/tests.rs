extern crate std;
use super::*;
use proptest::prelude::*;
use std::vec::Vec;

#[test]
fn wrapper_layout() {
    assert_eq!(core::mem::align_of::<Aligned16<u8>>(), 16);
    assert_eq!(core::mem::align_of::<Aligned32<u8>>(), 32);
    assert_eq!(core::mem::align_of::<Aligned64<u8>>(), 64);
    assert_eq!(core::mem::size_of::<Aligned64<[u8; 65]>>(), 128);
    assert_eq!(Aligned32::<u8>::ALIGN, 32);
    let a: Aligned16<i32> = 5.into();
    assert_eq!(*a, 5);
}

#[test]
fn is_aligned_cases() {
    assert!(is_aligned(core::ptr::null::<u8>(), 64));
    assert!(is_aligned(64usize as *const u8, 64));
    assert!(!is_aligned(65usize as *const u8, 64));
    assert!(!is_aligned(64usize as *const u8, 0));
    assert!(!is_aligned(64usize as *const u8, 48));
    assert!(is_aligned(7usize as *const u8, 1));
}

#[test]
fn load_store_roundtrip() {
    let mut buf = Aligned32([0.0f32; 16]);
    let v = Simd::<f32, 8>::from_fn(|i| i as f32);
    store_aligned(v, &mut buf.0);
    store_aligned(v + Simd::splat(8.0), &mut buf.0[8..]);
    let r = load_aligned::<f32, 8>(&buf.0[8..]);
    assert_eq!(r[0], 8.0);
    assert_eq!(load_aligned::<f32, 8>(&buf.0), v);
}

#[test]
fn try_variants() {
    let mut buf = Aligned32([0i16; 32]);
    assert!(try_load_aligned::<i16, 16>(&buf.0).is_some());
    assert!(try_load_aligned::<i16, 16>(&buf.0[8..]).is_none()); // 16 bytes in
    assert!(try_load_aligned::<i16, 16>(&buf.0[16..]).is_some());
    assert!(try_load_aligned::<i16, 16>(&buf.0[..0]).is_none());
    assert!(!try_store_aligned(
        Simd::<i16, 16>::splat(1),
        &mut buf.0[1..]
    ));
    assert_eq!(buf.0[1], 0);
}

#[test]
#[should_panic(expected = "not aligned")]
fn load_misaligned_panics() {
    let buf = Aligned32([0.0f32; 16]);
    let _ = load_aligned::<f32, 8>(&buf.0[1..]);
}

#[test]
#[should_panic(expected = "too short")]
fn store_short_panics() {
    let mut buf = Aligned32([0.0f32; 4]);
    store_aligned(Simd::<f32, 8>::splat(0.0), &mut buf.0);
}

#[cfg(feature = "alloc")]
mod alloc_tests {
    use super::*;

    #[test]
    fn buf_alignment_and_content() {
        for n in [0usize, 1, 7, 8, 100] {
            let b = AlignedBuf::<f32, 64>::filled(n, 3.0);
            assert_eq!(b.len(), n);
            assert_eq!(b.is_empty(), n == 0);
            assert!(is_aligned(b.as_ptr(), 64));
            assert!(b.iter().all(|&x| x == 3.0));
        }
    }

    #[test]
    fn buf_clone_eq_debug() {
        let mut a = AlignedBuf::<u8, 16>::from_slice(&[1, 2, 3]);
        let b = a.clone();
        assert_eq!(a, b);
        a[0] = 9;
        assert_ne!(a, b);
        assert_eq!(std::format!("{b:?}"), "[1, 2, 3]");
        let e = AlignedBuf::<u8, 16>::from_slice(&[]);
        assert!(e.is_empty());
        assert_eq!(e.clone(), e);
    }

    #[test]
    fn buf_low_align_uses_type_align() {
        assert_eq!(AlignedBuf::<u64, 1>::BASE_ALIGN, 8);
        let b = AlignedBuf::<u64, 1>::zeroed(3);
        assert!(is_aligned(b.as_ptr(), 8));
    }

    #[test]
    fn helpers_and_simd() {
        let mut v = aligned_vec_f32(16);
        store_aligned(Simd::<f32, 8>::splat(1.0), &mut v);
        assert_eq!(load_aligned::<f32, 8>(&v)[3], 1.0);
        assert_eq!(v[8], 0.0);
        assert!(aligned_vec_i32(0).is_empty());
        assert!(is_aligned(aligned_vec_i16(5).as_ptr(), 32));
    }

    proptest! {
        #[test]
        fn buf_from_slice_roundtrip(d in proptest::collection::vec(any::<i32>(), 0..300)) {
            let b = AlignedBuf::<i32, 32>::from_slice(&d);
            prop_assert_eq!(&*b, &d[..]);
            prop_assert!(is_aligned(b.as_ptr(), 32));
        }
    }
}

proptest! {
    #[test]
    fn load_matches_from_slice(d in proptest::array::uniform16(any::<i16>())) {
        let buf = Aligned32(d);
        let v = load_aligned::<i16, 16>(&buf.0);
        prop_assert_eq!(v.to_array(), d);
        let mut out = Aligned32([0i16; 16]);
        store_aligned(v, &mut out.0);
        let o: Vec<i16> = out.0.to_vec();
        prop_assert_eq!(o, d.to_vec());
    }
}
