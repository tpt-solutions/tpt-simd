use crate::*;

#[test]
fn complex_arithmetic() {
    let a = ComplexSimd::new(F32x4::splat(1.0), F32x4::splat(2.0));
    let b = ComplexSimd::new(F32x4::splat(3.0), F32x4::splat(4.0));
    let m = a * b;
    assert_eq!(m.real.to_array(), [-5.0; 4]);
    assert_eq!(m.imag.to_array(), [10.0; 4]);
    let q = m / b;
    assert_eq!(q.real.to_array(), [1.0; 4]);
    assert_eq!(q.imag.to_array(), [2.0; 4]);
    assert_eq!(a.conj().imag.to_array(), [-2.0; 4]);
    assert_eq!(b.mag().to_array(), [5.0; 4]);
    let p = ComplexSimd::new(F32x4::splat(0.0), F32x4::splat(1.0)).phase();
    assert!((p[0] - core::f32::consts::FRAC_PI_2).abs() < 1e-6);
}

#[test]
fn butterfly_in_place() {
    let mut a = ComplexSimd::new(F32x4::splat(1.0), F32x4::splat(1.0));
    let mut b = ComplexSimd::new(F32x4::splat(0.5), F32x4::splat(-1.0));
    a.butterfly(&mut b);
    assert_eq!(a.real.to_array(), [1.5; 4]);
    assert_eq!(a.imag.to_array(), [0.0; 4]);
    assert_eq!(b.real.to_array(), [0.5; 4]);
    assert_eq!(b.imag.to_array(), [2.0; 4]);
}

#[test]
fn width_and_detect() {
    assert_eq!(width::native_lanes::<f32>() * 4, width::NATIVE_VECTOR_BYTES);
    let _ = detect::Features::compile_time();
    let _ = detect::Features::runtime();
}

#[test]
fn backend_trait() {
    fn sum<V: SimdVector<i32, 4>>(v: V) -> i32 {
        v.to_array().iter().sum()
    }
    let v = <DefaultBackend as SimdOps<i32, 4>>::Vector::from_array([1, 2, 3, 4]);
    assert_eq!(sum(v), 10);
}
