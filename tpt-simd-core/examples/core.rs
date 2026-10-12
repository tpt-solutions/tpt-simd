//! The core vector types, complex multiply, and build-time feature detection.
//!
//! Run with: `cargo run -p tpt-simd-core --example core`

use tpt_simd_core::detect::Features;
use tpt_simd_core::width::NATIVE_F32_LANES;
use tpt_simd_core::{ComplexSimd, F32x8};

fn main() {
    // Complex multiply on 8 lanes at once: (1+2i)(3+4i) = -5 + 10i.
    let a = ComplexSimd::new(F32x8::splat(1.0), F32x8::splat(2.0));
    let b = ComplexSimd::new(F32x8::splat(3.0), F32x8::splat(4.0));
    let r = a * b;
    assert_eq!(r.real.to_array(), [-5.0; 8]);
    assert_eq!(r.imag.to_array(), [10.0; 8]);

    // What the compiler may assume for this build, and how wide a register is.
    let f = Features::compile_time();
    println!("avx2 = {}, native f32 lanes = {}", f.avx2, NATIVE_F32_LANES);

    println!("core: all checks passed");
}
