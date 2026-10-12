//! Number-theoretic transform: exact cyclic convolution modulo a prime.
//!
//! Run with: `cargo run -p tpt-simd-ntt --example ntt`

use tpt_simd_ntt::{NttPlan, cyclic_convolve_naive};

fn main() {
    // 998244353 is prime, 3 is a primitive root, and 8 divides q - 1.
    const Q: u32 = 998_244_353;
    let plan = NttPlan::new(Q, 3, 8);

    let mut a = [1u32, 2, 0, 0, 0, 0, 0, 0];
    let mut b = [3u32, 4, 0, 0, 0, 0, 0, 0];
    let expect = cyclic_convolve_naive(Q, &a, &b);
    plan.cyclic_convolve(&mut a, &mut b);
    assert_eq!(a[..], expect[..]);
    // (1 + 2x)(3 + 4x) = 3 + 10x + 8x^2
    assert_eq!(&a[..3], &[3, 10, 8]);

    println!("ntt: all checks passed");
}
