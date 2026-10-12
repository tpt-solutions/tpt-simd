//! Lane-parallel random number generators: uniform and normal fills.
//!
//! Run with: `cargo run -p tpt-simd-rng --example rng`

use tpt_simd_rng::{Philox4x32X8, Rng8, Xoshiro256ppX8};

fn main() {
    // Xoshiro256++, eight streams in lockstep, seeded deterministically.
    let mut rng = Xoshiro256ppX8::from_seed(42);

    let mut u = [0.0f32; 100];
    rng.fill_f32(&mut u);
    assert!(u.iter().all(|&x| (0.0..1.0).contains(&x)));

    let mut z = [0.0f64; 100];
    rng.fill_normal_f64(&mut z);

    // Philox is counter-based: any block can be reached with `seek`.
    let mut g = Philox4x32X8::new(1234, 7); // key 1234, stream 7
    let mut first = [0u32; 64];
    g.fill_u32(&mut first);
    g.seek(0); // back to block 0
    let mut again = [0u32; 64];
    g.fill_u32(&mut again);
    assert_eq!(first, again);

    println!("rng: all checks passed");
}
