//! Cache-blocked matrix transpose and strided copies.
//!
//! Run with: `cargo run -p tpt-simd-transpose --example transpose`

use tpt_simd_transpose::{transpose, transpose_square_in_place};

fn main() {
    // Out-of-place: a 2 x 3 row-major matrix becomes 3 x 2.
    let src = [1, 2, 3, 4, 5, 6];
    let mut dst = [0; 6];
    transpose(&src, 2, 3, &mut dst);
    assert_eq!(dst, [1, 4, 2, 5, 3, 6]);

    // In place for a square matrix.
    let mut m = [1, 2, 3, 4];
    transpose_square_in_place(&mut m, 2);
    assert_eq!(m, [1, 3, 2, 4]);

    println!("transpose: all checks passed");
}
