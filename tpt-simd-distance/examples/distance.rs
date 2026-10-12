//! Vector distances for search and clustering: f32 embeddings and integer codes.
//!
//! Run with: `cargo run -p tpt-simd-distance --example distance`

use tpt_simd_distance::{
    cosine_distance_f32, hamming_distance_u64, inner_product_f32, l2_squared_f32, l2_squared_u8,
    top_k_l2_f32,
};

fn main() {
    let query = [1.0f32, 0.0, 0.0];
    let a = [1.0f32, 0.0, 0.0];
    let b = [0.0f32, 1.0, 0.0];
    assert_eq!(inner_product_f32(&query, &a), 1.0);
    assert_eq!(l2_squared_f32(&query, &b), 2.0);
    assert!((cosine_distance_f32(&query, &b) - 1.0).abs() < 1e-6); // orthogonal

    // Integer codes (e.g. quantised embeddings) are exact.
    assert_eq!(l2_squared_u8(&[0, 3], &[4, 0]), 16 + 9);

    // Binary signatures: number of differing bits.
    assert_eq!(hamming_distance_u64(&[0b1011], &[0b0010]), 2);

    // Nearest-neighbour search: the two closest vectors to the query, by squared L2.
    // Rows are stored flat, `dim` values per row.
    let rows = [0.0f32, 0.0, 5.0, 5.0, 1.0, 0.0, 9.0, 9.0];
    let mut idx = [0usize; 2];
    let mut dist = [0.0f32; 2];
    let n = top_k_l2_f32(&[0.9, 0.1], &rows, 2, &mut idx, &mut dist);
    assert_eq!(n, 2);
    assert_eq!(idx, [2, 0]);

    println!("distance: all checks passed");
}
