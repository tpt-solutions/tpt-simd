//! Property tests: transposes against the index formula.

use proptest::prelude::*;
use tpt_simd_transpose::{transpose, transpose_square_in_place};

proptest! {
    #[test]
    fn transpose_matches_index_formula(
        rows in 0usize..40,
        cols in 0usize..40,
        seed in any::<u64>(),
    ) {
        let src: Vec<u32> = (0..rows * cols).map(|i| (i as u64 ^ seed) as u32).collect();
        let mut dst = vec![0u32; rows * cols];
        transpose(&src, rows, cols, &mut dst);
        for r in 0..rows {
            for c in 0..cols {
                prop_assert_eq!(dst[c * rows + r], src[r * cols + c]);
            }
        }
    }

    #[test]
    fn transpose_twice_is_identity(rows in 0usize..40, cols in 0usize..40) {
        let src: Vec<i16> = (0..rows * cols).map(|i| i as i16).collect();
        let mut once = vec![0i16; rows * cols];
        let mut twice = vec![0i16; rows * cols];
        transpose(&src, rows, cols, &mut once);
        transpose(&once, cols, rows, &mut twice);
        prop_assert_eq!(twice, src);
    }

    #[test]
    fn square_in_place_matches_out_of_place(n in 0usize..40) {
        let src: Vec<u8> = (0..n * n).map(|i| (i * 7 % 251) as u8).collect();
        let mut want = vec![0u8; n * n];
        transpose(&src, n, n, &mut want);
        let mut got = src;
        transpose_square_in_place(&mut got, n);
        prop_assert_eq!(got, want);
    }
}
