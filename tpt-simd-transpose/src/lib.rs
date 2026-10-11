//! Cache-blocked matrix transpose and strided 2D copies.
//!
//! | function | operation |
//! |---|---|
//! | [`transpose`] | out-of-place `rows x cols` row-major to `cols x rows` |
//! | [`transpose_square_in_place`] | in-place transpose of an `n x n` matrix |
//! | [`copy_strided`] | 2D copy between buffers with independent row strides |
//!
//! Work is done in `BLOCK x BLOCK` tiles so that both the reads and the writes
//! stay within a few cache lines per tile. The element type only needs `Copy`;
//! the inner loops are plain copies that compilers vectorise for small types.
//!
//! ## Length policy
//!
//! Buffers must hold the matrix they describe; otherwise the function **panics**
//! before touching any data. Zero rows or columns is valid and does nothing.
//!
//! ## Example
//!
//! ```
//! use tpt_simd_transpose::transpose;
//!
//! let src = [1, 2, 3, 4, 5, 6]; // 2 x 3
//! let mut dst = [0; 6];         // 3 x 2
//! transpose(&src, 2, 3, &mut dst);
//! assert_eq!(dst, [1, 4, 2, 5, 3, 6]);
//! ```
#![no_std]
#![deny(unsafe_code)]
#![warn(missing_docs)]

/// Tile edge length used by the blocked loops.
pub const BLOCK: usize = 16;

#[cold]
#[inline(never)]
fn too_small(func: &str, what: &str, need: usize, got: usize) -> ! {
    panic!("{func}: {what} holds {got} elements, needs at least {need}");
}

/// Transposes the row-major `rows x cols` matrix `src` into `dst`, which is
/// row-major `cols x rows`.
///
/// # Panics
///
/// Panics if `src` or `dst` is shorter than `rows * cols`.
pub fn transpose<T: Copy>(src: &[T], rows: usize, cols: usize, dst: &mut [T]) {
    let need = rows * cols;
    if src.len() < need {
        too_small("transpose", "src", need, src.len());
    }
    if dst.len() < need {
        too_small("transpose", "dst", need, dst.len());
    }
    for rb in (0..rows).step_by(BLOCK) {
        let r_end = (rb + BLOCK).min(rows);
        for cb in (0..cols).step_by(BLOCK) {
            let c_end = (cb + BLOCK).min(cols);
            for r in rb..r_end {
                for c in cb..c_end {
                    dst[c * rows + r] = src[r * cols + c];
                }
            }
        }
    }
}

/// Transposes the `n x n` row-major matrix `data` in place.
///
/// # Panics
///
/// Panics if `data` is shorter than `n * n`.
pub fn transpose_square_in_place<T: Copy>(data: &mut [T], n: usize) {
    let need = n * n;
    if data.len() < need {
        too_small("transpose_square_in_place", "data", need, data.len());
    }
    for rb in (0..n).step_by(BLOCK) {
        let r_end = (rb + BLOCK).min(n);
        for cb in (rb..n).step_by(BLOCK) {
            let c_end = (cb + BLOCK).min(n);
            for r in rb..r_end {
                // Diagonal tiles visit each pair once by starting the column
                // scan just past the row.
                let c_start = if cb == rb { r + 1 } else { cb };
                for c in c_start.max(cb)..c_end {
                    data.swap(r * n + c, c * n + r);
                }
            }
        }
    }
}

/// Copies a `rows x cols` block from `src` to `dst`, where row `r` starts at
/// `r * src_stride` in `src` and at `r * dst_stride` in `dst`.
///
/// # Panics
///
/// Panics if a stride is smaller than `cols` (when `rows > 1`), or if either
/// buffer is too short to hold the strided block.
pub fn copy_strided<T: Copy>(
    src: &[T],
    src_stride: usize,
    dst: &mut [T],
    dst_stride: usize,
    rows: usize,
    cols: usize,
) {
    if rows == 0 || cols == 0 {
        return;
    }
    if rows > 1 && src_stride < cols {
        panic!("copy_strided: src_stride {src_stride} is smaller than cols {cols}");
    }
    if rows > 1 && dst_stride < cols {
        panic!("copy_strided: dst_stride {dst_stride} is smaller than cols {cols}");
    }
    let src_need = (rows - 1) * src_stride + cols;
    let dst_need = (rows - 1) * dst_stride + cols;
    if src.len() < src_need {
        too_small("copy_strided", "src", src_need, src.len());
    }
    if dst.len() < dst_need {
        too_small("copy_strided", "dst", dst_need, dst.len());
    }
    for r in 0..rows {
        let s = &src[r * src_stride..r * src_stride + cols];
        dst[r * dst_stride..r * dst_stride + cols].copy_from_slice(s);
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec;
    use std::vec::Vec;

    fn naive(src: &[u32], rows: usize, cols: usize) -> Vec<u32> {
        let mut out = vec![0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = src[r * cols + c];
            }
        }
        out
    }

    #[test]
    fn transpose_matches_naive_for_awkward_shapes() {
        for (rows, cols) in [
            (0, 5),
            (1, 1),
            (1, 7),
            (7, 1),
            (3, 5),
            (16, 16),
            (17, 33),
            (40, 9),
        ] {
            let src: Vec<u32> = (0..(rows * cols) as u32).collect();
            let mut dst = vec![0; rows * cols];
            transpose(&src, rows, cols, &mut dst);
            assert_eq!(dst, naive(&src, rows, cols), "{rows}x{cols}");
        }
    }

    #[test]
    fn square_in_place_matches_out_of_place() {
        for n in [0usize, 1, 2, 15, 16, 17, 33, 50] {
            let src: Vec<u32> = (0..(n * n) as u32).collect();
            let mut data = src.clone();
            transpose_square_in_place(&mut data, n);
            assert_eq!(data, naive(&src, n, n), "n={n}");
            transpose_square_in_place(&mut data, n);
            assert_eq!(data, src, "double transpose n={n}");
        }
    }

    #[test]
    fn copy_strided_moves_block_only() {
        let src: Vec<u32> = (0..12).collect(); // 3 rows of stride 4, use 2 cols
        let mut dst = vec![99u32; 12]; // stride 5, last row ends at index 11
        copy_strided(&src, 4, &mut dst, 5, 3, 2);
        assert_eq!(dst, [0, 1, 99, 99, 99, 4, 5, 99, 99, 99, 8, 9]);
    }

    #[test]
    #[should_panic(expected = "transpose: dst holds 5 elements, needs at least 6")]
    fn transpose_short_buffer_panics() {
        let mut dst = [0u8; 5];
        transpose(&[0u8; 6], 2, 3, &mut dst);
    }

    #[test]
    #[should_panic(expected = "copy_strided: src_stride 2 is smaller than cols 3")]
    fn copy_strided_bad_stride_panics() {
        let mut dst = [0u8; 16];
        copy_strided(&[0u8; 16], 2, &mut dst, 3, 2, 3);
    }
}
