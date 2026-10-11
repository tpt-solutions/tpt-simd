# tpt-simd-transpose

Cache-blocked matrix transpose and strided 2D copies, generic over any `Copy`
element type:

- `transpose`: out-of-place `rows x cols` to `cols x rows`
- `transpose_square_in_place`: in-place transpose of a square matrix
- `copy_strided`: 2D copy between buffers with independent row strides

Fixed-size 4x4 and 8x8 register tiles are in `tpt-simd-permute`. `no_std`, no
dependencies, MIT OR Apache-2.0.
