# tpt-simd-bits

Bit-level and integer-stream kernels used by codecs and index structures:

- `popcount_u64` over a slice, and bitset `and` / `or` / `xor` / `andnot`
  into a destination slice
- zigzag encode and decode for `i32` and `i64`
- wrapping delta encode and decode (in place), and wrapping inclusive prefix sum

`no_std`, no dependencies, MIT OR Apache-2.0.
