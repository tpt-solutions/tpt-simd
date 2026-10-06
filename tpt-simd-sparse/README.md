# tpt-simd-sparse

Sparse kernels for iterative solvers, `f32` and `f64`, `no_std`:

- CSR and CSC SpMV `y = alpha*A*x + beta*y`, plain and transposed, on borrowed-slice
  views (`CsrView`, `CscView`) with O(nnz) validation (`try_new`) and an unsafe
  unchecked constructor.
- Dense vector kernels for CG / BiCGSTAB: `dot`, `sqnorm`, `axpy`, `xpay`,
  fused `axpy_dot`, `axpy_sqnorm`, `cg_update` (two updates plus `r.r` in one
  pass), `bicgstab_p_update`.
- Naive scalar `reference` module, owned `CsrMatrix` / `CscMatrix` helpers
  (`from_triplets`, `poisson_2d`) with the `alloc` feature.

```toml
[dependencies]
tpt-simd-sparse = "0.1.0"
```

See `examples/cg.rs` (CG on a 2D Poisson matrix built from these kernels) and
`examples/strategies.rs` (interleaved timing of the row strategies).

## Features

- `alloc` (default): owned matrices and builders.
- `std`: enables `alloc`, `std::error::Error` for `SparseError`, and `tpt-simd-core/std`.
- `nightly`, `scalar-only`: forwarded to `tpt-simd-core`.

## Results (f32, AVX2, `-C target-cpu=native`, noisy machine)

Interleaved best-of-150 (`examples/strategies.rs`), ns per stored entry:

| matrix | reference | 1 acc | lanes4 (default) | lanes8 | hw gather |
|---|---|---|---|---|---|
| Poisson 256x256 (5/row) | 0.573 | 0.575 | 0.410 | 0.502 | 0.461 |
| random 65536, 32/row | 0.572 | 0.556 | 0.484 | 0.490 | 0.435 |
| random 8192, 128/row | 0.515 | 0.389 | 0.252 | 0.242 | 0.210 |

SpMV is memory/latency bound: expect about 1.2-1.4x over a naive loop on short
and medium rows and up to 2x on long rows. Fused CG vector update: ~1.5x vs
three separate passes in cache, ~1.1x from memory. Details and caveats are in
the crate docs ("Measured findings").

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
