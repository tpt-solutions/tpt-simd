# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- SpMV kernels generic over `Real` (`f32`, `f64`): `spmv_csr`, `spmv_csr_with`, `spmv_csr_t`, `spmv_csc`, `spmv_csc_t`, computing `y = alpha*op(A)*x + beta*y`.
- `RowStrategy` (`Scalar`, `Lanes4`, `Lanes8`, `Hybrid`) and `DEFAULT_STRATEGY` (`Lanes4`) for the gather-style kernels.
- Vector kernels: `dot`, `sqnorm`, `axpy`, `xpay`, fused `axpy_dot`, `axpy_sqnorm`, `cg_update`, and `bicgstab_p_update`.
- Borrowed `CsrView` / `CscView` with validating `try_new`, unsafe `new_unchecked`, and accessors; `SparseError` (with `std::error::Error` under `std`).
- Owned `CsrMatrix` (`try_new`, `from_triplets`, `poisson_2d`, `view`, `to_csc`) and `CscMatrix` behind the default `alloc` feature.
- `reference` module with naive scalar versions of the kernels.
- Features: `alloc` (default), `std`, `nightly`, `scalar-only` (forwarded to `tpt-simd-core`).
- `#![no_std]`.
- Unit and property tests, Criterion benchmarks (`cargo bench -p tpt-simd-sparse`), and examples `cg` (conjugate gradient on the 2D Poisson matrix) and `strategies`.

### Notes

- Only `beta == 0` is special-cased (overwrite without reading `y`); `alpha == 0` is not.
- Duplicate entries are summed, indices may be unsorted, explicit zeros are kept.
- No FMA; accumulation order is fixed, so results are reproducible across targets. Results differ from a naive left-to-right loop by a few ulps of the sum of absolute terms.
- Kernels use unchecked loads that rely on validated views; `new_unchecked` is `unsafe`.
