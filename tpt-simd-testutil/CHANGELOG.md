# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Unpublished (`publish = false`) helper crate for workspace tests.
- proptest strategies: `lens`, `finite_f32`, `f32_with_specials`, `i16_edgy`, `i32_edgy`, `vec_of`, `vec_pair`, `simd_of`.
- Tolerant comparison helpers: `approx_eq_f32` and `assert_slice_close`.
- `#![forbid(unsafe_code)]`.
