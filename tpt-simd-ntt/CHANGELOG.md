# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release.
- `Montgomery`: modular multiplication for odd `q < 2^31`.
- `Barrett`: reduction of `u64` values modulo any `q >= 1`.
- `NttPlan`: in-place forward and inverse number-theoretic transform, and cyclic convolution.
- `cyclic_convolve_naive`: reference implementation for testing.
- Preconditions are documented, not checked: `q` must be prime, `g` a primitive root, and the transform length must divide `q - 1`. Operands must be reduced; NTT inputs are checked and panic otherwise.
- `#![no_std]`; no runtime dependencies.
