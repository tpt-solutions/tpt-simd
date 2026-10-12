# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release.
- `Dct`: orthonormal DCT-II (`forward`) and DCT-III, its inverse (`inverse`), using a precomputed cosine table.
- `Mdct`: MDCT (`forward`, 2N to N) and IMDCT (`inverse`, N to 2N).
- `sine_window`: window for MDCT analysis and synthesis, giving perfect reconstruction with overlap-add.
- Cost is O(N²) per block (matrix-vector product); intended for block sizes up to a few thousand.
- Buffer lengths must match the transform size; mismatches panic.
- `#![no_std]`; no runtime dependencies.
