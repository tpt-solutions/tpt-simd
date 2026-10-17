# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- AVX2 forward transforms for `Dct::forward` and `Mdct::forward` on x86_64 builds with `avx2` enabled. Eight outputs per accumulator, summed in the same order as the scalar code, so results are bit-identical (checked against the scalar sums, including the sign of zero). The `Dct` and `Mdct` structs keep a transposed copy of the cosine table for this path, so their memory doubles when AVX2 is compiled in. Criterion (`dct2_forward_256`): 21.1 → 5.6 µs. Harness, forward: DCT 943 → 247 ns at N = 64; MDCT 2.2 → 0.46 µs at N = 64 and 49 → 8.8 µs at N = 256.
- `scalar-only` feature forces the scalar sums. The umbrella crate forwards it.

### Changed

- The crate root is `deny(unsafe_code)`; `unsafe` is confined to the `x86` module. `inverse` is unchanged.

## [0.1.1] - 2026-10-17

### Added
- FFT-based DCT and orthonormal Haar wavelets.

## [0.1.0] - 2026-10-12

### Added

- Initial release.
- `Dct`: orthonormal DCT-II (`forward`) and DCT-III, its inverse (`inverse`), using a precomputed cosine table.
- `Mdct`: MDCT (`forward`, 2N to N) and IMDCT (`inverse`, N to 2N).
- `sine_window`: window for MDCT analysis and synthesis, giving perfect reconstruction with overlap-add.
- Cost is O(N²) per block (matrix-vector product); intended for block sizes up to a few thousand.
- Buffer lengths must match the transform size; mismatches panic.
- `#![no_std]`; no runtime dependencies.
