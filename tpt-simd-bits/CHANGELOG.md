# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- AVX2 kernels for `prefix_sum_u32` and `prefix_sum_u64` (and so `delta_decode_u32`) on x86_64 builds with `avx2` enabled. Bit-identical to the scalar scan (wrapping integer adds). Criterion (`prefix_sum_u32`, 4096 values): 1.23 → 1.01 µs in the default build and 1.13 → 1.05 µs in the AVX2 build. An earlier in-process timing had suggested about 1.8×; criterion does not reproduce that, so the smaller figure is the one to rely on.
- `scalar-only` feature forces the scalar loops. The umbrella crate forwards it.

### Changed

- The crate root is `forbid(unsafe_code)` unless the AVX2 path is compiled. `unsafe` is confined to the `x86` module.

### Not changed

- `popcount_u64` has no explicit path. Every AVX2 build already has the hardware `popcnt` instruction, so a vector popcount would have to beat that, and no such path was built or measured.

## [0.1.1] - 2026-10-17

### Added
- Varint and bitpacking codecs.

## [0.1.0] - 2026-10-12

### Added

- Initial release.
- Bitsets: `popcount_u64`, `bitset_and`, `bitset_or`, `bitset_xor`, `bitset_andnot`.
- Zigzag coding: `zigzag_encode_i32`, `zigzag_decode_u32`, `zigzag_encode_i64`, `zigzag_decode_u64`, and slice variants `zigzag_encode_slice_i32` and `zigzag_decode_slice_u32`.
- Streams: `delta_encode_u32`, `delta_decode_u32`, `prefix_sum_u32`, `prefix_sum_u64`.
- `RankSelect`: rank and select index over a bitset (`rank1` in O(1), `select1` in O(log words) plus an in-word scan).
- All integer arithmetic is wrapping, so debug and release builds agree.
- Paired slices must have equal lengths; mismatches panic. Empty input is valid.
- `#![no_std]`; no runtime dependencies.
