# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release.
- Bitsets: `popcount_u64`, `bitset_and`, `bitset_or`, `bitset_xor`, `bitset_andnot`.
- Zigzag coding: `zigzag_encode_i32`, `zigzag_decode_u32`, `zigzag_encode_i64`, `zigzag_decode_u64`, and slice variants `zigzag_encode_slice_i32` and `zigzag_decode_slice_u32`.
- Streams: `delta_encode_u32`, `delta_decode_u32`, `prefix_sum_u32`, `prefix_sum_u64`.
- `RankSelect`: rank and select index over a bitset (`rank1` in O(1), `select1` in O(log words) plus an in-word scan).
- All integer arithmetic is wrapping, so debug and release builds agree.
- Paired slices must have equal lengths; mismatches panic. Empty input is valid.
- `#![no_std]`; no runtime dependencies.
