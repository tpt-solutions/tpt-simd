# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-10-17

### Fixed
- UTF-8 validator accepted some invalid input: lead byte `0xFD` (and other `F5..FF`) followed by a continuation byte, and `C0`, `C1` or `F5..FF` at the end of a 32-byte block when the next byte was ASCII. The AVX2 and NEON paths shared the tables and were both affected. 0.1.0 is affected; upgrade.

### Added
- aarch64: NEON UTF-8 validation and hex encoding.

### Changed
- Inlined the `hex_decode` entry point (about 20% faster at 64 B).

## [0.1.0] - 2026-10-12

### Added

- Initial release.
- `is_valid_utf8`: UTF-8 validation with `core` by default; `simdutf8` with the `simdutf8` feature.
- `hex_encode_lower` and `hex_decode`: own implementation, no allocation.
- `utf8_to_utf16` and `utf16_to_utf8`: own transcoding with surrogate checks.
- `utf8_len_of_utf16` and `utf16_len`: exact output sizes.
- `TextError` for invalid input, unpaired surrogates, odd or invalid hex, and buffers that are too small.
- Functions that write into a caller buffer validate input and size first, so an error leaves the buffer untouched.
- Feature flags `std`, `runtime-dispatch` (x86_64 CPU detection once at run time) and `simdutf8` (optional dependency).
- `simdutf8` and `faster-hex` are dev-dependencies used as test oracles.
- `#![no_std]`; no runtime dependencies by default.
