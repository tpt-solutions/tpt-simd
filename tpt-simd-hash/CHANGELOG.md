# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release.
- `crc32` and `Crc32`: IEEE CRC-32, slicing-by-8 kernel, streaming (`new`, `update`, `finish`).
- `adler32` and `Adler32`: Adler-32 with deferred modulo, streaming.
- `fnv1a_64`: FNV-1a, 64-bit.
- `Rolling`: polynomial rolling hash mod 2^64 over a fixed window (`new`, `push`, `roll`, `hash`).
- Feature flags `std` and `runtime-dispatch` (x86_64: PCLMULQDQ + SSE4.1 folding CRC-32 kernel, selected at run time).
- Dev-dependencies `crc32fast` and `simd-adler32` serve as differential-test oracles only.
- `#![no_std]`; no `unsafe` code; no runtime dependencies.
