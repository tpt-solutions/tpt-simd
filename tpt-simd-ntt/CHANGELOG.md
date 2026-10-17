# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `bigint_mul`: multiplication of little-endian `u32` limb numbers with the NTT (two primes, 16-bit digits, Chinese remaindering). Checked against schoolbook multiplication (`tests/bigint_matches_schoolbook.rs`). Criterion, default build: schoolbook is faster at 512 limbs (175 µs vs 252 µs), NTT at 1024 limbs (544 µs vs 721 µs).
- AVX2 butterfly stages for `NttPlan::forward`, `inverse` and `cyclic_convolve` on x86_64 builds with `avx2` enabled. Eight lanes of Montgomery multiply and modular add/subtract per instruction, bit-identical to the scalar code. Stages narrower than eight are scalar. Criterion (`ntt_forward_4096`, 31-bit prime): 40 µs in the default build, 17 µs in the AVX2 build.
- `scalar-only` feature forces the scalar butterflies. The umbrella crate forwards it.

### Changed

- The plan stores per-stage twiddle tables (`n - 1` entries per direction, instead of `n / 2`), so each stage reads a contiguous slice. Output is unchanged.
- The crate root is `deny(unsafe_code)`; `unsafe` is confined to the `x86` module.

## [0.1.1] - 2026-10-17

### Added
- `is_prime` for checking NTT moduli.

## [0.1.0] - 2026-10-12

### Added

- Initial release.
- `Montgomery`: modular multiplication for odd `q < 2^31`.
- `Barrett`: reduction of `u64` values modulo any `q >= 1`.
- `NttPlan`: in-place forward and inverse number-theoretic transform, and cyclic convolution.
- `cyclic_convolve_naive`: reference implementation for testing.
- Preconditions are documented, not checked: `q` must be prime, `g` a primitive root, and the transform length must divide `q - 1`. Operands must be reduced; NTT inputs are checked and panic otherwise.
- `#![no_std]`; no runtime dependencies.
