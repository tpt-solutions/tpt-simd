# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- SSE2 `premultiply_rgba8`: two pixels per vector, bit-identical to the scalar formula (checked exhaustively). Criterion, 1 MiB: 289 → 165 µs in the default build; in the AVX2 build the scalar baseline is autovectorised and the gap is smaller (167 → 144 µs). SSE2 is part of the x86_64 baseline, so it is used on every x86_64 build unless `scalar-only` is set.
- `scalar-only` feature forces the scalar code. The umbrella crate forwards it.

## [0.1.1] - 2026-10-17

- No code changes; version bumped with the workspace.

## [0.1.0] - 2026-10-12

### Added

- Initial release.
- `rgb8_to_rgba8` and `rgba8_to_rgb8`: channel expansion (with alpha value) and drop.
- `planar_to_interleaved_rgb8` and `interleaved_to_planar_rgb8`: three 8-bit planes.
- `u8_to_unit_f32` and `unit_f32_to_u8`: conversion between `[0, 255]` and `[0, 1]`.
- `premultiply_rgba8`: colour times alpha, rounded.
- `rgb_to_hsv` and `hsv_to_rgb`: `f32`, hue in degrees `[0, 360)`.
- `histogram_u8`: 256-bin histogram.
- Length rules checked up front; mismatches panic.
- `#![no_std]`; no runtime dependencies.
