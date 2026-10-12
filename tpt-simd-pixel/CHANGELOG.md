# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
