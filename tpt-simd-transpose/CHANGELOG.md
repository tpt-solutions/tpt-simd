# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-10-17

- No code changes; version bumped with the workspace.

## [0.1.0] - 2026-10-12

### Added

- Initial release.
- `transpose`: out-of-place transpose of a `rows x cols` row-major matrix.
- `transpose_square_in_place`: in-place transpose of an `n x n` matrix.
- `copy_strided`: 2D copy between buffers with independent row strides.
- Cache-blocked in `BLOCK x BLOCK` tiles (`BLOCK` constant).
- Works for any `Copy` element type.
- Buffers must hold the described matrix; mismatches panic before any data is touched. Zero rows or columns is valid.
- `#![no_std]`; no runtime dependencies.
