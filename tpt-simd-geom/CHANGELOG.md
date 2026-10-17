# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- AVX2 slab test for `ray_aabb` on x86_64 builds with `avx2` enabled, on the SoA layout. Eight boxes per step, bit-identical to the scalar loop (compare-and-blend in place of `min`/`max`, so ties and signed zeros match). Criterion (`ray_aabb`, 16K boxes): 30.7 → 15.2 µs in the AVX2 build against the committed SoA code. A plain array-of-structs loop (`aos_naive`) runs at 9.7 µs, but it reads a different layout, so it is a reference for AoS data, not a competitor to the SoA path.
- `scalar-only` feature forces the scalar loop. The umbrella crate forwards it.

### Changed

- `ray_aabb` checks that every box coordinate slice holds at least as many values as there are boxes, and panics otherwise, before any SIMD load. Previously a short slice panicked on an index in the loop; the message now names the problem.
- The scalar loop runs over zipped slices, so it has no per-element bounds checks. In a default build it is about the same as before (criterion: 47 µs committed, 45 µs now).
- The crate root is `deny(unsafe_code)`; `unsafe` is confined to the `x86` module.

## [0.1.1] - 2026-10-17

- No code changes; version bumped with the workspace.

## [0.1.0] - 2026-10-12

### Added

- Initial release.
- `distance_sq` and `distance`: point-to-query distance per point.
- `dot`: dot product per point against a direction.
- `aabb_overlap`: box-against-query-box overlap test per box.
- `ray_aabb`: ray entry distance per box, `+inf` on a miss.
- `quat_mul` and `quat_rotate`: batched quaternion product and rotation.
- `mat4_transform_points`: batched 4x4 matrix transform of points.
- `Soa3` and `Soa3Mut`: structure-of-arrays views of three-axis data; `BoxesSoa` for boxes; `QuatSoa` and `QuatSoaMut` for quaternions.
- Slices describing the same items must match in length; mismatches panic.
- Plain `f32` arithmetic with no FMA, so results are the same on every target. NaN coordinates are not supported.
- `#![no_std]`; no runtime dependencies.
