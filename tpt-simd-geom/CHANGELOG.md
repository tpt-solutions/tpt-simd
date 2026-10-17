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
