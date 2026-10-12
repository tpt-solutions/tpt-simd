# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release.
- `sort_unstable` and `sort_by`: introsort with 3-way partition and heapsort fallback.
- `sort_u32`: sort for `u32` slices.
- `select_nth` and `top_k_smallest`: quickselect with the same fallback.
- `argsort`: sorts indices by keys; not stable.
- `partition`: in-place, unstable, returns the count of matching elements.
- Worst case is O(n log n) for every sort and select function; many duplicates are handled in one pass.
- `#![no_std]`; no runtime dependencies.
