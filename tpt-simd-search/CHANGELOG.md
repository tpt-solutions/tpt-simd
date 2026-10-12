# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Initial release.
- `find_byte`, `rfind_byte`, `count_byte`: word-at-a-time (SWAR) byte search.
- `find`: substring search using an own Two-Way implementation, linear in the haystack length.
- `find_ascii_case_insensitive`: substring search with ASCII-only case folding.
- `count_lines`: counts `\n`, plus an unterminated last line.
- Feature flags `std`, `runtime-dispatch` (x86_64 CPU detection once at run time) and `memchr` (delegates byte and substring search to `memchr`).
- `memchr` is a dev-dependency used as a differential-test oracle.
- `#![no_std]`; no runtime dependencies by default.
