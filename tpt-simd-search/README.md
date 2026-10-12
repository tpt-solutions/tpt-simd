# tpt-simd-search

Byte search, substring search, ASCII case-insensitive search and line counting, with no runtime dependencies by default.

## Overview

Use this crate to find bytes and substrings in buffers, count lines in text, and do ASCII case-insensitive matching, without pulling in a dependency tree.

| Function | Operation | Notes |
|---|---|---|
| `find_byte` / `rfind_byte` | first / last position of a byte | Word-at-a-time (SWAR) code; `memchr` with the `memchr` feature |
| `count_byte` | number of occurrences of a byte | Same kernels as `find_byte` |
| `find` | first position of a substring | Own Two-Way algorithm, linear in the haystack; `memchr::memmem` with the `memchr` feature |
| `find_ascii_case_insensitive` | first position, ASCII case folded | Own code; only `A`-`Z` and `a`-`z` are folded |
| `count_lines` | number of lines | Counts `\n`, plus an unterminated last line |

The crate is `#![no_std]`. The `memchr` crate is an optional runtime dependency, and is also a dev-dependency used as a differential-test oracle.

## Installation

```toml
[dependencies]
tpt-simd-search = "0.1.0"
```

The umbrella crate `tpt-simd` re-exports this crate.

## Quick start

```rust
use tpt_simd_search::{count_byte, count_lines, find, find_ascii_case_insensitive, find_byte};

assert_eq!(find_byte(b"hello", b'l'), Some(2));
assert_eq!(find(b"hello world", b"world"), Some(6));
assert_eq!(find_ascii_case_insensitive(b"Hello WORLD", b"world"), Some(6));
assert_eq!(count_byte(b"banana", b'a'), 3);
assert_eq!(count_lines(b"a\nb\nc"), 3);
```

## API overview

| Item | Description |
|---|---|
| `find_byte(haystack: &[u8], needle: u8) -> Option<usize>` | Index of the first match |
| `rfind_byte(haystack: &[u8], needle: u8) -> Option<usize>` | Index of the last match |
| `count_byte(haystack: &[u8], needle: u8) -> usize` | Number of matches |
| `find(haystack: &[u8], needle: &[u8]) -> Option<usize>` | Index of the first occurrence of a substring |
| `find_ascii_case_insensitive(haystack: &[u8], needle: &[u8]) -> Option<usize>` | As `find`, with ASCII case folding |
| `count_lines(bytes: &[u8]) -> usize` | Line count |

## Feature flags

| Feature | Default | Effect |
|---|---|---|
| `std` | off | Enables `std`; the crate itself stays `no_std` |
| `runtime-dispatch` | off | Implies `std`. On x86_64, detects CPU features once at run time and picks the fastest kernel. |
| `memchr` | off | Delegates byte and substring search to the `memchr` crate (optional dependency) |

## Platform support and semantics

- The default build has no runtime dependencies and runs on every target.
- Results are the same with and without `runtime-dispatch` and `memchr`; these only change speed.
- `find` is linear in the haystack length in the worst case, because it uses the Two-Way algorithm.
- Empty needles return `Some(0)` from `find` and `find_ascii_case_insensitive`.
- Only ASCII letters are case-folded. Non-ASCII bytes are compared exactly.

## Example

Runnable example: [examples/search.rs](examples/search.rs). Run it with `cargo run -p tpt-simd-search --example search`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
