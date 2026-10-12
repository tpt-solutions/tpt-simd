# tpt-simd-text

Text kernels: UTF-8 validation, hex encoding and decoding, and UTF-8 to UTF-16 transcoding, with no runtime dependencies by default.

## Overview

| Function / type | Notes |
|---|---|
| `is_valid_utf8` | `core` validation by default; `simdutf8` with the `simdutf8` feature |
| `hex_encode_lower`, `hex_decode` | own implementation, no allocation |
| `utf8_to_utf16`, `utf16_to_utf8` | own implementation, with surrogate checks and exact sizes |
| `utf8_len_of_utf16`, `utf16_len` | exact output sizes, to size buffers before transcoding |
| `TextError` | invalid input, unpaired surrogates, bad hex, or a buffer that is too small |

Every function that writes into a caller buffer validates its input and size before writing, so an error leaves the buffer untouched.

The crate is `#![no_std]` and forbids `unsafe` code outside its SIMD kernels, which are documented in the source.

## Installation

```toml
[dependencies]
tpt-simd-text = "0.1.0"
```

The umbrella crate `tpt-simd` re-exports this crate.

## Quick start

```rust
use tpt_simd_text::{hex_decode, hex_encode_lower, is_valid_utf8, utf16_to_utf8, utf8_to_utf16};

assert!(is_valid_utf8("h€😀".as_bytes()));
assert!(!is_valid_utf8(&[0xC3, 0x28]));

// Hex round trip, no allocation.
let mut hex = [0u8; 8];
let n = hex_encode_lower(&[0xDE, 0xAD, 0xBE, 0xEF], &mut hex).unwrap();
assert_eq!(&hex[..n], b"deadbeef");
let mut bytes = [0u8; 4];
hex_decode(&hex[..n], &mut bytes).unwrap();
assert_eq!(bytes, [0xDE, 0xAD, 0xBE, 0xEF]);

// UTF-8 -> UTF-16 and back. The emoji becomes a surrogate pair.
let mut units = [0u16; 4];
let n = utf8_to_utf16("h€😀", &mut units).unwrap();
assert_eq!(&units[..n], &[0x68, 0x20AC, 0xD83D, 0xDE00]);

let mut back = [0u8; 8];
let m = utf16_to_utf8(&units[..n], &mut back).unwrap();
assert_eq!(core::str::from_utf8(&back[..m]).unwrap(), "h€😀");
```

## Platform support and semantics

- Input validation happens before any write, so a failed call leaves the output buffer unchanged.
- `hex_decode` rejects odd-length input and non-hex digits.
- `utf16_to_utf8` rejects unpaired surrogates.

## Feature flags

| Feature | Default | Effect |
|---|---|---|
| `std` | off | Enables `std`; the crate itself stays `no_std` |
| `runtime-dispatch` | off | Implies `std`. On x86_64, detects CPU features once at run time and picks the fastest kernel. Results are unchanged. |
| `simdutf8` | off | Uses the `simdutf8` crate for `is_valid_utf8` (optional dependency) |

## Example

Runnable example: [examples/text.rs](examples/text.rs). Run it with `cargo run -p tpt-simd-text --example text`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
