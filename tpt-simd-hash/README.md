# tpt-simd-hash

Checksums and hashes: CRC-32, Adler-32, FNV-1a (64-bit) and a polynomial rolling hash, with no runtime dependencies.

## Overview

Use this crate when you need a checksum for data integrity (CRC-32, Adler-32), a fast non-cryptographic hash for keys (FNV-1a), or a rolling hash for substring search and deduplication (`Rolling`). None of these are cryptographic.

| Item | Kind | Notes |
|---|---|---|
| `crc32` / `Crc32` | IEEE CRC-32 | Slicing-by-8; `Crc32` is streaming (`new`, `update`, `finish`) |
| `adler32` / `Adler32` | Adler-32 | Deferred modulo; `Adler32` is streaming |
| `fnv1a_64` | FNV-1a, 64-bit | One-shot |
| `Rolling` | Polynomial rolling hash mod 2^64 | Fixed window: `push` while filling, `roll` to slide |

The crate is `#![no_std]` and contains no `unsafe` code. The `crc32fast` and `simd-adler32` crates are dev-dependencies only, used as differential-test oracles.

## Installation

```toml
[dependencies]
tpt-simd-hash = "0.1.0"
```

The umbrella crate `tpt-simd` re-exports this crate.

## Quick start

```rust
use tpt_simd_hash::{adler32, crc32, fnv1a_64, Crc32};

assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);

// Streaming: the result equals the one-shot CRC of the concatenated input.
let mut c = Crc32::new();
c.update(b"12345");
c.update(b"6789");
assert_eq!(c.finish(), 0xCBF4_3926);
```

## API overview

| Item | Description |
|---|---|
| `crc32(&[u8]) -> u32` | IEEE CRC-32 of a buffer |
| `Crc32::new()`, `update(&mut self, &[u8])`, `finish(&self) -> u32` | Streaming CRC-32 |
| `adler32(&[u8]) -> u32` | Adler-32 of a buffer |
| `Adler32::new()`, `update(&mut self, &[u8])`, `finish(&self) -> u32` | Streaming Adler-32 |
| `fnv1a_64(&[u8]) -> u64` | FNV-1a, 64-bit |
| `Rolling::new(window, base)` | Empty hash for windows of `window` bytes; panics if `window` is zero |
| `Rolling::push(u8)` | Append a byte while the window is still filling |
| `Rolling::roll(out, inb)` | Remove the oldest byte `out` and append `inb` |
| `Rolling::hash() -> u64` | Current hash |

## Rolling hash

`Rolling` holds the hash of the last `window` bytes. Fill the window with `push`, then call `roll` once per new byte, passing the byte that leaves the window. Use an odd `base` (for example a large prime): an even base discards high bits of the input.

## Feature flags

| Feature | Default | Effect |
|---|---|---|
| `std` | off | Reserved; no item depends on it today. The crate stays `no_std`. |
| `runtime-dispatch` | off | Implies `std`. On x86_64, detects PCLMULQDQ and SSE4.1 at run time and uses a folding CRC-32 kernel. Without it, the portable slicing-by-8 kernel is used, unless the build already enables those features with `-C target-cpu`. |

Output is identical with and without `runtime-dispatch`.

## Platform support and semantics

- All functions are portable; the default build runs on every target.
- Results are bit-exact across targets and build configurations.
- `crc32` uses the IEEE 802.3 polynomial in reflected form (the same CRC as zlib, gzip and PNG).
- `Rolling` arithmetic wraps modulo 2^64.

## Example

Runnable example: [examples/hash.rs](examples/hash.rs). Run it with `cargo run -p tpt-simd-hash --example hash`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
