# tpt-simd-ntt

Modular arithmetic and number-theoretic transforms (NTT) for odd moduli below 2^31: exact cyclic convolution modulo a prime.

## Overview

| Type | Purpose |
|---|---|
| `Montgomery` | `a * b mod q` for odd `q < 2^31`: reduction, multiplication, addition, subtraction, exponentiation |
| `Barrett` | reduction of `u64` values modulo any `q >= 1` |
| `NttPlan` | in-place forward and inverse NTT of a power-of-two length, and cyclic convolution |
| `cyclic_convolve_naive` | O(n²) reference, for testing |

The NTT needs a prime modulus `q`, a primitive root `g` modulo `q`, and a transform length that divides `q - 1`. For example, `q = 998244353` with `g = 3` supports lengths up to 2^23.

The crate is `#![no_std]` (it uses `alloc`) and forbids `unsafe` code.

## Installation

```toml
[dependencies]
tpt-simd-ntt = "0.1.0"
```

The umbrella crate `tpt-simd` re-exports this crate.

## Quick start

```rust
use tpt_simd_ntt::{NttPlan, cyclic_convolve_naive};

const Q: u32 = 998_244_353; // prime; 3 is a primitive root; 8 divides Q - 1
let plan = NttPlan::new(Q, 3, 8);

let mut a = [1u32, 2, 0, 0, 0, 0, 0, 0];
let mut b = [3u32, 4, 0, 0, 0, 0, 0, 0];
let expect = cyclic_convolve_naive(Q, &a, &b);
plan.cyclic_convolve(&mut a, &mut b);
assert_eq!(a[..], expect[..]);
// (1 + 2x)(3 + 4x) = 3 + 10x + 8x^2
assert_eq!(&a[..3], &[3, 10, 8]);
```

## Platform support and semantics

- Operands passed to the arithmetic methods must be reduced (`< q`). NTT inputs are checked, and a violation causes a **panic**.
- Primality of `q` and the primitive-root property of `g` are **not** checked. Supplying a composite modulus or a wrong root gives wrong results without an error.
- All arithmetic is exact integer arithmetic, so results are identical on every target.

## Feature flags

This crate has no optional features.

## Example

Runnable example: [examples/ntt.rs](examples/ntt.rs). Run it with `cargo run -p tpt-simd-ntt --example ntt`.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
