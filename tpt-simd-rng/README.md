# tpt-simd-rng

Lane-parallel pseudo-random number generation for `no_std`: 8 independent
xoshiro256++ streams (or 8 Philox4x32-10 blocks) advanced in lock step, with
slice fills for `u32`, `u64`, uniform `f32`/`f64` in `[0, 1)` and standard
normal `f32`/`f64` (Box-Muller with branch-free polynomial `ln`/`sin`/`cos`
and Newton `sqrt`, no libm in the hot loop).

```rust
use tpt_simd_rng::{Rng8, Xoshiro256ppX8};

let mut rng = Xoshiro256ppX8::from_seed(42);
let mut x = [0.0f32; 1000];
rng.fill_normal_f32(&mut x);
```

Not cryptographically secure. Build with `-C target-cpu=native` (ADR 0001:
compile-time dispatch) for AVX2 code; see the crate docs for streams, layout
and accuracy details.

## Features

- `std`: forwards to `tpt-simd-core/std`
- `nightly`, `scalar-only`: forwarded to `tpt-simd-core`

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
