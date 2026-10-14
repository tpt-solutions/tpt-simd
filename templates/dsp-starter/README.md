# dsp-starter

A starting point for a DSP or codec crate that depends on
[tpt-simd](https://github.com/tpt-solutions/tpt-simd).

## Use it

With [cargo-generate](https://github.com/cargo-generate/cargo-generate):

```sh
cargo generate --git https://github.com/tpt-solutions/tpt-simd templates/dsp-starter --name my-codec
```

Or copy this directory and replace `{{project-name}}` in `Cargo.toml`.

## What is in it

* `dot`, `fir`: SIMD kernels from `tpt-simd` with a scalar reference next to each.
* Tests that compare the two across tail lengths, and check the FIR bit for bit.

## Before the first build

`tpt-simd = "0.1"` resolves from crates.io once 0.1.0 is published. Until then,
change the dependency to a path or git source.

## Run the checks

```sh
cargo test
cargo clippy --all-targets -- -D warnings
```

Run the tests under `RUSTFLAGS="-C target-cpu=native"` too, so the AVX2 and
AVX-512 paths are exercised. `cargo run -p tpt-simd --example cpu_report` in the
tpt-simd repo shows which path a build will pick.
