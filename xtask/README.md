# xtask

Developer tasks for the tpt-simd workspace, run as `cargo xtask <task>`. Unpublished (`publish = false`).

The tasks mirror the CI jobs in `.github/workflows/ci.yml`, so a passing `cargo xtask ci` on your machine is a good predictor of a green build. The crate has no dependencies and runs the same on Windows, macOS and Linux.

## Tasks

| Task | What it does |
|---|---|
| `ci` | fmt, clippy, docs and tests (std and scalar-only), as in the CI lint and test jobs |
| `no-std` | `no_std` builds for the bare-metal and wasm targets installed with rustup (`thumbv7em-none-eabihf`, `wasm32-unknown-unknown`) |
| `msrv` | `cargo check` on the minimum supported toolchain (1.95) |
| `cpu` | prints the SIMD features this build and CPU support (runs `examples/cpu_report`) |
| `help` | shows the usage text |

```sh
cargo xtask ci
```

## License

Licensed under either of Apache License, Version 2.0 ([LICENSE-APACHE](../LICENSE-APACHE)) or MIT license ([LICENSE-MIT](../LICENSE-MIT)), at your option.
