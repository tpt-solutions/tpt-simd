# Crate-done audit

Workspace-wide mechanical sweep against the per-crate definition of done in `todo.md`.
Run on Windows 11, stable Rust 1.97.1 (`rust-toolchain.toml`), x86_64. Commands run exactly as in `.github/workflows/ci.yml` where applicable.

## Commands run (all passed)

| Command | Result |
|---|---|
| `cargo fmt --all --check` | pass |
| `cargo clippy --workspace --all-targets -- -D warnings` | pass |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | pass |
| `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` (also `--all-features`) | pass |
| `cargo test --workspace --features std` | pass (all test binaries and doctests) |
| `cargo test --workspace --features std,scalar-only` | pass |
| `RUSTFLAGS="-D warnings" cargo build --workspace --exclude tpt-simd-testutil --target thumbv7em-none-eabihf` | pass (target installed) |
| same, `--target wasm32-unknown-unknown` | pass (target installed) |

`missing_docs = "warn"` and `unsafe_op_in_unsafe_fn = "deny"` are set in `[workspace.lints]` and every crate has `[lints] workspace = true`, so the clippy `-D warnings` run enforces missing_docs on every crate. Every `pub unsafe fn` (gather 4, scatter 4, sparse 1) has a `# Safety` section; `clippy::missing_safety_doc` also passes. No source changes were needed.

## Per-crate table

Legend: y = present / passing, n = missing, n/a = not applicable. Columns for scalar ref, SIMD impl and unit tests are presence checks (source/test files exist, tests run and pass); edge-case coverage (NaN/inf/overflow/empty/tail) was spot-checked only for the smallest crates (mul, select, shift, rounding, matrix), not audited exhaustively. fmt/clippy/doc/no_std are workspace-wide commands that pass for every member; the license column was checked per `Cargo.toml` (`license.workspace = true` -> `MIT OR Apache-2.0`).

| crate | scalar ref | SIMD impl | unit tests | proptest | docs (missing_docs, # Safety) | bench | no_std | clippy/fmt | doc | license | show-asm |
|---|---|---|---|---|---|---|---|---|---|---|---|
| core | y | y (arch + portable) | y (4) | n (none) | y | n (none) | y | y | y | y | not done |
| vector | y | y | y (12) | y | y | n (none) | y | y | y | y | not done |
| complex | y | y | y | y | y | y | y | y | y | y | not done |
| fixed | y | y | y | y | y | y | y | y | y | y | not done |
| horizontal | y | y | y | y | y | y | y | y | y | y | not done |
| dot | y | y | y | y | y | y | y | y | y | y | not done |
| butterfly | y | y | y | y | y | y | y | y | y | y | not done |
| saturate | y | y | y | y | y | y | y | y | y | y | not done |
| permute | y | y | y | y | y | y | y | y | y | y | not done |
| gather | y | y | y | y | y | y | y | y | y | y | not done |
| scatter | y | y | y | y | y | y | y | y | y | y | not done |
| aligned | n/a (container) | n/a | y (12) | y | y | n (none) | y | y | y | y | not done |
| mul | y | y | y (4, incl. specials) | y | y | n (none) | y | y | y | y | not done |
| rounding | y | y | y | y | y | y | y | y | y | y | not done |
| shift | y | y | y (4) | y | y | y | y | y | y | y | not done |
| compare | y | y | y | y | y | y | y | y | y | y | not done |
| blend | y | y | y | y | y | y | y | y | y | y | not done |
| select | y | y | y (4) | y | y | n (none) | y | y | y | y | not done |
| convolve | y | y | y | y | y | y | y | y | y | y | not done |
| interpolate | y | y | y | y | y | y | y | y | y | y | not done |
| matrix | y | y | y (6) | y | y | n (none) | y | y | y | y | not done |
| window | y | y | y | y | y | y | y | y | y | y | not done |
| blas | y | y (x86 backend) | y | y | y | y | y | y | y | y | not done |
| math | y | y | y | y | y | y | y | y | y | y | not done |
| rng | y | y | y | y | y | y | y | y | y | y | not done |
| sparse | y | y | y | y | y | y | y | y | y | y | not done |
| reduce | y | y | y | y | y | y | y | y | y | y | not done |
| testutil | n/a | n/a | n/a (helper) | n/a | y | n/a | n/a (excluded in CI; std) | y | y | y | not done |
| tpt-simd (umbrella) | n/a | n/a | y | n/a | y | n/a | y | y | y | y | not done |

## Not verified / not done

* **show-asm: not done for every crate.** `cargo asm` is not installed (`cargo asm --version` -> no such command) and no asm audit was performed or claimed.
* Crates with no criterion bench: core, vector, aligned, mul, select, matrix (no `benches/` dir). testutil is a helper crate.
* tpt-simd-core has no proptest (it only has 4 unit tests); testutil has no tests of its own.
* "SIMD implementation for each target in scope" is only verified for the host (x86_64: SSE2 default, AVX2/FMA paths via cfg). aarch64, RISC-V, AVX-512 and wasm *runtime* behaviour are not tested here; wasm32/thumbv7em were only built, not run. Intel-SDE/qemu/miri CI jobs were not run locally.
* Doc examples: doctest fences exist in each crate (blas, math, rng, reduce, sparse, vector and core have few), but "an example on every public item" was not audited.
* The `x86-features` (`+sse4.1`, `+avx2,+fma`) CI matrix and nightly toolchain were not run in this sweep.
