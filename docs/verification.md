# Verification environments

What was run, where, and what each result does and does not show. This covers
the todo items in Phase 8 (cross-architecture hardening) and the Phase 1, 3 and
6 exits.

**Emulated results prove correctness, not speed.** Timings are only taken on
the native x86_64 host.

## Environments

| Environment | What it is | Used for |
|---|---|---|
| Host | Intel Core i5-1135G7 (Tiger Lake, 4C/8T), AVX2, FMA, AVX-512 F/BW/DQ/VL, Windows 11, rustc 1.97 | Native tests, benchmarks, Miri, asm audit |
| `rust:1.97-bookworm` arm64 container | Docker Desktop, `linux/arm64`, QEMU user-mode emulation. CPU reports `asimd`, `sve`, `sve2`, `sme` | aarch64 correctness (NEON paths, portable paths, golden digests) |
| `rust:1.97-trixie` riscv64 container | Docker Desktop, `linux/riscv64`, QEMU user-mode. CPU reports `rv64gc` + `v` (`zve64d`, `zvbb`, ...) | riscv64 correctness; RVV auto-vectorised code |
| Host toolchain 1.95 | `rust-version` in the workspace | MSRV check |
| Host nightly + Miri | `cargo +nightly miri` | Undefined-behaviour check of unsafe code |
| amd64 container | `linux/amd64` Rust image | `cargo-fuzz` (libFuzzer does not build on Windows MSVC) |

No Intel SDE, QEMU system emulator, or AVX-512 cloud runner was needed:
AVX-512 is present on the host, and Docker Desktop provides the other
architectures.

## Commands

```sh
# x86_64 variants (must produce identical golden digests)
cargo test --workspace --features std                                   # SSE2 baseline
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test --workspace --features std
RUSTFLAGS="-C target-cpu=native" cargo test --workspace --features std  # AVX-512 on this host
cargo test --workspace --features std,scalar-only                       # portable fallback

# MSRV
cargo +1.95 test --workspace --features std

# no_std and cross-target builds
cargo check --workspace --exclude tpt-simd-testutil --no-default-features --target <t>
#   t = thumbv7em-none-eabihf, thumbv6m-none-eabi, wasm32-unknown-unknown,
#       riscv32imc-unknown-none-elf, aarch64-unknown-none-softfloat,
#       aarch64-unknown-linux-gnu, riscv64gc-unknown-linux-gnu

# Emulated targets (inside the containers)
cargo test --locked --workspace --features std
RUSTFLAGS="-C target-feature=+v" cargo test ...    # riscv64: RVV auto-vectorisation (unstable flag, warns)
RUSTFLAGS="-C target-feature=+sve" cargo test ...  # arm64: SVE auto-vectorisation (accepted on stable)

# Miri (CI configuration)
PROPTEST_CASES=8 MIRIFLAGS=-Zmiri-disable-isolation cargo +nightly miri test --workspace --features std,scalar-only
```

## Cross-architecture conformance

`tpt-simd/tests/cross_arch_golden.rs` runs ten kernels with fixed inputs and
hashes the outputs (FNV-1a over the bit patterns). The golden digests were
computed on x86_64. The same test runs unchanged on every target. Only
bit-exact kernels are included: integer arithmetic, IEEE add/sub/mul in a fixed
order, correctly rounded rounding functions, and fused complex multiply.
Reassociating reductions are excluded (ADR 0003 tolerance tier). NaN outputs
are canonicalised before hashing, because NaN sign and payload are not
specified across architectures.

## Results

See the status table at the end of this file. Entries marked *pending* were
not finished when this file was written.

| Check | Environment | Result |
|---|---|---|
| Workspace tests, SSE2 | Host | pass (59 test binaries) at the start of the session; not re-run in full after the later changes |
| Workspace tests, AVX2+FMA | Host | not run in full this session; golden digests and the dot and matrix suites re-run |
| Workspace tests, AVX-512 (`target-cpu=native`) | Host | pass (61 test results) before the AVX-512BW dot kernel and the NEON changes; the dot suite and the golden digests re-run after the dot change |
| Workspace tests, scalar-only | Host | pass (per the earlier audit; not re-run this session) |
| MSRV 1.95, `cargo test --locked --workspace --features std` | Host | pass: exit 0, 64 test results, 0 failed |
| no_std / cross builds (7 targets) | Host | pass (before the NEON changes; re-checked for aarch64 with and without `scalar-only`) |
| clippy `--all-targets -D warnings` (default and all features) | Host | pass |
| `cargo doc` with `-D warnings` | Host | pass |
| `cargo deny check licenses` (rustfft added) | Host | pass |
| `cargo fmt --check` | Host | pass |
| Golden digests, 10 kernels | Host, SSE2 / AVX2+FMA / AVX-512 | identical |
| Golden digests, 10 kernels | arm64 container (NEON paths) | identical to x86 (`cross_arch_golden` passes) |
| Targeted tests incl. NEON paths (rounding, mul, butterfly, blend, permute, dot, saturate, matrix, core, umbrella) | arm64 container | pass: exit 0, 19 test results, 0 failed |
| Golden digests, 10 kernels, and targeted tests (riscv64 portable path) | riscv64 container | pass: exit 0, 26 test results, golden digests identical to x86 |
| Miri, CI configuration | Host nightly | partial: workspace stage in progress (see status) |
| Fuzzing, 4 targets, 120 s each | amd64 container, `cargo-fuzz` | pass: no crashes or assertion failures (6.5M dot, 5.3M gather/scatter, 4.7M transpose, 115k gemm executions) |

## Status table

*pending* entries are updated as the runs finish; see the todo for what is
still open.
