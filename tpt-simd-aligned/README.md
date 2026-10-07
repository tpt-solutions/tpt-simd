# tpt-simd-aligned

SIMD-aligned memory types and helpers: alignment wrappers, alignment-checked vector loads and stores, and a densely packed aligned heap buffer.

## Overview

Aligned loads and stores are faster (and on some targets required) when the address is a multiple of the vector size. This crate provides:

- `Aligned16`, `Aligned32`, `Aligned64`: `#[repr(C, align(N))]` wrappers that force a value onto a 16/32/64-byte boundary (SSE, AVX2, AVX-512 / cache line).
- `is_aligned` and `vector_align`: alignment checks.
- `load_aligned` / `store_aligned` and `try_load_aligned` / `try_store_aligned`: safe, alignment-checked `Simd` loads and stores on slices.
- `AlignedBuf` and `aligned_vec_f32` / `aligned_vec_i16` / `aligned_vec_i32` (feature `alloc`, on by default): a heap buffer whose first element is aligned to a const-generic boundary. Unlike `Vec<Aligned32<f32>>`, which pads every `f32` to 32 bytes, elements are densely packed.

Scalar versus SIMD: the load/store helpers check alignment and then copy through the `Simd` backend (`Simd::from_slice` / `copy_to_slice`); LLVM can emit aligned vector moves. Dispatch is compile-time; see `docs/adr`. The crate is `#![no_std]` (the buffer needs `alloc`). `unsafe` appears only in `AlignedBuf` (allocation and slice construction), each block documented with a `// SAFETY:` comment.

Panic policy (ADR 0002): the plain functions panic when the slice is too short or misaligned (see `# Panics`); the `try_*` variants return `None` / `false` instead.

## Installation

```toml
[dependencies]
tpt-simd-aligned = "0.1.0"
```

The crate is also re-exported by the umbrella crate `tpt-simd` as `tpt_simd::aligned`.

## Quick start

```rust
use tpt_simd_aligned::{
    Aligned32, AlignedBuf, is_aligned, load_aligned, store_aligned, try_load_aligned,
};
use tpt_simd_core::Simd;

// Stack/static data forced onto a 32-byte boundary.
let mut buf = Aligned32([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
let v = load_aligned::<f32, 8>(&buf.0);
assert_eq!(v[7], 8.0);
store_aligned(Simd::<f32, 8>::splat(2.5), &mut buf.0);
assert_eq!(buf.0, [2.5; 8]);

// A misaligned slice is rejected by the try_ form instead of panicking.
assert!(try_load_aligned::<f32, 8>(&buf.0[1..]).is_none());

// Heap buffer with 64-byte base alignment and densely packed elements.
let mut b = AlignedBuf::<f32, 64>::filled(100, 1.5);
assert!(is_aligned(b.as_ptr(), 64));
b[3] = 2.0;
assert_eq!(b.len(), 100);
```

## API overview

| Item | Description |
| --- | --- |
| `Aligned16<T>`, `Aligned32<T>`, `Aligned64<T>` | Alignment wrappers with a public field; `ALIGN`, `new`, `into_inner`, `Deref` / `DerefMut`, `From<T>`; derive `Clone`, `Copy`, `Debug`, `Default`, `PartialEq`, `Eq`, `PartialOrd`, `Ord`, `Hash` |
| `is_aligned(ptr, alignment)` | True if `ptr` is a multiple of a power-of-two `alignment`; a non-power-of-two (including 0) returns false |
| `vector_align::<T, N>()` | `const fn`; required byte alignment of an `N`-lane vector of `T` (its size rounded up to a power of two) |
| `load_aligned::<T, N>(slice)` | Load `N` lanes; panics if too short or misaligned |
| `store_aligned(v, slice)` | Store a `Simd`; panics if too short or misaligned |
| `try_load_aligned::<T, N>(slice)` | Returns `Option<Simd<T, N>>` |
| `try_store_aligned(v, slice)` | Returns `bool`; writes nothing on failure |
| `AlignedBuf<T: Copy, const ALIGN: usize = 32>` | Heap buffer; `filled`, `from_slice`, `zeroed`, `len`, `is_empty`, `BASE_ALIGN`; derefs to `[T]`; `Clone`, `Debug`, `PartialEq`; `Send` / `Sync` when `T` is |
| `aligned_vec_f32(len)`, `aligned_vec_i16(len)`, `aligned_vec_i32(len)` | Zero-initialised buffers aligned to 32 bytes |

## Feature flags

| Feature | Default | Effect |
| --- | --- | --- |
| `alloc` | yes | Enables `AlignedBuf` and the `aligned_vec_*` helpers (needs the `alloc` crate). |
| `std` | no | Implies `alloc` and forwards to `tpt-simd-core/std`. |
| `nightly` | no | Forwards to `tpt-simd-core/nightly` (currently a no-op). |
| `scalar-only` | no | Forwards to `tpt-simd-core/scalar-only` (currently a no-op). |

For an allocation-free build use `default-features = false`; the wrappers and slice helpers remain.

## Platform and safety notes

- All targets are supported; the wrappers use `repr(align)` only. Use `Aligned64` for AVX-512 or to avoid false sharing across cache lines.
- `AlignedBuf`'s effective alignment is `max(ALIGN, align_of::<T>())`. Construction panics if `ALIGN` is not a power of two or the size overflows `isize`, and aborts through `handle_alloc_error` if the allocator fails. Zero-length buffers do not allocate.
- Alignment checks use pointer addresses, so a sub-slice taken from the middle of an aligned buffer is generally not aligned.

## Related crates

- `tpt-simd-core` (dependency): `Simd` and `SimdElement`.
- `tpt-simd-vector`: backend types.
- `tpt-simd-testutil` (unpublished): dev-dependency for the property tests.
- `tpt-simd`: umbrella crate.

## MSRV

Rust 1.95, edition 2024.

## License

Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](../LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](../LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contributing

This project accepts issues only; pull requests are not accepted. Please open an issue to report bugs or request features — see [CONTRIBUTING.md](../CONTRIBUTING.md). Unless you explicitly state otherwise, anything you intentionally submit for inclusion (e.g. in an issue) is dual licensed as above, without any additional terms or conditions.
