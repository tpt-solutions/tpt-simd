# From manual intrinsics to tpt-simd

Manual SIMD intrinsics (`core::arch`) are verbose, unsafe, and hard to port
across x86_64/ARM64. tpt-simd gives you the same building blocks through a
safe, portable API.

## Before: manual AVX2 / NEON

```rust
// x86_64 AVX2: 8-lane f32 add
use core::arch::x86_64::*;
let a = _mm256_loadu_ps(ptr_a);
let b = _mm256_loadu_ps(ptr_b);
let c = _mm256_add_ps(a, b);
_mm256_storeu_si256(ptr_c, c);
```

## After: tpt-simd

```rust
use tpt_simd_core::F32x8;
use tpt_simd_core::Simd;

let a = F32x8::from_array(unsafe { ptr_a.read_unaligned() });
let b = F32x8::from_array(unsafe { ptr_b.read_unaligned() });
let c = a + b;
// ptr_c.write_unaligned(c.to_array());
```

## Mapping table

| Manual intrinsic | tpt-simd |
| --- | --- |
| `_mm256_add_ps` | `a + b` (or `add`) |
| `_mm256_mul_ps` | `a * b` (or `mul`) |
| `_mm256_mul_add_ps` | `a.mul_add(b, c)` |
| `_mm256_blendv_ps` | `blend_f32(mask, a, b)` |
| `_mm256_permutevar8x32_ps` | `permute_f32` |
| `_mm256_i32gather_epi32` | `gather_i32` |
| `_mm256_storeu_si256` | `to_array()` + `write_unaligned` (or `AlignedBuf`) |

## Key differences

- **Safety.** Intrinsics are `unsafe` at every load/store. tpt-simd wraps
  them behind safe APIs where the bounds are checked (`try_` variants) or
  documented (`# Safety`).
- **Portability.** One codebase compiles to AVX2, SSE2, NEON, SVE, RVV,
  depending only on the compile flags.
- **Type safety.** Lane types and lane counts are in the type (`F32x8`,
  `I16x16`); mixing widths is a compile error.
- **Semantics.** The workspace documents integer wrap vs. saturation, float
  NaN/infinity behaviour and rounding tie rules in [ADR 0002](../../docs/adr/0002-api-surface-and-policies.md).

## Migration steps

1. Replace load/store intrinsics with `Simd::from_array`/`to_array`.
2. Replace arithmetic intrinsics with operators `+ - * /` and `mul_add`.
3. Replace `cfg(target_feature)` branches with the build flags
   (`-C target-cpu=native`) described in [crate-overview](../../crate-overview.md).
4. Replace gather/scatter with the safe `try_` variants or use the checked
   `gather`/`scatter` crate.
5. Keep the portable scalar path as the reference for tests (proptest).

## See also

- [From scalar code](scalar-code.md)
- [From other SIMD crates](other-crates.md)
