# How to do motion compensation in a video codec

Motion compensation is the per-block search for the best-matching
reference block, scored by a fixed-point sum of absolute difference (SAD) or
sum of squared differences (SSD). tpt-simd contributes: fixed-point arithmetic
(`tpt-simd-fixed`), saturating arithmetic (`tpt-simd-saturate`), dot products
(`tpt-simd-dot`), and the integer neon/vector backend.

## Fixed-point score

Use `Fixed<i32, 16, 16, N>` (Q16.16) for sample values. Compute the SAD/SSD in
fixed point, then rescale with `tpt_simd_shift::descale_i32` (arithmetic
right shift with rounding) for the final score.

```rust
use tpt_simd_fixed::Fixed;
use tpt_simd_shift::descale_i32;
use tpt_simd_core::{Simd, I32x8};

/// Sum of absolute differences, fixed-point (Q16.16).
fn sad(a: &[i16], b: &[i16]) -> i32 {
    let mut acc = I32x8::splat(0);
    let mut i = 0;
    while i + 8 <= a.len() {
        let x: I32x8 = Simd::from_array(*a[i..i + 8].try_into().unwrap());
        let y: I32x8 = Simd::from_array(*b[i..i + 8].try_into().unwrap());
        let d = (x.abs() - y.abs()).wrapping_add(acc); // wrapping, wrap at 2^32
        // note: abs on i16 lanes -> convert to i32; see production code for
        // the saturating widening that avoids overflow on the 8-lane sum.
        acc = d;
        i += 8;
    }
    // horizontal sum
    let s = tpt_simd_horizontal::sum_i32(acc);
    s.reduce_sum()
}

/// Rescale an accumulated fixed-point score to a plain integer.
fn rescale(score: i32, frac_bits: i32) -> i32 {
    descale_i32(score, frac_bits)
}
```

## Search

The search loop (exhaustive or a fast heuristic) calls the block-match
function for each candidate offset. Because `Fixed` operators wrap at the
storage width, the hot block-match uses `__wrapping_add`/`__wrapping_sub` /
`__mul` and explicit `saturating_*` for the boundary handling.

```rust
use tpt_simd_fixed::Fixed;

type Q = Fixed<i32, 16, 16, 8>;

/// Matching cost for one candidate block (Q16.16 residual, accumulated).
fn block_cost(pred: &[i16], ref_block: &[i16]) -> i32 {
    let mut acc = 0i32;
    for (p, r) in pred.iter().zip(ref_block) {
        let diff = (p as i32).saturating_sub(*r as i32); // Q16.16: 16 frac bits
        acc += diff.abs();
    }
    acc << 16 // convert to Q16.16
}

fn main() {
    let pred: Vec<i16> = (0..64).map(|i| (i % 256) as i16).collect();
    let ref_block: Vec<i16> = (100..164).map(|i| (i % 256) as i16).collect();
    let cost = block_cost(&pred, &ref_block);
    let score = rescale(cost, 16);
    eprintln!("block score: {}", score);
}
```

## Performance notes

- Use the **portable** fixed-point path (no `scalar-only`); on x86_64 with
  AVX2 it uses `_mm256_mul_epi32` on even/odd lanes and is bit-identical to
  the portable path.
- SAD/SSD are integer reductions; when the 8-lane accumulator is consumed,
  `tpt_simd_horizontal::sum_i32` (or the `horizontal` crate) reduces with a
  documented order.
- `saturating_*` is for the boundary/prediction handling, never for the block
  score itself (the score must wrap, not saturate, to stay unbiased).
- Motion compensation is a search; the SIMD kernels here are the per-block
  scoring kernels. The search strategy (full search vs. coarse-to-fine) is
  outside tpt-simd's scope.

## See also

- [How to convert between fixed-point formats](fixed-point-conversion.md)
- [How to implement an FFT using tpt-simd](fft.md)
