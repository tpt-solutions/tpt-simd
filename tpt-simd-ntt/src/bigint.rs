//! Big-integer multiplication with the NTT.
//!
//! Numbers are little-endian `u32` limbs. Each limb is split into two 16-bit
//! digits, and the digit sequences are convolved with NTT over two primes. The
//! convolution coefficients are exact integers below `2^56` (at most `2^23`
//! terms, each below `2^32`), which is below the product of the two primes,
//! so Chinese remaindering recovers them exactly. A carry pass then turns the
//! coefficients back into limbs.
//!
//! Cost is `O(n log n)` in the limb count, against `O(n^2)` for schoolbook
//! multiplication. Measured with criterion (`bigint_mul` group, default build,
//! equal-length inputs): schoolbook is faster at 512 limbs (175 µs against 252 µs),
//! and the NTT is faster at 1024 limbs (544 µs against 721 µs). For smaller
//! inputs, use schoolbook.

use crate::{Montgomery, NttPlan};
use alloc::vec;
use alloc::vec::Vec;

/// The two NTT primes. Both are `1 mod 2^23`, so transforms up to `2^23` points exist.
const P1: u32 = 998_244_353;
const P2: u32 = 469_762_049;
const G: u32 = 3;
/// Largest transform, in digits. The coefficient bound needs `min(len) <= 2^23`.
const MAX_LOG2: u32 = 23;

/// Multiplies two little-endian `u32` limb numbers.
///
/// The result has `a.len() + b.len()` limbs and is not trimmed: leading zero
/// limbs are kept. Empty inputs give an empty result.
///
/// # Panics
///
/// Panics if the digit count of the transform would exceed `2^23` points.
///
/// # Examples
///
/// ```
/// use tpt_simd_ntt::bigint_mul;
///
/// // (2^32 + 1) * (2^32 + 1) = 2^64 + 2^33 + 1
/// assert_eq!(bigint_mul(&[1, 1], &[1, 1]), [1, 2, 1, 0]);
/// ```
#[must_use]
pub fn bigint_mul(a: &[u32], b: &[u32]) -> Vec<u32> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let out_limbs = a.len() + b.len();
    let da = digits(a);
    let db = digits(b);
    let conv_len = da.len() + db.len() - 1;
    let n = conv_len.next_power_of_two();
    assert!(
        n.trailing_zeros() <= MAX_LOG2,
        "bigint_mul: operands too large for the two-prime NTT ({} limbs)",
        a.len().max(b.len())
    );

    let c1 = convolve_mod(&da, &db, n, P1);
    let c2 = convolve_mod(&da, &db, n, P2);

    // Chinese remaindering: c = c1 + P1 * ((c2 - c1) * P1^-1 mod P2).
    let p1_inv_mod_p2 = Montgomery::new(P2).pow(P1 % P2, u64::from(P2) - 2);
    let m2 = Montgomery::new(P2);
    let mut coeffs = Vec::with_capacity(conv_len);
    for k in 0..conv_len {
        let r1 = c1[k];
        let r2 = c2[k];
        // (r2 - r1) mod P2, then times P1^-1 mod P2.
        let diff = m2.sub(r2, r1 % P2);
        let t = m2.mul(diff, p1_inv_mod_p2);
        coeffs.push(u64::from(r1) + u64::from(P1) * u64::from(t));
    }

    // Carry pass over 16-bit digits: coefficient k has weight 2^(16 k).
    let mut out_digits = Vec::with_capacity(2 * out_limbs);
    let mut carry: u128 = 0;
    for c in coeffs {
        carry += u128::from(c);
        out_digits.push((carry & 0xFFFF) as u32);
        carry >>= 16;
    }
    while carry > 0 {
        out_digits.push((carry & 0xFFFF) as u32);
        carry >>= 16;
    }
    // The product fits in out_limbs limbs, so the digits above that are zero.
    out_digits.resize(2 * out_limbs, 0);

    out_digits
        .chunks_exact(2)
        .map(|d| d[0] | (d[1] << 16))
        .collect()
}

/// Splits each `u32` limb into its low and high 16-bit digits.
fn digits(limbs: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(2 * limbs.len());
    for &x in limbs {
        out.push(x & 0xFFFF);
        out.push(x >> 16);
    }
    out
}

/// Linear convolution of `a` and `b` modulo `p`, computed with transforms of length `n`
/// (`n` at least `a.len() + b.len() - 1`, a power of two).
fn convolve_mod(a: &[u32], b: &[u32], n: usize, p: u32) -> Vec<u32> {
    let plan = NttPlan::new(p, G, n);
    let mut fa = vec![0u32; n];
    let mut fb = vec![0u32; n];
    for (dst, &x) in fa.iter_mut().zip(a) {
        *dst = x % p;
    }
    for (dst, &x) in fb.iter_mut().zip(b) {
        *dst = x % p;
    }
    // With n >= the full product length, cyclic convolution has no wrap-around.
    plan.cyclic_convolve(&mut fa, &mut fb);
    fa
}
