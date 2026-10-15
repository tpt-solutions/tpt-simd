//! Modular arithmetic and number-theoretic transforms.
//!
//! | type | purpose |
//! |---|---|
//! | [`Montgomery`] | fast `a * b mod q` for odd `q < 2^31` |
//! | [`Barrett`] | fast `x mod q` for `u64` `x` and any `q >= 1` |
//! | [`NttPlan`] | in-place forward / inverse NTT and cyclic convolution |
//!
//! ## Preconditions
//!
//! - Montgomery and NTT require an odd modulus below 2^31. The NTT additionally
//!   requires `q` to be prime, `g` to be a primitive root modulo `q`, and the
//!   transform length to divide `q - 1`. Primality and `g` are **not** checked.
//! - Operands passed to the arithmetic methods must be reduced (`< q`). NTT
//!   inputs are checked and the function panics otherwise.
//!
//! ## Example
//!
//! ```
//! use tpt_simd_ntt::{NttPlan, cyclic_convolve_naive};
//!
//! let plan = NttPlan::new(998_244_353, 3, 8);
//! let mut a = [1u32, 2, 0, 0, 0, 0, 0, 0];
//! let mut b = [3u32, 4, 0, 0, 0, 0, 0, 0];
//! let expect = cyclic_convolve_naive(998_244_353, &a, &b);
//! plan.cyclic_convolve(&mut a, &mut b);
//! assert_eq!(a[..], expect[..]);
//! ```
#![no_std]
#![deny(unsafe_code)]
#![warn(missing_docs)]

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

/// Whether `n` is prime, by trial division.
///
/// Intended for checking a modulus before building a [`Montgomery`] or [`Barrett`] context,
/// which accept any valid modulus and do not check primality themselves. Trial division costs
/// `O(sqrt(n))`, so it is fast for the `u32` moduli these types take.
///
/// ```
/// assert!(tpt_simd_ntt::is_prime(998_244_353));
/// assert!(!tpt_simd_ntt::is_prime(998_244_351));
/// ```
#[must_use]
pub fn is_prime(n: u32) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n.is_multiple_of(2) {
        return false;
    }
    let mut d = 3u64;
    while d * d <= u64::from(n) {
        if u64::from(n).is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// Montgomery arithmetic for an odd modulus `q < 2^31`, with `R = 2^32`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Montgomery {
    q: u32,
    /// `-q^{-1} mod 2^32`.
    q_neg_inv: u32,
    /// `R^2 mod q`, for converting into Montgomery form.
    r2: u32,
}

impl Montgomery {
    /// Creates the context. Panics unless `q` is odd and in `[3, 2^31)`.
    #[must_use]
    pub fn new(q: u32) -> Self {
        assert!(
            q >= 3 && q % 2 == 1 && q < (1 << 31),
            "Montgomery::new: modulus must be odd and in [3, 2^31), got {q}"
        );
        // Newton iteration for q^{-1} mod 2^32; q is its own inverse to 3 bits.
        let mut inv = q;
        for _ in 0..4 {
            inv = inv.wrapping_mul(2u32.wrapping_sub(q.wrapping_mul(inv)));
        }
        let r2 = ((1u128 << 64) % u128::from(q)) as u32;
        Self {
            q,
            q_neg_inv: inv.wrapping_neg(),
            r2,
        }
    }

    /// The modulus.
    #[must_use]
    pub fn modulus(&self) -> u32 {
        self.q
    }

    /// Montgomery reduction: returns `t * 2^-32 mod q`, for `t < q * 2^32`.
    #[must_use]
    pub fn reduce(&self, t: u64) -> u32 {
        let m = (t as u32).wrapping_mul(self.q_neg_inv);
        let u = ((t + u64::from(m) * u64::from(self.q)) >> 32) as u32;
        if u >= self.q { u - self.q } else { u }
    }

    /// Converts `a` into Montgomery form, `a * 2^32 mod q`.
    #[must_use]
    pub fn to_mont(&self, a: u32) -> u32 {
        self.reduce(u64::from(a % self.q) * u64::from(self.r2))
    }

    /// Converts `a` out of Montgomery form.
    #[must_use]
    pub fn from_mont(&self, a: u32) -> u32 {
        self.reduce(u64::from(a))
    }

    /// `a * b mod q` where `a < q` is normal and `b_mont` is in Montgomery form.
    /// Returns a normal value. This is the one-reduction fast path for fixed
    /// multipliers such as twiddle factors.
    #[must_use]
    pub fn mul_mont(&self, a: u32, b_mont: u32) -> u32 {
        self.reduce(u64::from(a) * u64::from(b_mont))
    }

    /// `a * b mod q` for normal inputs below `q`.
    #[must_use]
    pub fn mul(&self, a: u32, b: u32) -> u32 {
        self.mul_mont(a, self.to_mont(b))
    }

    /// `a + b mod q` for inputs below `q`.
    #[must_use]
    pub fn add(&self, a: u32, b: u32) -> u32 {
        let s = a + b;
        if s >= self.q { s - self.q } else { s }
    }

    /// `a - b mod q` for inputs below `q`.
    #[must_use]
    pub fn sub(&self, a: u32, b: u32) -> u32 {
        if a >= b { a - b } else { a + self.q - b }
    }

    /// `base^exp mod q` by square-and-multiply.
    #[must_use]
    pub fn pow(&self, base: u32, mut exp: u64) -> u32 {
        let mut result = 1 % self.q;
        let mut b = base % self.q;
        while exp > 0 {
            if exp & 1 == 1 {
                result = self.mul(result, b);
            }
            b = self.mul(b, b);
            exp >>= 1;
        }
        result
    }
}

/// Barrett reduction of `u64` values modulo `q`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Barrett {
    q: u64,
    /// `floor(2^64 / q)`.
    m: u64,
}

impl Barrett {
    /// Creates the context. Panics if `q` is less than 2 (`floor(2^64 / q)` must fit in `u64`).
    #[must_use]
    pub fn new(q: u64) -> Self {
        assert!(q >= 2, "Barrett::new: modulus must be at least 2, got {q}");
        let m = ((1u128 << 64) / u128::from(q)) as u64;
        Self { q, m }
    }

    /// `x mod q` for any `u64` `x`.
    ///
    /// The quotient estimate is at most one below the true quotient, so the
    /// remainder is below `2q` and one conditional subtraction finishes it.
    #[must_use]
    pub fn reduce(&self, x: u64) -> u64 {
        let qhat = ((u128::from(x) * u128::from(self.m)) >> 64) as u64;
        let mut r = x - qhat * self.q;
        if r >= self.q {
            r -= self.q;
        }
        r
    }
}

#[cold]
#[inline(never)]
fn bad_plan(n: usize, q: u32) -> ! {
    panic!(
        "NttPlan::new: length {n} must be a power of two dividing q - 1 = {}",
        q - 1
    );
}

/// In-place NTT of a fixed length and modulus.
#[derive(Clone, Debug)]
pub struct NttPlan {
    n: usize,
    mont: Montgomery,
    /// `w^j` in Montgomery form, `j < n / 2`, where `w` is a primitive n-th root.
    fwd: Vec<u32>,
    /// Same for `w^{-1}`.
    inv: Vec<u32>,
    /// `n^{-1}` in Montgomery form.
    n_inv: u32,
}

impl NttPlan {
    /// Creates a plan for length `n` modulo prime `q` with primitive root `g`.
    ///
    /// Panics unless `n` is a power of two that divides `q - 1`, or if `q` is
    /// not a valid Montgomery modulus.
    #[must_use]
    pub fn new(q: u32, g: u32, n: usize) -> Self {
        let mont = Montgomery::new(q);
        let qm1 = u64::from(q) - 1;
        if !n.is_power_of_two() || qm1 % (n as u64) != 0 {
            bad_plan(n, q);
        }
        let w = mont.pow(g, qm1 / n as u64);
        let w_inv = mont.pow(w, u64::from(q) - 2);
        let half = n / 2;
        let mut fwd = vec![0u32; half];
        let mut inv = vec![0u32; half];
        let (mut cf, mut ci) = (1u32, 1u32);
        for j in 0..half {
            fwd[j] = mont.to_mont(cf);
            inv[j] = mont.to_mont(ci);
            cf = mont.mul(cf, w);
            ci = mont.mul(ci, w_inv);
        }
        let n_inv = mont.to_mont(mont.pow(n as u32, u64::from(q) - 2));
        Self {
            n,
            mont,
            fwd,
            inv,
            n_inv,
        }
    }

    /// Transform length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns `true` if the length is 0 (never, by construction).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Forward NTT in place. Input and output are reduced residues.
    ///
    /// # Panics
    ///
    /// Panics if `a.len() != len()` or any element is `>= q`.
    pub fn forward(&self, a: &mut [u32]) {
        self.check(a);
        self.transform(a, &self.fwd);
    }

    /// Inverse NTT in place, including the `1/n` scaling, so
    /// `inverse(forward(a)) == a`.
    ///
    /// # Panics
    ///
    /// Panics if `a.len() != len()` or any element is `>= q`.
    pub fn inverse(&self, a: &mut [u32]) {
        self.check(a);
        self.transform(a, &self.inv);
        for x in a.iter_mut() {
            *x = self.mont.mul_mont(*x, self.n_inv);
        }
    }

    /// Cyclic convolution: `a <- a * b` modulo `x^n - 1` and `q`.
    ///
    /// `b` is used as scratch and is overwritten with its transform.
    ///
    /// # Panics
    ///
    /// Panics as [`NttPlan::forward`] does, for either slice.
    pub fn cyclic_convolve(&self, a: &mut [u32], b: &mut [u32]) {
        self.forward(a);
        self.forward(b);
        for (x, &y) in a.iter_mut().zip(b.iter()) {
            *x = self.mont.mul(*x, y);
        }
        self.inverse(a);
    }

    fn check(&self, a: &[u32]) {
        if a.len() != self.n {
            panic!(
                "NttPlan: input length {} does not match plan length {}",
                a.len(),
                self.n
            );
        }
        let q = self.mont.modulus();
        if a.iter().any(|&x| x >= q) {
            panic!("NttPlan: input element is not reduced modulo {q}");
        }
    }

    fn transform(&self, a: &mut [u32], roots: &[u32]) {
        let n = self.n;
        let mont = &self.mont;
        // Bit-reversal permutation.
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                a.swap(i, j);
            }
        }
        // Iterative Cooley-Tukey butterflies.
        let mut half = 1;
        while half < n {
            let len = 2 * half;
            let step = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..half {
                    let w = roots[k * step];
                    let u = a[start + k];
                    let v = mont.mul_mont(a[start + k + half], w);
                    a[start + k] = mont.add(u, v);
                    a[start + k + half] = mont.sub(u, v);
                }
            }
            half = len;
        }
    }
}

/// Reference cyclic convolution with direct summation, for testing plans.
#[must_use]
pub fn cyclic_convolve_naive(q: u32, a: &[u32], b: &[u32]) -> Vec<u32> {
    let n = a.len();
    assert_eq!(b.len(), n, "cyclic_convolve_naive: lengths differ");
    let mut out = vec![0u32; n];
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            let k = (i + j) % n;
            out[k] = ((u64::from(out[k]) + u64::from(x) * u64::from(y)) % u64::from(q)) as u32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use std::vec;

    const Q: u32 = 998_244_353;
    const G: u32 = 3;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 16
        }
        fn residue(&mut self, q: u32) -> u32 {
            (self.next() % u64::from(q)) as u32
        }
    }

    #[test]
    fn montgomery_matches_u64_mod() {
        let mut rng = Lcg(1);
        for q in [3u32, 17, 65_537, Q, (1 << 31) - 1] {
            let m = Montgomery::new(q);
            for _ in 0..2000 {
                let a = rng.residue(q);
                let b = rng.residue(q);
                let want = ((u64::from(a) * u64::from(b)) % u64::from(q)) as u32;
                assert_eq!(m.mul(a, b), want, "q={q} a={a} b={b}");
                assert_eq!(m.from_mont(m.to_mont(a)), a);
                assert_eq!(
                    m.add(a, b),
                    ((u64::from(a) + u64::from(b)) % u64::from(q)) as u32
                );
                assert_eq!(
                    m.sub(a, b),
                    ((u64::from(a) + u64::from(q) - u64::from(b)) % u64::from(q)) as u32
                );
            }
        }
    }

    #[test]
    fn montgomery_pow_matches_fermat() {
        let m = Montgomery::new(Q);
        assert_eq!(m.pow(2, 10), 1024);
        assert_eq!(m.pow(G, u64::from(Q) - 1), 1);
        assert_eq!(m.pow(5, 0), 1);
    }

    #[test]
    fn barrett_matches_rem() {
        let mut rng = Lcg(7);
        for q in [2u64, 17, 998_244_353, 0xFFFF_FFFB, u64::MAX / 3] {
            let b = Barrett::new(q);
            for _ in 0..2000 {
                let x = rng.next() << (rng.next() % 40);
                assert_eq!(b.reduce(x), x % q, "q={q} x={x}");
            }
            assert_eq!(b.reduce(u64::MAX), u64::MAX % q);
        }
    }

    #[test]
    fn ntt_round_trip_for_power_of_two_lengths() {
        let mut rng = Lcg(3);
        for log in 0..=10 {
            let n = 1usize << log;
            let plan = NttPlan::new(Q, G, n);
            let orig: vec::Vec<u32> = (0..n).map(|_| rng.residue(Q)).collect();
            let mut a = orig.clone();
            plan.forward(&mut a);
            plan.inverse(&mut a);
            assert_eq!(a, orig, "n={n}");
        }
    }

    #[test]
    fn ntt_of_delta_is_all_ones() {
        let plan = NttPlan::new(Q, G, 16);
        let mut a = [0u32; 16];
        a[0] = 1;
        plan.forward(&mut a);
        assert!(a.iter().all(|&x| x == 1));
    }

    #[test]
    fn cyclic_convolution_matches_naive() {
        let mut rng = Lcg(11);
        for (q, g, n) in [(Q, G, 8usize), (Q, G, 64), (17, 3, 8), (17, 3, 4)] {
            let plan = NttPlan::new(q, g, n);
            let a0: vec::Vec<u32> = (0..n).map(|_| rng.residue(q)).collect();
            let b0: vec::Vec<u32> = (0..n).map(|_| rng.residue(q)).collect();
            let want = cyclic_convolve_naive(q, &a0, &b0);
            let mut a = a0.clone();
            let mut b = b0.clone();
            plan.cyclic_convolve(&mut a, &mut b);
            assert_eq!(a, want, "q={q} n={n}");
        }
    }

    #[test]
    #[should_panic(expected = "must be a power of two dividing q - 1")]
    fn plan_rejects_length_not_dividing_q_minus_one() {
        // 17 - 1 = 16, so length 32 does not divide it.
        let _ = NttPlan::new(17, 3, 32);
    }

    #[test]
    #[should_panic(expected = "does not match plan length")]
    fn forward_rejects_wrong_length() {
        let plan = NttPlan::new(Q, G, 8);
        plan.forward(&mut [0u32; 4]);
    }

    #[test]
    #[should_panic(expected = "not reduced")]
    fn forward_rejects_unreduced_input() {
        let plan = NttPlan::new(Q, G, 2);
        plan.forward(&mut [Q, 0]);
    }
}
