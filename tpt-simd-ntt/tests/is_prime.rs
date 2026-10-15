//! `is_prime` against a sieve, and against known NTT primes.

use proptest::prelude::*;
use tpt_simd_ntt::is_prime;

fn sieve(limit: usize) -> Vec<bool> {
    let mut is = vec![true; limit + 1];
    is[0] = false;
    if limit >= 1 {
        is[1] = false;
    }
    let mut i = 2;
    while i * i <= limit {
        if is[i] {
            let mut j = i * i;
            while j <= limit {
                is[j] = false;
                j += i;
            }
        }
        i += 1;
    }
    is
}

#[test]
fn agrees_with_a_sieve_below_ten_thousand() {
    let table = sieve(10_000);
    for n in 0..=10_000u32 {
        assert_eq!(is_prime(n), table[n as usize], "n = {n}");
    }
}

#[test]
fn known_ntt_moduli() {
    for q in [12_289u32, 998_244_353, 2_147_483_647] {
        assert!(is_prime(q), "{q} is prime");
    }
    // 998244351 = 3 * 332748117, not prime.
    assert!(!is_prime(998_244_351));
    assert!(!is_prime(u32::MAX));
}

proptest! {
    #[test]
    fn composites_have_a_small_factor(a in 2u32..60_000, b in 2u32..60_000) {
        let n = a * b;
        prop_assert!(!is_prime(n));
    }
}
