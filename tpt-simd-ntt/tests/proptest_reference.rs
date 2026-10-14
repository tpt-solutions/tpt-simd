//! Property tests: Montgomery and Barrett reduction against `u64` arithmetic.

use proptest::prelude::*;
use tpt_simd_ntt::{Barrett, Montgomery};

/// Odd moduli in `[3, 2^31)`, including NTT primes.
const MODULI: &[u32] = &[3, 5, 12289, 998_244_353, 2_147_483_647];

fn modulus() -> impl Strategy<Value = u32> {
    prop::sample::select(MODULI)
}

proptest! {
    #[test]
    fn montgomery_mul_matches_u64(q in modulus(), a in any::<u32>(), b in any::<u32>()) {
        let m = Montgomery::new(q);
        let (a, b) = (a % q, b % q);
        let want = ((u64::from(a) * u64::from(b)) % u64::from(q)) as u32;
        prop_assert_eq!(m.mul(a, b), want);
    }

    #[test]
    fn montgomery_form_round_trips(q in modulus(), a in any::<u32>()) {
        let m = Montgomery::new(q);
        let a = a % q;
        prop_assert_eq!(m.from_mont(m.to_mont(a)), a);
    }

    #[test]
    fn montgomery_add_and_sub_match_modular_arithmetic(q in modulus(), a in any::<u32>(), b in any::<u32>()) {
        let m = Montgomery::new(q);
        let (a, b) = (a % q, b % q);
        let sum = ((u64::from(a) + u64::from(b)) % u64::from(q)) as u32;
        let diff = ((u64::from(a) + u64::from(q) - u64::from(b)) % u64::from(q)) as u32;
        prop_assert_eq!(m.add(a, b), sum);
        prop_assert_eq!(m.sub(a, b), diff);
    }

    #[test]
    fn montgomery_pow_matches_repeated_multiplication(q in modulus(), base in any::<u32>(), exp in 0u64..64) {
        let m = Montgomery::new(q);
        let base = base % q;
        let mut want = 1u32 % q;
        for _ in 0..exp {
            want = ((u64::from(want) * u64::from(base)) % u64::from(q)) as u32;
        }
        prop_assert_eq!(m.pow(base, exp), want);
    }

    #[test]
    fn barrett_reduce_matches_rem(q in 2u64..=(1u64 << 32), x in any::<u64>()) {
        let b = Barrett::new(q);
        prop_assert_eq!(b.reduce(x), x % q);
    }
}
