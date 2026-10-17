//! `bigint_mul` must equal schoolbook multiplication limb for limb. The schoolbook
//! reference uses `u64` products and carries. The inputs include all-ones limbs,
//! which maximise every convolution coefficient and so stress the carry pass.

use tpt_simd_ntt::bigint_mul;

fn schoolbook(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = vec![0u32; a.len() + b.len()];
    for (i, &x) in a.iter().enumerate() {
        let mut carry = 0u64;
        for (j, &y) in b.iter().enumerate() {
            let t = u64::from(out[i + j]) + u64::from(x) * u64::from(y) + carry;
            out[i + j] = t as u32;
            carry = t >> 32;
        }
        let mut k = i + b.len();
        while carry > 0 {
            let t = u64::from(out[k]) + carry;
            out[k] = t as u32;
            carry = t >> 32;
            k += 1;
        }
    }
    out
}

fn random_limbs(n: usize, seed: &mut u64) -> Vec<u32> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            *seed as u32
        })
        .collect()
}

#[test]
fn empty_and_single_limb_cases() {
    assert!(bigint_mul(&[], &[1, 2]).is_empty());
    assert!(bigint_mul(&[3], &[]).is_empty());
    assert_eq!(bigint_mul(&[7], &[6]), schoolbook(&[7], &[6]));
    assert_eq!(
        bigint_mul(&[u32::MAX], &[u32::MAX]),
        schoolbook(&[u32::MAX], &[u32::MAX])
    );
}

#[test]
fn all_ones_limbs_match_schoolbook() {
    for (na, nb) in [
        (1usize, 1usize),
        (2, 3),
        (5, 5),
        (16, 9),
        (64, 64),
        (100, 31),
        (257, 255),
    ] {
        let a = vec![u32::MAX; na];
        let b = vec![u32::MAX; nb];
        assert_eq!(bigint_mul(&a, &b), schoolbook(&a, &b), "{na}x{nb}");
    }
}

#[test]
fn random_limbs_match_schoolbook() {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    for na in [1usize, 2, 3, 7, 8, 9, 31, 64, 129, 300] {
        for nb in [1usize, 4, 17, 128, 300] {
            let a = random_limbs(na, &mut seed);
            let b = random_limbs(nb, &mut seed);
            assert_eq!(bigint_mul(&a, &b), schoolbook(&a, &b), "{na}x{nb}");
        }
    }
}

#[test]
fn zero_limbs_and_sparse_inputs() {
    let zeros = vec![0u32; 40];
    let mut sparse = vec![0u32; 40];
    sparse[39] = 1;
    assert_eq!(bigint_mul(&zeros, &sparse), schoolbook(&zeros, &sparse));
    assert!(bigint_mul(&zeros, &sparse).iter().all(|&x| x == 0));
    assert_eq!(bigint_mul(&sparse, &sparse), schoolbook(&sparse, &sparse));
}
