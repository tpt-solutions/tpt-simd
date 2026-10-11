//! Differential tests: our byte and substring search against `memchr`, which
//! is a dev-dependency only, and against a naive reference.

use tpt_simd_search::{count_byte, find, find_byte, rfind_byte};

fn sample(n: usize, seed: u32, alphabet: &[u8]) -> Vec<u8> {
    let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            alphabet[(x as usize) % alphabet.len()]
        })
        .collect()
}

fn naive_find(h: &[u8], n: &[u8]) -> Option<usize> {
    if n.is_empty() {
        return Some(0);
    }
    if h.len() < n.len() {
        return None;
    }
    (0..=h.len() - n.len()).find(|&i| &h[i..i + n.len()] == n)
}

#[test]
fn byte_search_matches_memchr_over_lengths_and_bytes() {
    for n in (0..200).chain([1023, 4096]) {
        let h = sample(n, n as u32, &(0u8..=255).collect::<Vec<_>>());
        for needle in [0u8, 1, 0x7F, 0x80, 0xFF, b'a'] {
            assert_eq!(
                find_byte(&h, needle),
                memchr::memchr(needle, &h),
                "n={n} b={needle:#x}"
            );
            assert_eq!(
                rfind_byte(&h, needle),
                memchr::memrchr(needle, &h),
                "n={n} b={needle:#x}"
            );
            assert_eq!(
                count_byte(&h, needle),
                memchr::memchr_iter(needle, &h).count(),
                "n={n} b={needle:#x}"
            );
        }
    }
}

#[test]
fn byte_search_at_every_position() {
    for len in 1..70 {
        for at in 0..len {
            let mut h = vec![b'.'; len];
            h[at] = b'#';
            assert_eq!(find_byte(&h, b'#'), Some(at), "len={len} at={at}");
            assert_eq!(rfind_byte(&h, b'#'), Some(at), "len={len} at={at}");
        }
    }
}

#[test]
fn substring_find_matches_memmem_and_naive() {
    let alphabet = b"abcab ";
    for seed in 0..300u32 {
        let h = sample(20 + (seed as usize % 200), seed, alphabet);
        let needle_len = (seed % 6) as usize;
        let n = sample(needle_len, seed + 999, alphabet);
        let ours = find(&h, &n);
        assert_eq!(ours, memchr::memmem::find(&h, &n), "seed={seed}");
        assert_eq!(ours, naive_find(&h, &n), "seed={seed}");
    }
}

#[test]
fn substring_find_edge_cases() {
    assert_eq!(find(b"", b""), Some(0));
    assert_eq!(find(b"", b"a"), None);
    assert_eq!(find(b"abc", b"abcd"), None);
    assert_eq!(find(b"aaaa", b"aa"), Some(0));
    assert_eq!(find(b"xxab", b"ab"), Some(2));
    assert_eq!(find(b"ab", b"ab"), Some(0));
}
