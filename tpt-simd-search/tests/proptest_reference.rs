//! Property tests: byte and substring search against naive scans.

use proptest::prelude::*;
use tpt_simd_search::{
    count_byte, count_lines, find, find_ascii_case_insensitive, find_byte, rfind_byte,
};

fn naive_find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

proptest! {
    #[test]
    fn find_byte_matches_position(hay in prop::collection::vec(any::<u8>(), 0..400), needle in any::<u8>()) {
        prop_assert_eq!(find_byte(&hay, needle), hay.iter().position(|&b| b == needle));
    }

    #[test]
    fn rfind_byte_matches_rposition(hay in prop::collection::vec(any::<u8>(), 0..400), needle in any::<u8>()) {
        prop_assert_eq!(rfind_byte(&hay, needle), hay.iter().rposition(|&b| b == needle));
    }

    #[test]
    fn count_byte_matches_filter(hay in prop::collection::vec(any::<u8>(), 0..400), needle in any::<u8>()) {
        prop_assert_eq!(count_byte(&hay, needle), hay.iter().filter(|&&b| b == needle).count());
    }

    #[test]
    fn find_matches_naive(
        hay in prop::collection::vec(0u8..4, 0..300),
        needle in prop::collection::vec(0u8..4, 0..8),
    ) {
        prop_assert_eq!(find(&hay, &needle), naive_find(&hay, &needle));
    }

    #[test]
    fn find_ascii_case_insensitive_matches_naive(
        hay in "[a-cA-C ]{0,200}",
        needle in "[a-cA-C]{1,5}",
    ) {
        let lower_hay = hay.to_ascii_lowercase();
        let lower_needle = needle.to_ascii_lowercase();
        prop_assert_eq!(
            find_ascii_case_insensitive(hay.as_bytes(), needle.as_bytes()),
            naive_find(lower_hay.as_bytes(), lower_needle.as_bytes()),
        );
    }

    #[test]
    fn count_lines_matches_newlines(bytes in prop::collection::vec(prop::sample::select(vec![b'a', b'\n']), 0..300)) {
        let newlines = bytes.iter().filter(|&&b| b == b'\n').count();
        let tail = usize::from(!bytes.is_empty() && bytes.last() != Some(&b'\n'));
        prop_assert_eq!(count_lines(&bytes), newlines + tail);
    }
}
