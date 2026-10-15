//! UTF-8 validation on inputs long enough for the vector paths, with one byte corrupted: errors
//! land on every block and tail position.

use proptest::prelude::*;
use tpt_simd_text::is_valid_utf8;

fn text_strategy() -> impl Strategy<Value = String> {
    // Characters of 1 to 4 bytes, so multi-byte sequences straddle the 16- and 32-byte blocks.
    prop::collection::vec(
        prop_oneof![
            (0x20u32..0x7f),
            (0xa0u32..0x800),
            (0x800u32..0xd800),
            (0xe000u32..0x10000),
            (0x10000u32..0x11000),
        ]
        .prop_map(|c| char::from_u32(c).unwrap_or('?')),
        0..120,
    )
    .prop_map(|v| v.into_iter().collect())
}

proptest! {
    #[test]
    fn valid_text_is_accepted(s in text_strategy()) {
        prop_assert!(is_valid_utf8(s.as_bytes()));
    }

    #[test]
    fn one_corrupted_byte_matches_std(
        s in text_strategy(),
        at in any::<usize>(),
        replacement in any::<u8>(),
    ) {
        let mut bytes = s.into_bytes();
        if !bytes.is_empty() {
            let i = at % bytes.len();
            bytes[i] = replacement;
        }
        prop_assert_eq!(is_valid_utf8(&bytes), core::str::from_utf8(&bytes).is_ok());
    }

    #[test]
    fn truncation_matches_std(s in text_strategy(), cut in any::<usize>()) {
        let bytes = s.as_bytes();
        let k = if bytes.is_empty() { 0 } else { cut % (bytes.len() + 1) };
        prop_assert_eq!(is_valid_utf8(&bytes[..k]), core::str::from_utf8(&bytes[..k]).is_ok());
    }
}
