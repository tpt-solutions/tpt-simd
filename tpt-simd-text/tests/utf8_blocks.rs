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

/// Every byte value at every position of a few inputs long enough for the vector paths. Proptest
/// found the 0xFD lead case only by chance (it was accepted when it followed a 4-byte lead
/// pattern), so this exhaustive check pins it down.
#[test]
fn every_single_byte_replacement_matches_std() {
    let bases = [
        // 80 bytes: two 32-byte blocks and a tail, so the NEON path (from 32 bytes) and the AVX2
        // path (from 64 bytes) both run.
        "ascii only, long enough for two 32-byte blocks plus a tail, so both vector paths run.",
        "ࠀ𐀀𐀀ࠀ 𐀀\u{a0}ࠀ    \u{a0}\u{a0}ࠀ𐀀𐀀ࠀࠀ𐀀ࠀ𐀀ࠀ\u{a0}   𐀀\u{a0}\u{a0}ࠀ\u{a0}ࠀ𐀀\u{a0}ࠀࠀ ࠀ  \u{a0}\u{a0} ࠀ \u{a0}ࠀ\u{a0} ࠀ𐀀ࠀ𐀀ࠀ\u{a0}ࠀ¢ı𐻖ꊴͼځz\u{10210}％Ҡᦜ\u{107c8}\u{f4b6})𐴐Ȣćz𐴠\u{ed9f}\u{10a01}𐀫\u{f2aa}} 𐠟Ė\u{61a}𐨢\u{108d6}{𐬿\u{fadd}𐲕챞\u{1095e}𐗦Şۆ",
    ];
    for base in bases {
        let base = base.as_bytes();
        for i in 0..base.len() {
            for r in 0..=255u8 {
                let mut bytes = base.to_vec();
                bytes[i] = r;
                assert_eq!(
                    is_valid_utf8(&bytes),
                    core::str::from_utf8(&bytes).is_ok(),
                    "position {i}, byte {r:#04x}"
                );
            }
        }
    }
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
