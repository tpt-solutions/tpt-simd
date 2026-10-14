//! Property tests: UTF-8, UTF-16 and hex kernels against `std`.

use proptest::prelude::*;
use tpt_simd_text::{
    hex_decode, hex_encode_lower, is_valid_utf8, utf8_len_of_utf16, utf8_to_utf16, utf16_len,
    utf16_to_utf8,
};

proptest! {
    #[test]
    fn is_valid_utf8_matches_std(bytes in prop::collection::vec(any::<u8>(), 0..300)) {
        prop_assert_eq!(is_valid_utf8(&bytes), std::str::from_utf8(&bytes).is_ok());
    }

    #[test]
    fn valid_strings_are_accepted(s in ".*") {
        prop_assert!(is_valid_utf8(s.as_bytes()));
    }

    #[test]
    fn utf8_to_utf16_matches_std(s in ".*") {
        let want: Vec<u16> = s.encode_utf16().collect();
        prop_assert_eq!(utf16_len(&s), want.len());
        let mut dst = vec![0u16; want.len()];
        let n = utf8_to_utf16(&s, &mut dst).expect("buffer is exact");
        prop_assert_eq!(n, want.len());
        prop_assert_eq!(dst, want);
    }

    #[test]
    fn utf16_to_utf8_matches_std(units in prop::collection::vec(any::<u16>(), 0..200)) {
        let want = String::from_utf16(&units);
        let mut dst = vec![0u8; units.len() * 3];
        let got = utf16_to_utf8(&units, &mut dst);
        match want {
            Ok(s) => {
                let n = got.expect("valid UTF-16 is accepted");
                prop_assert_eq!(&dst[..n], s.as_bytes());
                prop_assert_eq!(utf8_len_of_utf16(&units).expect("valid"), s.len());
            }
            Err(_) => prop_assert!(got.is_err()),
        }
    }

    #[test]
    fn hex_encode_matches_format(data in prop::collection::vec(any::<u8>(), 0..200)) {
        let want: String = data.iter().map(|b| format!("{b:02x}")).collect();
        let mut dst = vec![0u8; data.len() * 2];
        let n = hex_encode_lower(&data, &mut dst).expect("buffer is exact");
        prop_assert_eq!(&dst[..n], want.as_bytes());
    }

    #[test]
    fn hex_round_trips(data in prop::collection::vec(any::<u8>(), 0..200)) {
        let mut encoded = vec![0u8; data.len() * 2];
        hex_encode_lower(&data, &mut encoded).expect("buffer is exact");
        let mut decoded = vec![0u8; data.len()];
        let n = hex_decode(&encoded, &mut decoded).expect("valid hex");
        prop_assert_eq!(n, data.len());
        prop_assert_eq!(decoded, data);
    }
}
