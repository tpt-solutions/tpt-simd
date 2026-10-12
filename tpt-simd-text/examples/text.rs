//! Text kernels: UTF-8 validation, hex encoding, and UTF-8 <-> UTF-16 transcoding.
//!
//! Run with: `cargo run -p tpt-simd-text --example text`

use tpt_simd_text::{hex_decode, hex_encode_lower, is_valid_utf8, utf8_to_utf16, utf16_to_utf8};

fn main() {
    assert!(is_valid_utf8("h€😀".as_bytes()));
    assert!(!is_valid_utf8(&[0xC3, 0x28]));

    // Hex round trip, no allocation.
    let mut hex = [0u8; 8];
    let n = hex_encode_lower(&[0xDE, 0xAD, 0xBE, 0xEF], &mut hex).unwrap();
    assert_eq!(&hex[..n], b"deadbeef");
    let mut bytes = [0u8; 4];
    hex_decode(&hex[..n], &mut bytes).unwrap();
    assert_eq!(bytes, [0xDE, 0xAD, 0xBE, 0xEF]);

    // UTF-8 -> UTF-16 and back. Units: 'h', '€', and a surrogate pair for the emoji.
    let mut units = [0u16; 4];
    let n = utf8_to_utf16("h€😀", &mut units).unwrap();
    assert_eq!(&units[..n], &[0x68, 0x20AC, 0xD83D, 0xDE00]);

    let mut back = [0u8; 8];
    let m = utf16_to_utf8(&units[..n], &mut back).unwrap();
    assert_eq!(core::str::from_utf8(&back[..m]).unwrap(), "h€😀");

    println!("text: all checks passed");
}
