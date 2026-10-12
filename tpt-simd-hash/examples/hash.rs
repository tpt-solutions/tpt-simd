//! Checksums and hashes: CRC-32, Adler-32, FNV-1a and a rolling hash.
//!
//! Run with: `cargo run -p tpt-simd-hash --example hash`

use tpt_simd_hash::{Crc32, adler32, crc32, fnv1a_64};

fn main() {
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);

    // Streaming: the result equals the one-shot CRC of the concatenated input.
    // Useful when the data arrives in chunks, e.g. from a file or socket.
    let mut c = Crc32::new();
    c.update(b"12345");
    c.update(b"6789");
    assert_eq!(c.finish(), 0xCBF4_3926);

    println!("hash: all checks passed");
}
