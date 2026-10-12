//! Byte and substring search, case-insensitive search, and line counting.
//!
//! Run with: `cargo run -p tpt-simd-search --example search`

use tpt_simd_search::{count_byte, count_lines, find, find_ascii_case_insensitive, find_byte};

fn main() {
    assert_eq!(find_byte(b"hello", b'l'), Some(2));
    assert_eq!(find(b"hello world", b"world"), Some(6));
    assert_eq!(
        find_ascii_case_insensitive(b"Hello WORLD", b"world"),
        Some(6)
    );
    assert_eq!(count_byte(b"banana", b'a'), 3);
    assert_eq!(count_lines(b"a\nb\nc"), 3);

    println!("search: all checks passed");
}
