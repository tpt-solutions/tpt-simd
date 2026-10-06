//! Shared argument validation (panics with descriptive messages).

/// Checks that a column-major `rows x cols` matrix with leading dimension `ld`
/// fits in a slice of `len` elements.
pub(crate) fn check_mat(func: &str, name: &str, rows: usize, cols: usize, ld: usize, len: usize) {
    if rows == 0 || cols == 0 {
        return;
    }
    assert!(
        ld >= rows,
        "{func}: leading dimension of {name} is {ld}, must be >= {rows} rows"
    );
    let need = (cols - 1)
        .checked_mul(ld)
        .and_then(|v| v.checked_add(rows))
        .unwrap_or(usize::MAX);
    assert!(
        len >= need,
        "{func}: {name} has {len} elements, needs at least {need} ({rows}x{cols}, ld {ld})"
    );
}

pub(crate) fn check_ws(func: &str, have: usize, need: usize) {
    assert!(
        have >= need,
        "{func}: workspace has {have} elements, needs at least {need}"
    );
}
