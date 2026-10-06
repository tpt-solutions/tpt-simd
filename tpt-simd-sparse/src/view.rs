//! Borrowed CSR/CSC views, validation, and (with `alloc`) owned matrices.

use crate::real::Real;
use core::fmt;

/// Why a compressed-sparse structure was rejected by validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseError {
    /// `indptr.len()` is not `major + 1`.
    IndptrLength {
        /// Required length.
        expected: usize,
        /// Actual length.
        got: usize,
    },
    /// `indptr[0]` is not 0.
    IndptrStart(usize),
    /// `indptr[at] > indptr[at + 1]`.
    IndptrNotMonotone {
        /// Position of the first decreasing pair.
        at: usize,
    },
    /// `indptr[last]`, `indices.len()` and `data.len()` disagree.
    NnzMismatch {
        /// `indptr[major]`.
        indptr_last: usize,
        /// `indices.len()`.
        indices: usize,
        /// `data.len()`.
        data: usize,
    },
    /// `indices[pos] >= minor`.
    IndexOutOfRange {
        /// Position in `indices` (or in the triplet list).
        pos: usize,
        /// The offending index (saturated to `u32::MAX`).
        index: u32,
        /// The exclusive bound (`ncols` for CSR, `nrows` for CSC).
        bound: usize,
    },
    /// The minor dimension does not fit a `u32` index.
    DimensionTooLarge(usize),
}

impl fmt::Display for SparseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::IndptrLength { expected, got } => {
                write!(f, "indptr has length {got}, expected {expected}")
            }
            Self::IndptrStart(v) => write!(f, "indptr[0] is {v}, expected 0"),
            Self::IndptrNotMonotone { at } => {
                write!(f, "indptr decreases between positions {at} and {}", at + 1)
            }
            Self::NnzMismatch {
                indptr_last,
                indices,
                data,
            } => write!(
                f,
                "indptr ends at {indptr_last} but indices has {indices} and data has {data} entries"
            ),
            Self::IndexOutOfRange { pos, index, bound } => {
                write!(
                    f,
                    "index {index} at position {pos} is out of range (bound {bound})"
                )
            }
            Self::DimensionTooLarge(d) => write!(f, "dimension {d} does not fit a u32 index"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SparseError {}

/// Shared representation: `major` compressed lines (rows for CSR, columns
/// for CSC) over a `minor` dimension. Fields are private so that a value
/// can only be built through [`validate`] or an `unsafe` constructor, which
/// is what makes the unchecked loads in the kernels sound.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Compressed<'a, T> {
    pub(crate) major: usize,
    pub(crate) minor: usize,
    pub(crate) indptr: &'a [usize],
    pub(crate) indices: &'a [u32],
    pub(crate) data: &'a [T],
}

/// Checks the compressed-storage invariants (everything except sortedness
/// and uniqueness of indices, which are deliberately not required).
pub(crate) fn validate<T>(
    major: usize,
    minor: usize,
    indptr: &[usize],
    indices: &[u32],
    data: &[T],
) -> Result<(), SparseError> {
    if (minor as u64) > (1u64 << 32) {
        return Err(SparseError::DimensionTooLarge(minor));
    }
    if indptr.len() != major + 1 {
        return Err(SparseError::IndptrLength {
            expected: major + 1,
            got: indptr.len(),
        });
    }
    if indptr[0] != 0 {
        return Err(SparseError::IndptrStart(indptr[0]));
    }
    if let Some(at) = indptr.windows(2).position(|w| w[0] > w[1]) {
        return Err(SparseError::IndptrNotMonotone { at });
    }
    let last = indptr[major];
    if last != indices.len() || last != data.len() {
        return Err(SparseError::NnzMismatch {
            indptr_last: last,
            indices: indices.len(),
            data: data.len(),
        });
    }
    if let Some(pos) = indices.iter().position(|&i| i as usize >= minor) {
        return Err(SparseError::IndexOutOfRange {
            pos,
            index: indices[pos],
            bound: minor,
        });
    }
    Ok(())
}

macro_rules! view_type {
    (
        $(#[$doc:meta])* $name:ident, $major:ident, $minor:ident,
        $mdoc:literal, $ndoc:literal
    ) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug)]
        pub struct $name<'a, T: Real> {
            pub(crate) c: Compressed<'a, T>,
        }

        impl<'a, T: Real> $name<'a, T> {
            /// Validates and wraps borrowed slices.
            ///
            #[doc = $mdoc]
            /// Indices within a line may be **unsorted** and may contain
            /// **duplicates**; see the crate docs, "Index policy".
            ///
            /// # Errors
            /// Returns the first violated invariant as a [`SparseError`].
            pub fn try_new(
                $major: usize,
                $minor: usize,
                indptr: &'a [usize],
                indices: &'a [u32],
                data: &'a [T],
            ) -> Result<Self, SparseError> {
                validate($major, $minor, indptr, indices, data)?;
                Ok(Self {
                    c: Compressed {
                        major: $major,
                        minor: $minor,
                        indptr,
                        indices,
                        data,
                    },
                })
            }

            /// Wraps borrowed slices **without validation**: the unchecked
            /// fast path for data already known to be good (for example
            /// produced by a trusted builder), skipping the O(nnz) scan of
            /// [`try_new`](Self::try_new). The SpMV kernels themselves are
            /// identical for both constructors.
            ///
            /// # Safety
            /// The slices must satisfy every invariant that
            /// [`try_new`](Self::try_new) checks. The kernels rely on them
            /// for unchecked loads from the dense vectors, so a violation is
            /// undefined behaviour.
            #[must_use]
            pub unsafe fn new_unchecked(
                $major: usize,
                $minor: usize,
                indptr: &'a [usize],
                indices: &'a [u32],
                data: &'a [T],
            ) -> Self {
                Self {
                    c: Compressed {
                        major: $major,
                        minor: $minor,
                        indptr,
                        indices,
                        data,
                    },
                }
            }

            /// Number of stored entries (duplicates counted).
            #[must_use]
            pub fn nnz(&self) -> usize {
                self.c.data.len()
            }
            /// Line pointers (`len == major + 1`).
            #[must_use]
            pub fn indptr(&self) -> &'a [usize] {
                self.c.indptr
            }
            /// Minor-dimension indices of every stored entry.
            #[must_use]
            pub fn indices(&self) -> &'a [u32] {
                self.c.indices
            }
            /// Stored values.
            #[must_use]
            pub fn data(&self) -> &'a [T] {
                self.c.data
            }
        }
    };
}

view_type!(
    /// Borrowed compressed-sparse-row matrix: row `i` holds the entries
    /// `indptr[i]..indptr[i + 1]` of `indices` (column) and `data`.
    CsrView, nrows, ncols,
    "Checks that `indptr` has length `nrows + 1`, starts at 0, is non-decreasing and ends at `indices.len() == data.len()`, and that every index is below `ncols`.",
    "ncols"
);
view_type!(
    /// Borrowed compressed-sparse-column matrix: column `j` holds the entries
    /// `indptr[j]..indptr[j + 1]` of `indices` (row) and `data`.
    CscView, ncols, nrows,
    "Checks that `indptr` has length `ncols + 1`, starts at 0, is non-decreasing and ends at `indices.len() == data.len()`, and that every index is below `nrows`.",
    "nrows"
);

impl<T: Real> CsrView<'_, T> {
    /// Number of rows.
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.c.major
    }
    /// Number of columns.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.c.minor
    }
}

impl<T: Real> CscView<'_, T> {
    /// Number of rows.
    #[must_use]
    pub fn nrows(&self) -> usize {
        self.c.minor
    }
    /// Number of columns.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.c.major
    }
}

#[cfg(feature = "alloc")]
mod owned {
    extern crate alloc;
    use super::{CscView, CsrView, SparseError};
    use crate::real::Real;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Owned CSR matrix (convenience for building test/benchmark inputs).
    #[derive(Clone, Debug, PartialEq)]
    pub struct CsrMatrix<T: Real> {
        nrows: usize,
        ncols: usize,
        indptr: Vec<usize>,
        indices: Vec<u32>,
        data: Vec<T>,
    }

    /// Owned CSC matrix.
    #[derive(Clone, Debug, PartialEq)]
    pub struct CscMatrix<T: Real> {
        nrows: usize,
        ncols: usize,
        indptr: Vec<usize>,
        indices: Vec<u32>,
        data: Vec<T>,
    }

    impl<T: Real> CsrMatrix<T> {
        /// Builds from owned vectors, validating them like
        /// [`CsrView::try_new`].
        ///
        /// # Errors
        /// The first violated invariant.
        pub fn try_new(
            nrows: usize,
            ncols: usize,
            indptr: Vec<usize>,
            indices: Vec<u32>,
            data: Vec<T>,
        ) -> Result<Self, SparseError> {
            super::validate(nrows, ncols, &indptr, &indices, &data)?;
            Ok(Self {
                nrows,
                ncols,
                indptr,
                indices,
                data,
            })
        }

        /// Builds from `(row, col, value)` triplets. Entries are ordered by
        /// row then (stably) by input order; **duplicates are kept as
        /// separate entries**, which SpMV sums. Columns within a row keep
        /// input order (they are not sorted).
        ///
        /// # Errors
        /// [`SparseError::IndexOutOfRange`] if a coordinate is outside the
        /// shape; `pos` is the triplet position.
        pub fn from_triplets(
            nrows: usize,
            ncols: usize,
            triplets: &[(usize, usize, T)],
        ) -> Result<Self, SparseError> {
            let (indptr, indices, data) =
                bucket(nrows, ncols, triplets.iter().map(|&(r, c, v)| (r, c, v)))?;
            Self::try_new(nrows, ncols, indptr, indices, data)
        }

        /// The 2D Poisson (5-point Laplacian) matrix on an `n x n` grid with
        /// Dirichlet boundaries: `4` on the diagonal, `-1` for each of the up
        /// to four grid neighbours. Size `n² x n²`, symmetric positive
        /// definite, columns sorted within each row.
        #[must_use]
        pub fn poisson_2d(n: usize) -> Self {
            let dim = n * n;
            let mut indptr = Vec::with_capacity(dim + 1);
            let mut indices: Vec<u32> = Vec::with_capacity(5 * dim);
            let mut data = Vec::with_capacity(5 * dim);
            let four = T::ONE + T::ONE + T::ONE + T::ONE;
            let neg = T::ZERO - T::ONE;
            indptr.push(0);
            for i in 0..n {
                for j in 0..n {
                    let k = i * n + j;
                    if i > 0 {
                        indices.push((k - n) as u32);
                        data.push(neg);
                    }
                    if j > 0 {
                        indices.push((k - 1) as u32);
                        data.push(neg);
                    }
                    indices.push(k as u32);
                    data.push(four);
                    if j + 1 < n {
                        indices.push((k + 1) as u32);
                        data.push(neg);
                    }
                    if i + 1 < n {
                        indices.push((k + n) as u32);
                        data.push(neg);
                    }
                    indptr.push(indices.len());
                }
            }
            Self {
                nrows: dim,
                ncols: dim,
                indptr,
                indices,
                data,
            }
        }

        /// Borrowed view (no re-validation needed; the invariants were
        /// established at construction).
        #[must_use]
        pub fn view(&self) -> CsrView<'_, T> {
            // SAFETY: every constructor of `CsrMatrix` validates or builds
            // the invariants, and the fields are private and immutable.
            unsafe {
                CsrView::new_unchecked(
                    self.nrows,
                    self.ncols,
                    &self.indptr,
                    &self.indices,
                    &self.data,
                )
            }
        }

        /// The same matrix in CSC layout (entries within a column keep
        /// ascending row order of the CSR traversal; duplicates preserved).
        #[must_use]
        pub fn to_csc(&self) -> CscMatrix<T> {
            let mut trip = Vec::with_capacity(self.data.len());
            for r in 0..self.nrows {
                for k in self.indptr[r]..self.indptr[r + 1] {
                    trip.push((self.indices[k] as usize, r, self.data[k]));
                }
            }
            // Here "major" is the column.
            let (indptr, indices, data) =
                bucket(self.ncols, self.nrows, trip.into_iter()).expect("indices were validated");
            CscMatrix {
                nrows: self.nrows,
                ncols: self.ncols,
                indptr,
                indices,
                data,
            }
        }

        /// Number of rows.
        #[must_use]
        pub fn nrows(&self) -> usize {
            self.nrows
        }
        /// Number of columns.
        #[must_use]
        pub fn ncols(&self) -> usize {
            self.ncols
        }
    }

    impl<T: Real> CscMatrix<T> {
        /// Borrowed view.
        #[must_use]
        pub fn view(&self) -> CscView<'_, T> {
            // SAFETY: built by `CsrMatrix::to_csc` from validated data.
            unsafe {
                CscView::new_unchecked(
                    self.ncols,
                    self.nrows,
                    &self.indptr,
                    &self.indices,
                    &self.data,
                )
            }
        }
    }

    /// Stable counting sort of `(major, minor, value)` items into compressed
    /// storage.
    #[allow(clippy::type_complexity)]
    fn bucket<T: Real>(
        major_dim: usize,
        minor_dim: usize,
        items: impl Iterator<Item = (usize, usize, T)>,
    ) -> Result<(Vec<usize>, Vec<u32>, Vec<T>), SparseError> {
        let items: Vec<(usize, usize, T)> = items.collect();
        let mut indptr = vec![0usize; major_dim + 1];
        for (pos, &(m, n, _)) in items.iter().enumerate() {
            if m >= major_dim || n >= minor_dim {
                let (bad, bound) = if m >= major_dim {
                    (m, major_dim)
                } else {
                    (n, minor_dim)
                };
                return Err(SparseError::IndexOutOfRange {
                    pos,
                    index: u32::try_from(bad).unwrap_or(u32::MAX),
                    bound,
                });
            }
            indptr[m + 1] += 1;
        }
        for i in 0..major_dim {
            indptr[i + 1] += indptr[i];
        }
        let mut next = indptr.clone();
        let mut indices = vec![0u32; items.len()];
        let mut data = vec![T::ZERO; items.len()];
        for &(m, n, v) in &items {
            let p = next[m];
            next[m] += 1;
            indices[p] = n as u32;
            data[p] = v;
        }
        Ok((indptr, indices, data))
    }
}

#[cfg(feature = "alloc")]
pub use owned::{CscMatrix, CsrMatrix};
