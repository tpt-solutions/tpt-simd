//! Heap buffer with a guaranteed base alignment (feature `alloc`).

use alloc::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use core::fmt;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;

/// A fixed-length heap buffer of `Copy` elements whose first element is
/// aligned to `ALIGN` bytes (default 32, one AVX2 vector).
///
/// Elements are densely packed, so it holds exactly `len * size_of::<T>()`
/// bytes; contrast `Vec<Aligned32<f32>>`, which pads every element to 32
/// bytes. The effective alignment is `max(ALIGN, align_of::<T>())`.
///
/// Dereferences to `[T]`, so every slice API (and the
/// [`load_aligned`](crate::load_aligned) wrappers) works on it.
///
/// # Panics
/// Construction panics if `ALIGN` is not a power of two or the size overflows
/// `isize`; it aborts via [`handle_alloc_error`] if the allocator fails.
///
/// ```
/// use tpt_simd_aligned::{AlignedBuf, is_aligned};
/// let mut b = AlignedBuf::<f32, 64>::filled(100, 1.5);
/// assert!(is_aligned(b.as_ptr(), 64));
/// b[3] = 2.0;
/// assert_eq!(b.len(), 100);
/// assert_eq!(b.iter().sum::<f32>(), 99.0 * 1.5 + 2.0);
/// ```
pub struct AlignedBuf<T: Copy, const ALIGN: usize = 32> {
    ptr: NonNull<T>,
    len: usize,
}

// SAFETY: `AlignedBuf` uniquely owns its allocation, like `Box<[T]>`.
unsafe impl<T: Copy + Send, const ALIGN: usize> Send for AlignedBuf<T, ALIGN> {}
// SAFETY: shared access only hands out `&[T]`, like `Box<[T]>`.
unsafe impl<T: Copy + Sync, const ALIGN: usize> Sync for AlignedBuf<T, ALIGN> {}

impl<T: Copy, const ALIGN: usize> AlignedBuf<T, ALIGN> {
    /// Effective alignment of the buffer in bytes.
    pub const BASE_ALIGN: usize = if ALIGN > core::mem::align_of::<T>() {
        ALIGN
    } else {
        core::mem::align_of::<T>()
    };

    fn layout(len: usize) -> Layout {
        let size = len
            .checked_mul(core::mem::size_of::<T>())
            .expect("AlignedBuf size overflow");
        Layout::from_size_align(size, Self::BASE_ALIGN)
            .expect("AlignedBuf: ALIGN must be a power of two and size must fit isize")
    }

    /// Creates a buffer of `len` copies of `value`.
    ///
    /// # Panics
    /// See the type-level docs.
    pub fn filled(len: usize, value: T) -> Self {
        let layout = Self::layout(len);
        let ptr = if layout.size() == 0 {
            // Zero-sized allocation: a dangling, correctly aligned pointer.
            NonNull::new(core::ptr::without_provenance_mut::<T>(layout.align()))
                .expect("alignment is non-zero")
        } else {
            // SAFETY: `layout` has non-zero size.
            let raw = unsafe { alloc(layout) }.cast::<T>();
            let Some(p) = NonNull::new(raw) else {
                handle_alloc_error(layout)
            };
            // Initialise every element before the buffer can be observed.
            for i in 0..len {
                // SAFETY: `i < len`, so the write is inside the allocation,
                // which is aligned for `T` (BASE_ALIGN >= align_of::<T>()).
                unsafe { p.as_ptr().add(i).write(value) };
            }
            p
        };
        Self { ptr, len }
    }

    /// Creates a buffer holding a copy of `src`.
    ///
    /// ```
    /// use tpt_simd_aligned::AlignedBuf;
    /// let b = AlignedBuf::<i16, 32>::from_slice(&[1, 2, 3]);
    /// assert_eq!(&*b, &[1, 2, 3]);
    /// ```
    pub fn from_slice(src: &[T]) -> Self {
        let Some(&first) = src.first() else {
            return Self::filled_empty();
        };
        let mut b = Self::filled(src.len(), first);
        b.copy_from_slice(src);
        b
    }

    fn filled_empty() -> Self {
        let layout = Self::layout(0);
        Self {
            ptr: NonNull::new(core::ptr::without_provenance_mut::<T>(layout.align()))
                .expect("alignment is non-zero"),
            len: 0,
        }
    }

    /// Number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` if the buffer has no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T: Copy + Default, const ALIGN: usize> AlignedBuf<T, ALIGN> {
    /// Creates a buffer of `len` default (zero for numeric types) elements.
    ///
    /// ```
    /// use tpt_simd_aligned::AlignedBuf;
    /// let b = AlignedBuf::<u8, 32>::zeroed(5);
    /// assert_eq!(&*b, &[0u8; 5]);
    /// ```
    pub fn zeroed(len: usize) -> Self {
        Self::filled(len, T::default())
    }
}

impl<T: Copy, const ALIGN: usize> Drop for AlignedBuf<T, ALIGN> {
    fn drop(&mut self) {
        let layout = Self::layout(self.len);
        if layout.size() != 0 {
            // SAFETY: allocated in `filled` with this exact layout; `T: Copy`
            // so there are no element destructors to run.
            unsafe { dealloc(self.ptr.as_ptr().cast(), layout) };
        }
    }
}

impl<T: Copy, const ALIGN: usize> Deref for AlignedBuf<T, ALIGN> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        // SAFETY: `ptr` is valid, aligned and initialised for `len` elements
        // (dangling-but-aligned when `len == 0`, which is allowed).
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy, const ALIGN: usize> DerefMut for AlignedBuf<T, ALIGN> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in `deref`, and `&mut self` guarantees uniqueness.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy, const ALIGN: usize> Clone for AlignedBuf<T, ALIGN> {
    fn clone(&self) -> Self {
        Self::from_slice(self)
    }
}

impl<T: Copy + fmt::Debug, const ALIGN: usize> fmt::Debug for AlignedBuf<T, ALIGN> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: Copy + PartialEq, const ALIGN: usize> PartialEq for AlignedBuf<T, ALIGN> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

/// Zero-initialised `f32` buffer of `len` elements aligned to 32 bytes
/// (replacement for the spec's `Vec<Aligned32<f32>>`, which wastes 28 bytes
/// per element).
///
/// ```
/// use tpt_simd_aligned::{aligned_vec_f32, is_aligned};
/// let v = aligned_vec_f32(10);
/// assert!(is_aligned(v.as_ptr(), 32));
/// assert!(v.iter().all(|&x| x == 0.0));
/// ```
pub fn aligned_vec_f32(len: usize) -> AlignedBuf<f32, 32> {
    AlignedBuf::zeroed(len)
}

/// Zero-initialised `i16` buffer aligned to 32 bytes. See [`aligned_vec_f32`].
///
/// ```
/// use tpt_simd_aligned::aligned_vec_i16;
/// assert_eq!(aligned_vec_i16(3).len(), 3);
/// ```
pub fn aligned_vec_i16(len: usize) -> AlignedBuf<i16, 32> {
    AlignedBuf::zeroed(len)
}

/// Zero-initialised `i32` buffer aligned to 32 bytes. See [`aligned_vec_f32`].
///
/// ```
/// use tpt_simd_aligned::aligned_vec_i32;
/// assert_eq!(aligned_vec_i32(3).len(), 3);
/// ```
pub fn aligned_vec_i32(len: usize) -> AlignedBuf<i32, 32> {
    AlignedBuf::zeroed(len)
}
