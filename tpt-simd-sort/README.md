# tpt-simd-sort

Slice algorithms with no runtime dependencies:

- `sort_unstable`, `sort_by`: introsort with 3-way partitioning, insertion sort
  for small slices, and a heapsort fallback, so the worst case is O(n log n)
- `select_nth`: quickselect with the same guard, plus `top_k_smallest`
- `argsort`: indices that sort a key slice
- `partition`: in-place predicate partition

Sorting networks for fixed small blocks are not built yet. `no_std`,
MIT OR Apache-2.0.
