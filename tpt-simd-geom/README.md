# tpt-simd-geom

Batched 3D geometry over structure-of-arrays (SoA) `f32` data:

- point distance and squared distance to a query point, dot product with a direction
- AABB overlap against a query box
- ray versus AABB slab test, returning entry distances

Inputs are borrowed slices, so nothing is allocated. `no_std`, MIT OR Apache-2.0.
Quaternions and 4x4 matrices are not included yet.
