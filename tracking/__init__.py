"""Per-object rigid-pose tracking on top of 2DGS surfel reconstructions.

This package implements Strategy 1: per-timestep surfel reconstructions
are run independently (Stage B), then post-hoc grouped into objects
(Stage C), classified static/dynamic (Stage D), associated across time
(Stage E), aligned by rigid ICP (Stage F), and assembled into per-object
SE(3) trajectories (Stage G).

The deformation field for intra-object articulation (Strategy 4) is
explicitly out of scope for this module. Reconstruction / rasterizer
code is never modified from this package.
"""
