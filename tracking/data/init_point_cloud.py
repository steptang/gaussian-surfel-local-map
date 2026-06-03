"""Seed a Stage A scene with a ``points3d.ply`` near the cameras'
convergence point so the Blender reader picks it up as the initial
point cloud, instead of falling back to the synthetic-Blender default
of uniform random points in ``[-1.3, 1.3]^3``.

Why this exists
---------------
``scene/dataset_readers.readNerfSyntheticInfo`` does, for Blender-style
scenes:

    ply_path = <scene>/points3d.ply
    if not os.path.exists(ply_path):
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3   # synthetic-Blender default
        ...
        storePly(ply_path, xyz, ...)

That default is appropriate for Blender synthetic scenes that *are*
centred at the origin. For the DMV (Deep 3D Mask Volume) dynamics scenes
this Stage A is built to consume, the cameras' rays converge somewhere
like ``z ≈ -62`` -- 60+ units from the random-init cube. The position
learning rate cannot bridge that gap in the usual 7-30k iterations, so
training is bimodal: it sometimes finds the scene region and converges
nicely, and sometimes never reaches it and settles for "render mostly-
black", with the outcome determined by cuDNN nondeterminism in early
iterations. Investigated extensively in the Jun 2026 debugging
session; see git log around commit 222fad6.

This module fixes the root cause by writing a ``points3d.ply``
populated near the cameras' actual look-at point, so the Blender
reader uses *those* points as the init instead of synthesising random
ones at the wrong location. No changes to training, the rasterizer,
the reader, or the rest of the pipeline.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np


def compute_camera_convergence(
    c2w_matrices: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Least-squares intersection of cameras' forward rays.

    For each camera at world position ``P_i`` looking in unit direction
    ``F_i`` (= OpenCV ``c2w[:3, 2]``), the perpendicular distance from a
    point ``X`` to the ray is
    ``|(X - P_i) - ((X - P_i) · F_i) F_i|``. Minimising the sum-of-squares
    over all cameras yields

        sum_i (I - F_i F_i^T) X = sum_i (I - F_i F_i^T) P_i

    which is a 3x3 linear system in X. The returned RMSE is the root-
    mean-squared perpendicular distance at the solution -- a measure of
    how tightly the rays actually converge (small = sharp focus point,
    large = nearly-parallel cameras with a poorly-defined "look-at").

    Args:
        c2w_matrices: (N, 4, 4) OpenCV camera-to-world matrices.

    Returns:
        (convergence_point, rmse) -- (3,) float64 and a scalar.
    """
    if c2w_matrices.ndim != 3 or c2w_matrices.shape[1:] != (4, 4):
        raise ValueError(f"expected (N, 4, 4) c2w; got {c2w_matrices.shape}")
    positions = c2w_matrices[:, :3, 3]      # (N, 3)
    forwards = c2w_matrices[:, :3, 2]       # (N, 3)
    # Normalise forwards (defensive: c2w columns should already be unit).
    norms = np.linalg.norm(forwards, axis=1, keepdims=True)
    forwards = forwards / np.maximum(norms, 1e-12)

    # Vectorised: M_i = I - F_i F_i^T, then A = sum_i M_i, b = sum_i M_i P_i.
    FFt = np.einsum("ni,nj->nij", forwards, forwards)        # (N, 3, 3)
    M = np.eye(3) - FFt                                       # (N, 3, 3)
    A = M.sum(axis=0)                                         # (3, 3)
    b = (M @ positions[:, :, None]).squeeze(-1).sum(axis=0)   # (3,)
    X, *_ = np.linalg.lstsq(A, b, rcond=None)                 # (3,)

    # RMSE of perpendicular distances at the solution.
    delta = X[None, :] - positions                            # (N, 3)
    along = (delta * forwards).sum(axis=1, keepdims=True) * forwards
    perp = delta - along
    rmse = float(np.sqrt(np.mean((perp ** 2).sum(axis=1))))
    return X.astype(np.float64), rmse


def _max_camera_baseline(positions: np.ndarray) -> float:
    """Max pairwise distance between camera centres.

    Used as a spread floor: the init cloud must be at least as wide as
    the camera rig so every camera can see *some* of it through its
    frustum.
    """
    # (N, N, 3) diffs; bottleneck is the outer product but N is small.
    diffs = positions[:, None, :] - positions[None, :, :]
    return float(np.linalg.norm(diffs, axis=-1).max())


def init_point_cloud_in_frustums(
    c2w_matrices: np.ndarray,
    fov_x: np.ndarray,
    fov_y: np.ndarray,
    bounds: np.ndarray,
    n_pts: int = 100_000,
    depth_distribution: str = "uniform",
    near_floor: float = 1e-2,
    far_ceiling: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Sample init points within the union of camera frustums.

    Per point:
        1. Pick a camera index uniformly at random.
        2. Sample a depth ``d`` in ``[near_i, far_i]`` per
           ``depth_distribution``:
             "uniform" -- uniform in depth (default; equivalent to
                          vanilla 3DGS random init when restricted to
                          the frustum).
             "log"     -- log-uniform in depth; biases toward near
                          depths where image resolution is highest.
        3. Sample u, v world-space lateral offsets uniformly in
           ``[-d * tan(fov / 2), +d * tan(fov / 2)]``. This is
           mathematically equivalent to drawing image-plane coords
           ``img_uv`` uniformly in ``[-1, 1]`` and projecting at depth
           ``d`` -- the two formulations are interchangeable, so
           per-pixel density is flat across the image (and per-world-
           volume density goes as ``1/d^2`` under the default
           depth distribution).
        4. World position = cam_pos + d * forward + u * right + v * down.

    Every emitted point sits inside at least one camera's frustum by
    construction, so the optimizer has projection signal on every
    Gaussian. Compared to fixed cubes around the world origin
    (``scene/dataset_readers.py:279`` random fallback) or the LSQ
    ray-meeting point (ill-conditioned for nearly-parallel rigs),
    this scales correctly to any rig shape -- the ``poses_bounds.npy``
    near/far values encode the data producer's prior on where content
    lives, and we trust them.

    ``near_floor`` guards against ``bounds[:, 0] <= 0`` (some LLFF
    files store unreasonable near values); ``far_ceiling`` caps wildly
    large far values that would otherwise spray points at irrelevant
    depths.

    Args:
        c2w_matrices: (N, 4, 4) OpenCV c2w; columns 0/1/2 are
            right/down/forward in world coords.
        fov_x, fov_y: (N,) per-camera horizontal/vertical FoV in radians.
        bounds: (N, 2) per-camera (near, far) depths -- the last two
            columns of an LLFF ``poses_bounds.npy`` row.
        n_pts: how many points to sample.
        depth_distribution: "uniform" or "log"; see point 2 above.

    Returns:
        xyz:  (n_pts, 3) float64 world positions.
        rgb:  (n_pts, 3) float32 colours in [0, 1].
        meta: diagnostic dict with ``near_used``, ``far_used``,
              ``per_camera_count``, and the per-axis bbox of the
              resulting cloud -- handed back to the writer so logs
              make the actual init region visible at Stage A time.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if c2w_matrices.ndim != 3 or c2w_matrices.shape[1:] != (4, 4):
        raise ValueError(f"expected (N, 4, 4) c2w; got {c2w_matrices.shape}")
    N = c2w_matrices.shape[0]
    fov_x = np.broadcast_to(np.asarray(fov_x, dtype=np.float64), (N,))
    fov_y = np.broadcast_to(np.asarray(fov_y, dtype=np.float64), (N,))
    if bounds.shape != (N, 2):
        raise ValueError(f"expected bounds shape ({N}, 2); got {bounds.shape}")

    # Sanitise near/far -- positive near, finite far.
    near_used = np.maximum(bounds[:, 0].astype(np.float64), near_floor)
    far_used = bounds[:, 1].astype(np.float64)
    if far_ceiling is not None:
        far_used = np.minimum(far_used, far_ceiling)
    # If somehow near >= far for a camera, pin a tiny range so we don't crash.
    bad = near_used >= far_used
    if bad.any():
        far_used = np.where(bad, near_used + 1e-3, far_used)

    cam_idx = rng.integers(0, N, size=n_pts)
    near_per_pt = near_used[cam_idx]
    far_per_pt = far_used[cam_idx]
    if depth_distribution == "uniform":
        depths = rng.uniform(near_per_pt, far_per_pt)
    elif depth_distribution == "log":
        u_log = rng.uniform(np.log(near_per_pt), np.log(far_per_pt))
        depths = np.exp(u_log)
    else:
        raise ValueError(
            f"unknown depth_distribution={depth_distribution!r}; "
            "expected 'uniform' or 'log'"
        )

    half_w = np.tan(fov_x[cam_idx] / 2.0) * depths
    half_h = np.tan(fov_y[cam_idx] / 2.0) * depths
    u = rng.uniform(-half_w, half_w)
    v = rng.uniform(-half_h, half_h)

    # OpenCV c2w: cols 0 = right, 1 = down, 2 = forward.
    cam_pos = c2w_matrices[cam_idx, :3, 3]      # (n_pts, 3)
    right = c2w_matrices[cam_idx, :3, 0]
    down = c2w_matrices[cam_idx, :3, 1]
    forward = c2w_matrices[cam_idx, :3, 2]

    xyz = (cam_pos
           + depths[:, None] * forward
           + u[:, None] * right
           + v[:, None] * down).astype(np.float64)
    rgb = rng.random(size=(n_pts, 3)).astype(np.float32)

    per_camera_count = np.bincount(cam_idx, minlength=N)
    return xyz, rgb, {
        "near_used": near_used,
        "far_used": far_used,
        "per_camera_count": per_camera_count,
        "xyz_min": xyz.min(axis=0),
        "xyz_max": xyz.max(axis=0),
        "xyz_median": np.median(xyz, axis=0),
    }


def write_points3d_ply(path: str, xyz: np.ndarray, rgb_u8: np.ndarray) -> None:
    """Write a ``points3d.ply`` compatible with ``scene/dataset_readers.fetchPly``.

    Format matches ``scene.dataset_readers.storePly`` exactly: vertex
    element with ``x, y, z, nx, ny, nz, red, green, blue`` where
    normals are all zero and RGB is uint8 in ``[0, 255]``. The Blender
    reader path picks the file up unchanged.

    Kept here (rather than importing storePly) to avoid a circular
    import: scene/dataset_readers.py imports nothing from tracking/,
    and tracking/data/init_point_cloud.py should keep that direction.
    """
    from plyfile import PlyData, PlyElement
    n = xyz.shape[0]
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    elements = np.empty(n, dtype=dtype)
    elements["x"] = xyz[:, 0].astype(np.float32)
    elements["y"] = xyz[:, 1].astype(np.float32)
    elements["z"] = xyz[:, 2].astype(np.float32)
    elements["nx"] = 0
    elements["ny"] = 0
    elements["nz"] = 0
    elements["red"] = rgb_u8[:, 0].astype(np.uint8)
    elements["green"] = rgb_u8[:, 1].astype(np.uint8)
    elements["blue"] = rgb_u8[:, 2].astype(np.uint8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
