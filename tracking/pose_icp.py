"""Stage F: per-object rigid-pose estimation via Open3D point-to-plane ICP.

The pipeline target is a single 4x4 SE(3) that aligns one object's
surfels at time t to its surfels at time t+1. Strategy 1's correctness
guarantee -- "rigid transforms can't skew the surface" -- is preserved
exactly because we estimate at the cluster level, not per-surfel.

Why point-to-plane: 2DGS surfels carry an explicit local normal (the
third column of the rotation-matrix-from-quaternion). Point-to-plane
ICP uses these to align "surface-tangent-to-surface-tangent" rather
than "point-to-point", which converges faster and tolerates the
sparser ~10-view reconstructions far better than vanilla
point-to-point. Centroid alignment seeds ICP so it doesn't fall into
a local minimum from a far-from-identity initial pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .sequence import SurfelSnapshot, surfel_normals


@dataclass
class IcpConfig:
    """Knobs for ``estimate_rigid_transform``."""
    # ICP correspondence distance threshold, in world units. Should be
    # comparable to expected per-frame displacement plus surfel-extent
    # noise. ``centroid-init + threshold ~ 0.2 * cluster_radius`` is a
    # safe default.
    threshold: Optional[float] = None
    max_iterations: int = 60
    # If True, attempt a centroid-only translation init before running
    # ICP. Helps when the per-frame translation is larger than the
    # cluster radius.
    centroid_init: bool = True
    # Whether to verify normals are available; if a snapshot's
    # rotation_quat is all zeros (untrained / synthetic without normals)
    # this lets us fall back to point-to-point.
    require_normals: bool = False


def _normals_for(snap: SurfelSnapshot, indices: np.ndarray) -> np.ndarray:
    n = surfel_normals(snap)
    return n[indices].astype(np.float64)


def _build_open3d_pcd(points: np.ndarray, normals: Optional[np.ndarray] = None):
    """Construct an Open3D PointCloud with optional precomputed normals."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points, dtype=np.float64))
    if normals is not None and normals.size > 0:
        pcd.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(normals, dtype=np.float64))
    return pcd


def estimate_rigid_transform(
    src_snap: SurfelSnapshot,
    src_indices: np.ndarray,
    tgt_snap: SurfelSnapshot,
    tgt_indices: np.ndarray,
    cfg: Optional[IcpConfig] = None,
) -> np.ndarray:
    """Run Open3D point-to-plane ICP on the given cluster subsets.

    Returns a 4x4 SE(3) matrix ``T`` such that ``T @ [src_x; 1] ~ tgt_x``
    for points x in the cluster.
    """
    cfg = cfg or IcpConfig()
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError(
            "tracking.pose_icp requires open3d. Install with `pip install open3d` "
            "(or whatever wheel matches your CUDA/Python; the CPU wheel is enough)."
        ) from e

    src_xyz = src_snap.xyz[src_indices].astype(np.float64)
    tgt_xyz = tgt_snap.xyz[tgt_indices].astype(np.float64)
    if src_xyz.shape[0] < 4 or tgt_xyz.shape[0] < 4:
        raise ValueError(
            f"ICP needs >=4 points per cluster; got src={src_xyz.shape[0]}, tgt={tgt_xyz.shape[0]}"
        )

    src_normals = _normals_for(src_snap, src_indices)
    tgt_normals = _normals_for(tgt_snap, tgt_indices)

    # The target needs normals for point-to-plane (Open3D evaluates the
    # cost using only tgt normals, but we precompute src normals for
    # downstream symmetry / debugging).
    method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    use_normals = True
    if cfg.require_normals:
        if np.allclose(tgt_normals, 0) or np.allclose(src_normals, 0):
            raise ValueError("require_normals=True but surfel normals are zero")
    else:
        # Soft fallback: if normals look degenerate, drop to point-to-point.
        if np.allclose(tgt_normals, 0) or np.allclose(src_normals, 0):
            method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
            use_normals = False

    src_pcd = _build_open3d_pcd(src_xyz, src_normals if use_normals else None)
    tgt_pcd = _build_open3d_pcd(tgt_xyz, tgt_normals if use_normals else None)

    init = np.eye(4, dtype=np.float64)
    if cfg.centroid_init:
        src_c = src_xyz.mean(axis=0)
        tgt_c = tgt_xyz.mean(axis=0)
        init[:3, 3] = tgt_c - src_c

    if cfg.threshold is None:
        # Heuristic: 20% of the source cluster radius, bounded below.
        src_radius = float(np.linalg.norm(src_xyz - src_xyz.mean(axis=0), axis=1).max())
        threshold = max(0.2 * src_radius, 1e-3)
    else:
        threshold = float(cfg.threshold)

    result = o3d.pipelines.registration.registration_icp(
        src_pcd, tgt_pcd, threshold, init, method,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg.max_iterations),
    )
    return np.asarray(result.transformation, dtype=np.float64)


def apply_se3(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Apply a 4x4 SE(3) to (N, 3) points. Returned dtype matches input."""
    homog = np.concatenate([xyz.astype(np.float64),
                            np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
    return ((T @ homog.T).T[:, :3]).astype(xyz.dtype)
