"""Stage D: classify each cluster as static or dynamic.

Two paths, exposed as separate callables so the orchestrator can mix
and match per scene:

* ``per_surfel_fg_score_from_views`` -- the **primary** path. Projects
  each surfel centre into every training view, looks up the GT fg/bg
  mask the DMV dataset provides at the projected pixel, and averages
  the result. Output: (N,) float in [0, 1] giving the fraction of
  visible views where the surfel landed inside the foreground.

  The brief calls this "use the dataset's provided fg/bg mask"; this
  is the GT-driven branch.

* ``per_surfel_fg_score_from_semantic`` -- the **fallback** for scenes
  without GT masks. Picks "dynamic-prior" classes (person, car, ...)
  from the scene's region-embedding vocabulary using SigLIP2 text
  similarity. Implemented as a stub here; the brief says only "modular
  so motion-based classification can be added later".

Both routes feed into ``classify_clusters`` which reduces per-surfel
scores to per-cluster static/dynamic labels via a configurable
threshold.

Center-only projection (no depth test) is intentionally simple; the
DMV dataset's ~10-view multi-view setup means a single occluded
projection is averaged out across the other ~9 views. A surfel behind
the moving object in one view will be labeled bg by most of the other
views and end up with a low fg_score (correctly background).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .sequence import SurfelSnapshot


@dataclass
class FgProjectionConfig:
    """Knobs for per_surfel_fg_score_from_views."""
    # If the surfel's clip-w (camera-space z) is <=0 it's behind the camera
    # and we don't count that view as "visible" for the surfel.
    require_in_view: bool = True
    # Minimum number of views a surfel must be in-bounds in before its
    # score is considered reliable. Surfels seen in fewer views get the
    # ``insufficient_views_score`` so they don't dominate cluster majority
    # votes with one or two flukes.
    min_views_visible: int = 2
    insufficient_views_score: float = 0.0


def _project_to_pixels(xyz: np.ndarray, full_proj_transform: np.ndarray,
                       image_width: int, image_height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """World-points -> (px, py, cam_z) for one camera.

    full_proj_transform is the 2DGS Camera.full_proj_transform (4x4,
    row-vector right-multiply convention). cam_z (== clip-w because the
    project matrix has P[3,2]=1, P[3,3]=0; same as mesh_utils does) is
    the surfel's camera-space depth.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3), got {xyz.shape}")
    N = xyz.shape[0]
    homog = np.concatenate([xyz, np.ones((N, 1), dtype=xyz.dtype)], axis=1)  # (N, 4)
    clip = homog @ full_proj_transform                                        # (N, 4)
    cam_z = clip[:, 3]
    safe_w = np.where(cam_z != 0, cam_z, 1e-8)
    ndc_xy = clip[:, :2] / safe_w[:, None]
    px = (ndc_xy[:, 0] + 1.0) * 0.5 * image_width
    py = (ndc_xy[:, 1] + 1.0) * 0.5 * image_height
    return px, py, cam_z


@dataclass(frozen=True)
class ProjectionView:
    """A single view's data for fg-mask projection.

    Decoupled from the 2DGS Camera class so this module is importable
    without CUDA / the rest of the repo's training infra (and so the
    synthetic test can construct views with plain numpy).
    """
    full_proj_transform: np.ndarray   # (4, 4) row-vector convention
    image_width: int
    image_height: int
    fg_mask: np.ndarray               # (H, W) bool


def per_surfel_fg_score_from_views(
    snap: SurfelSnapshot,
    views: list[ProjectionView],
    cfg: Optional[FgProjectionConfig] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project each surfel into every view; majority-vote fg vs bg.

    Returns:
        fg_score:     (N,) float32 in [0, 1] -- frac of visible views
                      where the surfel projected into the fg mask.
        n_visible:    (N,) int32          -- how many views saw it in-bounds.
    """
    cfg = cfg or FgProjectionConfig()
    N = snap.n_surfels
    fg_hits = np.zeros(N, dtype=np.float32)
    n_visible = np.zeros(N, dtype=np.int32)

    for v in views:
        H, W = v.image_height, v.image_width
        if v.fg_mask.shape != (H, W):
            raise ValueError(
                f"fg_mask shape {v.fg_mask.shape} disagrees with "
                f"(image_height={H}, image_width={W})"
            )
        px, py, cam_z = _project_to_pixels(snap.xyz, v.full_proj_transform, W, H)
        in_bounds = (cam_z > 0) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        if not in_bounds.any():
            continue
        ix = np.clip(px.astype(np.int64), 0, W - 1)
        iy = np.clip(py.astype(np.int64), 0, H - 1)
        # Sample the fg mask at each surfel's projection; only the
        # in-bounds entries count.
        sample = v.fg_mask[iy, ix]
        fg_hits[in_bounds] += sample[in_bounds].astype(np.float32)
        n_visible[in_bounds] += 1

    fg_score = np.where(
        n_visible >= cfg.min_views_visible,
        fg_hits / np.maximum(n_visible, 1).astype(np.float32),
        np.float32(cfg.insufficient_views_score),
    )
    return fg_score, n_visible


def per_surfel_fg_score_from_semantic(
    snap: SurfelSnapshot,
    dynamic_prototype_embedding: np.ndarray,    # (K_target,) unit vector
    threshold: float = 0.6,
) -> np.ndarray:
    """Fallback when GT fg masks aren't available.

    Score each surfel by cosine similarity between its semantic feature
    and a pre-computed "dynamic" prototype embedding (e.g., averaged
    SigLIP2 text embeddings of {person, car, bicycle, ...}). Returns
    fg_score in [0, 1] (clipped to >= 0). The caller is responsible for
    producing the prototype upstream; this module just performs the dot
    product.
    """
    if snap.semantic.size == 0:
        return np.zeros(snap.n_surfels, dtype=np.float32)
    sem = snap.semantic.astype(np.float32)
    sem_n = sem / np.maximum(np.linalg.norm(sem, axis=1, keepdims=True), 1e-12)
    proto = dynamic_prototype_embedding.astype(np.float32)
    proto = proto / max(float(np.linalg.norm(proto)), 1e-12)
    sims = sem_n @ proto                          # (N,)
    # Above-threshold fg_score in [0, 1]; below-threshold -> 0. Linear
    # ramp is fine for the threshold-based reducer downstream.
    sims_clipped = np.clip(sims, 0.0, 1.0)
    return np.where(sims_clipped >= threshold, sims_clipped, 0.0).astype(np.float32)


def classify_clusters(
    object_ids: np.ndarray,
    fg_score: np.ndarray,
    cluster_fg_threshold: float = 0.5,
) -> dict[int, str]:
    """Reduce per-surfel fg_score to per-cluster {"dynamic", "static"}.

    A cluster is "dynamic" if the mean fg_score of its member surfels
    exceeds ``cluster_fg_threshold``. Noise (id == -1) is skipped.
    """
    if object_ids.shape[0] != fg_score.shape[0]:
        raise ValueError("object_ids and fg_score must have the same length")
    out: dict[int, str] = {}
    unique_ids = np.unique(object_ids)
    for oid in unique_ids:
        oid_int = int(oid)
        if oid_int == -1:
            continue
        mean_fg = float(fg_score[object_ids == oid_int].mean())
        out[oid_int] = "dynamic" if mean_fg >= cluster_fg_threshold else "static"
    return out
