"""Stage C: cluster surfels into objects within one timestep.

Joint spatial + semantic clustering: each surfel becomes a feature
vector ``[w_spatial * (xyz / scene_radius), w_semantic * (semantic /
||semantic||)]`` and the resulting cloud is DBSCAN-clustered. The
spatial term gives two pedestrians distinct identities; the semantic
term keeps a single rigid object together across parts whose surface
appearance varies.

This module is purposely small and library-backed: we use
``sklearn.cluster.DBSCAN`` (KD-tree backend, scales well to ~10^5
surfels) rather than rolling our own. For very large surfel counts
(~10^6) callers can pass ``subsample`` to cluster on a random subset
and propagate labels to the rest via nearest-neighbour assignment;
that's a quality/speed tradeoff and is off by default.

The default weighting (w_spatial=1.0, w_semantic=0.3) emphasises
spatial -- the user's brief explicitly notes "two pedestrians have
similar features but are distinct objects", and spatial proximity is
the discriminating signal there. Crank w_semantic up when objects of
the same kind are adjacent and should still be separated by their
fine-grained appearance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .sequence import SurfelSnapshot


@dataclass
class ClusterConfig:
    """Knobs for cluster_surfels."""
    spatial_weight: float = 1.0
    semantic_weight: float = 0.3
    # DBSCAN reach in normalised-feature units. With the default weights,
    # 0.05 means surfels closer than 5% of scene radius (and consistent
    # semantically) get merged. Tune up for sparser clouds.
    eps: float = 0.05
    min_samples: int = 50
    # Optional scene_radius override; if None, derived from the snapshot's
    # bounding-box diagonal so eps stays scale-invariant.
    scene_radius: Optional[float] = None
    # If set, DBSCAN runs on a random subsample of this size and labels
    # are propagated to the rest via 1-NN. Useful for ~10^6-surfel scenes
    # where the full distance matrix is too expensive.
    subsample: Optional[int] = None
    random_state: int = 0


def _build_feature_vector(snap: SurfelSnapshot, cfg: ClusterConfig) -> tuple[np.ndarray, float]:
    """Construct the (N, 3 + K) joint feature for DBSCAN.

    Returns (features, scene_radius_used). Surfels with all-zero
    semantic vectors (e.g., scenes trained without --lambda_semantic)
    fall through to spatial-only clustering with the semantic block
    zeroed out -- DBSCAN ignores zero columns.
    """
    xyz = snap.xyz.astype(np.float64)
    if cfg.scene_radius is None:
        bbox = xyz.max(axis=0) - xyz.min(axis=0)
        scene_radius = float(np.linalg.norm(bbox) / 2.0) or 1.0
    else:
        scene_radius = float(cfg.scene_radius) or 1.0
    xyz_norm = xyz / scene_radius

    sem = snap.semantic.astype(np.float64)
    if sem.size == 0:
        sem_norm = np.zeros((xyz.shape[0], 0), dtype=np.float64)
    else:
        n = np.linalg.norm(sem, axis=1, keepdims=True)
        sem_norm = sem / np.maximum(n, 1e-12)

    feats = np.concatenate(
        [cfg.spatial_weight * xyz_norm, cfg.semantic_weight * sem_norm],
        axis=1,
    )
    return feats, scene_radius


def cluster_surfels(snap: SurfelSnapshot, cfg: Optional[ClusterConfig] = None) -> np.ndarray:
    """Cluster surfels into object IDs.

    Returns:
        (N,) int32 -- per-surfel object ID; -1 marks DBSCAN noise.
    """
    cfg = cfg or ClusterConfig()

    feats, _ = _build_feature_vector(snap, cfg)
    N = feats.shape[0]

    if cfg.subsample is not None and cfg.subsample < N:
        rng = np.random.default_rng(cfg.random_state)
        idx = rng.choice(N, size=cfg.subsample, replace=False)
        sub_feats = feats[idx]
        sub_labels = _run_dbscan(sub_feats, cfg)
        # Propagate to all surfels by nearest neighbour on the same feature
        # space (so spatial+semantic agreement is what assigns the label).
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(sub_feats)
        _, nn_idx = nn.kneighbors(feats, return_distance=True)
        labels = sub_labels[nn_idx[:, 0]].astype(np.int32)
        return labels

    return _run_dbscan(feats, cfg).astype(np.int32)


def _run_dbscan(feats: np.ndarray, cfg: ClusterConfig) -> np.ndarray:
    """Fit sklearn DBSCAN; return (N,) int labels (-1 == noise)."""
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=cfg.eps, min_samples=cfg.min_samples,
                metric="euclidean", n_jobs=-1).fit(feats)
    return db.labels_.astype(np.int32)


def cluster_stats(snap: SurfelSnapshot, object_ids: np.ndarray) -> dict[int, dict]:
    """Per-cluster summary stats: centroid, size, mean semantic, ids range.

    Built once and reused by Stages D, E, F -- avoids each downstream
    consumer recomputing per-cluster reductions on the same labels.
    """
    if object_ids.shape[0] != snap.n_surfels:
        raise ValueError("object_ids length must match snap.n_surfels")

    out: dict[int, dict] = {}
    unique_ids = np.unique(object_ids)
    for oid in unique_ids:
        oid_int = int(oid)
        if oid_int == -1:
            continue   # skip noise points; they don't form a cluster
        mask = object_ids == oid_int
        sel = snap.xyz[mask]
        centroid = sel.mean(axis=0).astype(np.float32)
        # Approximate spatial extent = max distance from centroid. Useful
        # for matching in Stage E without standing up full KD-trees.
        radius = float(np.linalg.norm(sel - centroid, axis=1).max())
        sem_mean = (
            snap.semantic[mask].mean(axis=0).astype(np.float32)
            if snap.semantic.size > 0 else np.zeros(0, dtype=np.float32)
        )
        out[oid_int] = {
            "indices": np.flatnonzero(mask).astype(np.int64),
            "n_surfels": int(mask.sum()),
            "centroid": centroid,
            "radius": radius,
            "semantic_mean": sem_mean,
        }
    return out
