"""Stage E: associate dynamic clusters across consecutive timesteps.

For v1 the dataset is single-dynamic-object (the brief says so
explicitly). With one dynamic cluster per timestep there's nothing to
disambiguate -- the match is trivial. We still ship a generic API that
returns ``dict[curr_id -> prev_id]`` so multi-object data association
(Hungarian on a cost matrix, etc.) can drop in later without changes
upstream or downstream.

Two scoring signals are pre-computed by ``cluster_stats`` in
``tracking.cluster``:

* spatial: distance between cluster centroids (smaller is better)
* semantic: cosine similarity between mean semantic features (larger
  is better)

A composite cost combines both with configurable weights. The
single-object case ignores cost entirely; the multi-object stub builds
the cost matrix and solves with Hungarian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AssociationConfig:
    """Knobs for ``associate_clusters``."""
    # Cost = spatial_weight * normalised_distance + semantic_weight * (1 - cos_sim).
    # spatial term is normalised by the larger of (prev_radius, curr_radius)
    # so a tracked object's expected per-frame displacement is comparable
    # to its own scale.
    spatial_weight: float = 1.0
    semantic_weight: float = 0.5
    # Costs above this threshold are not eligible matches (return -1 from
    # ``associate_clusters``). Set to inf to always match the best pair.
    max_cost: float = float("inf")
    # When True and exactly one dynamic cluster appears on each side, skip
    # the cost matrix entirely.
    single_object_shortcut: bool = True


def _semantic_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    na = float(np.linalg.norm(a)) or 1.0
    nb = float(np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / (na * nb))


def _pairwise_cost(prev_stats: dict[int, dict], curr_stats: dict[int, dict],
                   cfg: AssociationConfig) -> tuple[np.ndarray, list[int], list[int]]:
    """Build (cost matrix, prev_ids_in_order, curr_ids_in_order).

    Cost rows are previous clusters; cost columns are current clusters.
    """
    prev_ids = sorted(prev_stats.keys())
    curr_ids = sorted(curr_stats.keys())
    cost = np.zeros((len(prev_ids), len(curr_ids)), dtype=np.float64)
    for i, p in enumerate(prev_ids):
        for j, c in enumerate(curr_ids):
            d = float(np.linalg.norm(prev_stats[p]["centroid"] - curr_stats[c]["centroid"]))
            # Normalise by the larger cluster's own extent so the spatial
            # cost is dimensionless and ~O(1) for sensible matches.
            scale = max(prev_stats[p]["radius"], curr_stats[c]["radius"], 1e-6)
            sp = d / scale
            sem = _semantic_cosine(prev_stats[p]["semantic_mean"],
                                   curr_stats[c]["semantic_mean"])
            cost[i, j] = cfg.spatial_weight * sp + cfg.semantic_weight * (1.0 - sem)
    return cost, prev_ids, curr_ids


def associate_clusters(
    prev_stats: dict[int, dict],
    curr_stats: dict[int, dict],
    cfg: Optional[AssociationConfig] = None,
) -> dict[int, int]:
    """Match cluster IDs across one transition.

    Returns ``dict[curr_id -> prev_id]``. Curr clusters with no
    acceptable match (cost > max_cost) map to ``-1``.

    For the single-object case (one cluster on each side), this is a
    no-op trivial match. For multi-object we use Hungarian assignment.
    """
    cfg = cfg or AssociationConfig()
    if not prev_stats or not curr_stats:
        return {c: -1 for c in curr_stats.keys()}

    if cfg.single_object_shortcut and len(prev_stats) == 1 and len(curr_stats) == 1:
        only_prev = next(iter(prev_stats.keys()))
        only_curr = next(iter(curr_stats.keys()))
        return {only_curr: only_prev}

    cost, prev_ids, curr_ids = _pairwise_cost(prev_stats, curr_stats, cfg)
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as e:
        raise ImportError(
            "tracking.associate.associate_clusters requires scipy for multi-object "
            "matching; install it or stick to single_object_shortcut for "
            "single-object scenes"
        ) from e

    # Hungarian. Pad the cost matrix so the rectangular case still works
    # (scipy supports rectangular matrices natively in modern versions but
    # we'd rather be explicit about which side has unmatched IDs).
    n_p, n_c = cost.shape
    big = float(max(cost.max() * 10.0, 1e6))
    if n_p == n_c:
        padded = cost
    elif n_p < n_c:
        padded = np.full((n_c, n_c), big)
        padded[:n_p, :] = cost
    else:
        padded = np.full((n_p, n_p), big)
        padded[:, :n_c] = cost
    row_ind, col_ind = linear_sum_assignment(padded)

    result: dict[int, int] = {}
    for r, c in zip(row_ind, col_ind):
        if r >= n_p or c >= n_c:
            # padded-region match -- this column had no real partner.
            if c < n_c:
                result[curr_ids[c]] = -1
            continue
        if cost[r, c] > cfg.max_cost:
            result[curr_ids[c]] = -1
            continue
        result[curr_ids[c]] = prev_ids[r]
    # Curr ids that didn't get touched (rare with padding) get -1.
    for c in curr_ids:
        result.setdefault(c, -1)
    return result
