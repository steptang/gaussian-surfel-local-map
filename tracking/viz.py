"""Visualisation helpers for the tracking pipeline.

These write color-coded PLYs so the per-stage output can be inspected
in any meshlab-style viewer (the existing 2DGS rendering pipeline
already lives elsewhere; this module is for offline inspection
artifacts):

* ``write_object_id_ply`` -- color surfels by their Stage C object id.
* ``write_static_dynamic_ply`` -- bicolour by Stage D fg/bg label.
* ``write_transformed_overlay_ply`` -- overlays src_at_t transformed
  by the Stage F pose onto tgt_at_t+1, with src in red and tgt in
  green; quick visual confirmation that ICP recovered a sensible
  alignment.

The PLY format mirrors what scene.dataset_readers.storePly already
uses -- vertex elements with x, y, z, nx, ny, nz, red, green, blue --
so the output is loadable by the same tooling that already exists in
the repo.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
# plyfile is imported lazily inside _write_ply so the synthetic test
# / dataclass-only flows don't require it.

from .pose_icp import apply_se3
from .sequence import SurfelSnapshot


_PLY_DTYPE = [
    ("x", "f4"), ("y", "f4"), ("z", "f4"),
    ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
]


def _write_ply(path: str, xyz: np.ndarray, rgb_u8: np.ndarray,
                normals: Optional[np.ndarray] = None) -> None:
    """Write a colored PLY mesh-style point cloud."""
    from plyfile import PlyData, PlyElement    # lazy: see module top
    n = xyz.shape[0]
    if normals is None:
        normals = np.zeros_like(xyz)
    elements = np.empty(n, dtype=_PLY_DTYPE)
    elements["x"] = xyz[:, 0].astype(np.float32)
    elements["y"] = xyz[:, 1].astype(np.float32)
    elements["z"] = xyz[:, 2].astype(np.float32)
    elements["nx"] = normals[:, 0].astype(np.float32)
    elements["ny"] = normals[:, 1].astype(np.float32)
    elements["nz"] = normals[:, 2].astype(np.float32)
    elements["red"] = rgb_u8[:, 0].astype(np.uint8)
    elements["green"] = rgb_u8[:, 1].astype(np.uint8)
    elements["blue"] = rgb_u8[:, 2].astype(np.uint8)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def _palette(n: int, seed: int = 0) -> np.ndarray:
    """N visually distinct uint8 RGB colors (deterministic by seed)."""
    rng = np.random.default_rng(seed)
    hsv = np.zeros((n, 3), dtype=np.float32)
    hsv[:, 0] = (np.arange(n) / max(n, 1)) % 1.0      # spread hue evenly
    hsv[:, 1] = 0.7 + 0.3 * rng.random(n)              # saturated-ish
    hsv[:, 2] = 0.85                                    # bright but not blown
    # HSV -> RGB
    import colorsys
    rgb = np.array([colorsys.hsv_to_rgb(*h) for h in hsv])
    return (rgb * 255).astype(np.uint8)


def write_object_id_ply(path: str, snap: SurfelSnapshot, object_ids: np.ndarray,
                         noise_color: tuple[int, int, int] = (40, 40, 40)) -> None:
    """Color-by-cluster PLY: noise (-1) gets ``noise_color``."""
    unique_ids = np.unique(object_ids)
    fg_ids = [int(i) for i in unique_ids if int(i) != -1]
    palette = _palette(len(fg_ids), seed=0)
    id_to_color: dict[int, np.ndarray] = {oid: palette[k] for k, oid in enumerate(fg_ids)}

    rgb = np.tile(np.array(noise_color, dtype=np.uint8), (snap.n_surfels, 1))
    for oid, color in id_to_color.items():
        rgb[object_ids == oid] = color
    _write_ply(path, snap.xyz, rgb)


def write_static_dynamic_ply(path: str, snap: SurfelSnapshot,
                              fg_score: np.ndarray,
                              threshold: float = 0.5) -> None:
    """Bicolour PLY: green if fg_score >= threshold, grey otherwise."""
    rgb = np.zeros((snap.n_surfels, 3), dtype=np.uint8)
    dyn = fg_score >= threshold
    rgb[dyn] = (30, 200, 60)        # green = dynamic
    rgb[~dyn] = (130, 130, 140)     # grey = static
    _write_ply(path, snap.xyz, rgb)


def write_transformed_overlay_ply(
    path: str,
    src_snap: SurfelSnapshot, src_indices: np.ndarray,
    tgt_snap: SurfelSnapshot, tgt_indices: np.ndarray,
    T: np.ndarray,
    src_color: tuple[int, int, int] = (220, 50, 50),
    tgt_color: tuple[int, int, int] = (50, 200, 50),
) -> None:
    """Stack src@t (transformed by T) and tgt@t+1 into one colored PLY.

    Eyeball check: if Stage F recovered the correct rigid pose, the red
    and green clouds should overlap completely.
    """
    src_xyz = apply_se3(T, src_snap.xyz[src_indices])
    tgt_xyz = tgt_snap.xyz[tgt_indices].astype(np.float32)
    xyz = np.concatenate([src_xyz, tgt_xyz], axis=0)
    rgb = np.zeros((xyz.shape[0], 3), dtype=np.uint8)
    rgb[: src_xyz.shape[0]] = src_color
    rgb[src_xyz.shape[0]:] = tgt_color
    _write_ply(path, xyz, rgb)
