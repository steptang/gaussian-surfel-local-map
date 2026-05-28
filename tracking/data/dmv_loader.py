"""Lazy reader for the Deep 3D Mask Volume (ken2576/multiview_preprocessing) format.

Per-scene there's an h5 file and a poses_bounds.npy. The h5 holds three
datasets (verbatim from compress_data.py in the producer repo):

    rgb     : (n_cams, T, H, W, 3) uint8   -- full multi-view video frames
    fg_rgb  : (n_cams, T, H, W, 4) uint8   -- RGBA; alpha = ground-truth
                                              foreground mask
    bg_rgb  : (n_cams, H, W, 3)    uint8   -- per-camera static background
                                              (median across time)

The npy is the LLFF poses_bounds (n_cams, 17); see llff_conversion.py
for the exact axis-flip conversion to the convention 2DGS expects.

This loader is *lazy*: the h5 stays open via a context manager and
frames are pulled on demand per (timestep, camera) so we never blow up
memory loading the full (n_cams * T * H * W) tensor.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from .llff_conversion import llff_to_2dgs


# Verbatim from compress_data.py in ken2576/multiview_preprocessing.
H5_KEY_RGB = "rgb"
H5_KEY_FG = "fg_rgb"
H5_KEY_BG = "bg_rgb"


@dataclass(frozen=True)
class DMVMeta:
    """Static (timestep-independent) metadata for a DMV scene."""
    h5_path: str
    poses_path: str
    n_cams: int
    n_timesteps: int
    height: int
    width: int
    # Shared (across all timesteps) camera intrinsics + extrinsics, already
    # converted to the 2DGS Blender-reader convention.
    R: np.ndarray             # (n_cams, 3, 3) glm-transposed w2c rotation
    T: np.ndarray             # (n_cams, 3)
    FovX: np.ndarray          # (n_cams,) radians
    FovY: np.ndarray          # (n_cams,) radians
    bounds: np.ndarray        # (n_cams, 2) LLFF (near, far)


class DMVScene:
    """Opens the h5 once; serves per-(timestep, camera) reads on demand.

    Use as a context manager:

        with DMVScene("scene.h5", "poses_bounds.npy") as scene:
            for t in scene.timesteps():
                view = scene.read_timestep(t)
                ...
    """

    def __init__(self, h5_path: str, poses_path: str):
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"h5 not found: {h5_path}")
        if not os.path.exists(poses_path):
            raise FileNotFoundError(f"poses_bounds.npy not found: {poses_path}")
        self._h5_path = h5_path
        self._poses_path = poses_path
        self._h5 = None       # opened in __enter__
        self._meta: Optional[DMVMeta] = None

    def __enter__(self):
        import h5py
        self._h5 = h5py.File(self._h5_path, "r")
        self._meta = self._build_meta()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    # -- introspection ----------------------------------------------------

    @property
    def meta(self) -> DMVMeta:
        if self._meta is None:
            raise RuntimeError("DMVScene must be used as a context manager")
        return self._meta

    def timesteps(self) -> range:
        """All available timestep indices (0..T-1)."""
        return range(self.meta.n_timesteps)

    def cameras(self) -> range:
        return range(self.meta.n_cams)

    # -- per-timestep reads -----------------------------------------------

    def read_timestep(self, t: int) -> dict:
        """Pull (n_cams, H, W, 3) RGB and (n_cams, H, W) fg mask for timestep t.

        Returns dict with keys:
            rgb:     (n_cams, H, W, 3) uint8
            fg_mask: (n_cams, H, W)    bool   (alpha > 0)
            fg_rgb:  (n_cams, H, W, 3) uint8  (foreground-only RGB; alpha-composited)
        """
        if self._h5 is None:
            raise RuntimeError("DMVScene must be used as a context manager")
        if not (0 <= t < self.meta.n_timesteps):
            raise IndexError(f"timestep {t} out of range [0, {self.meta.n_timesteps})")

        rgb = self._h5[H5_KEY_RGB][:, t]                # (n_cams, H, W, 3) uint8
        fg_rgba = self._h5[H5_KEY_FG][:, t]              # (n_cams, H, W, 4) uint8
        fg_rgb = fg_rgba[..., :3].copy()
        fg_alpha = fg_rgba[..., 3]
        # Treat any non-zero alpha as "this pixel is foreground." The
        # producer pipeline uses 0/255 binary alpha in practice, but the
        # > 0 rule is robust to soft alpha if a downstream tool changes it.
        fg_mask = fg_alpha > 0

        return {
            "rgb": np.asarray(rgb),
            "fg_rgb": fg_rgb,
            "fg_mask": fg_mask,
        }

    def read_background(self) -> np.ndarray:
        """(n_cams, H, W, 3) uint8 -- per-camera static-background plate."""
        if self._h5 is None:
            raise RuntimeError("DMVScene must be used as a context manager")
        return np.asarray(self._h5[H5_KEY_BG][:])

    # -- internal ----------------------------------------------------------

    def _build_meta(self) -> DMVMeta:
        rgb_ds = self._h5[H5_KEY_RGB]
        if rgb_ds.ndim != 5 or rgb_ds.shape[-1] != 3:
            raise ValueError(
                f"unexpected '{H5_KEY_RGB}' shape {rgb_ds.shape}; "
                "expected (n_cams, T, H, W, 3)"
            )
        n_cams, n_timesteps, H, W, _ = rgb_ds.shape

        # Spot-check fg_rgb and bg_rgb shapes for early-failure clarity.
        fg_ds = self._h5[H5_KEY_FG]
        if fg_ds.shape[:2] != (n_cams, n_timesteps) or fg_ds.shape[-1] != 4:
            raise ValueError(
                f"unexpected '{H5_KEY_FG}' shape {fg_ds.shape}; "
                f"expected ({n_cams}, {n_timesteps}, H, W, 4)"
            )
        bg_ds = self._h5[H5_KEY_BG]
        if bg_ds.shape != (n_cams, H, W, 3):
            raise ValueError(
                f"unexpected '{H5_KEY_BG}' shape {bg_ds.shape}; "
                f"expected ({n_cams}, H, W, 3)"
            )

        poses_raw = np.load(self._poses_path)
        if poses_raw.shape != (n_cams, 17):
            raise ValueError(
                f"poses_bounds.npy shape {poses_raw.shape} disagrees with "
                f"h5 n_cams={n_cams}; expected ({n_cams}, 17)"
            )

        conv = llff_to_2dgs(poses_raw)
        # Sanity: LLFF hwf may differ from the resized frames in the h5.
        # We trust the h5 for (H, W) at frame-load time and use the LLFF
        # focal as-is for the FoV (the focal-vs-resolution rescaling is
        # the *user's* responsibility upstream; ken2576 stores resized
        # frames with the matching focal).
        return DMVMeta(
            h5_path=self._h5_path,
            poses_path=self._poses_path,
            n_cams=int(n_cams),
            n_timesteps=int(n_timesteps),
            height=int(H),
            width=int(W),
            R=conv["R"],
            T=conv["T"],
            FovX=conv["FovX"],
            FovY=conv["FovY"],
            bounds=conv["bounds"],
        )


def select_timesteps(
    available: range,
    explicit: Optional[list[int]] = None,
    stride: Optional[int] = None,
    count: Optional[int] = None,
) -> list[int]:
    """Resolve the user's timestep-selection knobs into a concrete list.

    Precedence: `explicit` > `stride` > `count` > all.
    """
    available_list = list(available)
    if explicit is not None:
        bad = [t for t in explicit if t not in available]
        if bad:
            raise ValueError(f"timesteps not in scene: {bad} (have {available})")
        return list(explicit)
    if stride is not None:
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        return available_list[::stride]
    if count is not None:
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        return available_list[:count]
    return available_list
