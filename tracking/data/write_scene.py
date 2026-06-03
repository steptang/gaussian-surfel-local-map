"""Write a Blender-style 2DGS scene directory for one DMV timestep.

Layout produced (matches what scene/dataset_readers.py:readNerfSyntheticInfo
expects, with sam3/ added for downstream semantic preprocessing):

    <work_root>/timestep_{t:05d}/
        transforms_train.json   -- all DMV cameras as train frames
        transforms_test.json    -- empty (Blender reader requires the file)
        images/
            cam_00.png
            cam_01.png
            ...
        masks/
            cam_00.png          -- ground-truth fg/bg mask (uint8 0/255)
            cam_01.png
            ...
        (sam3/ is populated later by preprocess_semantic.py)

Two cautions baked into the code:

* The Blender reader at scene/dataset_readers.py:215-217 applies
  ``c2w[:3, 1:3] *= -1`` after loading, expecting NeRF/OpenGL convention
  (Y up, Z back). Our DMVScene gives us OpenCV/COLMAP c2w (Y down, Z
  forward), so we pre-flip to OpenGL before serialising. The reader's
  flip then cancels ours and the loaded camera is back in COLMAP.

* The Blender JSON stores a single global ``camera_angle_x`` (FovX).
  Real multi-view rigs have small per-camera FoV variation that's just
  calibration noise rather than genuine intrinsic differences (the DMV
  cameras show ~2% spread in focal length, far below what a per-camera
  intrinsics representation could meaningfully encode). We average the
  per-camera FoVX, emit a warning so the approximation is visible in
  logs, and only raise if the spread exceeds a configurable threshold
  -- which would indicate either a broken rig or a real heterogeneous
  setup. The fidelity upgrade if reconstruction ever proves
  intrinsics-limited would be a COLMAP-style writer (not yet
  implemented in ``tracking.data``), which supports per-camera
  intrinsics natively.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from PIL import Image

from .dmv_loader import DMVScene
from .init_point_cloud import (
    init_point_cloud_in_frustums,
    write_points3d_ply,
)


# Below this fractional spread we don't bother warning -- avoids noise
# for ideal/synthetic rigs whose per-camera FoVX is bit-identical.
_FOV_SPREAD_WARN_FLOOR_FRAC = 1e-6


@dataclass(frozen=True)
class WriteOptions:
    work_root: str
    image_ext: str = ".png"
    # If False, skip writing per-camera masks (saves disk if you don't
    # need Stage D's gt fg/bg).
    write_masks: bool = True
    # Optional resize cap. Set to None to keep the h5 resolution as-is.
    max_image_side: int | None = None
    # Reject the rig if per-camera FoVX disagreement exceeds this
    # fraction of the mean. The DMV cameras show ~2.1% (calibration
    # noise); 5% is the default cutoff for "this rig is probably broken
    # or genuinely heterogeneous and the Blender single-FoV layout
    # can't represent it." If you hit this on a real scene, the
    # principled fix is a COLMAP-style writer with per-camera intrinsics.
    max_fov_spread_frac: float = 0.05
    # Seed each per-timestep scene with a points3d.ply whose points are
    # sampled uniformly inside the union of the cameras' frustums
    # (per-camera near/far from the LLFF poses_bounds, per-camera FoV
    # from the JSON). Without this, the Blender reader at
    # scene/dataset_readers.py:279 falls back to uniform random init in
    # [-1.3, 1.3]^3 -- catastrophically wrong for DMV-style scenes
    # whose content sits far from the world origin. See
    # tracking.data.init_point_cloud and the Jun 2026 debugging session
    # for the bimodal-training failure this fixes.
    init_points3d_ply: bool = True
    init_n_pts: int = 100_000
    init_depth_distribution: str = "uniform"
    init_near_floor: float = 1e-2
    init_far_ceiling: Optional[float] = None
    init_seed: int = 0


def _opencv_c2w_to_opengl(c2w_opencv: np.ndarray) -> np.ndarray:
    """Invert the Blender reader's axis flip so the round-trip recovers c2w_opencv.

    The reader does ``c2w[:3, 1:3] *= -1`` after JSON load. To make that
    flip undo itself, we serialise the same operation here. Net effect:
    JSON-on-disk uses OpenGL convention; reader-in-memory uses OpenCV.
    """
    c2w_gl = c2w_opencv.copy()
    c2w_gl[..., :3, 1:3] *= -1.0
    return c2w_gl


def _resolve_global_fovx(FovX: np.ndarray, max_spread_frac: float) -> float:
    """Reduce per-camera FoVX to one global value for the Blender JSON.

    Calibration-noise-scale spread (e.g., DMV's ~2% from per-camera
    focal jitter) is treated as approximation and the mean is used,
    with a one-time warning so the substitution is visible in logs.

    Genuine heterogeneity above ``max_spread_frac`` (default 5%)
    raises -- that's beyond what mean-FoV-substitution can paper over
    and the Blender single-FoV JSON can't express it. The principled
    fix is a COLMAP-style writer with per-camera intrinsics; not yet
    implemented in ``tracking.data``.
    """
    mean_fov = float(FovX.mean())
    spread_rad = float(FovX.max() - FovX.min())
    # Guard against division by zero on a degenerate rig.
    spread_frac = spread_rad / mean_fov if mean_fov > 0 else 0.0

    if spread_frac > max_spread_frac:
        raise ValueError(
            f"per-camera FoVX spread is {spread_frac * 100:.2f}% of the mean "
            f"({spread_rad:.4f} rad over mean {mean_fov:.4f} rad), exceeding "
            f"max_fov_spread_frac={max_spread_frac * 100:.2f}%. The Blender "
            "JSON layout stores a single global camera_angle_x; this much "
            "intrinsic disagreement can't be papered over by averaging. "
            "Either tighten the rig calibration or use a COLMAP-style "
            "writer with per-camera intrinsics (not yet implemented in "
            "tracking.data; see module docstring)."
        )

    if spread_frac > _FOV_SPREAD_WARN_FLOOR_FRAC:
        warnings.warn(
            f"per-camera FoVX averaged: mean={mean_fov:.4f} rad "
            f"({math.degrees(mean_fov):.2f} deg), spread={spread_rad:.4f} rad "
            f"({spread_frac * 100:.2f}% of mean). Treating as calibration "
            "noise; if reconstruction looks intrinsics-limited consider "
            "switching to a per-camera-intrinsics writer.",
            UserWarning,
            stacklevel=2,
        )
    return mean_fov


def _resize_to_cap(img: np.ndarray, max_side: int | None) -> np.ndarray:
    if max_side is None:
        return img
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    new_h = int(round(h * max_side / s))
    new_w = int(round(w * max_side / s))
    # PIL handles uint8 RGB / single-channel masks identically.
    pil = Image.fromarray(img)
    pil = pil.resize((new_w, new_h), Image.NEAREST if img.ndim == 2 else Image.BILINEAR)
    return np.array(pil)


def _build_c2w_from_RT(R_2dgs: np.ndarray, T_2dgs: np.ndarray) -> np.ndarray:
    """Reverse opencv_c2w_to_2dgs_RT to recover the per-camera (4, 4) c2w.

    DMVScene exposes R, T in the *2DGS convention* (R is w2c[:3,:3].T,
    transposed for the CUDA glm layout). We need the c2w to write into
    transform_matrix. Reconstruct it cleanly here so write_scene
    doesn't reach back into the loader internals.
    """
    w2c = np.zeros((R_2dgs.shape[0], 4, 4), dtype=np.float64)
    w2c[:, :3, :3] = R_2dgs.transpose(0, 2, 1)
    w2c[:, :3, 3] = T_2dgs
    w2c[:, 3, 3] = 1.0
    return np.linalg.inv(w2c)


def write_timestep(t: int, scene: DMVScene, options: WriteOptions) -> str:
    """Write one timestep's Blender-style scene dir. Returns the dir path.

    Idempotent: skips PNG files that already exist (so re-running after a
    partial crash doesn't re-encode every frame), but always rewrites
    the JSON (cheap and lets you change resolution / mask flags later).
    """
    meta = scene.meta
    dest = os.path.join(options.work_root, f"timestep_{t:05d}")
    images_dir = os.path.join(dest, "images")
    masks_dir = os.path.join(dest, "masks")
    os.makedirs(images_dir, exist_ok=True)
    if options.write_masks:
        os.makedirs(masks_dir, exist_ok=True)

    view = scene.read_timestep(t)
    rgb = view["rgb"]                    # (n_cams, H, W, 3) uint8
    fg_mask = view["fg_mask"]            # (n_cams, H, W)    bool

    for i in range(meta.n_cams):
        rgb_path = os.path.join(images_dir, f"cam_{i:02d}{options.image_ext}")
        if not os.path.exists(rgb_path):
            arr = _resize_to_cap(rgb[i], options.max_image_side)
            Image.fromarray(arr).save(rgb_path)
        if options.write_masks:
            mask_path = os.path.join(masks_dir, f"cam_{i:02d}.png")
            if not os.path.exists(mask_path):
                m = (fg_mask[i].astype(np.uint8)) * 255
                m = _resize_to_cap(m, options.max_image_side)
                Image.fromarray(m, mode="L").save(mask_path)

    # ---- transforms_*.json ----
    global_fovx = _resolve_global_fovx(meta.FovX, options.max_fov_spread_frac)
    c2w_opencv = _build_c2w_from_RT(meta.R, meta.T)
    c2w_blender = _opencv_c2w_to_opengl(c2w_opencv)
    frames = []
    for i in range(meta.n_cams):
        frames.append({
            "file_path": f"./images/cam_{i:02d}",   # extension appended by reader
            "transform_matrix": c2w_blender[i].tolist(),
        })
    train_doc = {
        "camera_angle_x": global_fovx,
        "frames": frames,
    }
    test_doc = {
        "camera_angle_x": global_fovx,
        "frames": [],     # readNerfSyntheticInfo requires the file but tolerates empty
    }
    with open(os.path.join(dest, "transforms_train.json"), "w") as f:
        json.dump(train_doc, f, indent=2)
    with open(os.path.join(dest, "transforms_test.json"), "w") as f:
        json.dump(test_doc, f, indent=2)

    # ---- points3d.ply (init seed for the Blender reader) ----
    # Sample uniformly inside the union of camera frustums using the
    # per-camera near/far from poses_bounds.npy. Without this, the
    # Blender reader at scene/dataset_readers.py:279 falls back to
    # uniform random init in [-1.3, 1.3]^3 -- wrong for any scene
    # whose content doesn't happen to sit at the world origin.
    # Always overwrite on re-run so a fresh Stage A overrides any
    # stale init from a previous (random-fallback) run.
    if options.init_points3d_ply:
        # Diagnostic: dump the RAW per-camera near/far from poses_bounds
        # *before* near_floor / far_ceiling sanitisation. Lets you spot
        # LLFF files with bad bounds (negative near, far <= near, or
        # absurd magnitudes that suggest the file is in different
        # units than the camera positions) at Stage A time -- before
        # any Stage B reconstruction wastes compute on a bad init.
        raw_near = meta.bounds[:, 0].astype(np.float64)
        raw_far = meta.bounds[:, 1].astype(np.float64)
        print(
            f"[write_scene] timestep_{t:05d}: raw bounds from poses_bounds.npy: "
            f"near=[{raw_near.min():.3g}, {raw_near.max():.3g}] median={np.median(raw_near):.3g}, "
            f"far=[{raw_far.min():.3g}, {raw_far.max():.3g}] median={np.median(raw_far):.3g}"
        )
        if (raw_near <= 0).any():
            n_bad = int((raw_near <= 0).sum())
            print(
                f"[write_scene]   WARN: {n_bad}/{meta.n_cams} cameras have near <= 0; "
                f"clamped to init_near_floor={options.init_near_floor}. "
                "If this is wrong for your data, pass --init-near-floor or fix poses_bounds."
            )
        if (raw_far <= raw_near).any():
            n_bad = int((raw_far <= raw_near).sum())
            print(
                f"[write_scene]   WARN: {n_bad}/{meta.n_cams} cameras have far <= near; "
                "the writer extends those by 1e-3 to avoid a degenerate frustum, but the "
                "init for those cameras will be a paper-thin shell -- check poses_bounds."
            )

        # Per-camera FoVY derived from FoVX + image aspect (mirrors the
        # Blender reader at scene/dataset_readers.py:236).
        fov_x_per_cam = meta.FovX.astype(np.float64)
        focal_per_cam = float(meta.width) / (2.0 * np.tan(fov_x_per_cam / 2.0))
        fov_y_per_cam = 2.0 * np.arctan(float(meta.height) / (2.0 * focal_per_cam))
        xyz, rgb, init_meta = init_point_cloud_in_frustums(
            c2w_matrices=c2w_opencv,
            fov_x=fov_x_per_cam,
            fov_y=fov_y_per_cam,
            bounds=meta.bounds,
            n_pts=options.init_n_pts,
            depth_distribution=options.init_depth_distribution,
            near_floor=options.init_near_floor,
            far_ceiling=options.init_far_ceiling,
            rng=np.random.default_rng(options.init_seed),
        )
        write_points3d_ply(
            os.path.join(dest, "points3d.ply"),
            xyz,
            (rgb * 255).astype(np.uint8),
        )
        bbox_min = init_meta["xyz_min"]
        bbox_max = init_meta["xyz_max"]
        bbox_med = init_meta["xyz_median"]
        near_lo, near_hi = float(init_meta["near_used"].min()), float(init_meta["near_used"].max())
        far_lo, far_hi = float(init_meta["far_used"].min()), float(init_meta["far_used"].max())
        print(
            f"[write_scene] timestep_{t:05d}: seeded points3d.ply with "
            f"{xyz.shape[0]} frustum-sampled points; "
            f"per-cam near in [{near_lo:.3g}, {near_hi:.3g}], "
            f"far in [{far_lo:.3g}, {far_hi:.3g}]; "
            f"bbox x=[{bbox_min[0]:+.2f}, {bbox_max[0]:+.2f}] "
            f"y=[{bbox_min[1]:+.2f}, {bbox_max[1]:+.2f}] "
            f"z=[{bbox_min[2]:+.2f}, {bbox_max[2]:+.2f}]; "
            f"median ({bbox_med[0]:+.2f}, {bbox_med[1]:+.2f}, {bbox_med[2]:+.2f})"
        )

    return dest


def write_timesteps(scene: DMVScene, timesteps: Iterable[int],
                     options: WriteOptions) -> list[str]:
    """Convenience wrapper. Returns the list of per-timestep dirs."""
    return [write_timestep(int(t), scene, options) for t in timesteps]
