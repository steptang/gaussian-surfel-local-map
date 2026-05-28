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
from typing import Iterable

import numpy as np
from PIL import Image

from .dmv_loader import DMVScene


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

    return dest


def write_timesteps(scene: DMVScene, timesteps: Iterable[int],
                     options: WriteOptions) -> list[str]:
    """Convenience wrapper. Returns the list of per-timestep dirs."""
    return [write_timestep(int(t), scene, options) for t in timesteps]
