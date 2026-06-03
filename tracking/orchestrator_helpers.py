"""Pure-numpy helpers the orchestrator uses to bridge Stage A's outputs to
Stage D's ``ProjectionView`` API.

Kept separate from the orchestrator itself so the synthetic test in
``diagnostics/test_tracking_synthetic.py`` and any unit tests can import
the helpers without dragging the orchestrator's heavier IO surface.

Why pure-numpy: Stage D's projection check is the only non-trivial
geometric computation the orchestrator does, and it needs the camera's
``full_proj_transform`` matrix in exactly the convention the 2DGS
rasterizer uses at training time -- otherwise projected pixel
coordinates won't line up with the fg-mask pixel grid. We mirror the
existing math (``scene/cameras.py:Camera.__init__`` and
``utils/graphics_utils.{getWorld2View2, getProjectionMatrix}``) on
CPU/float64 so no CUDA / torch is required.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np

from .classify_dynamic import ProjectionView


# Defaults mirror ``scene/cameras.py`` for znear/zfar. The Blender reader
# hardcodes these constants on every Camera, so we use the same values
# for the projection matrix we build here. Override only if the upstream
# Camera defaults change.
DEFAULT_ZNEAR = 0.01
DEFAULT_ZFAR = 100.0


def _get_world2view(R_stored: np.ndarray, T_stored: np.ndarray) -> np.ndarray:
    """Numpy port of ``utils.graphics_utils.getWorld2View2`` for the
    default ``translate=0, scale=1`` case (which is what Stage A's
    ``Camera.__init__`` uses).

    ``R_stored`` is the convention the Blender reader produces:
    ``R = w2c[:3, :3].T`` (transposed for the CUDA glm layout). This
    function returns the actual ``w2c`` 4x4 matrix.
    """
    w2c = np.zeros((4, 4), dtype=np.float64)
    w2c[:3, :3] = R_stored.T          # un-transpose to recover w2c rotation
    w2c[:3, 3] = T_stored
    w2c[3, 3] = 1.0
    return w2c


def _get_projection(znear: float, zfar: float, fovX: float, fovY: float) -> np.ndarray:
    """Numpy port of ``utils.graphics_utils.getProjectionMatrix``.

    With left = -right and bottom = -top (the always-true case in the
    upstream code), the matrix simplifies to:
        P[0,0] = 1 / tan(FovX/2)
        P[1,1] = 1 / tan(FovY/2)
        P[2,2] = zfar / (zfar - znear)
        P[2,3] = -zfar*znear / (zfar - znear)
        P[3,2] = 1
    See the upstream comment about z_sign=+1.
    """
    tan_half_fy = math.tan(fovY / 2.0)
    tan_half_fx = math.tan(fovX / 2.0)
    P = np.zeros((4, 4), dtype=np.float64)
    P[0, 0] = 1.0 / tan_half_fx
    P[1, 1] = 1.0 / tan_half_fy
    P[2, 2] = zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = 1.0
    return P


def build_full_proj_transform(R_stored: np.ndarray, T_stored: np.ndarray,
                                fovX: float, fovY: float,
                                znear: float = DEFAULT_ZNEAR,
                                zfar: float = DEFAULT_ZFAR) -> np.ndarray:
    """Reproduce ``Camera.full_proj_transform`` on CPU.

    The 2DGS Camera computes (per ``scene/cameras.py``):

        world_view_transform = getWorld2View2(R, T).T
        projection_matrix    = getProjectionMatrix(...).T
        full_proj_transform  = world_view_transform @ projection_matrix

    i.e., everything is in row-vector convention: ``clip = pt @ full``.
    Returns a (4, 4) float64 matrix. Used by Stage D's projection check
    and by any other code path that wants to project surfels into
    training-resolution pixel coordinates without instantiating a CUDA
    Camera.
    """
    w2c = _get_world2view(R_stored, T_stored)
    P = _get_projection(znear, zfar, fovX, fovY)
    # row-vector: full = w2c.T @ P.T = (P @ w2c).T
    return (P @ w2c).T


def _blender_json_to_R_T(c2w_in_json: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the Blender reader's c2w flip + R/T extraction.

    ``scene/dataset_readers.py:readCamerasFromTransforms`` does:

        c2w = transform_matrix
        c2w[:3, 1:3] *= -1            # OpenGL -> OpenCV flip
        w2c = inv(c2w)
        R = w2c[:3, :3].T              # glm-transposed
        T = w2c[:3, 3]
    """
    c2w = c2w_in_json.copy()
    c2w[:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3].T
    T = w2c[:3, 3]
    return R, T


def _load_mask_png(path: str) -> np.ndarray:
    """Read a uint8 0/255 mask PNG into a (H, W) bool array."""
    from PIL import Image
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr > 127


def load_projection_views_for_timestep(
    timestep_work_dir: str,
    znear: float = DEFAULT_ZNEAR,
    zfar: float = DEFAULT_ZFAR,
) -> list[ProjectionView]:
    """Build one ``ProjectionView`` per camera in this timestep.

    Reads:
        <timestep_work_dir>/transforms_train.json   -- camera_angle_x + per-cam transform_matrix
        <timestep_work_dir>/masks/cam_XX.png        -- fg/bg mask, sibling of the rendered image

    Camera ``i`` in the JSON's ``frames`` list pairs with ``cam_{i:02d}.png``
    in the masks/ subdirectory (which mirrors how Stage A's
    ``write_scene.write_timestep`` emits the files).

    Skipped silently per-camera when its mask PNG is missing -- ICP /
    classification can still proceed on the remaining views; only Stage
    D's per-surfel fg_score becomes less reliable.
    """
    json_path = os.path.join(timestep_work_dir, "transforms_train.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(json_path)
    with open(json_path, "r") as f:
        doc = json.load(f)
    fovx_global = float(doc["camera_angle_x"])
    frames = doc.get("frames", [])
    masks_dir = os.path.join(timestep_work_dir, "masks")

    views: list[ProjectionView] = []
    for i, frame in enumerate(frames):
        mask_path = os.path.join(masks_dir, f"cam_{i:02d}.png")
        if not os.path.exists(mask_path):
            print(f"[load_projection_views] {os.path.basename(timestep_work_dir)} "
                  f"cam_{i:02d}: no mask PNG; skipping")
            continue
        fg_mask = _load_mask_png(mask_path)
        H, W = fg_mask.shape

        # Per-camera FoVY is derived from FoVX + aspect ratio, exactly like
        # the Blender reader does at scene/dataset_readers.py:236.
        # focal = W / (2 * tan(fovx/2));  fovy = 2 * arctan(H / (2 * focal)).
        f_px = W / (2.0 * math.tan(fovx_global / 2.0))
        fovY = 2.0 * math.atan(H / (2.0 * f_px))

        c2w_json = np.array(frame["transform_matrix"], dtype=np.float64)
        R_stored, T_stored = _blender_json_to_R_T(c2w_json)

        full = build_full_proj_transform(
            R_stored=R_stored, T_stored=T_stored,
            fovX=fovx_global, fovY=fovY, znear=znear, zfar=zfar,
        )
        views.append(ProjectionView(
            full_proj_transform=full,
            image_width=int(W),
            image_height=int(H),
            fg_mask=fg_mask,
        ))
    return views


def load_views_for_sequence(work_root: str, timesteps: list[int],
                              znear: float = DEFAULT_ZNEAR,
                              zfar: float = DEFAULT_ZFAR
                              ) -> dict[int, list[ProjectionView]]:
    """Load per-timestep ``ProjectionView`` lists for every timestep in
    ``timesteps``. Missing per-timestep dirs are silently skipped (the
    orchestrator decides whether to fail or proceed)."""
    out: dict[int, list[ProjectionView]] = {}
    for t in timesteps:
        d = os.path.join(work_root, f"timestep_{t:05d}")
        if not os.path.isdir(d):
            print(f"[load_views_for_sequence] {d} missing; skipping")
            continue
        out[t] = load_projection_views_for_timestep(d, znear=znear, zfar=zfar)
    return out
