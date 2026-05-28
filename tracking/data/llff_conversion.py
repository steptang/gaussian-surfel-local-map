"""LLFF -> COLMAP/OpenCV camera-pose conversion.

References (this module ports the canonical conversion, does NOT
hand-derive the axis flips -- which is a known silent-failure mode that
yields mirrored or rotated geometry with no error thrown):

  * LLFF: https://github.com/Fyusion/LLFF/blob/master/llff/poses/pose_utils.py
  * 3DGS' LLFF handling: https://github.com/graphdeco-inria/gaussian-splatting/issues/85
  * nerfstudio's load_llff.py: https://github.com/nerfstudio-project/nerfstudio

LLFF stores poses_bounds.npy as (N, 17) float64:
    columns 0..14 reshape to (3, 5) per camera:
        [R_3x3 | T_3x1 | (h, w, f)_3x1]
    columns 15..16 are (near, far) depth bounds.

LLFF camera basis (the columns of R) is [-y, x, z] expressed in the
OpenGL "right, up, back" convention -- equivalently, the camera looks
down its -Z axis with +Y up. COLMAP / OpenCV uses [x, -y, -z] with the
camera looking down +Z and +Y down.

The canonical conversion is therefore:
    poses_colmap_basis = concat([poses[:, 1:2], -poses[:, 0:1], -poses[:, 2:3]], axis=1)
applied to the basis columns 0..2, while the translation column (3) is
left untouched and the (h, w, f) column (4) is kept for the intrinsics
extraction.

This module exposes ``llff_to_w2c`` returning per-camera (R, T) in the
exact convention the 2DGS Blender reader expects: ``R = w2c[:3,:3].T``
(transposed for the CUDA glm convention; see
``scene/dataset_readers.py:readCamerasFromTransforms``) and
``T = w2c[:3, 3]``.
"""

from __future__ import annotations

import numpy as np


def parse_poses_bounds(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split an LLFF poses_bounds.npy array into (poses_3x5, hwf, bounds).

    ``arr`` is the raw (N, 17) array. Returns:
        poses_3x5: (N, 3, 5) -- the 15 pose columns reshaped.
        hwf:       (N, 3)    -- (height, width, focal) per camera.
        bounds:    (N, 2)    -- (near, far) depth bounds per camera.
    """
    if arr.ndim != 2 or arr.shape[1] != 17:
        raise ValueError(f"expected poses_bounds shape (N, 17); got {arr.shape}")
    poses = arr[:, :15].reshape(-1, 3, 5)
    bounds = arr[:, 15:].astype(np.float64)
    hwf = poses[:, :, 4].astype(np.float64)         # (N, 3) = (h, w, f) per camera
    return poses.astype(np.float64), hwf, bounds


def llff_to_opencv_c2w(poses_3x5: np.ndarray) -> np.ndarray:
    """Convert LLFF (N, 3, 5) poses to (N, 4, 4) OpenCV/COLMAP c2w matrices.

    Canonical permutation, ported verbatim from the dataset's own README
    (ken2576/multiview_preprocessing -- the producer of this h5+npy
    format):

        ext = np.concatenate([ext[:, 1:2],
                              ext[:, 0:1],
                              -ext[:, 2:3],
                              ext[:, 3:4]], axis=1)

    Applied to the (3, 5) per-camera matrix [R_3x3 | T_3x1 | (h,w,f)]:
        new col 0 (R) = +old col 1
        new col 1 (R) = +old col 0
        new col 2 (R) = -old col 2
        new col 3 (T) = +old col 3  (translation untouched)
        col 4 (hwf)   = ignored here (handled separately by hwf_to_fov)

    DO NOT hand-derive this -- LLFF's basis convention is documented
    inconsistently across the literature ("down-right-back" vs
    "right-up-back") and the wrong permutation silently produces
    mirrored or rotated reconstructions with no error thrown. If the
    ken2576 README is ever updated, update this function to match.
    """
    if poses_3x5.ndim != 3 or poses_3x5.shape[1:] != (3, 5):
        raise ValueError(f"expected (N, 3, 5) LLFF poses; got {poses_3x5.shape}")
    R_llff = poses_3x5[:, :, :3]            # (N, 3, 3)
    t_world = poses_3x5[:, :, 3]            # (N, 3)
    # ken2576 canonical permutation -- see docstring.
    R_colmap = np.concatenate(
        [R_llff[:, :, 1:2], R_llff[:, :, 0:1], -R_llff[:, :, 2:3]],
        axis=2,
    )                                        # (N, 3, 3)
    N = R_colmap.shape[0]
    c2w = np.zeros((N, 4, 4), dtype=np.float64)
    c2w[:, :3, :3] = R_colmap
    c2w[:, :3, 3] = t_world
    c2w[:, 3, 3] = 1.0
    return c2w


def opencv_c2w_to_2dgs_RT(c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the (R, T) convention used by scene/dataset_readers.py.

    The 2DGS Blender reader does (see readCamerasFromTransforms):

        w2c = inv(c2w)
        R = w2c[:3, :3].T            # transposed for the CUDA glm convention
        T = w2c[:3, 3]

    Returns (R, T) with shapes (N, 3, 3) and (N, 3). T is the
    camera-space translation; R is the row-major-stored world-to-camera
    rotation matrix.
    """
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"expected (N, 4, 4) c2w; got {c2w.shape}")
    w2c = np.linalg.inv(c2w)                    # (N, 4, 4)
    R = np.transpose(w2c[:, :3, :3], axes=(0, 2, 1))   # (N, 3, 3), transposed
    T = w2c[:, :3, 3]                                    # (N, 3)
    return R.astype(np.float64), T.astype(np.float64)


def hwf_to_fov(hwf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute (FovX, FovY) in radians from LLFF's (h, w, f) column.

    Standard pinhole: FoV = 2 * arctan(half_extent / focal). f is in
    pixels, matching h and w.
    """
    h = hwf[:, 0]
    w = hwf[:, 1]
    f = hwf[:, 2]
    FovY = 2.0 * np.arctan(h / (2.0 * f))
    FovX = 2.0 * np.arctan(w / (2.0 * f))
    return FovX.astype(np.float64), FovY.astype(np.float64)


def llff_to_2dgs(arr: np.ndarray) -> dict:
    """One-shot helper: poses_bounds.npy array -> dict of 2DGS-ready arrays.

    Returns:
        R:      (N, 3, 3) world-to-camera rotation, transposed for glm.
        T:      (N, 3)    world-to-camera translation.
        FovX:   (N,)      horizontal FoV in radians.
        FovY:   (N,)      vertical FoV in radians.
        height: (N,) int  per-camera image height (from LLFF hwf).
        width:  (N,) int
        bounds: (N, 2)    LLFF (near, far) bounds.
        c2w:    (N, 4, 4) OpenCV camera-to-world (for diagnostics).
    """
    poses_3x5, hwf, bounds = parse_poses_bounds(arr)
    c2w = llff_to_opencv_c2w(poses_3x5)
    R, T = opencv_c2w_to_2dgs_RT(c2w)
    FovX, FovY = hwf_to_fov(hwf)
    return {
        "R": R,
        "T": T,
        "FovX": FovX,
        "FovY": FovY,
        "height": hwf[:, 0].astype(np.int64),
        "width": hwf[:, 1].astype(np.int64),
        "bounds": bounds,
        "c2w": c2w,
    }
