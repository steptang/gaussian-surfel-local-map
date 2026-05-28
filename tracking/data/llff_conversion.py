"""LLFF -> OpenCV/COLMAP camera-pose conversion.

poses_bounds.npy stores (N, 17) float64:
    columns 0..14 reshape to (3, 5) per camera:
        [R_3x3 | T_3x1 | (h, w, f)_3x1]
    columns 15..16 are (near, far) depth bounds.

**Convention surprise** (verified empirically; see
``llff_to_opencv_c2w`` for the full story): for the Deep 3D Mask
Volume datasets produced by ken2576's pipeline -- the actual format
Stage A consumes -- the stored R is **already** the OpenCV c2w
rotation. No LLFF axis permutation is needed.

The textbook LLFF [down, right, back] convention with the
``[col1, col0, -col2]`` permutation applies to other sources (the
original Fyusion/LLFF, or ken2576's COLMAP-input branch). The h5+npy
files we read here are OpenCV-c2w-as-is.

Two silent-failure modes led to that finding:

  1. ``det(R) == +1`` alone does NOT distinguish "correct rotation"
     from "correct rotation composed with a 180° flip about a
     perpendicular axis"; both have det +1.
  2. Training PSNR does NOT either: with mostly-black GT (foreground-
     masked DMV frames) and mis-oriented cameras the optimizer
     happily settles on a "render black everywhere" minimum, hitting
     PSNR ~24 while producing a model that renders pure zero at
     evaluation time.

Regression test:
``tests/test_llff_conversion.py::test_multi_camera_convergence_dot``
constructs a synthetic rig where every camera looks at a known
target, runs the conversion, and asserts ``forward · to-target > 0``
for every camera. det == +1 wouldn't catch the bug; this geometric
check does.

This module exposes ``llff_to_2dgs`` returning per-camera (R, T) in
the exact convention the 2DGS Blender reader expects:
``R = w2c[:3, :3].T`` (transposed for the CUDA glm convention; see
``scene/dataset_readers.py:readCamerasFromTransforms``) and
``T = w2c[:3, 3]``.

References:
  * Fyusion/LLFF: https://github.com/Fyusion/LLFF/blob/master/llff/poses/pose_utils.py
  * ken2576: https://github.com/ken2576/multiview_preprocessing
  * nerfstudio's load_llff.py (extra ``poses[:, 1:3] *= -1`` world flip not needed for this source)
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
    """Convert (N, 3, 5) LLFF-formatted poses to (N, 4, 4) OpenCV/COLMAP c2w.

    For the Deep 3D Mask Volume datasets produced by ken2576's pipeline
    (the format Stage A consumes), poses_bounds.npy stores the rotation
    block as the **OpenCV c2w rotation directly**, not in LLFF's native
    [down, right, back] convention. We verified this empirically:

        Multi-camera ray-convergence test on scene14 poses_bounds.npy:

            candidate                  dot mean  notes
            ------------------------   --------  ---------------------------
            [col1, col0, -col2]          -0.998  cameras all look AWAY
            identity (no permute)        +0.998  cameras all look AT scene  <-- correct
            transpose then [c1,c0,-c2]   +0.996  also plausible but higher RMSE

    The earlier `[col1, col0, -col2]` permutation -- inferred from the
    ken2576 README description of `save_poses` -- silently produces
    cameras pointing 180° away from the scene, an unsalvageable
    "trained-black" failure mode at training time (the renderer
    returns pure zeros because no surfel ever enters the frustum).
    det(R) == +1 alone does NOT catch this; you need a geometric
    test against the scene direction. See
    tests/test_llff_conversion.py::test_dmv_cameras_look_at_each_other
    for the regression check.

    If a future scene's pipeline really does use LLFF's documented
    [col1, col0, -col2] permutation, gate the conversion on a
    `--llff-permutation` flag and pick at Stage A time. Adding this
    flag is straightforward; we deliberately don't ship it yet
    because the current dataset doesn't need it.
    """
    if poses_3x5.ndim != 3 or poses_3x5.shape[1:] != (3, 5):
        raise ValueError(f"expected (N, 3, 5) LLFF poses; got {poses_3x5.shape}")
    R_opencv = poses_3x5[:, :, :3]              # (N, 3, 3) -- already in OpenCV c2w form
    t_world = poses_3x5[:, :, 3]                # (N, 3)
    N = R_opencv.shape[0]
    c2w = np.zeros((N, 4, 4), dtype=np.float64)
    c2w[:, :3, :3] = R_opencv
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
