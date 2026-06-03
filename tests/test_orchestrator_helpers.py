"""Sanity checks on tracking/orchestrator_helpers.

The functions here port pieces of utils/graphics_utils + scene/cameras
from CUDA/torch to pure numpy so the Stage D projection check can run
without GPU. If the upstream changes (e.g., znear/zfar defaults, axis-
flip conventions) and the port drifts, the orchestrator will silently
project surfels into the wrong pixel coordinates and Stage D will look
broken. These tests pin the contract.

Run from repo root with: pytest tests/test_orchestrator_helpers.py -v
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.orchestrator_helpers import (
    DEFAULT_ZFAR,
    DEFAULT_ZNEAR,
    build_full_proj_transform,
    load_projection_views_for_timestep,
)


def _camera_looks_at(eye: np.ndarray, target: np.ndarray,
                      world_up: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """Build (R_stored, T_stored) -- the convention the Blender reader
    hands to Camera.__init__ -- for an OpenCV camera at `eye` looking
    at `target`.
    """
    if world_up is None:
        world_up = np.array([0.0, 0.0, 1.0])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= max(np.linalg.norm(right), 1e-12)
    down = np.cross(forward, right)
    R_c2w = np.stack([right, down, forward], axis=1)   # OpenCV [right, down, forward] columns
    c2w = np.eye(4)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = eye
    w2c = np.linalg.inv(c2w)
    R_stored = w2c[:3, :3].T     # glm-transposed
    T_stored = w2c[:3, 3]
    return R_stored, T_stored


def test_origin_projects_to_image_centre_for_camera_looking_at_it():
    """A camera at +z looking at the origin should project the origin to
    the image centre.

    Tests the entire chain (R, T) -> full_proj_transform -> NDC ->
    pixel coordinates. If any sign in getWorld2View2 or
    getProjectionMatrix is flipped, this catches it.
    """
    H, W = 480, 640
    fovx = math.radians(60.0)
    fovy = 2.0 * math.atan(H / (2.0 * (W / (2.0 * math.tan(fovx / 2.0)))))

    eye = np.array([0.0, 0.0, 5.0])
    target = np.array([0.0, 0.0, 0.0])
    R_stored, T_stored = _camera_looks_at(eye, target,
                                            world_up=np.array([0.0, 1.0, 0.0]))
    full = build_full_proj_transform(R_stored, T_stored, fovx, fovy)

    # Project the origin (in row-vector convention).
    pt_h = np.array([0.0, 0.0, 0.0, 1.0])
    clip = pt_h @ full
    ndc = clip[:2] / clip[3]
    px = (ndc[0] + 1.0) * 0.5 * W
    py = (ndc[1] + 1.0) * 0.5 * H
    # Allow generous slop -- this isn't checking exact pixels, just that
    # we're not off by a sign or a flip.
    assert abs(px - W / 2) < 2.0, f"origin x-projected to {px}, expected ~{W/2}"
    assert abs(py - H / 2) < 2.0, f"origin y-projected to {py}, expected ~{H/2}"
    # And the camera-space depth (clip_w under z_sign=+1 projection) is positive.
    assert clip[3] > 0, f"camera-space z (clip_w) {clip[3]} <= 0; point appears behind camera"


def test_point_behind_camera_has_negative_clip_w():
    """A point on the opposite side of the camera from its forward
    direction must have clip_w <= 0 so the in-bounds frustum check can
    correctly reject it.
    """
    H, W = 100, 100
    fovx = math.radians(60.0)
    fovy = math.radians(60.0)

    # Camera at origin looking down +z. Place a point at -1 z (behind).
    R_stored, T_stored = _camera_looks_at(
        eye=np.array([0.0, 0.0, 0.0]),
        target=np.array([0.0, 0.0, 1.0]),
        world_up=np.array([0.0, 1.0, 0.0]),
    )
    full = build_full_proj_transform(R_stored, T_stored, fovx, fovy)
    pt_h = np.array([0.0, 0.0, -1.0, 1.0])
    clip = pt_h @ full
    assert clip[3] < 0


def test_translation_invariant():
    """If you move a target and the camera by the same world offset,
    the projected pixel of the target shouldn't change (relative
    geometry preserved).
    """
    H, W = 200, 200
    fovx = math.radians(60.0)
    fovy = math.radians(60.0)

    rng = np.random.default_rng(0)
    for trial in range(3):
        offset = rng.uniform(-10, 10, size=3)
        eye_a = np.array([0.0, 0.0, 5.0])
        eye_b = eye_a + offset
        tgt_a = np.array([0.5, 0.3, 0.0])
        tgt_b = tgt_a + offset

        for eye, tgt in [(eye_a, tgt_a), (eye_b, tgt_b)]:
            R, T = _camera_looks_at(eye, tgt, world_up=np.array([0.0, 1.0, 0.0]))
            full = build_full_proj_transform(R, T, fovx, fovy)
            pt = np.array([*tgt, 1.0]) @ full
            ndc = pt[:2] / pt[3]
            if eye is eye_a:
                ndc_a = ndc
            else:
                ndc_b = ndc
        np.testing.assert_allclose(ndc_a, ndc_b, atol=1e-9,
                                    err_msg="translation-invariance broken")


def _make_synthetic_timestep_dir(tmp_path: Path, n_cams: int = 4,
                                   H: int = 80, W: int = 120) -> Path:
    """Mirror what tracking/data/write_scene.py produces for one timestep."""
    d = tmp_path / "timestep_00000"
    (d / "images").mkdir(parents=True)
    (d / "masks").mkdir(parents=True)
    fovx = math.radians(60.0)
    frames = []
    for i in range(n_cams):
        theta = 2 * math.pi * i / n_cams
        eye = np.array([3.0 * math.cos(theta), 3.0 * math.sin(theta), 0.0])
        target = np.zeros(3)
        # OpenCV c2w with z-up world (matches Stage A's writer convention).
        forward = target - eye
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up); right /= max(np.linalg.norm(right), 1e-12)
        down = np.cross(forward, right)
        R_c2w = np.stack([right, down, forward], axis=1)
        c2w_opencv = np.eye(4)
        c2w_opencv[:3, :3] = R_c2w
        c2w_opencv[:3, 3] = eye
        # Apply the OpenCV -> OpenGL flip that the writer does so the
        # Blender reader's flip cancels.
        c2w_blender = c2w_opencv.copy()
        c2w_blender[:3, 1:3] *= -1
        frames.append({
            "file_path": f"./images/cam_{i:02d}",
            "transform_matrix": c2w_blender.tolist(),
        })
        # Write a dummy mask (all-zero is fine for the load test).
        Image.fromarray(np.zeros((H, W), dtype=np.uint8), mode="L").save(d / "masks" / f"cam_{i:02d}.png")
    with open(d / "transforms_train.json", "w") as f:
        json.dump({"camera_angle_x": fovx, "frames": frames}, f)
    return d


def test_load_projection_views_for_timestep(tmp_path):
    d = _make_synthetic_timestep_dir(tmp_path, n_cams=4, H=80, W=120)
    views = load_projection_views_for_timestep(str(d))
    assert len(views) == 4
    for i, v in enumerate(views):
        assert v.image_width == 120
        assert v.image_height == 80
        assert v.fg_mask.shape == (80, 120)
        assert v.fg_mask.dtype == bool
        assert v.full_proj_transform.shape == (4, 4)
        # Origin should project somewhere inside the image for every
        # camera (they all look at the origin from a circle of radius 3).
        pt = np.array([0.0, 0.0, 0.0, 1.0]) @ v.full_proj_transform
        ndc = pt[:2] / pt[3]
        px = (ndc[0] + 1.0) * 0.5 * v.image_width
        py = (ndc[1] + 1.0) * 0.5 * v.image_height
        assert abs(px - v.image_width / 2) < 2.0, f"cam {i}: origin px={px}, expected ~{v.image_width/2}"
        assert abs(py - v.image_height / 2) < 2.0, f"cam {i}: origin py={py}, expected ~{v.image_height/2}"
        assert pt[3] > 0, f"cam {i}: clip_w={pt[3]} <= 0; origin appears behind camera"


def test_missing_mask_is_skipped_not_raised(tmp_path):
    d = _make_synthetic_timestep_dir(tmp_path, n_cams=4)
    # Delete one mask -- the loader should skip that camera but keep the
    # rest, not crash.
    os.remove(d / "masks" / "cam_01.png")
    views = load_projection_views_for_timestep(str(d))
    assert len(views) == 3
