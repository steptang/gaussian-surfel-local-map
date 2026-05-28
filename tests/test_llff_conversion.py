"""Sanity checks on the LLFF -> OpenCV/COLMAP pose conversion.

These tests can't verify the *absolute* axis-flip direction (that
requires running a reconstruction and inspecting a render -- exactly
the verification step Stephanie planned for the single-timestep
driver run). They DO verify properties that any correct conversion
must satisfy, which catches regressions if the canonical formula is
ever edited incorrectly:

1. orthogonal: converted R is orthogonal (R @ R.T == I).
2. proper_rotation: det(R) == +1 (NOT -1, which would mean we
   introduced a mirror via the wrong sign on an axis).
3. translation_preserved: T column is byte-for-byte unchanged.
4. shape_invariants: (3, 5) per-camera in, (4, 4) c2w out.

Run from repo root with: pytest tests/test_llff_conversion.py -v
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.data.llff_conversion import (
    hwf_to_fov,
    llff_to_2dgs,
    llff_to_opencv_c2w,
    opencv_c2w_to_2dgs_RT,
    parse_poses_bounds,
)


def _make_random_llff_poses(n: int, seed: int = 0) -> np.ndarray:
    """N random valid LLFF 3x5 poses (rotation columns are orthonormal)."""
    rng = np.random.default_rng(seed)
    poses = np.zeros((n, 3, 5), dtype=np.float64)
    for i in range(n):
        # Random rotation via QR of a Gaussian matrix.
        A = rng.normal(size=(3, 3))
        Q, _ = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1   # ensure proper rotation
        poses[i, :, :3] = Q
        poses[i, :, 3] = rng.normal(size=3) * 2.0   # translation
        poses[i, :, 4] = [480.0, 640.0, 500.0]       # h, w, f
    return poses


def test_shape_invariants():
    poses = _make_random_llff_poses(3)
    c2w = llff_to_opencv_c2w(poses)
    assert c2w.shape == (3, 4, 4)
    # Bottom row is exactly [0, 0, 0, 1].
    np.testing.assert_array_equal(c2w[:, 3, :3], 0)
    np.testing.assert_array_equal(c2w[:, 3, 3], 1)


def test_converted_R_is_orthogonal():
    # If the conversion ever broke (e.g., we forgot to negate a column
    # or swapped two columns inconsistently), R would no longer satisfy
    # R @ R.T == I.
    poses = _make_random_llff_poses(5)
    c2w = llff_to_opencv_c2w(poses)
    R = c2w[:, :3, :3]
    eye = np.broadcast_to(np.eye(3), (5, 3, 3))
    np.testing.assert_allclose(R @ R.transpose(0, 2, 1), eye, atol=1e-12)


def test_converted_R_is_proper_rotation_not_mirror():
    # det(R) should be exactly +1. A wrong axis-flip (e.g., -col0
    # instead of +col0 when going from LLFF "down" to COLMAP "down")
    # would produce det = -1 silently and the rendered geometry would
    # be mirrored.
    poses = _make_random_llff_poses(8)
    c2w = llff_to_opencv_c2w(poses)
    dets = np.linalg.det(c2w[:, :3, :3])
    np.testing.assert_allclose(dets, 1.0, atol=1e-12,
                                err_msg="converted rotation matrices are not proper rotations -- LLFF axis flip is wrong")


def test_translation_column_preserved():
    poses = _make_random_llff_poses(4, seed=42)
    c2w = llff_to_opencv_c2w(poses)
    # T (camera position in world) is the fourth column of the input
    # and the [:3, 3] of the output. The conversion must not touch it.
    np.testing.assert_array_equal(c2w[:, :3, 3], poses[:, :, 3])


def test_opencv_c2w_to_2dgs_RT_inverts_round_trip():
    # opencv_c2w_to_2dgs_RT returns R, T such that:
    #   inv_c2w = R.T (un-transposed) on top of T.
    # Verify we can reconstruct c2w from (R, T).
    poses = _make_random_llff_poses(3)
    c2w = llff_to_opencv_c2w(poses)
    R_2dgs, T_2dgs = opencv_c2w_to_2dgs_RT(c2w)
    # R_2dgs is w2c[:3, :3].T -> w2c rotation is R_2dgs.T.
    w2c = np.zeros_like(c2w)
    w2c[:, :3, :3] = R_2dgs.transpose(0, 2, 1)
    w2c[:, :3, 3] = T_2dgs
    w2c[:, 3, 3] = 1.0
    np.testing.assert_allclose(np.linalg.inv(w2c), c2w, atol=1e-10)


def test_hwf_to_fov_matches_pinhole_definition():
    # f = h / (2 * tan(FoVY/2))  =>  FoVY = 2 * arctan(h / (2f))
    hwf = np.array([[480, 640, 500.0]], dtype=np.float64)
    fx, fy = hwf_to_fov(hwf)
    assert math.isclose(float(fy), 2.0 * math.atan(240.0 / 500.0), abs_tol=1e-12)
    assert math.isclose(float(fx), 2.0 * math.atan(320.0 / 500.0), abs_tol=1e-12)


def test_llff_to_2dgs_keys_and_shapes():
    poses = _make_random_llff_poses(3)
    bounds = np.array([[0.5, 50.0], [0.4, 40.0], [0.6, 60.0]])
    raw = np.concatenate([poses.reshape(3, 15), bounds], axis=1)
    out = llff_to_2dgs(raw)
    assert set(out.keys()) == {"R", "T", "FovX", "FovY", "height", "width", "bounds", "c2w"}
    assert out["R"].shape == (3, 3, 3)
    assert out["T"].shape == (3, 3)
    assert out["FovX"].shape == (3,)
    assert out["FovY"].shape == (3,)
    assert out["height"].tolist() == [480, 480, 480]
    assert out["width"].tolist() == [640, 640, 640]
    np.testing.assert_array_equal(out["bounds"], bounds)
    # Same orthogonality + det invariants on the c2w side.
    R = out["c2w"][:, :3, :3]
    np.testing.assert_allclose(R @ R.transpose(0, 2, 1), np.broadcast_to(np.eye(3), (3, 3, 3)), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-12)


def _build_synthetic_look_at_poses(n_cameras: int = 8,
                                    target: np.ndarray = None,
                                    rig_radius: float = 4.0,
                                    rig_z: float = 0.0,
                                    seed: int = 0) -> np.ndarray:
    """Build a (n, 17) poses_bounds.npy-shaped array where every camera
    sits on a circle in the xy-plane at z = ``rig_z`` and is oriented to
    look at ``target``. Stored as **OpenCV c2w** -- the empirically-
    confirmed convention of the Deep 3D Mask Volume dataset (see
    ``tracking.data.llff_conversion`` module docstring).

    Returned poses_bounds row layout: 15 floats of 3x5 (in C order)
    followed by 2 bounds floats.
    """
    if target is None:
        target = np.zeros(3, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.zeros((n_cameras, 17), dtype=np.float64)
    for i in range(n_cameras):
        theta = 2 * math.pi * i / n_cameras
        eye = np.array([rig_radius * math.cos(theta),
                        rig_radius * math.sin(theta),
                        rig_z], dtype=np.float64)
        # OpenCV look-at: forward = (target - eye)/|.|; world_up = +z
        # right = forward x world_up; down = forward x right.
        forward = target - eye
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right /= max(np.linalg.norm(right), 1e-12)
        down = np.cross(forward, right)
        # c2w 3x3: columns are [right, down, forward] for OpenCV.
        R = np.stack([right, down, forward], axis=1)
        pose_3x5 = np.zeros((3, 5), dtype=np.float64)
        pose_3x5[:, :3] = R
        pose_3x5[:, 3] = eye
        # hwf -- arbitrary; just needs to be positive so hwf_to_fov is well-defined.
        pose_3x5[:, 4] = [480.0, 640.0, 500.0]
        out[i, :15] = pose_3x5.reshape(15)
        out[i, 15:] = [0.1, 100.0]
    return out


def test_multi_camera_convergence_dot():
    """Geometric look-at sanity: after llff_to_2dgs, every camera should
    have ``forward . (target - camera_position) > 0``.

    Catches the silent-failure mode that ``det(R) == +1`` misses: a
    180-degree rotation about a perpendicular axis is still a proper
    rotation, but every camera ends up pointing AWAY from the scene
    rather than toward it.

    This is the regression that the DMV scene-14 debugging surfaced.
    With the previously-committed ``[col1, col0, -col2]`` permutation
    every dot was ~ -1; with the corrected identity conversion every
    dot is ~ +1.
    """
    target = np.array([0.5, 0.3, 1.2], dtype=np.float64)
    arr = _build_synthetic_look_at_poses(n_cameras=8, target=target,
                                          rig_radius=4.0, rig_z=0.0)
    out = llff_to_2dgs(arr)
    R = out["R"]                 # (N, 3, 3) -- w2c rotation transposed for glm
    T = out["T"]                 # (N, 3)   -- w2c translation
    N = R.shape[0]
    # Reconstruct OpenCV c2w: per 2DGS convention, R_stored = c2w_R, T_stored = w2c_t.
    # Camera position in world = -R_stored @ T_stored.
    # Forward direction in world = R_stored[:, :, 2] (= c2w[:, :, 2]).
    cam_pos = -np.einsum("nij,nj->ni", R, T)
    forward = R[:, :, 2]
    to_target = target[None, :] - cam_pos
    to_target /= np.linalg.norm(to_target, axis=1, keepdims=True)
    dots = (forward * to_target).sum(axis=1)
    # Every camera must point toward the target. A failure here means
    # the conversion silently flipped the look-axis on us.
    assert (dots > 0.99).all(), (
        f"some cameras don't look at target: dots = {dots} "
        f"(min={dots.min():.4f}, max={dots.max():.4f}). "
        "Probable cause: rotation permutation in llff_to_opencv_c2w is wrong "
        "for this dataset's convention."
    )


def test_parse_poses_bounds_rejects_bad_shape():
    with pytest.raises(ValueError):
        parse_poses_bounds(np.zeros((3, 16)))   # missing bounds column
    with pytest.raises(ValueError):
        parse_poses_bounds(np.zeros((3,)))       # not 2D


def test_fov_is_resolution_invariant():
    """Pin the silent-failure guarantee for the DMV resize case.

    The h5 frames are downsized (e.g., 1080x1920 -> 360x640, 1/3 scale)
    while poses_bounds.npy carries the LLFF native focal at the
    pre-resize resolution. Because the loader stores *FoV* (not focal)
    and the Blender reader re-derives focal at the actual image size,
    the rescale is implicit and correct -- but only if FoV stays
    resolution-invariant. This test pins that:

        FoVX(1920, 984) == FoVX(640, 328)
        FoVY(1080, 984) == FoVY(360, 328)

    If anyone refactors llff_conversion to pass focal-in-pixels through
    directly, this test will fail and surface the silent-failure mode
    Stephanie called out (correct orientation, wrong projection scale).
    """
    # Native LLFF intrinsics.
    full_hwf = np.array([[1080.0, 1920.0, 984.0]])
    # H5-resized intrinsics with f scaled by the same 1/3 factor.
    h5_hwf = np.array([[360.0, 640.0, 984.0 / 3.0]])

    fovx_full, fovy_full = hwf_to_fov(full_hwf)
    fovx_h5, fovy_h5 = hwf_to_fov(h5_hwf)

    np.testing.assert_allclose(fovx_full, fovx_h5, atol=1e-12,
                                err_msg="FoVX changed across the 1/3 resize -- focal isn't being rescaled correctly")
    np.testing.assert_allclose(fovy_full, fovy_h5, atol=1e-12,
                                err_msg="FoVY changed across the 1/3 resize")

    # And the inverse-direction check: feeding FoVX through the Blender
    # reader's focal2fov(fov2focal(...)) chain at the h5 resolution must
    # return a focal of exactly 984/3 = 328 (the correctly-rescaled focal).
    # Inlined to avoid a torch dep just for the formula:
    #   fov2focal(fov, pixels) = pixels / (2 * tan(fov / 2))
    #   focal2fov(f,   pixels) = 2 * arctan(pixels / (2 * f))
    f_implied_at_640 = 640.0 / (2.0 * math.tan(float(fovx_full[0]) / 2.0))
    np.testing.assert_allclose(f_implied_at_640, 984.0 / 3.0, atol=1e-9,
                                err_msg="reader-side focal at h5 resolution doesn't match the LLFF focal / 3")
    # The FoVY the reader derives from (FovX, h5 H, h5 W) must match too.
    fovy_implied = 2.0 * math.atan(360.0 / (2.0 * f_implied_at_640))
    np.testing.assert_allclose(fovy_implied, float(fovy_h5[0]), atol=1e-9)
