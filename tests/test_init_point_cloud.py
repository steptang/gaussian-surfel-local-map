"""Tests for the Stage A init-point-cloud seeder.

The seeder writes a ``points3d.ply`` whose points are sampled
uniformly inside the union of the cameras' frustums. These tests pin:

* the LSQ convergence helper (still used by diagnostics) gets the
  geometric answer right -- catches axis / sign drift in
  ``compute_camera_convergence``;
* every sampled init point projects to the canonical image extent of
  at least one camera and sits inside its [near, far] depth range --
  the central contract of ``init_point_cloud_in_frustums``;
* the writer round-trips the chosen points through PLY without
  drift, and the writer integration emits a points3d.ply whose
  centroid is in the right neighbourhood.

Run from repo root with: pytest tests/test_init_point_cloud.py -v
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.data.init_point_cloud import (
    compute_camera_convergence,
    init_point_cloud_in_frustums,
    write_points3d_ply,
)


# ---------- helpers ----------

def _camera_looking_at(eye: np.ndarray, target: np.ndarray,
                        world_up: np.ndarray = None) -> np.ndarray:
    """Build a (4, 4) OpenCV c2w at ``eye`` pointing at ``target``."""
    if world_up is None:
        world_up = np.array([0.0, 0.0, 1.0])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, world_up)
    right /= max(np.linalg.norm(right), 1e-12)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward], axis=1)
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = eye
    return c2w


def _ring_of_cameras(target: np.ndarray, n: int = 8,
                       radius: float = 4.0, height: float = 0.0,
                       world_up: np.ndarray = None) -> np.ndarray:
    if world_up is None:
        world_up = np.array([0.0, 0.0, 1.0])
    out = np.zeros((n, 4, 4))
    for i in range(n):
        theta = 2 * math.pi * i / n
        eye = np.array([radius * math.cos(theta),
                        radius * math.sin(theta),
                        height])
        out[i] = _camera_looking_at(eye, target, world_up)
    return out


def _project_to_camera(world_pts: np.ndarray, c2w: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Project (M, 3) world points into camera space. Returns
    (cam_xy, cam_z) where cam_xy is (M, 2) and cam_z is (M,)."""
    w2c = np.linalg.inv(c2w)
    homog = np.concatenate([world_pts, np.ones((world_pts.shape[0], 1))], axis=1)
    cam_h = homog @ w2c.T
    return cam_h[:, :2], cam_h[:, 2]


# ---------- compute_camera_convergence (diagnostic helper) ----------

def test_convergence_recovers_target_for_converging_rig():
    target = np.array([0.5, 0.3, 1.2])
    c2ws = _ring_of_cameras(target, n=8, radius=4.0)
    conv, rmse = compute_camera_convergence(c2ws)
    np.testing.assert_allclose(conv, target, atol=1e-9)
    assert rmse < 1e-9


def test_convergence_with_parallel_cameras_has_large_rmse():
    """Parallel-looking rigs have ill-defined LSQ convergence -- the
    rationale for switching the init from "centred at convergence" to
    "uniform in frustums". RMSE should reflect the ill-conditioning."""
    c2ws = np.zeros((4, 4, 4))
    for i in range(4):
        c2ws[i] = _camera_looking_at(
            eye=np.array([float(i), 0.0, 0.0]),
            target=np.array([float(i), 0.0, 100.0]),
            world_up=np.array([0.0, 1.0, 0.0]),
        )
    _, rmse = compute_camera_convergence(c2ws)
    assert rmse > 0.1


def test_convergence_rejects_bad_shape():
    with pytest.raises(ValueError):
        compute_camera_convergence(np.zeros((3, 3)))


# ---------- init_point_cloud_in_frustums ----------

def test_init_every_point_lies_in_some_camera_frustum():
    """Strict contract: for every sampled point, there must exist at
    least one camera whose frustum contains it (depth in [near, far]
    AND |x_ndc|, |y_ndc| <= 1)."""
    target = np.array([0.0, 0.0, 5.0])
    c2ws = _ring_of_cameras(target, n=6, radius=4.0, height=2.0)
    fov_x = np.full(6, math.radians(60.0))
    fov_y = np.full(6, math.radians(45.0))
    bounds = np.tile(np.array([1.0, 20.0]), (6, 1))
    xyz, _, _ = init_point_cloud_in_frustums(
        c2ws, fov_x, fov_y, bounds, n_pts=2_000,
        rng=np.random.default_rng(0),
    )
    # For each point, check at least one camera sees it.
    n_in = 0
    eps = 1e-4
    for pt in xyz:
        for i in range(6):
            cam_xy, cam_z = _project_to_camera(pt[None, :], c2ws[i])
            if cam_z[0] < bounds[i, 0] - eps or cam_z[0] > bounds[i, 1] + eps:
                continue
            half_w = cam_z[0] * math.tan(fov_x[i] / 2)
            half_h = cam_z[0] * math.tan(fov_y[i] / 2)
            if abs(cam_xy[0, 0]) <= half_w + eps and abs(cam_xy[0, 1]) <= half_h + eps:
                n_in += 1
                break
    # The construction should make this exact: every point in at least one frustum.
    assert n_in == xyz.shape[0], f"only {n_in}/{xyz.shape[0]} points lie in some camera frustum"


def test_init_bounds_respect_near_floor():
    """near values <= 0 should be clamped, not crash."""
    c2ws = _ring_of_cameras(np.zeros(3), n=3, radius=2.0)
    bounds = np.array([[-0.1, 5.0], [0.0, 5.0], [0.5, 5.0]])
    xyz, _, meta = init_point_cloud_in_frustums(
        c2ws, np.full(3, 1.0), np.full(3, 1.0),
        bounds, n_pts=500, near_floor=0.05,
        rng=np.random.default_rng(0),
    )
    assert np.isfinite(xyz).all()
    assert meta["near_used"].min() >= 0.05


def test_init_far_ceiling_caps_unrealistic_far():
    c2ws = _ring_of_cameras(np.zeros(3), n=3, radius=2.0)
    bounds = np.tile(np.array([1.0, 1e6]), (3, 1))
    xyz, _, meta = init_point_cloud_in_frustums(
        c2ws, np.full(3, 1.0), np.full(3, 1.0),
        bounds, n_pts=500, far_ceiling=50.0,
        rng=np.random.default_rng(0),
    )
    assert meta["far_used"].max() <= 50.0
    # All sampled depths should respect the capped far.
    # Reconstruct depth from each point: depth = (pt - cam_pos) . forward.
    positions = c2ws[:, :3, 3]
    forwards = c2ws[:, :3, 2]
    for pt in xyz:
        depths = ((pt[None, :] - positions) * forwards).sum(axis=1)
        # Some camera sees it within the capped range.
        in_range = np.any((depths >= 1.0) & (depths <= 50.0 + 1e-3))
        assert in_range


def test_init_rejects_bad_bounds_shape():
    c2ws = _ring_of_cameras(np.zeros(3), n=3, radius=2.0)
    with pytest.raises(ValueError):
        init_point_cloud_in_frustums(
            c2ws, np.full(3, 1.0), np.full(3, 1.0),
            bounds=np.zeros((5, 2)), n_pts=10,
        )


def test_init_log_depth_biases_toward_near():
    """Log-uniform depth should produce more close-to-near samples than
    uniform-in-depth. Specifically, for a single camera with [near, far]
    spanning two orders of magnitude, the median sampled depth under
    log should sit at the geometric mean (sqrt(near*far)), while under
    uniform it sits at the arithmetic mean ((near+far)/2)."""
    c2ws = _ring_of_cameras(np.zeros(3), n=1, radius=1.0)
    bounds = np.array([[0.1, 10.0]])
    n = 20_000
    xyz_unif, _, _ = init_point_cloud_in_frustums(
        c2ws, np.full(1, math.radians(60)), np.full(1, math.radians(60)),
        bounds, n_pts=n, depth_distribution="uniform",
        rng=np.random.default_rng(0),
    )
    xyz_log, _, _ = init_point_cloud_in_frustums(
        c2ws, np.full(1, math.radians(60)), np.full(1, math.radians(60)),
        bounds, n_pts=n, depth_distribution="log",
        rng=np.random.default_rng(0),
    )
    cam_pos = c2ws[0, :3, 3]
    forward = c2ws[0, :3, 2]
    depths_unif = ((xyz_unif - cam_pos) * forward).sum(axis=1)
    depths_log = ((xyz_log - cam_pos) * forward).sum(axis=1)
    median_unif = float(np.median(depths_unif))
    median_log = float(np.median(depths_log))
    geo_mean = math.sqrt(0.1 * 10.0)        # = 1.0
    arith_mean = (0.1 + 10.0) / 2           # = 5.05
    assert abs(median_unif - arith_mean) < 0.4, f"uniform median {median_unif} != arithmetic {arith_mean}"
    assert abs(median_log - geo_mean) < 0.15, f"log median {median_log} != geometric {geo_mean}"
    # And log strongly biases toward near.
    assert median_log < median_unif


def test_init_rejects_unknown_depth_distribution():
    c2ws = _ring_of_cameras(np.zeros(3), n=2, radius=1.0)
    with pytest.raises(ValueError):
        init_point_cloud_in_frustums(
            c2ws, np.full(2, 1.0), np.full(2, 1.0),
            bounds=np.tile([1.0, 5.0], (2, 1)),
            n_pts=10, depth_distribution="cubic",
        )


def test_init_is_deterministic_with_fixed_rng():
    c2ws = _ring_of_cameras(np.zeros(3), n=4, radius=2.0)
    bounds = np.tile(np.array([0.5, 10.0]), (4, 1))
    xyz_a, _, _ = init_point_cloud_in_frustums(
        c2ws, np.full(4, 1.0), np.full(4, 1.0), bounds, n_pts=300,
        rng=np.random.default_rng(42),
    )
    xyz_b, _, _ = init_point_cloud_in_frustums(
        c2ws, np.full(4, 1.0), np.full(4, 1.0), bounds, n_pts=300,
        rng=np.random.default_rng(42),
    )
    np.testing.assert_array_equal(xyz_a, xyz_b)


def test_init_covers_all_cameras_roughly_evenly():
    c2ws = _ring_of_cameras(np.zeros(3), n=10, radius=3.0)
    bounds = np.tile(np.array([0.5, 5.0]), (10, 1))
    _, _, meta = init_point_cloud_in_frustums(
        c2ws, np.full(10, 1.0), np.full(10, 1.0), bounds, n_pts=10_000,
        rng=np.random.default_rng(0),
    )
    per_cam = meta["per_camera_count"]
    expected = 10_000 / 10
    # All cameras should see roughly 1000 points each (uniform draw).
    assert per_cam.min() > 0.7 * expected
    assert per_cam.max() < 1.3 * expected


# ---------- write_points3d_ply ----------

def test_write_points3d_ply_round_trip(tmp_path):
    plyfile = pytest.importorskip("plyfile")
    PlyData = plyfile.PlyData
    n = 500
    rng = np.random.default_rng(0)
    xyz = rng.uniform(-1, 1, size=(n, 3))
    rgb_u8 = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    path = tmp_path / "points3d.ply"
    write_points3d_ply(str(path), xyz, rgb_u8)
    assert path.exists()
    ply = PlyData.read(str(path))
    el = ply.elements[0]
    assert el.count == n
    pos = np.stack([el["x"], el["y"], el["z"]], axis=1)
    col = np.stack([el["red"], el["green"], el["blue"]], axis=1)
    nrm = np.stack([el["nx"], el["ny"], el["nz"]], axis=1)
    np.testing.assert_allclose(pos, xyz.astype(np.float32), atol=1e-6)
    np.testing.assert_array_equal(col.astype(np.uint8), rgb_u8)
    np.testing.assert_array_equal(nrm, np.zeros_like(nrm))


def test_write_points3d_ply_creates_parent_dir(tmp_path):
    pytest.importorskip("plyfile")
    path = tmp_path / "nested" / "subdir" / "points3d.ply"
    xyz = np.zeros((10, 3))
    rgb = np.zeros((10, 3), dtype=np.uint8)
    write_points3d_ply(str(path), xyz, rgb)
    assert path.exists()


# ---------- writer integration ----------

def test_writer_emits_points3d_ply_inside_camera_frustums(tmp_path):
    pytest.importorskip("h5py")
    pytest.importorskip("plyfile")
    plyfile = pytest.importorskip("plyfile")
    from PIL import Image
    from tracking.data.write_scene import WriteOptions, write_timestep
    PlyData = plyfile.PlyData

    target = np.array([0.0, 0.0, 5.0])
    c2ws_opencv = _ring_of_cameras(target, n=6, radius=3.0, height=0.0)
    # 2DGS storage: R = w2c[:3,:3].T, T = w2c[:3,3].
    w2c = np.linalg.inv(c2ws_opencv)
    R_stored = np.transpose(w2c[:, :3, :3], axes=(0, 2, 1))
    T_stored = w2c[:, :3, 3]
    H, W = 60, 80
    fovx = math.radians(60.0)
    bounds = np.tile(np.array([1.0, 12.0]), (6, 1))

    class _FakeMeta:
        n_cams = 6
        height = H
        width = W
        FovX = np.full(6, fovx)
        R = R_stored
        T = T_stored
        bounds_arr = bounds  # not the field name; set below

    class _FakeScene:
        meta = _FakeMeta()
        def read_timestep(self, t):
            return {
                "rgb": np.zeros((6, H, W, 3), dtype=np.uint8),
                "fg_mask": np.zeros((6, H, W), dtype=bool),
            }

    # DMVMeta's actual field name is ``bounds``.
    _FakeScene.meta.bounds = bounds

    options = WriteOptions(
        work_root=str(tmp_path),
        write_masks=False,
        init_points3d_ply=True,
        init_n_pts=2_000,
        init_seed=0,
    )
    write_timestep(0, _FakeScene(), options)
    ply_path = tmp_path / "timestep_00000" / "points3d.ply"
    assert ply_path.exists()

    ply = PlyData.read(str(ply_path))
    pos = np.stack([ply.elements[0]["x"], ply.elements[0]["y"], ply.elements[0]["z"]], axis=1)
    # Every emitted point must lie in at least one camera frustum.
    fov_y = 2.0 * math.atan(H / (2.0 * (W / (2.0 * math.tan(fovx / 2.0)))))
    n_in = 0
    eps = 1e-3
    for pt in pos:
        for i in range(6):
            cam_xy, cam_z = _project_to_camera(pt[None, :], c2ws_opencv[i])
            if cam_z[0] < bounds[i, 0] - eps or cam_z[0] > bounds[i, 1] + eps:
                continue
            half_w = cam_z[0] * math.tan(fovx / 2.0)
            half_h = cam_z[0] * math.tan(fov_y / 2.0)
            if abs(cam_xy[0, 0]) <= half_w + eps and abs(cam_xy[0, 1]) <= half_h + eps:
                n_in += 1
                break
    assert n_in == pos.shape[0], (
        f"emitted ply has {pos.shape[0] - n_in} points outside every camera frustum"
    )


def test_writer_skips_points3d_ply_when_disabled(tmp_path):
    pytest.importorskip("h5py")
    from tracking.data.write_scene import WriteOptions, write_timestep

    target = np.array([0.0, 0.0, 5.0])
    c2ws = _ring_of_cameras(target, n=4, radius=2.0)
    w2c = np.linalg.inv(c2ws)
    R_stored = np.transpose(w2c[:, :3, :3], axes=(0, 2, 1))
    T_stored = w2c[:, :3, 3]
    H, W = 30, 40
    fovx = math.radians(60.0)
    bounds = np.tile(np.array([1.0, 10.0]), (4, 1))

    class _FakeMeta:
        n_cams = 4
        height = H
        width = W
        FovX = np.full(4, fovx)
        R = R_stored
        T = T_stored
        bounds = None
    _FakeMeta.bounds = bounds

    class _FakeScene:
        meta = _FakeMeta()
        def read_timestep(self, t):
            return {
                "rgb": np.zeros((4, H, W, 3), dtype=np.uint8),
                "fg_mask": np.zeros((4, H, W), dtype=bool),
            }

    options = WriteOptions(
        work_root=str(tmp_path),
        write_masks=False,
        init_points3d_ply=False,
    )
    write_timestep(0, _FakeScene(), options)
    assert not (tmp_path / "timestep_00000" / "points3d.ply").exists()
