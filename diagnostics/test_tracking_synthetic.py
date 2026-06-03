"""End-to-end ground-truth test for tracking Stages C-G.

The strong correctness check the brief asks for: build a synthetic
``SurfelSequence`` with KNOWN ground-truth motion (a rigid cube
translating + rotating across timesteps with a static background
cluster), run every stage of the tracking pipeline, and assert each
stage's output matches the ground truth.

This is independent of Stages A and B: no real reconstruction, no
rasterizer, no CUDA. The whole thing runs on CPU under pytest. If
Stages C-G are correct, this passes; if any one regresses, this
fails before the real reconstruction pipeline ever runs.

Stages exercised:
    C. Cluster surfels into objects (DBSCAN on joint xyz + semantic).
    D. Classify clusters static/dynamic from per-surfel fg_score.
    E. Associate dynamic clusters across consecutive timesteps.
    F. Recover the per-transition rigid SE(3) via Open3D ICP.
    G. Assemble the trajectory and confirm it matches GT.

Run from repo root with:
    pytest diagnostics/test_tracking_synthetic.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.sequence import SurfelSnapshot, SurfelSequence
from tracking.cluster import ClusterConfig, cluster_stats, cluster_surfels
from tracking.classify_dynamic import classify_clusters
from tracking.associate import AssociationConfig, associate_clusters
from tracking.pose_icp import IcpConfig, estimate_rigid_transform, apply_se3
from tracking.trajectory import assemble_trajectory


# Open3D guard at use-site, NOT module level. test_cdeg_without_icp /
# test_extrapolation_modes run anywhere; only the full ICP test skips
# when Open3D isn't installed (e.g., Python 3.13 on macOS).
def _require_open3d():
    return pytest.importorskip("open3d")


# ----------------- synthetic-data construction --------------------------

def _normal_to_quat(normal: np.ndarray) -> np.ndarray:
    """Quaternion whose rotation maps +z to the given (unit) normal."""
    n = normal / max(float(np.linalg.norm(normal)), 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, n)
    s = float(np.linalg.norm(v))
    c = float(np.dot(z, n))
    if s < 1e-9:
        # Parallel: identity if same direction, 180-about-X if opposite.
        return np.array([1.0, 0.0, 0.0, 0.0]) if c > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    axis = v / s
    angle = np.arctan2(s, c)
    half = angle / 2.0
    return np.array(
        [np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)],
        dtype=np.float64,
    )


def _make_cube_surfels(rng: np.random.Generator, n_per_face: int = 40,
                       size: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Cube surfels distributed on its 6 faces. Normals point outward.

    Returns (xyz, rotation_quat). xyz is centred at origin.
    """
    faces = [
        ( 0, 0,  1),  # +Z
        ( 0, 0, -1),  # -Z
        ( 0,  1, 0),  # +Y
        ( 0, -1, 0),  # -Y
        ( 1, 0, 0),  # +X
        (-1, 0, 0),  # -X
    ]
    xyzs, quats = [], []
    for fn in faces:
        normal = np.array(fn, dtype=np.float64)
        # Pick the two in-plane axes.
        if abs(normal[2]) > 0.5:
            u = np.array([1, 0, 0], dtype=np.float64)
            v = np.array([0, 1, 0], dtype=np.float64)
        elif abs(normal[1]) > 0.5:
            u = np.array([1, 0, 0], dtype=np.float64)
            v = np.array([0, 0, 1], dtype=np.float64)
        else:
            u = np.array([0, 1, 0], dtype=np.float64)
            v = np.array([0, 0, 1], dtype=np.float64)
        # Random 2D coords on the face, mapped into 3D via u, v offset to face plane.
        uv = rng.uniform(-size, size, size=(n_per_face, 2))
        face_xyz = normal * size + uv[:, 0:1] * u + uv[:, 1:2] * v
        xyzs.append(face_xyz)
        face_quats = np.tile(_normal_to_quat(normal), (n_per_face, 1))
        quats.append(face_quats)
    return np.concatenate(xyzs, axis=0), np.concatenate(quats, axis=0)


def _make_background_surfels(rng: np.random.Generator, n: int = 1000,
                              bbox_half: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Diffuse static surfels in a uniform cube; random outward normals."""
    xyz = rng.uniform(-bbox_half, bbox_half, size=(n, 3))
    # Random unit normals -> quats.
    n_vec = rng.normal(size=(n, 3))
    n_vec /= np.maximum(np.linalg.norm(n_vec, axis=1, keepdims=True), 1e-12)
    quats = np.stack([_normal_to_quat(v) for v in n_vec])
    return xyz, quats


def _apply_se3(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Apply a 4x4 SE(3) to (N, 3) points."""
    H = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)
    return (T @ H.T).T[:, :3]


def _rotate_quaternions(R: np.ndarray, quats: np.ndarray) -> np.ndarray:
    """Apply a 3x3 rotation to a stack of (w, x, y, z) quats.

    Composes the rotation on the LEFT: the per-surfel rotation
    represents the surfel's local frame, and applying R to the parent
    object equals left-multiplying each rotation matrix by R, or
    equivalently quat-multiplying ``R_quat * q`` (Hamilton convention).
    """
    # Build R_quat from R (Shepperd's method, inline -- avoids importing
    # trajectory._R_to_quat for clarity in the test).
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        rq = np.array([0.25 / s,
                       (m[2, 1] - m[1, 2]) * s,
                       (m[0, 2] - m[2, 0]) * s,
                       (m[1, 0] - m[0, 1]) * s])
    else:
        # Fallback: simpler path via numpy linalg eigvals isn't needed for
        # the rotations we construct (rotations come from Rodrigues with
        # tr >> -1). The cases not hit here are exercised in trajectory.py.
        raise NotImplementedError("trace<0 fallback not needed in this test")
    out = np.zeros_like(quats)
    rw, rx, ry, rz = rq
    for i, q in enumerate(quats):
        w, x, y, z = q
        out[i] = [
            rw * w - rx * x - ry * y - rz * z,
            rw * x + rx * w + ry * z - rz * y,
            rw * y - rx * z + ry * w + rz * x,
            rw * z + rx * y - ry * x + rz * w,
        ]
    # Renormalise.
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out


def _build_gt_transforms(rng: np.random.Generator, n_frames: int
                          ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-transition SE(3) and absolute SE(3) sequences for the cube.

    Choose small per-frame rotations / translations so the ICP problem
    isn't degenerate but is far from identity.

    Returns (T_per_transition[len=n_frames-1], T_absolute[len=n_frames]).
    Absolute[0] is identity at frame 0.
    """
    # Fixed small rotations about an off-axis vector, translations of
    # ~0.15 per frame. Deterministic across runs for reproducibility.
    rotations_axis_angle = []
    translations = []
    for k in range(n_frames - 1):
        axis = np.array([0.5 + 0.1 * k, 1.0, -0.3], dtype=np.float64)
        axis /= np.linalg.norm(axis)
        angle = np.deg2rad(8.0 + 0.5 * k)
        rotations_axis_angle.append((axis, angle))
        translations.append(np.array([0.10 + 0.01 * k, -0.05, 0.07 + 0.005 * k]))

    per_transition: list[np.ndarray] = []
    absolute: list[np.ndarray] = [np.eye(4, dtype=np.float64)]
    for (axis, angle), t in zip(rotations_axis_angle, translations):
        # Rodrigues' rotation matrix.
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float64)
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        T_step = np.eye(4, dtype=np.float64)
        T_step[:3, :3] = R
        T_step[:3, 3] = t
        per_transition.append(T_step)
        absolute.append(T_step @ absolute[-1])
    return per_transition, absolute


def _build_synthetic_sequence(n_frames: int = 4, seed: int = 0
                                ) -> tuple[SurfelSequence, list[np.ndarray], list[np.ndarray]]:
    """Construct a sequence with a moving cube + static background.

    Returns:
        sequence: ``SurfelSequence`` of length ``n_frames``.
        per_transition: ground-truth (R, T) for each frame->frame+1.
        absolute_poses: ground-truth global pose of the cube at each frame
                        (anchored at frame 0 = identity).
    """
    rng = np.random.default_rng(seed)
    cube_xyz, cube_quats = _make_cube_surfels(rng, n_per_face=30, size=0.5)
    bg_xyz, bg_quats = _make_background_surfels(rng, n=600, bbox_half=4.0)

    # GT semantic features: distinguish cube from bg with non-overlapping
    # high-norm vectors (cluster step needs them to disambiguate two
    # spatially-close groups). bg gets near-zero semantic so the semantic
    # term doesn't fight the spatial term.
    K = 8
    cube_semantic = np.tile(np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32), (cube_xyz.shape[0], 1))
    bg_semantic = np.tile(np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=np.float32), (bg_xyz.shape[0], 1))

    per_transition, absolute = _build_gt_transforms(rng, n_frames)

    snapshots: list[SurfelSnapshot] = []
    for t in range(n_frames):
        T_t = absolute[t]
        moved_cube_xyz = _apply_se3(T_t, cube_xyz)
        moved_cube_quats = _rotate_quaternions(T_t[:3, :3], cube_quats)

        all_xyz = np.concatenate([moved_cube_xyz, bg_xyz], axis=0).astype(np.float32)
        all_quat = np.concatenate([moved_cube_quats, bg_quats], axis=0).astype(np.float32)
        # Other surfel attributes don't affect Stages C-G; fill with zeros / ones.
        N = all_xyz.shape[0]
        snap = SurfelSnapshot(
            timestep=t,
            xyz=all_xyz,
            rotation_quat=all_quat,
            scaling_2d_log=np.zeros((N, 2), dtype=np.float32),
            opacity_logit=np.zeros(N, dtype=np.float32),
            rgb_dc=np.zeros((N, 3), dtype=np.float32),
            semantic=np.concatenate([cube_semantic, bg_semantic], axis=0).astype(np.float32),
            model_path=f"synthetic/timestep_{t:05d}",
        )

        # Ground-truth per-surfel fg score: 1 for cube, 0 for bg.
        gt_fg = np.zeros(N, dtype=np.float32)
        gt_fg[: cube_xyz.shape[0]] = 1.0
        snap = snap.with_extras(gt_fg_score=gt_fg,
                                gt_cube_indices=np.arange(cube_xyz.shape[0]),
                                gt_bg_indices=np.arange(cube_xyz.shape[0], N))
        snapshots.append(snap)

    return SurfelSequence(snapshots=snapshots), per_transition, absolute


# ----------------- the actual end-to-end test ---------------------------

def _identify_cluster_ids(snap: SurfelSnapshot, object_ids: np.ndarray
                            ) -> tuple[int, int]:
    """Map (gt_cube, gt_bg) -> (cluster_id_for_cube, cluster_id_for_bg)."""
    cube_idx = snap.extras["gt_cube_indices"]
    bg_idx = snap.extras["gt_bg_indices"]
    # The cluster id that the majority of cube surfels got assigned to.
    def majority(indices):
        from collections import Counter
        c = Counter(int(x) for x in object_ids[indices] if int(x) != -1)
        if not c:
            return -1
        return c.most_common(1)[0][0]
    return majority(cube_idx), majority(bg_idx)


def test_end_to_end_recovers_ground_truth():
    """Single, opinionated test: build the sequence, run every stage, assert GT."""
    _require_open3d()    # Stage F is the only step that needs it
    sequence, gt_per_transition, gt_absolute = _build_synthetic_sequence(n_frames=4, seed=0)
    assert len(sequence) == 4

    # ---- Stage C: cluster every snapshot. ----
    cluster_cfg = ClusterConfig(
        spatial_weight=1.0,
        semantic_weight=0.3,
        # Bg surfels span ~8 units (bbox 4); cube ~1 unit. With
        # scene_radius set to the bbox half, eps=0.05 means 5% of that
        # radius -- separates the two and yields cohesive clusters.
        eps=0.20,
        min_samples=20,
    )
    per_snapshot_ids: list[np.ndarray] = []
    per_snapshot_stats: list[dict] = []
    for snap in sequence:
        oids = cluster_surfels(snap, cluster_cfg)
        # Sanity: at least 2 non-noise clusters (cube + bg).
        n_clusters = len({int(i) for i in np.unique(oids) if int(i) != -1})
        assert n_clusters >= 2, f"timestep {snap.timestep}: only {n_clusters} clusters"
        per_snapshot_ids.append(oids)
        per_snapshot_stats.append(cluster_stats(snap, oids))

    # ---- Stage D: classify with the GT per-surfel fg score. ----
    cube_cluster_ids: list[int] = []
    bg_cluster_ids: list[int] = []
    for snap, oids in zip(sequence, per_snapshot_ids):
        labels = classify_clusters(oids, snap.extras["gt_fg_score"],
                                   cluster_fg_threshold=0.5)
        # Exactly one dynamic cluster, at least one static cluster.
        dynamic = [oid for oid, lab in labels.items() if lab == "dynamic"]
        static = [oid for oid, lab in labels.items() if lab == "static"]
        assert len(dynamic) == 1, f"expected 1 dynamic cluster, got {dynamic}"
        assert len(static) >= 1, f"expected >=1 static cluster, got {static}"
        # The dynamic cluster IS the cube cluster the GT indices live in.
        gt_cube_cluster, gt_bg_cluster = _identify_cluster_ids(snap, oids)
        assert dynamic[0] == gt_cube_cluster, (
            f"timestep {snap.timestep}: dynamic={dynamic[0]} but gt cube cluster={gt_cube_cluster}"
        )
        cube_cluster_ids.append(gt_cube_cluster)
        bg_cluster_ids.append(gt_bg_cluster)

    # ---- Stage E: associate the dynamic cube across consecutive timesteps. ----
    # With one dynamic cluster on each side, single_object_shortcut applies.
    assoc_cfg = AssociationConfig(single_object_shortcut=True)
    for t in range(len(sequence) - 1):
        prev_stats = {cube_cluster_ids[t]: per_snapshot_stats[t][cube_cluster_ids[t]]}
        curr_stats = {cube_cluster_ids[t + 1]: per_snapshot_stats[t + 1][cube_cluster_ids[t + 1]]}
        match = associate_clusters(prev_stats, curr_stats, assoc_cfg)
        assert match[cube_cluster_ids[t + 1]] == cube_cluster_ids[t]

    # ---- Stage F: ICP each transition; compare against GT. ----
    icp_cfg = IcpConfig(centroid_init=True, max_iterations=80)
    recovered_per_transition: list[np.ndarray] = []
    for t in range(len(sequence) - 1):
        src_snap = sequence[t]
        tgt_snap = sequence[t + 1]
        src_idx = per_snapshot_stats[t][cube_cluster_ids[t]]["indices"]
        tgt_idx = per_snapshot_stats[t + 1][cube_cluster_ids[t + 1]]["indices"]
        T_hat = estimate_rigid_transform(src_snap, src_idx, tgt_snap, tgt_idx, icp_cfg)
        recovered_per_transition.append(T_hat)

        # Apply recovered T to the src cluster; assert mean residual to tgt
        # is small. Comparing the matrices directly is fragile because ICP
        # produces ANY rigid alignment that matches the points; the geometric
        # check is what we actually care about.
        src_xyz = src_snap.xyz[src_idx].astype(np.float64)
        tgt_xyz = tgt_snap.xyz[tgt_idx].astype(np.float64)
        transformed = apply_se3(T_hat, src_xyz)
        # Mean over per-point nearest-neighbour distance.
        from scipy.spatial import cKDTree
        kd = cKDTree(tgt_xyz)
        dists, _ = kd.query(transformed)
        rmse = float(np.sqrt(np.mean(dists ** 2)))
        # Surface sampling resolution is ~ size / sqrt(n_per_face).
        # Tight bound: well below the cube's edge length.
        assert rmse < 0.05, f"transition {t} -> {t + 1} ICP rmse {rmse:.4f} > 0.05"

        # Also: the matrix recovered should be close to the GT matrix.
        # Compare both translation and rotation directly.
        T_gt = gt_per_transition[t]
        t_err = float(np.linalg.norm(T_hat[:3, 3] - T_gt[:3, 3]))
        R_err = T_hat[:3, :3] @ T_gt[:3, :3].T
        trace = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
        ang_err = float(np.arccos(trace))
        assert t_err < 0.02, f"transition {t} translation error {t_err:.4f}"
        assert ang_err < np.deg2rad(2.0), (
            f"transition {t} rotation error {np.rad2deg(ang_err):.3f} deg"
        )

    # ---- Stage G: chain into a trajectory, compare against GT absolute poses. ----
    transitions = list(zip(range(1, len(sequence)), recovered_per_transition))
    traj = assemble_trajectory(object_id=cube_cluster_ids[0],
                               initial_timestep=0,
                               transitions=transitions)
    assert traj.timesteps == [0, 1, 2, 3]
    for k, (recovered, gt) in enumerate(zip(traj.poses, gt_absolute)):
        t_err = float(np.linalg.norm(recovered[:3, 3] - gt[:3, 3]))
        R_err = recovered[:3, :3] @ gt[:3, :3].T
        trace = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
        ang_err = float(np.arccos(trace))
        # Chained errors accumulate; allow looser bounds than per-transition.
        assert t_err < 0.05, f"absolute pose {k} translation error {t_err:.4f}"
        assert ang_err < np.deg2rad(5.0), (
            f"absolute pose {k} rotation error {np.rad2deg(ang_err):.3f} deg"
        )


def test_cdeg_without_icp():
    """C/D/E + trajectory-chain math without Open3D.

    Uses the GT per-transition poses in place of Stage F's ICP output to
    isolate everything except the ICP step. This lets the bulk of the
    pipeline be validated in environments without Open3D (e.g., Python
    3.13 on macOS where the Open3D wheel isn't published yet).
    """
    sequence, gt_per_transition, gt_absolute = _build_synthetic_sequence(n_frames=4, seed=0)

    cluster_cfg = ClusterConfig(spatial_weight=1.0, semantic_weight=0.3,
                                eps=0.20, min_samples=20)
    per_snapshot_ids: list[np.ndarray] = []
    per_snapshot_stats: list[dict] = []
    cube_cluster_ids: list[int] = []
    for snap in sequence:
        oids = cluster_surfels(snap, cluster_cfg)
        stats = cluster_stats(snap, oids)
        labels = classify_clusters(oids, snap.extras["gt_fg_score"], 0.5)
        dynamic = [oid for oid, lab in labels.items() if lab == "dynamic"]
        assert len(dynamic) == 1
        cube_cluster_ids.append(dynamic[0])
        per_snapshot_ids.append(oids)
        per_snapshot_stats.append(stats)

    # Association across consecutive timesteps (single-object shortcut).
    for t in range(len(sequence) - 1):
        prev_stats = {cube_cluster_ids[t]: per_snapshot_stats[t][cube_cluster_ids[t]]}
        curr_stats = {cube_cluster_ids[t + 1]: per_snapshot_stats[t + 1][cube_cluster_ids[t + 1]]}
        match = associate_clusters(prev_stats, curr_stats,
                                    AssociationConfig(single_object_shortcut=True))
        assert match[cube_cluster_ids[t + 1]] == cube_cluster_ids[t]

    # Trajectory math (G) using GT transitions -- no ICP involved.
    transitions = list(zip(range(1, len(sequence)), gt_per_transition))
    traj = assemble_trajectory(object_id=cube_cluster_ids[0], initial_timestep=0,
                               transitions=transitions)
    for k, (composed, gt) in enumerate(zip(traj.poses, gt_absolute)):
        np.testing.assert_allclose(composed, gt, atol=1e-9,
                                    err_msg=f"trajectory.poses[{k}] != gt_absolute[{k}]")


def test_extrapolation_modes():
    """Quick check of trajectory.query_at_time outside the observed range."""
    from tracking.trajectory import ObjectTrajectory

    # 2-pose trajectory with a clean +1 unit translation per timestep.
    P0 = np.eye(4); P1 = np.eye(4); P1[0, 3] = 1.0; P2 = np.eye(4); P2[0, 3] = 2.0
    traj = ObjectTrajectory(object_id=1, timesteps=[0, 1, 2], poses=[P0, P1, P2])

    # interpolation: t=0.5 -> translation 0.5 along x
    pose_mid = traj.query_at_time(0.5)
    assert abs(pose_mid[0, 3] - 0.5) < 1e-9

    # clamp: t=-3 stays at the first pose
    pose_clamped = traj.query_at_time(-3.0, extrapolation="clamp")
    np.testing.assert_allclose(pose_clamped, P0)

    # constant-velocity: t=4 extrapolates two more steps -> tx=4
    pose_extrap = traj.query_at_time(4.0, extrapolation="constant_velocity")
    assert abs(pose_extrap[0, 3] - 4.0) < 1e-9
