"""The B <-> C-G interface: a sequence of per-timestep surfel snapshots.

Stage B writes per-timestep point_cloud.ply files. Stages C-G consume
them through this module's loaders. Snapshots are frozen dataclasses so
downstream stages can't accidentally mutate Stage B's output.

PLY layout matches scene/gaussian_model.py:save_ply (the rest of the
2DGS pipeline's checkpoint format):

    x, y, z, nx, ny, nz                                -- (nx, ny, nz are zeros)
    f_dc_{0..2}                                         -- DC SH (3 channels)
    f_rest_{...}                                        -- higher-order SH (variable)
    opacity
    scale_{0..1}                                        -- 2DGS uses 2 scales
    rot_{0..3}                                          -- quaternion (w, x, y, z)
    sem_{0..K-1}                                        -- per-surfel semantic feature

We do NOT depend on scene.gaussian_model.GaussianModel at load time
(that would force CUDA + torch on a pure-numpy analysis pipeline);
instead we parse the same PLY format directly with plyfile.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Iterator, Optional

import numpy as np
# plyfile is imported lazily inside load_snapshot_from_ply -- the
# dataclass-only paths (synthetic test, tracking-only flows) don't
# need it.


@dataclass(frozen=True)
class SurfelSnapshot:
    """One per-timestep frozen view of a 2DGS reconstruction.

    Per-surfel arrays are all (N, ...) and indexed in the same order as
    the surfel index in the PLY file (also the same order
    GaussianModel uses internally during one training run).
    """
    timestep: int
    xyz: np.ndarray           # (N, 3) float32 -- world-space surfel centres
    rotation_quat: np.ndarray # (N, 4) float32 -- (w, x, y, z), un-normalised
    scaling_2d_log: np.ndarray# (N, 2) float32 -- log-scales (raw, pre-activation)
    opacity_logit: np.ndarray # (N,)   float32 -- raw, pre-sigmoid
    rgb_dc: np.ndarray        # (N, 3) float32 -- SH DC coefficients (RGB-ish)
    semantic: np.ndarray      # (N, K) float32 -- zero matrix if scene was trained without semantics
    model_path: str = ""      # source dir for traceability / re-rendering
    extras: dict = field(default_factory=dict)
                              # downstream-added per-surfel fields:
                              #   "fg_score": (N,) float32 in [0, 1]   (Stage D)
                              #   "object_id": (N,) int32 (-1 == noise) (Stage C)

    @property
    def n_surfels(self) -> int:
        return int(self.xyz.shape[0])

    def with_extras(self, **kwargs) -> "SurfelSnapshot":
        """Return a copy with additional / replaced extras."""
        new_extras = dict(self.extras)
        new_extras.update(kwargs)
        return replace(self, extras=new_extras)


def _quat_to_rotmat_np(quats: np.ndarray) -> np.ndarray:
    """Per-surfel (w, x, y, z) quaternion -> (N, 3, 3) rotation matrix.

    Mirrors utils.general_utils.build_rotation but on CPU/numpy. The
    quaternion is auto-normalised; this is the same convention 2DGS
    uses internally for the surfel's local frame.
    """
    norm = np.sqrt((quats * quats).sum(axis=1, keepdims=True))
    q = quats / np.maximum(norm, 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    N = q.shape[0]
    R = np.empty((N, 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def surfel_normals(snap: SurfelSnapshot) -> np.ndarray:
    """Per-surfel outward normal: third column of R(quat).

    2DGS surfels are 2D disks; the local +z axis (third basis vector of
    the per-surfel rotation matrix) is the disk normal. Returns (N, 3)
    unit vectors.
    """
    R = _quat_to_rotmat_np(snap.rotation_quat.astype(np.float64))
    n = R[:, :, 2].astype(np.float32)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return n


def load_snapshot_from_ply(
    ply_path: str,
    timestep: int,
    model_path: str = "",
) -> SurfelSnapshot:
    """Read a 2DGS-format PLY into a SurfelSnapshot. Pure numpy/CPU."""
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY not found: {ply_path}")
    from plyfile import PlyData    # lazy: see module top
    plydata = PlyData.read(ply_path)
    el = plydata.elements[0]

    xyz = np.stack([np.asarray(el["x"]), np.asarray(el["y"]), np.asarray(el["z"])], axis=1).astype(np.float32)
    opacity = np.asarray(el["opacity"]).astype(np.float32)
    f_dc = np.stack(
        [np.asarray(el["f_dc_0"]), np.asarray(el["f_dc_1"]), np.asarray(el["f_dc_2"])],
        axis=1,
    ).astype(np.float32)

    scale_names = sorted(
        [p.name for p in el.properties if p.name.startswith("scale_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    scaling = np.stack([np.asarray(el[n]) for n in scale_names], axis=1).astype(np.float32)

    rot_names = sorted(
        [p.name for p in el.properties if p.name.startswith("rot")],
        key=lambda x: int(x.split("_")[-1]),
    )
    rotation = np.stack([np.asarray(el[n]) for n in rot_names], axis=1).astype(np.float32)

    sem_names = sorted(
        [p.name for p in el.properties if p.name.startswith("sem_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    if sem_names:
        semantic = np.stack([np.asarray(el[n]) for n in sem_names], axis=1).astype(np.float32)
    else:
        semantic = np.zeros((xyz.shape[0], 0), dtype=np.float32)

    return SurfelSnapshot(
        timestep=timestep,
        xyz=xyz,
        rotation_quat=rotation,
        scaling_2d_log=scaling,
        opacity_logit=opacity,
        rgb_dc=f_dc,
        semantic=semantic,
        model_path=model_path,
    )


def find_latest_ply(model_dir: str) -> Optional[str]:
    """Return the highest-iteration point_cloud.ply under model_dir, or None.

    Matches the layout train.py writes: <model_dir>/point_cloud/iteration_N/point_cloud.ply.
    """
    pc_root = os.path.join(model_dir, "point_cloud")
    if not os.path.isdir(pc_root):
        return None
    candidates = []
    for name in os.listdir(pc_root):
        if name.startswith("iteration_"):
            try:
                it = int(name.split("_", 1)[1])
            except ValueError:
                continue
            full = os.path.join(pc_root, name, "point_cloud.ply")
            if os.path.exists(full):
                candidates.append((it, full))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


@dataclass
class SurfelSequence:
    """A list of per-timestep snapshots, sorted by timestep ascending."""
    snapshots: list[SurfelSnapshot]

    def __post_init__(self):
        self.snapshots = sorted(self.snapshots, key=lambda s: s.timestep)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __iter__(self) -> Iterator[SurfelSnapshot]:
        return iter(self.snapshots)

    def __getitem__(self, i: int) -> SurfelSnapshot:
        return self.snapshots[i]

    @classmethod
    def from_out_root(cls, out_root: str) -> "SurfelSequence":
        """Discover every <out_root>/timestep_NNNNN/point_cloud/.../point_cloud.ply."""
        if not os.path.isdir(out_root):
            raise FileNotFoundError(f"out_root not found: {out_root}")
        snaps: list[SurfelSnapshot] = []
        for name in sorted(os.listdir(out_root)):
            if not name.startswith("timestep_"):
                continue
            full = os.path.join(out_root, name)
            if not os.path.isdir(full):
                continue
            try:
                t = int(name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            ply = find_latest_ply(full)
            if ply is None:
                print(f"[SurfelSequence] no PLY under {full}; skipping")
                continue
            snaps.append(load_snapshot_from_ply(ply, timestep=t, model_path=full))
        if not snaps:
            raise RuntimeError(f"no timestep PLYs found under {out_root}")
        return cls(snapshots=snaps)
