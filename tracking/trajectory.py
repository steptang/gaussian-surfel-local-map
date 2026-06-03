"""Stage G: assemble per-transition rigid poses into per-object trajectories.

Definitions:

* A *per-transition pose* ``T_t->t+1`` aligns an object's surfels at
  time t to the same object's surfels at time t+1, in the world frame
  (Stage F's output).
* An *absolute trajectory* anchors the object at its first observed
  timestep as the identity and composes all subsequent per-transition
  poses. So ``pose_at_timestep[i] @ src_at_first_seen ~ src_at_i``.

A planner querying "where is this object at time t" reads
``ObjectTrajectory.query_at_time(t)``, which interpolates between the
two surrounding observed poses (SLERP for rotation, linear for
translation) or extrapolates by constant velocity beyond the observed
range. The extrapolation policy is exposed as a config knob so the
planner can choose: clamp, constant-velocity, or raise.

Static background is handled outside this module -- it isn't a
trajectory, it's just the surfel snapshot of one chosen anchor
timestep (the persistent map). ``persistent_static_map`` exposes that
data structure for completeness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .sequence import SurfelSnapshot


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Quaternion SLERP, (w, x, y, z) convention."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    omega = np.arccos(dot)
    sin_o = np.sin(omega)
    return (np.sin((1 - t) * omega) / sin_o) * q0 + (np.sin(t * omega) / sin_o) * q1


def _R_to_quat(R: np.ndarray) -> np.ndarray:
    """(3, 3) rotation -> (w, x, y, z) unit quaternion (Shepperd's method)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """(w, x, y, z) -> (3, 3)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _interpolate_se3(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    """SE(3) interpolation: SLERP on R, linear on t. ``alpha`` in [0, 1]."""
    R0, t0 = T0[:3, :3], T0[:3, 3]
    R1, t1 = T1[:3, :3], T1[:3, 3]
    q0 = _R_to_quat(R0)
    q1 = _R_to_quat(R1)
    qi = _slerp(q0, q1, alpha)
    Ri = _quat_to_R(qi)
    ti = (1 - alpha) * t0 + alpha * t1
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = Ri
    out[:3, 3] = ti
    return out


@dataclass
class ObjectTrajectory:
    """Per-object SE(3) trajectory anchored at the first observed timestep.

    ``timesteps[i]`` is the integer frame index of the i-th observation;
    ``poses[i]`` is the absolute pose at that timestep s.t.
    ``poses[i] @ x0 = x_i`` for x0 expressed at ``timesteps[0]``.
    """
    object_id: int                       # stable across the sequence
    timesteps: list[int]
    poses: list[np.ndarray] = field(default_factory=list)   # list of (4, 4)

    def __post_init__(self):
        if len(self.timesteps) != len(self.poses):
            raise ValueError(
                f"timesteps ({len(self.timesteps)}) and poses ({len(self.poses)}) "
                "must be the same length"
            )

    def query_at_time(self, t: float,
                      extrapolation: str = "constant_velocity") -> np.ndarray:
        """Pose interpolated/extrapolated at fractional time t.

        ``extrapolation`` controls behaviour outside the observed range:
            "clamp"             -- return the nearest endpoint pose.
            "constant_velocity" -- extrapolate using the last observed
                                   transition as a velocity (planner-
                                   friendly default).
            "raise"             -- raise ValueError.
        """
        ts = self.timesteps
        if not ts:
            raise ValueError("empty trajectory")
        if t <= ts[0]:
            return self._handle_before(t, extrapolation)
        if t >= ts[-1]:
            return self._handle_after(t, extrapolation)

        # Find bracket [ts[i], ts[i+1]] containing t.
        for i in range(len(ts) - 1):
            if ts[i] <= t <= ts[i + 1]:
                span = ts[i + 1] - ts[i] or 1
                alpha = (t - ts[i]) / span
                return _interpolate_se3(self.poses[i], self.poses[i + 1], alpha)
        # Shouldn't reach here.
        raise RuntimeError(f"trajectory bracket lookup failed for t={t}")

    def _handle_before(self, t: float, mode: str) -> np.ndarray:
        if mode == "clamp" or len(self.poses) < 2:
            return np.array(self.poses[0])
        if mode == "raise":
            raise ValueError(f"t={t} before first observed timestep {self.timesteps[0]}")
        # constant_velocity: extrapolate backward using the first transition.
        delta = self.timesteps[0] - t
        span = self.timesteps[1] - self.timesteps[0] or 1
        alpha = -delta / span
        return _interpolate_se3(self.poses[0], self.poses[1], alpha)

    def _handle_after(self, t: float, mode: str) -> np.ndarray:
        if mode == "clamp" or len(self.poses) < 2:
            return np.array(self.poses[-1])
        if mode == "raise":
            raise ValueError(f"t={t} after last observed timestep {self.timesteps[-1]}")
        # constant_velocity: extrapolate forward.
        delta = t - self.timesteps[-1]
        span = self.timesteps[-1] - self.timesteps[-2] or 1
        alpha = 1.0 + delta / span
        return _interpolate_se3(self.poses[-2], self.poses[-1], alpha)


def assemble_trajectory(
    object_id: int,
    initial_timestep: int,
    transitions: list[tuple[int, np.ndarray]],
) -> ObjectTrajectory:
    """Compose per-transition poses into an absolute trajectory.

    Args:
        object_id: stable id (matches the result of Stage E across the sequence)
        initial_timestep: the timestep this trajectory is anchored at (identity pose)
        transitions: list of (target_timestep, T_prev_to_target) ordered by time

    The 0th pose is the identity at ``initial_timestep``; subsequent
    poses compose the transitions: ``pose_at[i] = transitions[i-1].T @ pose_at[i-1]``.
    """
    timesteps = [int(initial_timestep)]
    poses = [np.eye(4, dtype=np.float64)]
    for target_t, T_step in transitions:
        if target_t <= timesteps[-1]:
            raise ValueError(
                f"transitions must be ordered with strictly increasing target_t; "
                f"got {target_t} after {timesteps[-1]}"
            )
        poses.append(T_step @ poses[-1])
        timesteps.append(int(target_t))
    return ObjectTrajectory(object_id=object_id, timesteps=timesteps, poses=poses)


@dataclass(frozen=True)
class PersistentStaticMap:
    """The static background, captured at one anchor timestep.

    Stored as a SurfelSnapshot for direct reuse with the existing
    rendering / visualisation tools, plus the mask of surfels deemed
    "static" by Stage D so the dynamic-side trajectory work can be
    re-applied on top of this map by callers.
    """
    snapshot: SurfelSnapshot
    static_surfel_indices: np.ndarray


def persistent_static_map(snap: SurfelSnapshot, object_ids: np.ndarray,
                          static_object_ids: list[int]) -> PersistentStaticMap:
    """Slice the anchor snapshot down to surfels in the static clusters."""
    mask = np.zeros(snap.n_surfels, dtype=bool)
    for oid in static_object_ids:
        mask |= (object_ids == oid)
    return PersistentStaticMap(
        snapshot=snap,
        static_surfel_indices=np.flatnonzero(mask).astype(np.int64),
    )
