"""End-to-end orchestrator: Stages B → C → D → E → F → G on a
reconstructed multi-timestep surfel sequence.

Per-stage inputs/outputs and the surrounding flow:

  Stage B  (already done by tracking.driver.run_per_timestep)
    Inputs:    work_root/timestep_*/{transforms_train.json, masks/, sam3/}
    Outputs:   out_root/timestep_*/point_cloud/iteration_*/point_cloud.ply

  Stage C  (cluster.py)              — cluster surfels per timestep
  Stage D  (classify_dynamic.py)     — per-surfel fg_score by projecting
                                       through the GT masks, then
                                       per-cluster static/dynamic.
  Stage E  (associate.py)            — match dynamic clusters across
                                       consecutive transitions.
  Stage F  (pose_icp.py)             — Open3D point-to-plane ICP per
                                       transition for each tracked object.
  Stage G  (trajectory.py)           — chain per-transition SE(3) into
                                       per-object absolute trajectories.

Outputs (under ``--tracking-out``):

    trajectories.json
        Per-object trajectory:
            object_id, timesteps[], poses[4x4][], per_timestep_cluster_id{}
        Anchored at the first observed timestep; pose[0] = identity.

    per_timestep/timestep_NNNNN/
        object_ids.npy       (N,) int32 -- DBSCAN labels (-1 = noise)
        fg_score.npy         (N,) float32 in [0, 1] -- Stage D output
        cluster_labels.json  {"cluster_id": "static"|"dynamic"}
        viz/ (optional, with --viz)
            clusters.ply, static_dynamic.ply

    transitions/NNNNN_to_MMMMM/
        T_object_OID.npy     (4, 4) recovered rigid pose for the object
        viz/ (optional)
            overlay_object_OID.ply

Single-object v1: the orchestrator picks the largest dynamic cluster per
timestep as the tracked object and chains it via single-object
association. The output schema already supports multi-object so adding
proper data association later doesn't require changes downstream.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .associate import AssociationConfig, associate_clusters
from .classify_dynamic import (
    FgProjectionConfig,
    classify_clusters,
    per_surfel_fg_score_from_views,
)
from .cluster import ClusterConfig, cluster_stats, cluster_surfels
from .pose_icp import IcpConfig, estimate_rigid_transform
from .orchestrator_helpers import load_views_for_sequence
from .sequence import SurfelSequence, SurfelSnapshot
from .trajectory import ObjectTrajectory, assemble_trajectory


# --- per-stage IO helpers --------------------------------------------------

def _save_per_timestep(
    tracking_out: str, snap: SurfelSnapshot,
    object_ids: np.ndarray, fg_score: np.ndarray,
    cluster_labels: dict[int, str],
    write_viz: bool,
) -> None:
    base = os.path.join(tracking_out, "per_timestep", f"timestep_{snap.timestep:05d}")
    os.makedirs(base, exist_ok=True)
    np.save(os.path.join(base, "object_ids.npy"), object_ids.astype(np.int32))
    np.save(os.path.join(base, "fg_score.npy"), fg_score.astype(np.float32))
    # Stringify keys for JSON (json requires string keys at the top level).
    label_doc = {str(int(k)): v for k, v in cluster_labels.items()}
    with open(os.path.join(base, "cluster_labels.json"), "w") as f:
        json.dump(label_doc, f, indent=2)

    if write_viz:
        from .viz import write_object_id_ply, write_static_dynamic_ply
        viz_dir = os.path.join(base, "viz")
        os.makedirs(viz_dir, exist_ok=True)
        write_object_id_ply(os.path.join(viz_dir, "clusters.ply"), snap, object_ids)
        write_static_dynamic_ply(os.path.join(viz_dir, "static_dynamic.ply"),
                                  snap, fg_score)


def _save_transition(
    tracking_out: str,
    src_snap: SurfelSnapshot, src_indices: np.ndarray,
    tgt_snap: SurfelSnapshot, tgt_indices: np.ndarray,
    persistent_object_id: int, T_step: np.ndarray,
    write_viz: bool,
) -> None:
    base = os.path.join(
        tracking_out, "transitions",
        f"{src_snap.timestep:05d}_to_{tgt_snap.timestep:05d}",
    )
    os.makedirs(base, exist_ok=True)
    np.save(os.path.join(base, f"T_object_{persistent_object_id}.npy"),
            T_step.astype(np.float64))

    if write_viz:
        from .viz import write_transformed_overlay_ply
        viz_dir = os.path.join(base, "viz")
        os.makedirs(viz_dir, exist_ok=True)
        write_transformed_overlay_ply(
            os.path.join(viz_dir, f"overlay_object_{persistent_object_id}.ply"),
            src_snap, src_indices, tgt_snap, tgt_indices, T_step,
        )


def _save_trajectories(tracking_out: str,
                        trajectories: dict[int, ObjectTrajectory],
                        per_t_cluster_ids: dict[int, dict[int, int]]) -> None:
    """Write trajectories.json.

    per_t_cluster_ids[object_id][timestep] = DBSCAN cluster id at that
    timestep -- preserved so downstream code can index into Stage C's
    per-timestep object_ids.npy without re-running matching.
    """
    doc = {
        "schema_version": 1,
        "trajectories": [],
    }
    for oid, traj in trajectories.items():
        doc["trajectories"].append({
            "object_id": int(oid),
            "timesteps": [int(t) for t in traj.timesteps],
            "poses": [p.tolist() for p in traj.poses],
            "per_timestep_cluster_id": {
                str(int(t)): int(c)
                for t, c in per_t_cluster_ids.get(oid, {}).items()
            },
        })
    out = os.path.join(tracking_out, "trajectories.json")
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)


# --- main flow -------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Knobs threaded through the orchestrator."""
    cluster: ClusterConfig
    fg_projection: FgProjectionConfig
    association: AssociationConfig
    icp: IcpConfig
    cluster_fg_threshold: float = 0.5
    write_viz: bool = False


def run_pipeline(
    out_root: str,
    work_root: str,
    tracking_out: str,
    config: PipelineConfig,
) -> int:
    """Returns the count of trajectories successfully assembled."""
    os.makedirs(tracking_out, exist_ok=True)
    sequence = SurfelSequence.from_out_root(out_root)
    print(f"[run_pipeline] loaded sequence with {len(sequence)} timesteps")
    timesteps = [snap.timestep for snap in sequence]

    print(f"[run_pipeline] loading per-timestep projection views from {work_root}")
    views_per_t = load_views_for_sequence(work_root, timesteps)
    timesteps_with_views = sorted(views_per_t.keys())
    if len(timesteps_with_views) < len(timesteps):
        missing = set(timesteps) - set(timesteps_with_views)
        print(f"[run_pipeline] missing work_root data for timesteps: {sorted(missing)}; "
              "their fg_scores will be all-zero (treated as static)")

    # ---- Pass 1: per-timestep C + D ----
    per_t_object_ids: dict[int, np.ndarray] = {}
    per_t_stats: dict[int, dict] = {}
    per_t_labels: dict[int, dict[int, str]] = {}
    for snap in sequence:
        print(f"[run_pipeline] t={snap.timestep}: clustering ({snap.n_surfels} surfels)")
        oids = cluster_surfels(snap, config.cluster)
        stats = cluster_stats(snap, oids)
        n_clusters = len(stats)
        print(f"  -> {n_clusters} clusters (noise: {int((oids == -1).sum())} surfels)")

        if snap.timestep in views_per_t:
            views = views_per_t[snap.timestep]
            fg_score, _ = per_surfel_fg_score_from_views(
                snap, views, config.fg_projection,
            )
        else:
            fg_score = np.zeros(snap.n_surfels, dtype=np.float32)
        labels = classify_clusters(oids, fg_score, config.cluster_fg_threshold)
        n_dynamic = sum(1 for lab in labels.values() if lab == "dynamic")
        print(f"  -> {n_dynamic} dynamic / {len(labels) - n_dynamic} static clusters")

        per_t_object_ids[snap.timestep] = oids
        per_t_stats[snap.timestep] = stats
        per_t_labels[snap.timestep] = labels
        _save_per_timestep(
            tracking_out, snap, oids, fg_score, labels,
            write_viz=config.write_viz,
        )

    # ---- Pass 2: E + F across consecutive transitions ----
    # Single-object v1: pick the largest dynamic cluster on each side and
    # chain. Track ONE persistent object whose object_id is anchored to
    # whatever the dynamic cluster id was in the first timestep that had
    # a dynamic cluster.
    transitions: list[tuple[int, np.ndarray]] = []
    per_object_per_t_cluster_id: dict[int, dict[int, int]] = {}
    persistent_object_id: Optional[int] = None
    anchor_timestep: Optional[int] = None
    anchor_src_indices: Optional[np.ndarray] = None  # for the overlay viz on the first transition

    for i in range(len(sequence) - 1):
        src_snap = sequence[i]
        tgt_snap = sequence[i + 1]
        src_stats = per_t_stats[src_snap.timestep]
        tgt_stats = per_t_stats[tgt_snap.timestep]
        src_labels = per_t_labels[src_snap.timestep]
        tgt_labels = per_t_labels[tgt_snap.timestep]

        src_dynamic = {oid: src_stats[oid] for oid in src_stats
                       if src_labels.get(oid) == "dynamic"}
        tgt_dynamic = {oid: tgt_stats[oid] for oid in tgt_stats
                       if tgt_labels.get(oid) == "dynamic"}
        if not src_dynamic or not tgt_dynamic:
            print(f"[run_pipeline] t={src_snap.timestep}->{tgt_snap.timestep}: "
                  f"no dynamic cluster on one side; skipping transition")
            continue

        # Pick the largest dynamic cluster on each side (v1).
        src_pick = max(src_dynamic, key=lambda oid: src_dynamic[oid]["n_surfels"])
        tgt_pick = max(tgt_dynamic, key=lambda oid: tgt_dynamic[oid]["n_surfels"])
        if len(src_dynamic) > 1 or len(tgt_dynamic) > 1:
            print(f"  warn: {len(src_dynamic)} src dynamic / "
                  f"{len(tgt_dynamic)} tgt dynamic; v1 picks largest on each side")

        # Confirm the match via the association API even though the choice
        # is forced -- this exercises the same data path multi-object will
        # use and surfaces an associate_clusters bug if there is one.
        match = associate_clusters(
            {src_pick: src_dynamic[src_pick]},
            {tgt_pick: tgt_dynamic[tgt_pick]},
            config.association,
        )
        assert match.get(tgt_pick) == src_pick, "single-object association failed"

        # First valid transition seeds the persistent object id.
        if persistent_object_id is None:
            persistent_object_id = int(src_pick)
            anchor_timestep = src_snap.timestep
            anchor_src_indices = src_dynamic[src_pick]["indices"]
            per_object_per_t_cluster_id[persistent_object_id] = {
                src_snap.timestep: int(src_pick),
            }

        # Stage F: ICP between the two clusters.
        try:
            T_step = estimate_rigid_transform(
                src_snap, src_dynamic[src_pick]["indices"],
                tgt_snap, tgt_dynamic[tgt_pick]["indices"],
                config.icp,
            )
        except Exception as e:
            print(f"  ICP failed at t={src_snap.timestep}->{tgt_snap.timestep}: {e}")
            continue

        transitions.append((tgt_snap.timestep, T_step))
        per_object_per_t_cluster_id[persistent_object_id][tgt_snap.timestep] = int(tgt_pick)
        _save_transition(
            tracking_out=tracking_out,
            src_snap=src_snap, src_indices=src_dynamic[src_pick]["indices"],
            tgt_snap=tgt_snap, tgt_indices=tgt_dynamic[tgt_pick]["indices"],
            persistent_object_id=persistent_object_id, T_step=T_step,
            write_viz=config.write_viz,
        )

    # ---- Pass 3: assemble trajectories ----
    trajectories: dict[int, ObjectTrajectory] = {}
    if persistent_object_id is not None and transitions:
        traj = assemble_trajectory(
            object_id=persistent_object_id,
            initial_timestep=anchor_timestep,
            transitions=transitions,
        )
        trajectories[persistent_object_id] = traj
        print(f"[run_pipeline] assembled trajectory for object {persistent_object_id}: "
              f"{len(traj.timesteps)} poses")
    else:
        print("[run_pipeline] no dynamic transitions; no trajectory produced")

    _save_trajectories(tracking_out, trajectories, per_object_per_t_cluster_id)
    print(f"[run_pipeline] outputs under {tracking_out}")
    return len(trajectories)


# --- CLI -------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run tracking Stages C-G end-to-end on a reconstructed sequence."
    )
    p.add_argument("--out-root", required=True,
                   help="Stage B output (contains timestep_*/point_cloud/...)")
    p.add_argument("--work-root", required=True,
                   help="Stage A output (contains timestep_*/{transforms_train.json, masks/})")
    p.add_argument("--tracking-out", required=True,
                   help="destination for trajectories.json + per_timestep/ + transitions/")

    # Stage C knobs
    p.add_argument("--cluster-spatial-weight", type=float, default=1.0)
    p.add_argument("--cluster-semantic-weight", type=float, default=0.3)
    p.add_argument("--cluster-eps", type=float, default=0.05)
    p.add_argument("--cluster-min-samples", type=int, default=50)
    p.add_argument("--cluster-subsample", type=int, default=None,
                   help="cluster on a random subset of N surfels, then 1-NN propagate "
                        "(use ~50000 for million-surfel scenes; default: no subsampling)")
    p.add_argument("--cluster-random-state", type=int, default=0)

    # Stage D knobs
    p.add_argument("--fg-min-views-visible", type=int, default=2)
    p.add_argument("--cluster-fg-threshold", type=float, default=0.5,
                   help="cluster mean fg_score >= this => dynamic")

    # Stage E knobs (single-object shortcut is the v1 default)
    p.add_argument("--assoc-spatial-weight", type=float, default=1.0)
    p.add_argument("--assoc-semantic-weight", type=float, default=0.5)
    p.add_argument("--assoc-max-cost", type=float, default=float("inf"))

    # Stage F knobs
    p.add_argument("--icp-threshold", type=float, default=None,
                   help="ICP correspondence distance; default heuristic = 20%% of src radius")
    p.add_argument("--icp-max-iterations", type=int, default=60)

    p.add_argument("--viz", action="store_true",
                   help="write per-timestep + per-transition PLYs (cluster colors, static/dynamic, ICP overlay)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = PipelineConfig(
        cluster=ClusterConfig(
            spatial_weight=args.cluster_spatial_weight,
            semantic_weight=args.cluster_semantic_weight,
            eps=args.cluster_eps,
            min_samples=args.cluster_min_samples,
            subsample=args.cluster_subsample,
            random_state=args.cluster_random_state,
        ),
        fg_projection=FgProjectionConfig(
            min_views_visible=args.fg_min_views_visible,
        ),
        association=AssociationConfig(
            spatial_weight=args.assoc_spatial_weight,
            semantic_weight=args.assoc_semantic_weight,
            max_cost=args.assoc_max_cost,
        ),
        icp=IcpConfig(
            threshold=args.icp_threshold,
            max_iterations=args.icp_max_iterations,
        ),
        cluster_fg_threshold=args.cluster_fg_threshold,
        write_viz=args.viz,
    )
    n = run_pipeline(args.out_root, args.work_root, args.tracking_out, config)
    print(f"[run_pipeline] DONE. {n} trajectory(ies) assembled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
