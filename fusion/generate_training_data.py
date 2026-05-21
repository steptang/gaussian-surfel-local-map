"""Training-data generation for the Path B fusion MLP.

Two flavours of pairs, unified on-disk schema (see SCHEMA below).

Distillation pairs (synthetic, ~100K)
    - Prior NIG states sampled from priors.json buckets, with optional
      jitter to cover post-update parameter regimes.
    - Observations sampled as a mixture of in-distribution, OOD, and
      uniform. Confidences sampled from a Beta mixture (high-conf base
      + small low-conf tail). All distribution choices are TODO'd for
      replacement with empirical VLM statistics.
    - target_local = target_asymptotic = closed_form_update(prior, obs).
    - Geometric and semantic features are zero-padded.
    - Purpose: 'imitate Path A' baseline.

Scene pairs (real, ~50K-100K per scene)
    - Pass 1: closed-form fusion (5c) over all train views; cache the
      per-surfel asymptotic posterior.
    - Pass 2: re-walk views in random order. For each (view, region,
      contributing surfel) emit one tuple with the *pre-update* state,
      observation, geometric features, semantic features. Apply the
      closed-form update in place so subsequent tuples reflect the
      growing posterior.
    - target_local = closed-form-after-this-update ('do no harm').
    - target_asymptotic = the cached all-views posterior ('extrapolate
      from partial info using surfel context').
    - 5e picks one or blends; the schema supports both.

Schema (saved as .pt with torch.save):
    schema_version:    1
    kind:              "distillation" | "scene"
    n_samples:         int
    prior_state:       (N, 4) float32  (tau, kappa, alpha, beta)
    observation:       (N, 2) float32  (value, omega post-normalisation)
    geometric_features:(N, G) float32  (zero-padded for distillation)
    semantic_features: (N, S) float32  (zero-padded for distillation)
    target_local:      (N, 4) float32  closed-form-after-this-update
    target_asymptotic: (N, 4) float32  = target_local for distillation
    has_features:      (N,)  bool      True for scene samples
    feature_meta:      dict             names + scaling info for geom features

Open questions (flagged, not answered in code):
  - Realistic VLM confidence distribution: empirical from RUGD vs assumed
    Beta mixture. Affects distillation pair quality.
  - Should out-of-distribution observation rate match real VLM error
    rate? Currently a fixed mixture.
  - Per-region surfel cap (default 64) vs uniform pixel-weighted sampling.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .fuse_surfels import (
    SurfelPhysicalState,
    _parse_region_record,
    _physical_path,
    apply_view_update,
    compute_region_contributions,
)
from .lookup import PriorTable
from .nig import closed_form_update


SCHEMA_VERSION = 1
DEFAULT_GEOMETRIC_DIM = 8       # see extract_geometric_features
DEFAULT_SEMANTIC_DIM = 32       # matches scene/gaussian_model.py SEMANTIC_DIM


# ============================================================================
# Distillation pair generation
# ============================================================================

def sample_prior_states(
    n: int,
    priors: PriorTable,
    rng: np.random.Generator,
    jitter_p: float = 0.5,
    jitter_log_scale: float = 0.3,
) -> np.ndarray:
    """Sample n prior NIG states from priors.json buckets +/- jitter.

    Returns (n, 4) float64. Each sample is drawn by:
      1. Pick a bucket uniformly at random (+ a fallback chance to pick
         the global prior).
      2. With probability `jitter_p`, multiply (kappa, alpha, beta) by
         lognormal(0, jitter_log_scale) and shift tau by a small Gaussian.
         The jitter covers post-update regimes (e.g., very high kappa
         after many observations) that the buckets themselves don't span.
    """
    buckets = list(priors._buckets.values())
    # Include the global prior as one of the candidates so its parameter
    # neighbourhood is also covered.
    pool = buckets + [{
        "tau": priors._global.tau, "kappa": priors._global.kappa,
        "alpha": priors._global.alpha, "beta": priors._global.beta,
    }]
    idx = rng.integers(0, len(pool), size=n)
    states = np.empty((n, 4), dtype=np.float64)
    for i, j in enumerate(idx):
        b = pool[j]
        states[i] = (float(b["tau"]), float(b["kappa"]),
                     float(b["alpha"]), float(b["beta"]))

    # Apply jitter to a random subset.
    mask = rng.random(n) < jitter_p
    if mask.any():
        m = mask.sum()
        scale_mult = rng.lognormal(mean=0.0, sigma=jitter_log_scale, size=(m, 3))
        states[mask, 1:] *= scale_mult       # kappa, alpha, beta
        states[mask, 0] += rng.normal(0.0, 0.05, size=m)  # tau drift
        # Keep all parameters in physically meaningful ranges.
        states[mask, 0] = np.clip(states[mask, 0], 0.01, 1.5)
        states[mask, 1] = np.clip(states[mask, 1], 0.1, 5000.0)   # kappa > 0
        states[mask, 2] = np.clip(states[mask, 2], 1.01, 5000.0)  # alpha > 1 for finite moments
        states[mask, 3] = np.clip(states[mask, 3], 1e-4, 100.0)
    return states


def sample_observations(
    n: int,
    prior_states: np.ndarray,           # (n, 4)
    priors: PriorTable,
    rng: np.random.Generator,
    mix_in_dist: float = 0.70,
    mix_ood: float = 0.25,
    # remainder is uniform
    conf_high_frac: float = 0.80,
) -> np.ndarray:
    """Sample n (value, omega) observation pairs.

    Mixture:
      * in_dist  (mix_in_dist): value drawn near the prior's tau (i.e.,
        the VLM agrees with the prior).
      * ood      (mix_ood):     value drawn from a *different* bucket's
        tau (the VLM disagrees, simulating misclassification).
      * uniform  (1 - mix_in_dist - mix_ood): value uniform on [0.05, 1.0]
        to expose the MLP to edge cases.

    Confidences are a Beta mixture: Beta(5, 2) for high-conf (mean ~0.71)
    with a Beta(1, 3) low-conf tail (mean ~0.25). Documented; replace
    with empirical VLM stats when available.
    """
    u = rng.random(n)
    values = np.empty(n, dtype=np.float64)

    # In-distribution: tau + small noise.
    in_dist_mask = u < mix_in_dist
    if in_dist_mask.any():
        m = in_dist_mask.sum()
        values[in_dist_mask] = prior_states[in_dist_mask, 0] + rng.normal(0.0, 0.05, size=m)

    # OOD: pick another bucket's tau (or perturbation thereof).
    ood_mask = (u >= mix_in_dist) & (u < mix_in_dist + mix_ood)
    if ood_mask.any():
        m = ood_mask.sum()
        buckets = list(priors._buckets.values())
        if buckets:
            taus = np.array([float(b["tau"]) for b in buckets])
            picks = rng.integers(0, len(buckets), size=m)
            values[ood_mask] = taus[picks] + rng.normal(0.0, 0.05, size=m)
        else:
            values[ood_mask] = rng.uniform(0.05, 1.0, size=m)

    # Uniform.
    uniform_mask = u >= mix_in_dist + mix_ood
    if uniform_mask.any():
        values[uniform_mask] = rng.uniform(0.05, 1.0, size=uniform_mask.sum())

    values = np.clip(values, 0.01, 1.5)

    # Confidence Beta mixture.
    high_mask = rng.random(n) < conf_high_frac
    confs = np.empty(n, dtype=np.float64)
    confs[high_mask] = rng.beta(5.0, 2.0, size=high_mask.sum())
    confs[~high_mask] = rng.beta(1.0, 3.0, size=(~high_mask).sum())

    return np.stack([values, confs], axis=1)   # (n, 2)


def _closed_form_batch(prior_states: np.ndarray, observations: np.ndarray) -> np.ndarray:
    """Vectorised closed-form update over an (N, 4) prior and (N, 2) observation."""
    tau, kappa, alpha, beta = (prior_states[:, i] for i in range(4))
    value, conf = observations[:, 0], observations[:, 1]
    new = closed_form_update(tau, kappa, alpha, beta, value, conf)
    return np.stack(new, axis=1)   # (N, 4)


def generate_distillation_pairs(
    n: int,
    priors_path: str,
    seed: int = 0,
    geometric_dim: int = DEFAULT_GEOMETRIC_DIM,
    semantic_dim: int = DEFAULT_SEMANTIC_DIM,
) -> dict:
    """Generate n distillation training pairs. Returns a dataset dict."""
    rng = np.random.default_rng(seed)
    priors = PriorTable.from_path(priors_path)
    prior_states = sample_prior_states(n, priors, rng)
    observations = sample_observations(n, prior_states, priors, rng)
    targets = _closed_form_batch(prior_states, observations)

    return _pack_dataset(
        kind="distillation",
        prior_state=prior_states,
        observation=observations,
        geometric_features=np.zeros((n, geometric_dim), dtype=np.float32),
        semantic_features=np.zeros((n, semantic_dim), dtype=np.float32),
        target_local=targets,
        target_asymptotic=targets,   # identical for distillation
        has_features=np.zeros(n, dtype=bool),
        feature_meta=_geometric_feature_meta(geometric_dim),
    )


# ============================================================================
# Scene pair generation
# ============================================================================

def _build_rotation_np(quats: np.ndarray) -> np.ndarray:
    """CPU/numpy port of utils/general_utils.build_rotation.

    quats: (N, 4) in (w, x, y, z) order; returned R has shape (N, 3, 3).
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


def _geometric_feature_meta(dim: int) -> dict:
    """Human-readable description of each geometric feature slot.

    Keep in sync with extract_geometric_features.
    """
    return {
        "dim": dim,
        "names": (
            ["normal_x", "normal_y", "normal_z",
             "pos_x_norm", "pos_y_norm", "pos_z_norm",
             "log_area", "log_opacity"]
            if dim == 8 else
            [f"f{i}" for i in range(dim)]
        ),
    }


def extract_geometric_features(
    xyz: np.ndarray,           # (N, 3)
    scaling: np.ndarray,       # (N, 2) post-activation
    rotation: np.ndarray,      # (N, 4) post-activation (unit quat)
    opacity: np.ndarray,       # (N, 1) post-activation
    scene_center: Optional[np.ndarray] = None,    # (3,)
    scene_radius: Optional[float] = None,
) -> np.ndarray:
    """Per-surfel geometric feature vector (N, 8) float32.

    Layout (see _geometric_feature_meta):
        0..2: world-space normal (3rd column of the rotation matrix; the
              local +z axis is the surfel's normal in 2DGS).
        3..5: position relative to scene center, normalised by scene
              radius. The MLP can learn that surfels near the ground vs
              ceiling behave differently.
        6:    log surfel area (log(s_u * s_v)). Larger surfels see more
              observations and aggregate more reliably.
        7:    log opacity. Translucent surfels are less load-bearing.
    """
    N = xyz.shape[0]
    R = _build_rotation_np(rotation)
    normals = R[:, :, 2]                              # (N, 3)
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    if scene_center is None:
        scene_center = xyz.mean(axis=0)
    if scene_radius is None or scene_radius <= 0:
        scene_radius = float(np.linalg.norm(xyz - scene_center, axis=1).max() + 1e-6)
    pos_norm = (xyz - scene_center) / scene_radius

    log_area = np.log(np.maximum(scaling[:, 0] * scaling[:, 1], 1e-12))
    log_opa = np.log(np.maximum(opacity[:, 0], 1e-12))

    feats = np.zeros((N, 8), dtype=np.float32)
    feats[:, 0:3] = normals.astype(np.float32)
    feats[:, 3:6] = pos_norm.astype(np.float32)
    feats[:, 6] = log_area.astype(np.float32)
    feats[:, 7] = log_opa.astype(np.float32)
    return feats


@dataclass
class _ScenePairAccumulator:
    """In-memory accumulator for one scene's training tuples.

    Lists are concatenated once at the end to avoid O(N^2) realloc.
    """
    prior_state: list
    observation: list
    geom: list
    sem: list
    target_local: list
    target_asymptotic: list

    @classmethod
    def empty(cls) -> "_ScenePairAccumulator":
        return cls([], [], [], [], [], [])

    def add(self, pre, obs, geom, sem, tgt_local, tgt_asym):
        self.prior_state.append(pre)
        self.observation.append(obs)
        self.geom.append(geom)
        self.sem.append(sem)
        self.target_local.append(tgt_local)
        self.target_asymptotic.append(tgt_asym)

    def __len__(self) -> int:
        return len(self.prior_state)

    def stack(self) -> dict:
        if len(self) == 0:
            return {k: np.zeros((0, 0), dtype=np.float32) for k in
                    ("prior_state", "observation", "geometric_features",
                     "semantic_features", "target_local", "target_asymptotic")}
        return {
            "prior_state": np.stack(self.prior_state).astype(np.float32),
            "observation": np.stack(self.observation).astype(np.float32),
            "geometric_features": np.stack(self.geom).astype(np.float32),
            "semantic_features": np.stack(self.sem).astype(np.float32),
            "target_local": np.stack(self.target_local).astype(np.float32),
            "target_asymptotic": np.stack(self.target_asymptotic).astype(np.float32),
        }


def emit_region_pairs(
    state: SurfelPhysicalState,
    asymptotic: SurfelPhysicalState,
    region_id: int,
    obs_value: float,
    vlm_confidence: float,
    region_map: np.ndarray,
    contrib_ids: np.ndarray,
    contrib_weights: np.ndarray,
    geometric_features: np.ndarray,    # (N_surfels, G)
    semantic_features: np.ndarray,     # (N_surfels, S)
    acc: _ScenePairAccumulator,
    max_per_region: int,
    rng: np.random.Generator,
) -> int:
    """Emit training tuples for one region; apply the in-place update.

    Returns the number of tuples emitted (== surfels updated, possibly
    after subsampling to max_per_region).
    """
    active, omega = compute_region_contributions(
        region_id=region_id,
        region_map=region_map,
        contrib_ids=contrib_ids,
        contrib_weights=contrib_weights,
        vlm_confidence=vlm_confidence,
        n_surfels=state.tau.size,
    )
    if active.size == 0:
        return 0

    # Compute the closed-form post-state for ALL active surfels (vectorised).
    pre_tau = state.tau[active]; pre_kappa = state.kappa[active]
    pre_alpha = state.alpha[active]; pre_beta = state.beta[active]
    post = closed_form_update(pre_tau, pre_kappa, pre_alpha, pre_beta,
                              obs_value, omega)
    post_tau, post_kappa, post_alpha, post_beta = post

    # Subsample which surfels we emit pairs for (cap memory). The state
    # update itself still applies to all active surfels — we just sample
    # which tuples to record.
    if active.size > max_per_region:
        keep = rng.choice(active.size, size=max_per_region, replace=False)
    else:
        keep = np.arange(active.size)

    for i in keep:
        sid = int(active[i])
        pre = np.array([pre_tau[i], pre_kappa[i], pre_alpha[i], pre_beta[i]], dtype=np.float32)
        tgt_local = np.array([post_tau[i], post_kappa[i], post_alpha[i], post_beta[i]], dtype=np.float32)
        tgt_asym = np.array([asymptotic.tau[sid], asymptotic.kappa[sid],
                              asymptotic.alpha[sid], asymptotic.beta[sid]],
                             dtype=np.float32)
        obs_arr = np.array([obs_value, omega[i]], dtype=np.float32)
        acc.add(pre, obs_arr, geometric_features[sid], semantic_features[sid],
                tgt_local, tgt_asym)

    # Apply the in-place update so the next region's emission sees the
    # post-update state for these surfels.
    state.tau[active] = post_tau
    state.kappa[active] = post_kappa
    state.alpha[active] = post_alpha
    state.beta[active] = post_beta
    return int(keep.size)


def generate_scene_pairs(
    scene,
    gaussians,
    pipe,
    background,
    priors_path: str,
    source_path: str,
    sam_dir: str = "sam3",
    max_samples: int = 100_000,
    max_per_region: int = 64,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Generate scene-anchored training pairs. Requires CUDA (renders views)."""
    from tqdm import tqdm
    from gaussian_renderer import render
    from .fuse_surfels import fuse_surfels

    # Pass 1: compute the asymptotic per-surfel posterior (all train views).
    if verbose:
        print("[gen_scene] pass 1: closed-form fusion over all views to "
              "compute asymptotic targets")
    asymptotic, _stats1, _proc1, _skip1 = fuse_surfels(
        scene=scene, gaussians=gaussians, pipe=pipe, background=background,
        priors_path=priors_path, source_path=source_path, sam_dir=sam_dir,
        verbose=verbose,
    )

    # Pre-extract per-surfel features once.
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    scaling = gaussians.get_scaling.detach().cpu().numpy()
    rotation = gaussians.get_rotation.detach().cpu().numpy()
    opacity = gaussians.get_opacity.detach().cpu().numpy()
    sem = gaussians.get_semantic.detach().cpu().numpy().astype(np.float32)
    if sem.size == 0:
        # Model was trained without the semantic stream; pad to default dim.
        sem = np.zeros((xyz.shape[0], DEFAULT_SEMANTIC_DIM), dtype=np.float32)
    geom = extract_geometric_features(xyz, scaling, rotation, opacity)

    rng = np.random.default_rng(seed)
    priors = PriorTable.from_path(priors_path)

    # Pass 2: walk views in random order, emit tuples.
    if verbose:
        print("[gen_scene] pass 2: emit training tuples")
    state = SurfelPhysicalState.from_global_prior(
        n_surfels=xyz.shape[0],
        prior_tau=priors._global.tau,
        prior_kappa=priors._global.kappa,
        prior_alpha=priors._global.alpha,
        prior_beta=priors._global.beta,
    )
    acc = _ScenePairAccumulator.empty()

    cameras = list(scene.getTrainCameras())
    perm = rng.permutation(len(cameras))
    iterator = tqdm(perm, desc="Emitting", disable=not verbose)
    for view_idx in iterator:
        if len(acc) >= max_samples:
            break
        view = cameras[view_idx]
        phys_path = _physical_path(source_path, view.image_name, sam_dir)
        if not os.path.exists(phys_path) or view.region_map is None:
            continue
        with open(phys_path, "r") as f:
            phys = json.load(f)
        records = phys.get("regions", [])
        if not records:
            continue

        pkg = render(view, gaussians, pipe, background,
                     render_semantic=False, record_contrib=True)
        cids_t = pkg["rendered_contrib_ids"]
        cw_t = pkg["rendered_contrib_weights"]
        if cids_t.numel() == 0:
            raise RuntimeError(
                "rendered_contrib_ids is empty; rebuild "
                "submodules/diff-surfel-rasterization with the top-K "
                "contribution buffer enabled."
            )
        contrib_ids = cids_t.cpu().numpy()
        contrib_weights = cw_t.cpu().numpy()
        region_map = view.region_map.cpu().numpy()[0]

        for record in records:
            if len(acc) >= max_samples:
                break
            try:
                rid, material, level, conf = _parse_region_record(record)
            except (KeyError, ValueError, TypeError) as e:
                print(f"[gen_scene] skipping malformed region: {e}")
                continue
            obs_value = priors.lookup(material, level).prior.tau
            emit_region_pairs(
                state=state, asymptotic=asymptotic,
                region_id=rid, obs_value=obs_value, vlm_confidence=conf,
                region_map=region_map, contrib_ids=contrib_ids,
                contrib_weights=contrib_weights,
                geometric_features=geom, semantic_features=sem,
                acc=acc, max_per_region=max_per_region, rng=rng,
            )

    packed = acc.stack()
    n = packed["prior_state"].shape[0]
    return _pack_dataset(
        kind="scene",
        prior_state=packed["prior_state"],
        observation=packed["observation"],
        geometric_features=packed["geometric_features"],
        semantic_features=packed["semantic_features"],
        target_local=packed["target_local"],
        target_asymptotic=packed["target_asymptotic"],
        has_features=np.ones(n, dtype=bool),
        feature_meta=_geometric_feature_meta(geom.shape[1]),
    )


# ============================================================================
# Shared packing + IO
# ============================================================================

def _pack_dataset(
    kind: str,
    prior_state: np.ndarray,
    observation: np.ndarray,
    geometric_features: np.ndarray,
    semantic_features: np.ndarray,
    target_local: np.ndarray,
    target_asymptotic: np.ndarray,
    has_features: np.ndarray,
    feature_meta: dict,
) -> dict:
    n = prior_state.shape[0]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "n_samples": int(n),
        "prior_state": prior_state.astype(np.float32),
        "observation": observation.astype(np.float32),
        "geometric_features": geometric_features.astype(np.float32),
        "semantic_features": semantic_features.astype(np.float32),
        "target_local": target_local.astype(np.float32),
        "target_asymptotic": target_asymptotic.astype(np.float32),
        "has_features": has_features.astype(bool),
        "feature_meta": feature_meta,
    }


def save_dataset(dataset: dict, output_path: str) -> None:
    """Save with torch (so 5e can load directly into tensors)."""
    import torch
    out = {k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v)
           for k, v in dataset.items()}
    torch.save(out, output_path)


def _main():
    import sys
    import torch
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generate Path B fusion training data")
    sub = parser.add_subparsers(dest="kind", required=True)

    dist = sub.add_parser("distillation", help="synthetic pairs from priors.json")
    dist.add_argument("--priors", required=True, type=str)
    dist.add_argument("--output", required=True, type=str)
    dist.add_argument("--n", type=int, default=100_000)
    dist.add_argument("--seed", type=int, default=0)
    dist.add_argument("--geometric-dim", type=int, default=DEFAULT_GEOMETRIC_DIM)
    dist.add_argument("--semantic-dim", type=int, default=DEFAULT_SEMANTIC_DIM)

    sc = sub.add_parser("scene", help="real pairs from a trained scene")
    sc.add_argument("--priors", required=True, type=str)
    sc.add_argument("--output", required=True, type=str)
    sc.add_argument("--iteration", default=-1, type=int)
    sc.add_argument("--sam-dir", default="sam3", type=str)
    sc.add_argument("--max-samples", default=100_000, type=int)
    sc.add_argument("--max-per-region", default=64, type=int)
    sc.add_argument("--seed", default=0, type=int)
    sc.add_argument("--quiet", action="store_true")
    # arguments for Scene loading (mirrors render.py)
    from arguments import ModelParams, PipelineParams, get_combined_args
    model = ModelParams(sc, sentinel=True)
    pipeline = PipelineParams(sc)

    args = parser.parse_args(sys.argv[1:])

    if args.kind == "distillation":
        ds = generate_distillation_pairs(
            n=args.n, priors_path=args.priors, seed=args.seed,
            geometric_dim=args.geometric_dim, semantic_dim=args.semantic_dim,
        )
        save_dataset(ds, args.output)
        print(f"[gen_distill] {ds['n_samples']} pairs -> {args.output}")
    elif args.kind == "scene":
        # Re-parse with the ModelParams/PipelineParams attached to the
        # 'scene' subparser so get_combined_args picks up cfg_args.
        args = get_combined_args(sc)
        dataset_cfg = model.extract(args)
        pipe = pipeline.extract(args)

        from scene import Scene
        from gaussian_renderer import GaussianModel

        gaussians = GaussianModel(dataset_cfg.sh_degree)
        scene = Scene(dataset_cfg, gaussians, load_iteration=args.iteration, shuffle=False)
        bg_color = [1, 1, 1] if dataset_cfg.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        ds = generate_scene_pairs(
            scene=scene, gaussians=gaussians, pipe=pipe, background=background,
            priors_path=args.priors, source_path=dataset_cfg.source_path,
            sam_dir=args.sam_dir, max_samples=args.max_samples,
            max_per_region=args.max_per_region, seed=args.seed,
            verbose=not args.quiet,
        )
        save_dataset(ds, args.output)
        print(f"[gen_scene] {ds['n_samples']} pairs -> {args.output}")


if __name__ == "__main__":
    _main()
