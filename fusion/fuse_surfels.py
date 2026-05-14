"""Per-surfel friction posterior fusion driver (Path A baseline).

Given a trained 2DGS checkpoint plus the per-image SAM3 region maps and
Qwen2.5-VL-7B physical labels produced by the shared preprocessing repo,
walk every training view, render with the top-K contribution buffer
enabled, and apply a closed-form NIG conjugate update (PhysGS Eq 12-13)
to every surfel that contributes to a labelled region.

Output: a dict of per-surfel posterior state ((N,) tau / kappa / alpha /
beta arrays) saved to `<model_path>/surfel_physical.pt`. Downstream code
(text query, Path B MLP) loads this and stacks into (N, 4) as needed.

The per-region aggregation logic (apply_region_update) is a pure function
over numpy arrays so it can be unit-tested without CUDA. The full driver
(fuse_surfels) requires the rasterizer extension and is invoked from the
__main__ CLI.

Open question (flagged, not answered here):
  - Multi-region-per-surfel-across-views: this driver applies one update
    per (view, region, contributing surfel) tuple. A surfel that sees
    "rocks" in 80 views and "grass" in 20 views collapses to a mixture
    posterior dominated by rocks. Alternative: cluster observations per
    surfel by (material, level) first, then update once per cluster.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .lookup import PriorTable, LookupResult
from .nig import closed_form_update


# Default update function: Path A (closed-form Bayesian). Path B will swap
# in a learned MLP wrapper with the same signature.
UpdateFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray],
    tuple,
]


@dataclass
class SurfelPhysicalState:
    """In-memory per-surfel NIG state. Mutated in place by apply_region_update."""
    tau: np.ndarray      # (N,) float64
    kappa: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray

    @classmethod
    def from_global_prior(cls, n_surfels: int, prior_tau: float, prior_kappa: float,
                          prior_alpha: float, prior_beta: float) -> "SurfelPhysicalState":
        return cls(
            tau=np.full(n_surfels, prior_tau, dtype=np.float64),
            kappa=np.full(n_surfels, prior_kappa, dtype=np.float64),
            alpha=np.full(n_surfels, prior_alpha, dtype=np.float64),
            beta=np.full(n_surfels, prior_beta, dtype=np.float64),
        )

    def to_torch_dict(self, lookup_stats: dict, n_views_processed: int,
                       n_views_skipped: int) -> dict:
        import torch
        return {
            "schema_version": 1,
            "tau": torch.from_numpy(self.tau),
            "kappa": torch.from_numpy(self.kappa),
            "alpha": torch.from_numpy(self.alpha),
            "beta": torch.from_numpy(self.beta),
            "n_surfels": int(self.tau.size),
            "lookup_stats": dict(lookup_stats),
            "n_views_processed": n_views_processed,
            "n_views_skipped_no_physical": n_views_skipped,
        }


def _parse_region_record(record: dict) -> tuple[int, str, str, float]:
    """Extract (region_id, material, friction_level, vlm_confidence) tolerantly.

    The shared preprocessing repo's _physical.json schema uses 'friction' in
    some buckets and 'friction_level' in others; accept either.
    """
    rid = int(record["region_id"])
    material = str(record["material"]).strip().lower()
    if "friction_level" in record:
        level = str(record["friction_level"]).strip().lower()
    elif "friction" in record:
        level = str(record["friction"]).strip().lower()
    else:
        raise KeyError(
            f"region {rid}: physical record missing 'friction' / 'friction_level'"
        )
    conf = float(record.get("vlm_confidence", record.get("confidence", 1.0)))
    return rid, material, level, conf


def apply_region_update(
    state: SurfelPhysicalState,
    region_id: int,
    obs_value: float,
    vlm_confidence: float,
    region_map: np.ndarray,        # (H, W) int
    contrib_ids: np.ndarray,       # (K, H, W) int
    contrib_weights: np.ndarray,   # (K, H, W) float
    update_fn: UpdateFn = closed_form_update,
    min_weight: float = 1e-6,
) -> int:
    """Apply one (region, VLM observation) update to contributing surfels.

    Per-surfel confidence is normalised by region size so a single (view,
    region) observation contributes <= vlm_confidence per surfel and the
    update strength is invariant to region pixel count. Returns the number
    of surfels touched by this call.
    """
    mask = (region_map == region_id)
    num_px = int(mask.sum())
    if num_px == 0:
        return 0

    ids = contrib_ids[:, mask].ravel()
    wts = contrib_weights[:, mask].ravel()
    valid = ids >= 0
    if not valid.any():
        return 0
    ids = ids[valid].astype(np.int64)
    wts = wts[valid].astype(np.float64)

    per_surfel_w = np.bincount(ids, weights=wts, minlength=state.tau.size)
    active = np.flatnonzero(per_surfel_w > min_weight)
    if active.size == 0:
        return 0

    omega = vlm_confidence * per_surfel_w[active] / float(num_px)
    new_tau, new_kappa, new_alpha, new_beta = update_fn(
        state.tau[active], state.kappa[active], state.alpha[active], state.beta[active],
        obs_value, omega,
    )
    state.tau[active] = new_tau
    state.kappa[active] = new_kappa
    state.alpha[active] = new_alpha
    state.beta[active] = new_beta
    return int(active.size)


def apply_view_update(
    state: SurfelPhysicalState,
    physical_records: list[dict],
    region_map: np.ndarray,        # (H, W) int
    contrib_ids: np.ndarray,       # (K, H, W) int
    contrib_weights: np.ndarray,   # (K, H, W) float
    priors: PriorTable,
    update_fn: UpdateFn = closed_form_update,
    lookup_stats: Optional[Counter] = None,
) -> int:
    """Apply every region's update from one view. Returns regions updated."""
    if lookup_stats is None:
        lookup_stats = Counter()
    regions_with_updates = 0
    for record in physical_records:
        try:
            rid, material, level, conf = _parse_region_record(record)
        except (KeyError, ValueError, TypeError) as e:
            # Malformed record; skip but don't crash the whole view.
            print(f"[fuse_surfels] skipping malformed region record: {e}")
            continue
        lookup: LookupResult = priors.lookup(material, level)
        lookup_stats[lookup.source] += 1
        n_updated = apply_region_update(
            state=state,
            region_id=rid,
            obs_value=lookup.prior.tau,
            vlm_confidence=conf,
            region_map=region_map,
            contrib_ids=contrib_ids,
            contrib_weights=contrib_weights,
            update_fn=update_fn,
        )
        if n_updated > 0:
            regions_with_updates += 1
    return regions_with_updates


def _physical_path(source_path: str, image_name: str, sam_dir: str) -> str:
    """Resolve a view's _physical.json sibling next to its _regions.png.

    The shared preprocessing repo writes <source_path>/<sam_dir>/<stem>_*.{png,json,npy}
    where <stem> is the image filename without extension.
    """
    return os.path.join(source_path, sam_dir, f"{image_name}_physical.json")


def fuse_surfels(
    scene,
    gaussians,
    pipe,
    background,
    priors_path: str,
    source_path: str,
    sam_dir: str = "sam3",
    update_fn: UpdateFn = closed_form_update,
    verbose: bool = True,
) -> tuple[SurfelPhysicalState, Counter, int, int]:
    """Drive Path A fusion across every train camera in the scene.

    Returns (state, lookup_stats, n_views_processed, n_views_skipped).
    State is mutated in place; caller serialises it.
    """
    # Local import so the module is importable on CPU-only machines for
    # pure-function unit tests of apply_region_update / apply_view_update.
    from tqdm import tqdm
    from gaussian_renderer import render

    priors = PriorTable.from_path(priors_path)
    n_surfels = int(gaussians.get_xyz.shape[0])
    state = SurfelPhysicalState.from_global_prior(
        n_surfels,
        prior_tau=priors._global.tau,
        prior_kappa=priors._global.kappa,
        prior_alpha=priors._global.alpha,
        prior_beta=priors._global.beta,
    )

    lookup_stats: Counter = Counter()
    n_processed = 0
    n_skipped = 0

    cameras = scene.getTrainCameras()
    iterator = tqdm(cameras, desc="Fusing", disable=not verbose)
    for view in iterator:
        phys_path = _physical_path(source_path, view.image_name, sam_dir)
        if not os.path.exists(phys_path):
            n_skipped += 1
            continue
        if view.region_map is None:
            # Region map is a load-time artifact; if missing, we can't map
            # pixels back to regions even if _physical.json exists.
            n_skipped += 1
            continue

        with open(phys_path, "r") as f:
            phys = json.load(f)
        records = phys.get("regions", [])
        if not records:
            n_skipped += 1
            continue

        # Render with contribution tracking. Semantic stream off (this pass
        # doesn't supervise features) to save memory and time.
        pkg = render(view, gaussians, pipe, background,
                     render_semantic=False, record_contrib=True)
        cids_t = pkg["rendered_contrib_ids"]
        cw_t = pkg["rendered_contrib_weights"]
        if cids_t.numel() == 0:
            raise RuntimeError(
                "rendered_contrib_ids is empty; rasterizer was not built with "
                "the contribution-buffer extension. Rebuild submodules/diff-surfel-rasterization."
            )

        contrib_ids = cids_t.cpu().numpy()
        contrib_weights = cw_t.cpu().numpy()
        region_map = view.region_map.cpu().numpy()[0]   # (H, W)

        apply_view_update(
            state=state,
            physical_records=records,
            region_map=region_map,
            contrib_ids=contrib_ids,
            contrib_weights=contrib_weights,
            priors=priors,
            update_fn=update_fn,
            lookup_stats=lookup_stats,
        )
        n_processed += 1

    return state, lookup_stats, n_processed, n_skipped


def _main():
    import sys
    import torch
    from argparse import ArgumentParser
    from arguments import ModelParams, PipelineParams, get_combined_args
    from scene import Scene
    from gaussian_renderer import GaussianModel

    parser = ArgumentParser(description="Per-surfel friction posterior fusion (Path A)")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int,
                        help="checkpoint iteration to load; -1 = latest")
    parser.add_argument("--priors", required=True, type=str,
                        help="path to calibration priors.json")
    parser.add_argument("--sam-dir", default="sam3", type=str,
                        help="subdirectory under source_path containing _physical.json")
    parser.add_argument("--output-name", default="surfel_physical.pt", type=str)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"[fuse_surfels] {gaussians.get_xyz.shape[0]} surfels, "
          f"{len(scene.getTrainCameras())} train views")
    state, lookup_stats, n_proc, n_skip = fuse_surfels(
        scene=scene, gaussians=gaussians, pipe=pipe, background=background,
        priors_path=args.priors,
        source_path=dataset.source_path,
        sam_dir=args.sam_dir,
        verbose=not args.quiet,
    )

    out_path = os.path.join(dataset.model_path, args.output_name)
    torch.save(state.to_torch_dict(lookup_stats, n_proc, n_skip), out_path)
    total = sum(lookup_stats.values()) or 1
    print(f"[fuse_surfels] processed {n_proc} views, skipped {n_skip}")
    print(f"[fuse_surfels] lookup distribution over {total} (region, view) updates:")
    for k in ("exact", "material_marginal", "global"):
        v = lookup_stats.get(k, 0)
        print(f"             {k:>20s}: {v:6d}  ({100.0 * v / total:5.1f}%)")
    print(f"[fuse_surfels] saved {out_path}")


if __name__ == "__main__":
    _main()
