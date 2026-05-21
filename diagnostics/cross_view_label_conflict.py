"""Diagnose per-surfel SAM3 label conflict across training views.

ONE-OFF DIAGNOSTIC. Lives in diagnostics/ deliberately separate from
core training/inference code. Reads only existing artifacts: the
trained gaussians, the per-view SAM3 region maps + SigLIP2 embeddings,
and the SemanticHead checkpoint. Does NOT modify training, the
rasterizer, or the render path; uses a pure-Python surfel-centroid
projection (the rasterizer's record_contrib mechanism is intentionally
absent from main, so we avoid it).

Question we are testing
-----------------------
The semantic loss in utils/semantic_loss.py supervises a 32x32 grid;
each cell's target is the SigLIP2 embedding of a single hard-assigned
SAM3 region (nearest-neighbour at the cell centre). A boundary surfel
whose footprint straddles two regions can receive contradictory
embeddings across views. This script tabulates, per surfel, the
distribution of SAM3 labels it falls into across all training views,
and surfaces the surfels whose label history conflicts most with
their semantic-query score.

Approximation
-------------
Instead of a per-pixel contribution buffer, we project each surfel
*centre* into each view via the same world-to-pixel math the existing
mesh / depth-back-projection code uses (see mesh_utils.compute_sdf_perframe
and point_utils.depths_to_points). We depth-test against the
rendered surf_depth using a scene-radius-scaled tolerance, and read
the SAM3 region map at the projected pixel.

CAVEAT: centre-only sampling under-reports conflict for boundary
surfels (a surfel whose footprint spans two regions is recorded as
belonging to whichever region the centre lands in). A "no-conflict"
verdict here is weaker evidence than a "conflict" verdict.

Usage
-----
    python diagnostics/cross_view_label_conflict.py \\
        -s data/scenes/scan105 -m output/scan105 \\
        --checkpoint output/scan105/chkpnt30000.pth \\
        --query "table"
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel
from scene.gaussian_model import SEMANTIC_DIM
from utils.semantic_loss import SemanticHead


# --- Config defaults (overridable via CLI) --------------------------------

DEFAULT_VOCAB = [
    "table", "grass", "paving stone", "brick wall",
    "foliage", "vase", "sky", "ground",
]
DEFAULT_QUERY = "table"
DEFAULT_DEPTH_TOL_REL = 0.02     # 2% of scene radius (cameras_extent)
DEFAULT_NUM_TRACE = 30           # how many noisiest surfels to print
DEFAULT_MIN_VIEWS = 5            # require >= this many visible views per surfel
DEFAULT_CONFLICT_FRAC = 0.20     # non-query labels >20% of views => CONFLICT

# Sentinel label IDs (do not collide with vocab indices)
BACKGROUND_LABEL = "background"  # SAM3 region id 0


# --- SigLIP2 text encoding (mirrors scripts/text_query.py) ----------------

def _unwrap_pooled(out):
    """Coerce SigLIP2 (text|image)-features call to (B, K_target) tensor."""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("pooler_output", "text_embeds", "image_embeds"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state.mean(dim=1)
    raise RuntimeError(f"unexpected SigLIP2 output type: {type(out).__name__}")


def encode_text(texts, variant):
    """Returns (len(texts), K_target) text embeddings, NOT normalised."""
    from transformers import AutoModel, AutoProcessor
    model = AutoModel.from_pretrained(variant, torch_dtype=torch.float32).cuda().eval()
    proc = AutoProcessor.from_pretrained(variant)
    inputs = proc(text=list(texts), padding="max_length", return_tensors="pt").to("cuda")
    with torch.no_grad():
        emb = _unwrap_pooled(model.get_text_features(**inputs))
    return emb


# --- Projection (matches mesh_utils.compute_sdf_perframe convention) ------

@torch.no_grad()
def project_to_pixels(xyz_N3, cam):
    """Project (N, 3) world-space surfel centres into one view.

    Returns (px, py, cam_z, in_bounds_mask) all length-N. cam_z is the
    surfel's camera-space depth (positive = in front of camera).
    """
    N = xyz_N3.shape[0]
    ones = torch.ones(N, 1, device=xyz_N3.device, dtype=xyz_N3.dtype)
    pts_h = torch.cat([xyz_N3, ones], dim=-1)               # (N, 4)
    clip = pts_h @ cam.full_proj_transform                   # (N, 4)
    # clip[:, 3] is the view-space z (P[3,2]=1, P[3,3]=0 in getProjectionMatrix).
    cam_z = clip[:, 3]
    safe_w = cam_z.clamp(min=1e-8)
    ndc_xy = clip[:, :2] / safe_w.unsqueeze(-1)              # (N, 2) in [-1, 1] inside view
    W, H = cam.image_width, cam.image_height
    px = (ndc_xy[:, 0] + 1.0) * 0.5 * W
    py = (ndc_xy[:, 1] + 1.0) * 0.5 * H
    in_bounds = (cam_z > 0) & (px >= 0) & (px < W) & (py >= 0) & (py < H)
    return px, py, cam_z, in_bounds


# --- Per-view region->label resolution ------------------------------------

@torch.no_grad()
def region_to_label_id(region_embeds_RT, vocab_emb_VK, vocab_size):
    """For a (R+1, K_target) per-region embedding table, return a (R+1,)
    int64 vector mapping region_id -> label_idx via cosine argmax against
    the vocab. Region 0 (SAM3 background) is forced to the background
    sentinel index `vocab_size`.
    """
    embeds = region_embeds_RT.cuda(non_blocking=True)
    emb_n = F.normalize(embeds, dim=-1, eps=1e-6)
    sim = emb_n @ vocab_emb_VK.T                              # (R+1, V)
    labels = sim.argmax(dim=1).cpu().numpy().astype(np.int64)
    labels[0] = vocab_size                                     # background sentinel
    return labels


# --- Main -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--checkpoint", default=None,
                        help="path to chkpnt*.pth (for the SemanticHead)")
    parser.add_argument("--iteration", type=int, default=-1,
                        help="trained-model iteration; -1 = latest")
    parser.add_argument("--variant", default="google/siglip2-base-patch16-512",
                        help="SigLIP2 model id (must match preprocessing)")
    parser.add_argument("--query", default=DEFAULT_QUERY,
                        help="label whose noisy surfels we inspect")
    parser.add_argument("--vocab", nargs="+", default=DEFAULT_VOCAB,
                        help="candidate labels for per-region argmax matching")
    parser.add_argument("--num-trace", type=int, default=DEFAULT_NUM_TRACE)
    parser.add_argument("--min-views", type=int, default=DEFAULT_MIN_VIEWS)
    parser.add_argument("--depth-tol-rel", type=float, default=DEFAULT_DEPTH_TOL_REL,
                        help="depth tolerance as a fraction of scene radius")
    parser.add_argument("--conflict-frac", type=float, default=DEFAULT_CONFLICT_FRAC)
    parser.add_argument("--max-views", type=int, default=None,
                        help="optional cap for quick smoke runs (default: all train views)")
    args = get_combined_args(parser)

    dataset = lp.extract(args)
    pipe = pp.extract(args)

    # ----- Loader (mirrors scripts/text_query.py) -----
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    semantic_head = SemanticHead(SEMANTIC_DIM, dataset.K_target).cuda().eval()
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required (need the trained SemanticHead)")
    ckpt = torch.load(args.checkpoint, weights_only=False)
    head_state = ckpt[1]
    if head_state is None:
        raise SystemExit("checkpoint has no SemanticHead; was lambda_semantic > 0?")
    semantic_head.load_state_dict(head_state)

    cameras = scene.getTrainCameras()
    if args.max_views is not None:
        cameras = cameras[:args.max_views]
    if not cameras:
        raise SystemExit("no train cameras loaded")

    # ----- Text vocab + query encoding -----
    vocab = list(args.vocab)
    if args.query not in vocab:
        # Prepend so the query label is matchable when we argmax regions.
        vocab.insert(0, args.query)
    vocab_size = len(vocab)
    query_idx = vocab.index(args.query)
    print(f"[cross_view] vocab: {vocab}")
    print(f"[cross_view] query='{args.query}' (vocab idx {query_idx})")

    print("[cross_view] encoding vocab + query through SigLIP2 text tower")
    all_text = vocab + [args.query]                  # last entry duplicates the query
    text_emb = encode_text(all_text, args.variant)   # (V+1, K_target)
    vocab_emb = F.normalize(text_emb[:vocab_size], dim=-1, eps=1e-6)
    query_emb = F.normalize(text_emb[-1:], dim=-1, eps=1e-6)
    if vocab_emb.shape[1] != dataset.K_target:
        raise SystemExit(
            f"SigLIP K_target={vocab_emb.shape[1]} != model K_target={dataset.K_target}; "
            "--variant must match preprocessing"
        )

    # ----- Per-surfel query score -----
    print("[cross_view] computing per-surfel query score")
    with torch.no_grad():
        sem = gaussians.get_semantic                  # (N, K_surfel)
        proj = F.normalize(semantic_head(sem), dim=-1, eps=1e-6)
        query_score = (proj @ query_emb.squeeze(0)).cpu().numpy()   # (N,)
    N = query_score.shape[0]
    print(f"[cross_view] {N} surfels, query_score range "
          f"[{query_score.min():.3f}, {query_score.max():.3f}]")

    # ----- Per-view label accumulation -----
    # counts[surfel_idx, label_idx]; label_idx in [0, vocab_size-1] are vocab,
    # label_idx == vocab_size is "background" (SAM3 region 0).
    counts = np.zeros((N, vocab_size + 1), dtype=np.int32)
    xyz = gaussians.get_xyz.detach()                     # (N, 3) cuda

    scene_radius = float(scene.cameras_extent)
    tol_abs = args.depth_tol_rel * scene_radius
    print(f"[cross_view] scene radius={scene_radius:.3f}, depth tol={tol_abs:.3f} "
          f"({args.depth_tol_rel*100:.1f}% of radius)")

    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    n_skipped_no_artifacts = 0
    n_processed = 0

    from tqdm import tqdm
    for cam in tqdm(cameras, desc="views"):
        if cam.region_map is None or cam.region_embeds is None:
            n_skipped_no_artifacts += 1
            continue

        # Per-view region -> vocab label id.
        region_labels = region_to_label_id(cam.region_embeds, vocab_emb, vocab_size)   # (R+1,)

        # Render depth (semantic stream off — we only need surf_depth).
        with torch.no_grad():
            pkg = render(cam, gaussians, pipe, bg, render_semantic=False)
            depth_HW = pkg["surf_depth"][0]              # (H, W) cuda float

        # Project all surfels into this view.
        with torch.no_grad():
            px, py, cam_z, in_bounds = project_to_pixels(xyz, cam)
            ix = px.long().clamp(0, depth_HW.shape[1] - 1)
            iy = py.long().clamp(0, depth_HW.shape[0] - 1)
            depth_at_px = depth_HW[iy, ix]
            visible = in_bounds & ((depth_at_px - cam_z).abs() < tol_abs)

            vis_idx = visible.nonzero(as_tuple=False).squeeze(-1).cpu().numpy()
            if vis_idx.size == 0:
                continue
            ix_cpu = ix[visible].cpu().numpy()
            iy_cpu = iy[visible].cpu().numpy()

        # region_map is (1, H, W) int16 on CPU; index it directly.
        rmap = cam.region_map[0].numpy()                  # (H, W) int16
        # Defensive: SAM3 region ids should fit in [0, R].
        R_plus_1 = region_labels.shape[0]
        region_at_surfel = rmap[iy_cpu, ix_cpu].astype(np.int64)
        region_at_surfel = np.clip(region_at_surfel, 0, R_plus_1 - 1)
        label_at_surfel = region_labels[region_at_surfel]
        np.add.at(counts, (vis_idx, label_at_surfel), 1)
        n_processed += 1

    print(f"[cross_view] processed {n_processed} views, "
          f"skipped {n_skipped_no_artifacts} for missing artifacts")
    if n_processed == 0:
        raise SystemExit("no views had both region_map and region_embeds")

    # ----- Filter + tabulate -----
    seen_counts = counts.sum(axis=1)                       # (N,)
    appeared_as_query = counts[:, query_idx] > 0
    enough_views = seen_counts >= args.min_views
    candidates = np.where(appeared_as_query & enough_views)[0]
    print(f"[cross_view] {len(candidates)} surfels match: "
          f">={args.min_views} views AND labelled '{args.query}' at least once")
    if len(candidates) == 0:
        raise SystemExit("no candidates; lower --min-views or check --query")

    # Sort candidates by query_score ascending (lowest = noisiest under the
    # query). Take the bottom N for tracing.
    order = candidates[np.argsort(query_score[candidates])]
    to_trace = order[:args.num_trace]

    # Label index -> human-readable string (vocab + background sentinel).
    label_names = vocab + [BACKGROUND_LABEL]
    n_conflict = 0

    print(f"\n[cross_view] noisiest {len(to_trace)} surfels under query='{args.query}':")
    for sid in to_trace:
        row = counts[sid]
        total = int(row.sum())
        # Sort labels by frequency descending; show all nonzero.
        nz = np.where(row > 0)[0]
        nz = nz[np.argsort(-row[nz])]
        frac_query = row[query_idx] / max(total, 1)
        is_conflict = (1.0 - frac_query) >= args.conflict_frac
        flag = "CONFLICT" if is_conflict else "consistent"
        if is_conflict:
            n_conflict += 1
        dist_str = "{" + ", ".join(
            f"{label_names[i]}: {100.0 * row[i] / total:.0f}%"
            for i in nz
        ) + "}"
        print(f"  surfel {sid:7d}  query_score={query_score[sid]:+.3f}  "
              f"views={total:3d}  [{flag}]  {dist_str}")

    frac = n_conflict / max(len(to_trace), 1)
    print(f"\n[cross_view] summary: "
          f"{n_conflict}/{len(to_trace)} traced surfels show CONFLICT "
          f"(non-'{args.query}' labels >= {args.conflict_frac*100:.0f}% of views)  "
          f"= {frac*100:.1f}%")
    print("[cross_view] CAVEAT: centre-only sampling under-reports boundary conflict; "
          "a low conflict fraction is weaker evidence than a high one.")


if __name__ == "__main__":
    main()
