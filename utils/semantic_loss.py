"""Semantic supervision: K-dim surfel features -> K_target SigLIP2 region embeds.

The rasterizer produces a (K, H, W) feature map by alpha-blending each
surfel's `_semantic` parameter. The supervision target is a (R+1, K_target)
table of SigLIP2 image embeddings, one per SAM3 region. We bridge the two
with a learned linear projection K -> K_target and a cosine loss.

Shapes (using SigLIP2-base K_target=768, default semantic_grid=32):
    rendered_K_HW:    (K, H, W)         -- raw rendered feature map
    region_map_HW:    (1, H, W) int     -- SAM3 region IDs at image res
    region_embeds:    (R+1, K_target)   -- SigLIP2 embeddings, row 0 = bg
    head:             SemanticHead      -- nn.Linear(K, K_target)

We avg-pool the rendered map to (K, G, G) and build per-cell coverage
fractions over the region IDs by avg-pooling a one-hot encoding of the
region map. Each cell's target is then a coverage-weighted blend of the
SigLIP2 embeddings of every region intersecting it (matmul; unit-norm
afterwards). Boundary cells whose pixels span two regions therefore
receive a blended target, symmetric with the honest average-pooled
rendered feature side -- whereas the prior nearest-neighbour-at-cell-
centre rule supervised the entire cell against a single region picked
by the centre pixel, mis-assigning ~20% of pixels at region boundaries.

Background (region id 0) coverage is excluded from the blend. A cell is
dropped if its foreground coverage falls below ``bg_threshold`` (default
0.5), preserving the spirit of the previous centre==0 drop rule.
Returns scalar mean cosine distance over valid cells; 0 if no cells
survive the threshold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticHead(nn.Module):
    """Project rendered surfel features to the encoder's embedding space.

    Linear (no bias, no activation) keeps the cosine-loss geometry simple:
    a linear map sends surfel-space directions to encoder-space directions.
    """

    def __init__(self, K: int, K_target: int):
        super().__init__()
        self.proj = nn.Linear(K, K_target, bias=False)

    def forward(self, x):
        # x: (..., K) -> (..., K_target)
        return self.proj(x)


def _pool_to_grid(rendered_K_HW: torch.Tensor, region_map_1HW: torch.Tensor,
                   region_embeds_RT: torch.Tensor, grid: int,
                   bg_threshold: float = 0.5, eps: float = 1e-6):
    """Avg-pool rendered features and compute soft per-cell region coverage.

    The rendered side is avg-pooled to (K, G, G) exactly as before.

    The region side now produces, for each cell, a vector of *coverage
    fractions* over the regions whose pixels intersect that cell --
    i.e., what fraction of the cell's pixels belong to each region.
    Implemented as F.adaptive_avg_pool2d over a per-pixel one-hot
    encoding of the region IDs (one matmul + one pool, vectorised).

    To keep the one-hot tensor small even when the embedding table has
    thousands of entries, we compress to the per-view *active* region
    set via torch.unique. Most images touch only ~10-100 distinct
    regions, so the one-hot is (R_present, H, W), not (R_full+1, H, W).

    Background (region id 0) and any IDs out of range for the embeds
    table are excluded from the foreground coverage. A cell is dropped
    (excluded from the loss) if its foreground coverage falls below
    ``bg_threshold`` -- the soft analogue of the previous centre==0
    drop rule.

    Returns:
        feat_grid:   (K, G, G) avg-pooled rendered features.
        coverage_NR: (N, R_fg) per-valid-cell coverage fractions over
                     the R_fg active foreground regions. Each row sums
                     to 1 by construction.
        fg_embeds:   (R_fg, K_target) embeddings for those regions,
                     on the same device as feat_grid.
        valid:       (G, G) bool, cells whose foreground coverage >=
                     bg_threshold.
    """
    K, H, W = rendered_K_HW.shape

    # Honest rendered side -- unchanged.
    feat_grid = F.adaptive_avg_pool2d(rendered_K_HW.unsqueeze(0), (grid, grid)).squeeze(0)

    # Compress to the per-view active region set.
    region_2D = region_map_1HW.squeeze(0).long()                            # (H, W)
    unique_regions, inverse = torch.unique(region_2D, return_inverse=True)  # (R_p,), (H, W)
    R_p = unique_regions.shape[0]

    embeds = region_embeds_RT.to(feat_grid.device, non_blocking=True)       # (R+1, K_target)
    fg_mask = (unique_regions > 0) & (unique_regions < embeds.shape[0])

    # Float one-hot built via scatter to skip the (long)->(float) intermediate
    # F.one_hot would create on a per-pixel tensor.
    one_hot = torch.zeros(R_p, H, W, device=region_2D.device, dtype=torch.float32)
    one_hot.scatter_(0, inverse.unsqueeze(0), 1.0)                          # (R_p, H, W)

    # Per-cell coverage of each active region.
    coverage_full = F.adaptive_avg_pool2d(
        one_hot.unsqueeze(0), (grid, grid)
    ).squeeze(0)                                                              # (R_p, G, G)

    fg_coverage = coverage_full[fg_mask]                                     # (R_fg, G, G)
    fg_weight = fg_coverage.sum(dim=0)                                        # (G, G)
    valid = fg_weight >= bg_threshold                                         # (G, G) bool

    if not valid.any():
        empty_cov = torch.empty(0, fg_coverage.shape[0],
                                device=embeds.device, dtype=embeds.dtype)
        empty_emb = torch.empty(0, embeds.shape[1],
                                device=embeds.device, dtype=embeds.dtype)
        return feat_grid, empty_cov, empty_emb, valid

    # Renormalise so each kept cell's foreground-coverage rows sum to 1.
    fg_norm = fg_coverage / fg_weight.clamp(min=eps).unsqueeze(0)            # (R_fg, G, G)
    coverage_NR = fg_norm.permute(1, 2, 0)[valid]                             # (N, R_fg)

    fg_unique_ids = unique_regions[fg_mask]                                   # (R_fg,)
    fg_embeds = embeds[fg_unique_ids]                                         # (R_fg, K_target)

    return feat_grid, coverage_NR, fg_embeds, valid


def cosine_region_loss(
    rendered_K_HW: torch.Tensor,
    region_map_1HW: torch.Tensor,
    region_embeds_RT: torch.Tensor,
    head: SemanticHead,
    grid: int = 32,
    bg_threshold: float = 0.5,
    eps: float = 1e-6,
):
    """Mean cosine distance between projected rendered features and
    coverage-weighted region targets.

    Each grid cell is supervised against a coverage-weighted blend of
    the SigLIP2 embeddings of the regions intersecting it (see
    ``_pool_to_grid``). This makes the target side symmetric with the
    average-pooled rendered side: a cell whose pixels are 60% table and
    40% chair is supervised against 0.6 * E(table) + 0.4 * E(chair)
    (re-normalised to unit length), not against whichever region's
    embedding the cell centre happened to fall in.

    Args:
        bg_threshold: minimum fraction of a cell that must be foreground
            (region_id > 0) for the cell to participate in the loss.
            Default 0.5 keeps cells that are at least half non-background.

    Returns a scalar tensor on the device of ``rendered_K_HW``. If no
    cells survive the foreground threshold, returns 0.0 with grad
    enabled so the loss term contributes nothing without crashing
    autograd.
    """
    feat_grid, coverage_NR, fg_embeds, valid = _pool_to_grid(
        rendered_K_HW, region_map_1HW, region_embeds_RT, grid,
        bg_threshold=bg_threshold, eps=eps,
    )
    if not valid.any():
        return rendered_K_HW.sum() * 0.0

    feats = feat_grid.permute(1, 2, 0)[valid]      # (N, K)
    projected = head(feats)                         # (N, K_target)

    # Blend = coverage matrix x embedding table -- one matmul, no loops.
    # The blended target is a non-unit vector (it's a convex combination
    # of unit-or-near-unit embeddings), so we re-normalise before the
    # cosine sim.
    targets = coverage_NR @ fg_embeds               # (N, K_target)
    targets = F.normalize(targets, dim=-1, eps=eps)
    p = F.normalize(projected, dim=-1, eps=eps)
    return (1.0 - (p * targets).sum(dim=-1)).mean()
