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

We avg-pool the rendered map to (K, G, G) and nearest-pool the region map
to (1, G, G) so the per-cell correspondence is exact. Background cells (id 0)
are excluded from the loss. Returns scalar mean cosine distance over valid
cells; 0 if there are no valid cells in the view.
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


def _pool_to_grid(rendered_K_HW: torch.Tensor, region_map_1HW: torch.Tensor, grid: int):
    """Avg-pool features, nearest-pool region map to (grid, grid).

    Both pools use F.adaptive_*_pool2d so any input H, W works (output is
    always exactly grid x grid). Region IDs are integers; nearest-pooling
    them via float averaging would be wrong -- we use max-pooling on a
    one-hot-ish trick: take the centre pixel of each cell.
    """
    K, H, W = rendered_K_HW.shape
    feat = F.adaptive_avg_pool2d(rendered_K_HW.unsqueeze(0), (grid, grid)).squeeze(0)  # (K, G, G)

    # For region IDs, sample at cell centres (nearest-neighbour). This
    # avoids interpolating across region boundaries.
    rmap = region_map_1HW.float().unsqueeze(0)                          # (1, 1, H, W)
    rmap_g = F.interpolate(rmap, size=(grid, grid), mode="nearest")
    region_grid = rmap_g.squeeze(0).squeeze(0).long()                   # (G, G)
    return feat, region_grid


def cosine_region_loss(
    rendered_K_HW: torch.Tensor,
    region_map_1HW: torch.Tensor,
    region_embeds_RT: torch.Tensor,
    head: SemanticHead,
    grid: int = 32,
    eps: float = 1e-6,
):
    """Mean cosine distance between projected rendered features and region targets.

    Returns a scalar tensor on the same device as `rendered_K_HW`. If no
    valid cells (all background), returns 0.0 with grad enabled (so the
    loss term contributes nothing but doesn't crash autograd).
    """
    feat_grid, region_grid = _pool_to_grid(rendered_K_HW, region_map_1HW, grid)
    K = feat_grid.shape[0]

    # Move embeds to GPU lazily (Camera holds them on CPU to save VRAM).
    embeds = region_embeds_RT.to(feat_grid.device, non_blocking=True)   # (R+1, K_target)

    # Mask out background cells (region_id == 0) and any out-of-range IDs
    # (defensive; nearest-pooling should never produce them).
    valid = (region_grid > 0) & (region_grid < embeds.shape[0])
    if not valid.any():
        return rendered_K_HW.sum() * 0.0

    # Gather: each valid cell -> (K,) feature and (K_target,) target.
    feats = feat_grid.permute(1, 2, 0)[valid]                           # (N, K)
    targets = embeds[region_grid[valid]]                                # (N, K_target)

    projected = head(feats)                                             # (N, K_target)

    # Cosine distance = 1 - cos_sim. Normalise both for stability; eps
    # avoids div-by-zero when a surfel cell has no contributing Gaussian.
    p = F.normalize(projected, dim=-1, eps=eps)
    t = F.normalize(targets, dim=-1, eps=eps)
    return (1.0 - (p * t).sum(dim=-1)).mean()
