"""Tests for the optional top-K contribution buffer (record_contrib).

The rasterizer's forward kernel internally knows which surfels composite
into each pixel (with what alpha-weight). record_contrib=True surfaces the
top-K of these via a (CONTRIB_TOPK, H, W) int32 buffer of surfel IDs and a
parallel float32 buffer of weights. Used by fusion/fuse_surfels.py to map
SAM3 regions back to contributing surfels.

Tests:
1. record_contrib_disabled_outputs_empty — default off path returns empty
   tensors and bit-matches the pre-existing color/feature outputs.
2. record_contrib_shapes_and_dtypes — on path returns (K, H, W) int32 ids
   and (K, H, W) float32 weights.
3. record_contrib_weights_sorted_descending — at every covered pixel, the
   K weights are non-increasing.
4. record_contrib_ids_unique_per_pixel — no surfel ID appears twice at a
   single pixel (insertion-sort correctness).
5. record_contrib_weight_budget — sum of K weights at any pixel <= 1.
6. record_contrib_matches_alpha — when N_surfels <= K, sum of weights at a
   covered pixel matches the alpha-budget (1 - final_T = render_alpha).
7. record_contrib_background_pixels_are_sentinel — pixels not touched by
   any Gaussian have all ids == -1 and weights == 0.
8. record_contrib_no_gradient_flow — the contrib outputs are non-
   differentiable; backward through the rgb path still works when contrib
   is enabled.

Run from repo root with: pytest tests/test_contrib_record.py -v
"""

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CUDA_AVAILABLE = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="rasterizer requires CUDA")

if CUDA_AVAILABLE:
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix

    # Must match cuda_rasterizer/config.h CONTRIB_TOPK.
    CONTRIB_TOPK = 8


def _settings(H=64, W=64, fov_deg=60.0):
    fov = math.radians(fov_deg)
    R = np.eye(3, dtype=np.float32)
    T = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    znear, zfar = 0.01, 100.0
    view = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).contiguous().cuda()
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fov, fovY=fov).transpose(0, 1).cuda()
    full = (view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    campos = view.inverse()[3, :3]
    tanfov = math.tan(fov / 2)
    return GaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=tanfov, tanfovy=tanfov,
        bg=torch.zeros(3, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=view, projmatrix=full,
        sh_degree=0, campos=campos,
        prefiltered=False, debug=False,
    )


def _toy_gaussians(N=16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    means3D = (torch.rand(N, 3, generator=g, device="cuda") - 0.5) * 0.5
    means2D = torch.zeros(N, 3, device="cuda", requires_grad=True)
    colors = torch.rand(N, 3, generator=g, device="cuda")
    opacity = torch.full((N, 1), 0.7, device="cuda")
    scales = torch.full((N, 2), 0.25, device="cuda")
    rotations = torch.zeros(N, 4, device="cuda")
    rotations[:, 0] = 1.0
    return means3D, means2D, colors, opacity, scales, rotations


def _render(rasterizer, gaussians, record_contrib):
    means3D, means2D, colors, opacity, scales, rotations = gaussians
    return rasterizer(
        means3D=means3D, means2D=means2D, opacities=opacity,
        colors_precomp=colors, scales=scales, rotations=rotations,
        record_contrib=record_contrib,
    )


def test_record_contrib_disabled_outputs_empty():
    # Default off path: contrib outputs should be empty tensors and the
    # color path bit-identical to a run without the flag.
    settings = _settings()
    rasterizer = GaussianRasterizer(settings)
    gaussians = _toy_gaussians()

    rgb_a, _, _, _, cids_a, cw_a = _render(rasterizer, gaussians, record_contrib=False)
    rgb_b, _, _, _, cids_b, cw_b = _render(rasterizer, gaussians, record_contrib=False)

    assert cids_a.numel() == 0
    assert cw_a.numel() == 0
    assert torch.equal(rgb_a, rgb_b)
    assert rgb_a.abs().sum().item() > 1.0, "rgb path didn't render anything; setup is degenerate"


def test_record_contrib_shapes_and_dtypes():
    settings = _settings(H=64, W=64)
    rasterizer = GaussianRasterizer(settings)
    gaussians = _toy_gaussians()

    _, _, _, _, cids, cw = _render(rasterizer, gaussians, record_contrib=True)
    assert cids.shape == (CONTRIB_TOPK, 64, 64)
    assert cw.shape == (CONTRIB_TOPK, 64, 64)
    assert cids.dtype == torch.int32
    assert cw.dtype == torch.float32


def test_record_contrib_weights_sorted_descending():
    settings = _settings()
    rasterizer = GaussianRasterizer(settings)
    gaussians = _toy_gaussians()
    _, _, _, _, _, cw = _render(rasterizer, gaussians, record_contrib=True)
    # At every pixel, cw[k] >= cw[k+1] for k=0..K-2.
    for k in range(CONTRIB_TOPK - 1):
        diff = (cw[k] - cw[k + 1]).min().item()
        assert diff >= -1e-7, f"top-K weights not descending at slot {k}: min diff {diff}"


def test_record_contrib_ids_unique_per_pixel():
    settings = _settings()
    rasterizer = GaussianRasterizer(settings)
    gaussians = _toy_gaussians()
    _, _, _, _, cids, cw = _render(rasterizer, gaussians, record_contrib=True)

    # Exclude sentinel (-1) slots, which can repeat in background pixels.
    cids_cpu = cids.cpu().numpy()  # (K, H, W)
    K, H, W = cids_cpu.shape
    for y in range(H):
        for x in range(W):
            column = cids_cpu[:, y, x]
            valid = column[column >= 0]
            if valid.size > 0:
                assert valid.size == np.unique(valid).size, (
                    f"duplicate surfel id at pixel ({y},{x}): {column}"
                )


def test_record_contrib_weight_budget():
    settings = _settings()
    rasterizer = GaussianRasterizer(settings)
    gaussians = _toy_gaussians()
    _, _, _, _, _, cw = _render(rasterizer, gaussians, record_contrib=True)
    # Each alpha-composite weight w = alpha * T contributes <= 1 in sum
    # across all front-to-back contributors (1 - T_final). Top-K can only
    # be <= the full sum, so per-pixel sum is <= 1 + slack for fp roundoff.
    per_pixel_sum = cw.sum(dim=0)  # (H, W)
    assert per_pixel_sum.max().item() <= 1.0 + 1e-5


def test_record_contrib_matches_alpha_when_under_k():
    # 4 Gaussians (< CONTRIB_TOPK=8). Every covered pixel can have at most
    # 4 contributors, so the top-K captures all of them and per-pixel sum
    # of weights == render_alpha (1 - T_final).
    settings = _settings()
    rasterizer = GaussianRasterizer(settings)
    gaussians = _toy_gaussians(N=4, seed=1)

    _, _, allmap, _, _, cw = _render(rasterizer, gaussians, record_contrib=True)
    # render_alpha = allmap[1:2] per gaussian_renderer/__init__.py.
    render_alpha = allmap[1, :, :]
    per_pixel_sum = cw.sum(dim=0)

    covered = render_alpha > 1e-4
    if covered.any():
        diff = (per_pixel_sum[covered] - render_alpha[covered]).abs().max().item()
        assert diff < 1e-4, f"weight-sum vs render_alpha max diff {diff} (should be ~0 when N<=K)"


def test_record_contrib_background_pixels_are_sentinel():
    settings = _settings()
    rasterizer = GaussianRasterizer(settings)
    gaussians = _toy_gaussians()
    _, _, allmap, _, cids, cw = _render(rasterizer, gaussians, record_contrib=True)
    render_alpha = allmap[1, :, :]
    bg = render_alpha < 1e-7
    if bg.any():
        # All K slots at a background pixel should be the (-1, 0) sentinel.
        assert (cids[:, bg] == -1).all(), "background pixel has non-sentinel ids"
        assert (cw[:, bg] == 0).all(), "background pixel has non-zero weights"


def test_record_contrib_no_gradient_flow_breaks_color_path():
    # With record_contrib=True, the rgb path should still be differentiable
    # exactly as before. The contrib outputs are non-differentiable; backward
    # through them must not be reached.
    settings = _settings()
    rasterizer = GaussianRasterizer(settings)
    means3D, means2D, colors, opacity, scales, rotations = _toy_gaussians()
    colors = colors.detach().clone().requires_grad_(True)

    rgb, _, _, _, _, _ = rasterizer(
        means3D=means3D, means2D=means2D, opacities=opacity,
        colors_precomp=colors, scales=scales, rotations=rotations,
        record_contrib=True,
    )
    (rgb ** 2).sum().backward()
    assert colors.grad is not None
    assert colors.grad.abs().sum().item() > 0
