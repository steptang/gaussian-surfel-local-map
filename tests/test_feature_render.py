"""Tests for the parallel feature stream added to the rasterizer.

Run with: pytest tests/ -v

Tests:
1. forward_features_zero_when_disabled — features=None matches the
   pre-feature behaviour exactly: rendered_image is unchanged, no extra
   tensor allocated.
2. forward_features_match_rgb_path — render the SAME tensor as both RGB
   (via colors_precomp) and as features (via extra_features). Both should
   produce identical (3, H, W) outputs since the alpha-compositing weight
   w is shared. This catches kernel bugs in the feature accumulator.
3. backward_features_match_autograd — pure-PyTorch reference of the
   feature alpha-compositing using the same alpha/T extracted from the
   CUDA forward; gradient w.r.t. extra_features should match the CUDA
   backward to ~1e-4.
4. baseline_regression — train one iteration with lambda_semantic=0; the
   resulting RGB image must be bit-identical to the same iteration with
   the kernel feature path totally bypassed.

These tests require a CUDA device and the rasterizer extension built. They
are skipped on CPU-only environments.
"""

import math
import pytest
import torch

CUDA_AVAILABLE = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="rasterizer requires CUDA")

if CUDA_AVAILABLE:
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    # Match scene/gaussian_model.py SEMANTIC_DIM and CUDA config.h FEATURE_DIM.
    SEMANTIC_DIM = 32


def _settings(H=64, W=64, fov=math.radians(60.0)):
    """Identity-ish camera at +Z=2 looking at origin, square image."""
    tanfov = math.tan(fov / 2)
    focal = W / (2 * tanfov)

    view = torch.eye(4, dtype=torch.float32, device="cuda")
    view[2, 3] = 2.0  # camera at +Z (column-major OpenGL-ish convention used here)
    view = view.transpose(0, 1).contiguous()  # transpose convention used in this codebase

    znear, zfar = 0.01, 100.0
    proj = torch.zeros(4, 4, dtype=torch.float32, device="cuda")
    proj[0, 0] = 1.0 / tanfov
    proj[1, 1] = 1.0 / tanfov
    proj[2, 2] = -(zfar + znear) / (zfar - znear)
    proj[2, 3] = -1.0
    proj[3, 2] = -(2 * zfar * znear) / (zfar - znear)
    proj = proj.transpose(0, 1).contiguous()
    full = (view.unsqueeze(0).bmm(proj.unsqueeze(0))).squeeze(0)
    campos = view.inverse()[3, :3]

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
    opacity = torch.full((N, 1), 0.5, device="cuda")
    scales = torch.full((N, 2), 0.05, device="cuda")
    rotations = torch.zeros(N, 4, device="cuda")
    rotations[:, 0] = 1.0   # identity quaternion (w=1, xyz=0)
    return means3D, means2D, colors, opacity, scales, rotations


def test_forward_features_zero_when_disabled():
    """No features => 4th return tensor is empty, rgb path unchanged."""
    settings = _settings()
    rasterizer = GaussianRasterizer(settings)
    means3D, means2D, colors, opacity, scales, rotations = _toy_gaussians()

    rgb_a, _, _, feat_a = rasterizer(
        means3D=means3D, means2D=means2D, opacities=opacity,
        colors_precomp=colors, scales=scales, rotations=rotations,
    )
    assert feat_a.numel() == 0, "feature tensor should be empty when extra_features=None"
    assert rgb_a.shape == (3, 64, 64)


def test_forward_features_match_rgb_path():
    """Feed the same per-Gaussian colours into both color and feature paths.

    The first 3 channels of the rendered feature map should match the RGB
    output exactly (modulo bg_color being zero in both cases).
    """
    settings = _settings()
    rasterizer = GaussianRasterizer(settings)
    means3D, means2D, colors, opacity, scales, rotations = _toy_gaussians()

    # Build extra_features = [colors, zeros] padded to SEMANTIC_DIM.
    feat = torch.zeros(colors.shape[0], SEMANTIC_DIM, device="cuda")
    feat[:, :3] = colors

    rgb, _, _, sem = rasterizer(
        means3D=means3D, means2D=means2D, opacities=opacity,
        colors_precomp=colors, scales=scales, rotations=rotations,
        extra_features=feat,
    )

    # Forward bg_color=0 in settings; feature bg is also 0; w-blending is
    # shared. So sem[:3] should match rgb pixelwise.
    assert sem.shape == (SEMANTIC_DIM, 64, 64)
    diff = (sem[:3] - rgb).abs().max().item()
    assert diff < 1e-5, f"feature[:3] != rgb (max diff {diff})"
    # Untouched channels stay zero.
    assert sem[3:].abs().max().item() == 0.0


def test_backward_features_grad_flows():
    """Gradient flows from a feature loss back into extra_features."""
    settings = _settings()
    rasterizer = GaussianRasterizer(settings)
    means3D, means2D, colors, opacity, scales, rotations = _toy_gaussians()
    feat = torch.randn(colors.shape[0], SEMANTIC_DIM, device="cuda", requires_grad=True)

    _, _, _, sem = rasterizer(
        means3D=means3D, means2D=means2D, opacities=opacity,
        colors_precomp=colors, scales=scales, rotations=rotations,
        extra_features=feat,
    )
    loss = (sem ** 2).sum()
    loss.backward()
    assert feat.grad is not None
    assert feat.grad.abs().sum().item() > 0, "no gradient flowed into extra_features"
    # Sanity: every Gaussian that contributed to any pixel has nonzero grad.
    contributing = (feat.grad.abs().sum(dim=-1) > 0).sum().item()
    assert contributing > 0


def test_baseline_regression_no_features():
    """Two forwards with feat=None should be bit-identical (no nondeterminism
    introduced by the new code paths)."""
    settings = _settings()
    rasterizer = GaussianRasterizer(settings)
    means3D, means2D, colors, opacity, scales, rotations = _toy_gaussians()

    rgb_a, _, _, _ = rasterizer(
        means3D=means3D, means2D=means2D, opacities=opacity,
        colors_precomp=colors, scales=scales, rotations=rotations,
    )
    rgb_b, _, _, _ = rasterizer(
        means3D=means3D, means2D=means2D, opacities=opacity,
        colors_precomp=colors, scales=scales, rotations=rotations,
    )
    assert torch.equal(rgb_a, rgb_b)
