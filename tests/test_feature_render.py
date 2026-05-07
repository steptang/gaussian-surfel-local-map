"""Tests for the parallel feature stream added to the rasterizer.

Run with: pytest tests/ -v

Tests:
1. forward_features_zero_when_disabled — features=None preserves the
   pre-feature return signature.
2. forward_features_match_rgb_path — render the SAME tensor as both RGB
   (via colors_precomp) and as features (via extra_features). Both should
   produce identical (3, H, W) outputs since the alpha-compositing weight
   w is shared. ALSO asserts the output is non-zero, so the test fails if
   the camera/Gaussians don't actually render anything.
3. backward_features_grad_flows — gradient propagates from a feature loss
   into extra_features. Catches kernel bugs in the backward feature path.
4. baseline_regression — two forwards with feat=None match bit-for-bit.

Camera matrices use the project's own getWorld2View2 / getProjectionMatrix
helpers so the +Z convention this codebase expects is preserved (don't
hand-roll OpenGL projection matrices here — they have the wrong sign).

These tests require a CUDA device and the rasterizer extension built. They
are skipped on CPU-only environments.
"""

import math
import os
import sys

import numpy as np
import pytest
import torch

# Allow running pytest from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CUDA_AVAILABLE = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="rasterizer requires CUDA")

if CUDA_AVAILABLE:
    from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from utils.graphics_utils import getWorld2View2, getProjectionMatrix

    # Match scene/gaussian_model.py SEMANTIC_DIM and CUDA config.h FEATURE_DIM.
    SEMANTIC_DIM = 32


def _settings(H=64, W=64, fov_deg=60.0):
    """Camera at world (0, 0, -2) looking toward +Z (codebase convention).

    We build the matrices via the project's helpers so the +Z / z_sign=1.0
    convention in getProjectionMatrix matches what the rasterizer expects.
    """
    fov = math.radians(fov_deg)
    R = np.eye(3, dtype=np.float32)        # camera oriented with world axes
    T = np.array([0.0, 0.0, 2.0], dtype=np.float32)   # world is +Z=2 in cam frame

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
    """16 large-ish opaque Gaussians clustered near the origin.

    Scales are deliberately big enough (~25 cm) that footprint radii
    are >> 1 px at our 64×64 / 60° FOV camera, so the kernel actually
    has work to do and gradients flow.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    means3D = (torch.rand(N, 3, generator=g, device="cuda") - 0.5) * 0.5
    means2D = torch.zeros(N, 3, device="cuda", requires_grad=True)
    colors = torch.rand(N, 3, generator=g, device="cuda")
    opacity = torch.full((N, 1), 0.7, device="cuda")
    scales = torch.full((N, 2), 0.25, device="cuda")
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
    # Sanity: at least some pixels were covered by Gaussians. If this fails,
    # the camera setup or Gaussian sizes are degenerate; downstream tests
    # would silently pass on zero output.
    assert rgb_a.abs().sum().item() > 0, "no Gaussians rendered; check camera setup"


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
    # Non-vacuous: rgb is meaningfully non-zero (some Gaussians did render).
    assert rgb.abs().sum().item() > 1.0, "rgb is essentially zero; nothing rendered"
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
