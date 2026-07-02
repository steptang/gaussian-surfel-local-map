"""Masked 2DGS reconstruction of a region (the person, or the static background).

Mirrors ``train.py``'s densification + normal/dist regularisers, but restricts the loss to a
per-view *keep* mask so only the wanted region is reconstructed. Used to build the canonical
object model (keep = person) and the static map (keep = non-person), optionally fusing several
timesteps (the person moves, so a multi-timestep static fill has no occlusion holes).
"""
import random

import numpy as np
import torch

from utils.loss_utils import ssim
from utils.graphics_utils import BasicPointCloud
from gaussian_renderer import render, GaussianModel


def reconstruct_masked(views, init_points, init_colors, opt, pipe, background, extent,
                       sh_degree, iters=7000, lambda_depth=1.0):
    """Reconstruct a masked region into a fresh GaussianModel.

    Args:
        views: list of (camera, keep_mask) pairs; keep_mask is (1,H,W) with 1 = reconstruct.
        init_points, init_colors: (M,3) world points / (M,3) RGB [0,1] to seed the model.
        opt/pipe/background/extent/sh_degree: the usual training params.
    Returns the trained (un-frozen) GaussianModel.
    """
    g = GaussianModel(sh_degree)
    g.create_from_pcd(BasicPointCloud(points=init_points, colors=init_colors,
                                      normals=np.zeros_like(init_points)), spatial_lr_scale=extent)
    g.training_setup(opt)
    for it in range(1, iters + 1):
        g.update_learning_rate(it)
        if it % 1000 == 0:
            g.oneupSHdegree()
        cam, keep = random.choice(views)
        pkg = render(cam, g, pipe, background)
        img, dep = pkg["render"], pkg["surf_depth"]
        gt = cam.original_image.cuda()
        Ll1 = (torch.abs(img - gt) * keep).sum() / (keep.sum() * 3 + 1e-8)
        loss = (1 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1 - ssim(img * keep, gt * keep))
        ln = opt.lambda_normal if it > 7000 else 0.0
        ld = opt.lambda_dist if it > 3000 else 0.0
        normal_err = (1 - (pkg["rend_normal"] * pkg["surf_normal"]).sum(0))[None]
        loss = loss + ln * normal_err.mean() + ld * pkg["rend_dist"].mean()
        if lambda_depth > 0 and getattr(cam, "gt_depth", None) is not None:
            gd = cam.gt_depth.cuda()
            valid = (gd > 0) & (keep > 0.5)
            if valid.any():
                loss = loss + lambda_depth * torch.abs(dep - gd)[valid].mean()
        loss.backward()
        with torch.no_grad():
            vis, radii = pkg["visibility_filter"], pkg["radii"]
            if it < opt.densify_until_iter:
                g.max_radii2D[vis] = torch.max(g.max_radii2D[vis], radii[vis])
                g.add_densification_stats(pkg["viewspace_points"], vis)
                if it > opt.densify_from_iter and it % opt.densification_interval == 0:
                    size_threshold = 20 if it > opt.opacity_reset_interval else None
                    g.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, extent, size_threshold)
                if it % opt.opacity_reset_interval == 0:
                    g.reset_opacity()
            g.optimizer.step()
            g.optimizer.zero_grad(set_to_none=True)
        if it % 1000 == 0:
            print(f"  [recon] iter {it:5d} loss {loss.item():.4f} surfels {g.get_xyz.shape[0]}")
    g.active_sh_degree = g.max_sh_degree
    return g


def freeze(model, keep=("opacity", "scaling", "rotation", "xyz", "f_dc", "f_rest", "semantic")):
    """Zero the learning rate of the named optimiser param groups (freeze the model)."""
    for group in model.optimizer.param_groups:
        if group["name"] in keep:
            group["lr"] = 0.0
