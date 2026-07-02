"""Composite render: draw several GaussianModels together in one rasterizer pass.

The unified map query at time t = static surfels + each mover's deformed surfels, rendered as
one image. This concatenates the models' attributes and uses render()'s means3D/rotations
overrides to substitute a mover's deformed geometry without mutating the model.
"""
import torch

from gaussian_renderer import render, GaussianModel


def render_composite(viewpoint_camera, models, pipe, background,
                     xyz_overrides=None, rot_overrides=None):
    """Render ``models`` (list of GaussianModel) together.

    xyz_overrides / rot_overrides: optional lists (one entry per model) of (N_i,3)/(N_i,4)
    tensors that replace that model's positions/rotations (e.g. a deformed mover); ``None``
    entries use the model's own get_xyz / get_rotation.
    """
    n = len(models)
    xyz_overrides = xyz_overrides or [None] * n
    rot_overrides = rot_overrides or [None] * n
    xyz = [m.get_xyz if xyz_overrides[i] is None else xyz_overrides[i] for i, m in enumerate(models)]
    rot = [m.get_rotation if rot_overrides[i] is None else rot_overrides[i] for i, m in enumerate(models)]

    tmp = GaussianModel(models[0].max_sh_degree)
    tmp._xyz = torch.cat(xyz, 0)                                  # placeholder; means override used
    tmp._opacity = torch.cat([m._opacity for m in models], 0)
    tmp._scaling = torch.cat([m._scaling for m in models], 0)
    tmp._rotation = torch.cat([m._rotation for m in models], 0)   # placeholder; rot override used
    tmp._features_dc = torch.cat([m._features_dc for m in models], 0)
    tmp._features_rest = torch.cat([m._features_rest for m in models], 0)
    tmp._semantic = torch.cat([m._semantic for m in models], 0)
    tmp.active_sh_degree = tmp.max_sh_degree
    return render(viewpoint_camera, tmp, pipe, background,
                  means3D_override=torch.cat(xyz, 0), rotations_override=torch.cat(rot, 0))
