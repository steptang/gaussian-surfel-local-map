#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

# Module-level guard so the "Custom device <X> failed" warning fires
# at most once per process even when hundreds of Cameras are constructed
# (typical training scenes have 100–500). Stores the value that was
# rejected so a different bad device down the line still warns.
_BAD_DEVICE_WARNED = set()


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 region_map=None, region_embeds=None, gt_depth=None,
                 px=0.5, py=0.5, dynamic_mask=None,
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            if data_device not in _BAD_DEVICE_WARNED:
                print(e)
                print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
                _BAD_DEVICE_WARNED.add(data_device)
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0) # move to device at dataloader to reduce VRAM requirement
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            # self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device) # do we need this?
            self.gt_alpha_mask = None

        # Semantic supervision artifacts (optional). Kept on CPU; moved to
        # GPU lazily at the loss site to avoid pinning VRAM for views that
        # aren't sampled in this iteration.
        # region_map:    (1, H, W) int16  -- per-pixel SAM3 region ID, 0 = background
        # region_embeds: (R+1, K_target)  -- per-region SigLIP2 image embedding
        self.region_map = region_map
        self.region_embeds = region_embeds

        # Optional RGB-D depth supervision. (1, H, W) float, metric units matching
        # the pose scale; 0 = no measurement. Kept on CPU, moved to GPU at the loss
        # site (like region_map) to avoid pinning VRAM for unsampled views.
        self.gt_depth = gt_depth

        # Optional dynamic-object mask (1, H, W) float in {0,1}; 1 = moving object, excluded
        # from the static-reconstruction loss. CPU-held, .cuda() at the loss site.
        self.dynamic_mask = dynamic_mask

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, px=px, py=py).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

