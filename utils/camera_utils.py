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

from scene.cameras import Camera
import numpy as np
import torch
from PIL import Image
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False


def _load_region_map(regions_path, resolution):
    """Load uint16 SAM3 region map and downsample to (W, H) with nearest.

    Returns int16 tensor of shape (1, H, W); int16 is enough for ~32k regions
    per image and is what cosine_region_loss indexes with.
    """
    if regions_path is None:
        return None
    img = Image.open(regions_path)
    if img.size != resolution:
        img = img.resize(resolution, Image.NEAREST)
    arr = np.array(img, dtype=np.int32)
    return torch.from_numpy(arr.astype(np.int16))[None]   # (1, H, W)


def _load_region_embeds(embeds_path):
    """Load (R+1, K_target) float16 embedding table; convert to float32."""
    if embeds_path is None:
        return None
    arr = np.load(embeds_path)
    return torch.from_numpy(arr.astype(np.float32))


def _load_dynamic_mask(mask_path, resolution):
    """Load a per-frame dynamic-object mask (png, white = moving object) and resize (W,H) nearest.

    Returns (1, H, W) float in {0, 1}; 1 = dynamic (excluded from the static-reconstruction loss).
    """
    if mask_path is None:
        return None
    img = Image.open(mask_path).convert("L")
    if img.size != resolution:
        img = img.resize(resolution, Image.NEAREST)
    arr = (np.array(img) > 127).astype(np.float32)
    return torch.from_numpy(arr)[None]   # (1, H, W)


def _load_depth(depth_path, resolution):
    """Load a per-frame metric depth map (.npy, H x W float) and resize to (W, H).

    Returns (1, H, W) float32 tensor; 0 = no measurement. Uses nearest resize so
    depth discontinuities aren't blurred (and invalid 0s aren't averaged in).
    """
    if depth_path is None:
        return None
    arr = np.load(depth_path).astype(np.float32)
    t = torch.from_numpy(arr)[None, None]            # (1, 1, H, W)
    W, H = resolution
    if (t.shape[3], t.shape[2]) != (W, H):
        t = torch.nn.functional.interpolate(t, size=(H, W), mode="nearest")
    return t[0]                                      # (1, H, W)

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if len(cam_info.image.split()) > 3:
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    region_map = _load_region_map(getattr(cam_info, "regions_path", None), resolution)
    region_embeds = _load_region_embeds(getattr(cam_info, "embeds_path", None))
    gt_depth = _load_depth(getattr(cam_info, "depth_path", None), resolution)
    dynamic_mask = _load_dynamic_mask(getattr(cam_info, "dynamic_mask_path", None), resolution)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  region_map=region_map, region_embeds=region_embeds, gt_depth=gt_depth,
                  px=getattr(cam_info, "px", 0.5), py=getattr(cam_info, "py", 0.5),
                  dynamic_mask=dynamic_mask)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry