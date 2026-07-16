"""Load the converted per-timestep BEHAVE scenes and the person masks for the deformation rig.

The converter (preprocess) writes one fork scene per timestep: ``timestep_XXXXX/`` with a
4-camera ``transforms_train.json``, ``depth/``, ``masks/k{0-3}.person.png`` and a
back-projected ``points3d.ply``. This module turns those into the in-memory structures the
rig needs: cameras + per-(timestep,camera) person masks + per-frame time, plus helpers to
back-project a masked region to world points and to compute per-timestep person centroids.
"""
import os
import glob
import math

import numpy as np
import torch
import cv2

from scene import Scene
from gaussian_renderer import GaussianModel


def _person_mask(ts_dir, cam):
    p = f"{ts_dir}/masks/{cam.image_name}.person.png"
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return torch.ones((1, cam.image_height, cam.image_width), device="cuda")
    if (m.shape[1], m.shape[0]) != (cam.image_width, cam.image_height):
        m = cv2.resize(m, (cam.image_width, cam.image_height), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy((m > 127).astype(np.float32))[None].cuda()


def load_timesteps(conv_root, dataset):
    """Return (TS, PMASK, extent).

    TS: list of {'dir', 'cams' (list of 4 Camera), 't' (float in [0,1])} in timestep order.
    PMASK: {(timestep_index, camera.image_name): (1,H,W) person mask}.
    Only ``timestep_*`` dirs with a written ``transforms_train.json`` are loaded (a crashed
    conversion may leave half-written dirs).
    """
    ts_dirs = sorted(d for d in glob.glob(f"{conv_root}/timestep_*")
                     if os.path.exists(f"{d}/transforms_train.json"))
    TS, PMASK, extent = [], {}, None
    for i, td in enumerate(ts_dirs):
        dataset.source_path = td
        g = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, g, shuffle=False)
        cams = scene.getTrainCameras()
        if extent is None:
            extent = scene.cameras_extent
        for cam in cams:
            PMASK[(i, cam.image_name)] = _person_mask(td, cam)
        TS.append({"dir": td, "cams": cams, "t": i / max(1, len(ts_dirs) - 1)})
    return TS, PMASK, extent


def keep_mask(PMASK, ti, cam, want, pad=0):
    """'person' -> the person mask; 'static' -> its complement (everything but the person).

    ``pad`` (px) optionally dilates the person mask before complementing — a safety margin against
    mask undersegmentation bleeding person pixels into static supervision (precaution, off by
    default; no such bleed has actually been confirmed in the BEHAVE masks).
    """
    pm = PMASK[(ti, cam.image_name)]
    if want == "person":
        return pm
    if pad > 0:
        k = 2 * int(pad) + 1
        pm = torch.nn.functional.max_pool2d(pm[None], k, stride=1, padding=int(pad))[0]
    return 1.0 - pm


def fuse_region_pad(want, pad=None):
    """Default: no dilation. (The pad mechanism exists in case mask-undersegmentation bleed into the
    static map is ever actually observed — it was once suspected and turned out to be a misreading.)"""
    return 0 if pad is None else pad


def backproject(cam, keep):
    """Back-project the kept (mask & depth>0) pixels of ``cam`` to world points + colours.

    Uses the camera's own pinhole intrinsics (FoV + principal-point fraction) and the same
    world convention the fork renders with, so init points and render cameras agree.
    """
    W, H = cam.image_width, cam.image_height
    fx = W / (2 * math.tan(cam.FoVx * 0.5))
    fy = H / (2 * math.tan(cam.FoVy * 0.5))
    cx = getattr(cam, "px", 0.5) * W
    cy = getattr(cam, "py", 0.5) * H
    d = cam.gt_depth.cuda().squeeze(0)
    m = keep.squeeze(0) > 0.5
    img = cam.original_image.cuda()
    vv, uu = torch.meshgrid(torch.arange(H, device="cuda"), torch.arange(W, device="cuda"), indexing="ij")
    valid = m & (d > 0)
    z = d[valid]; u = uu[valid].float(); v = vv[valid].float()
    x = (u - cx) / fx * z; y = (v - cy) / fy * z
    pc = torch.stack([x, y, z], 1).double().cpu().numpy()
    c2w = np.linalg.inv(cam.world_view_transform.T.cpu().numpy())
    world = (c2w[:3, :3] @ pc.T).T + c2w[:3, 3]
    return world, img[:, valid].T.cpu().numpy()


def fuse_region(TS, PMASK, tis, want, cap=300000, pad=None):
    """Back-project ``want`` across all 4 views of every timestep in ``tis`` -> (points, colours).

    Also returns the list of (camera, keep_mask) views for reconstruction supervision.
    ``pad``: person-mask dilation in px (default 12 for 'static', 0 for 'person' — see keep_mask).
    """
    pad = fuse_region_pad(want, pad)
    pts, cols, views = [], [], []
    for ti in tis:
        for cam in TS[ti]["cams"]:
            k = keep_mask(PMASK, ti, cam, want, pad)
            w, c = backproject(cam, k)
            pts.append(w); cols.append(c); views.append((cam, k))
    pts = np.concatenate(pts); cols = np.concatenate(cols)
    if len(pts) > cap:
        sel = np.random.choice(len(pts), cap, replace=False)
        pts, cols = pts[sel], cols[sel]
    return pts, cols, views


def person_centroids(TS, PMASK):
    """Per-timestep world centroid of the person (from all 4 views' back-projected points)."""
    cents = {}
    for ti in range(len(TS)):
        pts = [backproject(cam, keep_mask(PMASK, ti, cam, "person"))[0] for cam in TS[ti]["cams"]]
        cents[ti] = np.concatenate(pts).mean(0)
    return cents


def coarse_translations(cents, ref_ts):
    """Centroid-seeded rigid pose per timestep: centroid(t) - centroid(ref) (the M2 layer)."""
    return {i: torch.tensor(cents[i] - cents[ref_ts], dtype=torch.float, device="cuda") for i in cents}
