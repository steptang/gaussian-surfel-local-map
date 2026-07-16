"""Render the MAMMA (or GT) body as a SHADED TRIANGLE MESH inside the static surfel scene.

No surfels on the person: the static map renders via the fork rasterizer (RGB + depth), the body
renders as a classical mesh via pyrender (RGB + depth) with the SAME pinhole camera, and the two
composite by per-pixel DEPTH TEST — so the mesh correctly occludes / is occluded by the scene.
This is the "show the raw MAMMA output standing in our map" demo (cf. ma_vis, which overlays the
mesh on the RAW VIDEO — here the background is the reconstructed static scene, no real person).

Outputs under --out: frames/ts*.png (composite), overlay.mp4 (composite only), sequence.mp4 (GT |
composite side-by-side for context).

Run (Colab, GPU; needs `pip install pyrender`):
  python -m tracking.mesh_overlay --conv_root <scenes> --smpl_root <mamma-exports> \
    --smpl_source mamma --smpl_model_dir <smplx_dir> --static_ply <cache.ply> --out <dir>
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")      # headless GL on Colab

import math
import json
import argparse

import numpy as np


def cam_intrinsics(cam):
    """Pinhole (fx, fy, cx, cy) from a fork camera (same convention as behave_data.backproject)."""
    W, H = cam.image_width, cam.image_height
    fx = W / (2 * math.tan(cam.FoVx * 0.5))
    fy = H / (2 * math.tan(cam.FoVy * 0.5))
    cx = getattr(cam, "px", 0.5) * W
    cy = getattr(cam, "py", 0.5) * H
    return fx, fy, cx, cy


def render_mesh(renderer, pyrender, verts, faces, cam, color=(0.55, 0.65, 0.85)):
    """Shaded mesh RGB(A) + depth from the fork camera's viewpoint (OpenCV->OpenGL convention)."""
    import trimesh
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=(*color, 1.0),
                                             metallicFactor=0.0, roughnessFactor=0.6)
    mesh = pyrender.Mesh.from_trimesh(tm, material=mat, smooth=True)

    fx, fy, cx, cy = cam_intrinsics(cam)
    c2w = np.linalg.inv(cam.world_view_transform.T.cpu().numpy())
    pose_gl = c2w @ np.diag([1.0, -1.0, -1.0, 1.0])    # OpenCV cam -> OpenGL cam

    scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4], bg_color=[0, 0, 0, 0])
    scene.add(mesh)
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=50.0)
    scene.add(camera, pose=pose_gl)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=pose_gl)
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    return rgb.astype(np.float32) / 255.0, depth       # (H,W,4) in [0,1], (H,W) metres (0 = empty)


def main():
    p = argparse.ArgumentParser(description="Shaded body MESH depth-composited into the static scene")
    p.add_argument("--conv_root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--smpl_source", choices=["gt", "mamma"], default="mamma")
    p.add_argument("--smpl_root", required=True)
    p.add_argument("--smpl_model_dir", default=None, help="SMPL-X dir (mamma faces; gt uses person.ply)")
    p.add_argument("--static_ply", required=True, help="cached static map (run smpl_person/surfel_avatar once to create)")
    p.add_argument("--behave_frames", nargs="*", default=None)
    p.add_argument("--n_select", type=int, default=24)
    p.add_argument("--select_stride", type=int, default=None)
    p.add_argument("--view", type=int, default=0)
    p.add_argument("--mesh_color", type=float, nargs=3, default=(0.55, 0.65, 0.85))
    p.add_argument("--align_s", type=float, default=1.0)
    p.add_argument("--align_t", type=float, nargs=3, default=None)
    p.add_argument("--force_smplx", action="store_true")
    p.add_argument("--fps", type=int, default=8)
    args = p.parse_args()

    import torch
    import pyrender
    import imageio.v2 as imageio
    import tracking.behave_data as bd
    import tracking.smpl_person as sp
    import tracking.surfel_avatar as sa
    from gaussian_renderer import render, GaussianModel

    os.makedirs(f"{args.out}/frames", exist_ok=True)
    dataset, opt, pipe = sp._fork_params()
    dataset.white_background = False
    bg = torch.zeros(3, device="cuda")

    TS, PMASK, extent = bd.load_timesteps(args.conv_root, dataset)
    assert os.path.exists(args.static_ply), f"static cache missing: {args.static_ply}"
    static = GaussianModel(dataset.sh_degree)
    static.load_ply(args.static_ply)
    print(f"static map loaded ({static.get_xyz.shape[0]} surfels)")

    frames = sp.resolve_frames(TS, args.smpl_root, args.behave_frames, args.n_select,
                               args.select_stride, args.smpl_source)
    valid = [(ti, frames[ti]) for ti in range(len(TS)) if frames[ti] and os.path.isdir(frames[ti])]
    assert valid, "no SMPL frames resolved"

    cam0 = TS[valid[0][0]]["cams"][args.view]
    renderer = pyrender.OffscreenRenderer(cam0.image_width, cam0.image_height)
    faces = None
    vframes, oframes = [], []
    for ti, frame in valid:
        verts, _ = sp.get_person_verts(frame, args)
        verts = sp.align_to_scene(verts, t=args.align_t, s=args.align_s)
        if faces is None:
            faces = sa.load_faces(args, len(verts))

        cam = TS[ti]["cams"][args.view]
        with torch.no_grad():
            pkg = render(cam, static, pipe, bg, render_semantic=False)
        srgb = pkg["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy()          # (H,W,3)
        sdep = pkg["surf_depth"][0].cpu().numpy()                                 # (H,W)

        mrgb, mdep = render_mesh(renderer, pyrender, verts, faces, cam, args.mesh_color)
        # depth test: mesh pixel wins where the mesh exists AND is nearer than the scene surface
        hit = (mdep > 0) & ((sdep <= 0) | (mdep < sdep))
        comp = srgb.copy()
        comp[hit] = mrgb[hit][:, :3]

        imageio.imwrite(f"{args.out}/frames/ts{ti:03d}.png", (comp * 255).astype(np.uint8))
        gt = cam.original_image.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        vframes.append((np.concatenate([gt, comp], 1) * 255).astype(np.uint8))
        oframes.append((comp * 255).astype(np.uint8))
        print(f"ts {ti:03d} mesh px: {int(hit.sum())}")
    renderer.delete()

    imageio.mimwrite(f"{args.out}/overlay.mp4", oframes, fps=args.fps, macro_block_size=None)
    imageio.mimwrite(f"{args.out}/sequence.mp4", vframes, fps=args.fps, macro_block_size=None)
    json.dump({"frames": len(valid), "config": vars(args)}, open(f"{args.out}/meta.json", "w"),
              indent=2, default=str)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
