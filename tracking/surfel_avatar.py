"""#2 (M10) — appearance(+semantic) SURFEL AVATAR bound to the body-model mesh.

The moving person becomes REAL surfels instead of a gray placeholder: one Gaussian per body-mesh
vertex with **shared, trainable** appearance (SH colour, opacity, scale, 32-d SigLIP2 semantic),
while each frame's **positions + orientations** are the posed mesh verts + normals (from MAMMA
`pred_vertices` or GT `person.ply`, already in world). Optimise the shared appearance against the 4
**masked person views** per timestep (photometric), then composite with the static surfel map.

Why it's tractable / on-thesis: the body model gives per-frame verts with **fixed correspondence**
(vertex i = same body point every frame), so we don't need LBS — the canonical surfels share
appearance and each frame supplies their pose. This is the surfel-preserving human (HUGS/GauHuman/
surfel-SMPL recipe): a body-model prior + optimised surfel appearance, posed per frame, composited
into the semantic surfel map.

Run (Colab, GPU):
  python -m tracking.surfel_avatar --conv_root <converted-scenes> \
     --smpl_root <mamma-exports | raw-BEHAVE> --smpl_source mamma \
     --smpl_model_dir <smplx_locked_head> --out <dir> [--iters 4000] [--person_embed person.npy]

VERIFY on first run: surfel normals orient the disks (facing the cameras) — if the body looks like
edge-on flecks, the normal→quat frame is off; the loss/PSNR + the compare grid tell you if appearance
converged. Semantic: `--person_embed` sets a constant per-surfel feature (e.g. SigLIP2('person')) so
the body is text-queryable; per-vertex body-part semantics is a future extension.
"""
import os
import json
import glob
import random
import argparse

import numpy as np


# ----------------------------------------------------------------------------- mesh topology + normals
def load_faces(args, n_verts):
    """Body-mesh faces (F,3). MAMMA/smplx -> from the SMPL-X model; GT -> from a BEHAVE person.ply."""
    if args.smpl_source == "mamma":
        import smplx
        m = smplx.create(args.smpl_model_dir, model_type="smplx", use_pca=False, batch_size=1)
        return np.asarray(m.faces, dtype=np.int64)
    # gt: read faces off any person.ply (topology is fixed across frames)
    import tracking.smpl_person as sp
    import trimesh
    for ts in sorted(glob.glob(f"{args.smpl_root}/t*.000")):
        mp = sp.find_person_mesh(ts)
        if mp:
            return np.asarray(trimesh.load(mp, process=False).faces, dtype=np.int64)
    raise FileNotFoundError("no person.ply to read faces from")


def vertex_normals(verts, faces):
    """Area-weighted per-vertex normals (N,3), unit."""
    n = np.zeros_like(verts)
    tri = verts[faces]                                              # (F,3,3)
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])     # (F,3)
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True); ln[ln == 0] = 1
    return n / ln


def normals_to_quats(normals):
    """Rotation (as w,x,y,z quats) whose 3rd axis = the vertex normal (2DGS disk lies in axes 1,2)."""
    from scipy.spatial.transform import Rotation
    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (len(n), 1))
    par = np.abs((n * ref).sum(1)) > 0.9                           # avoid ref parallel to n
    ref[par] = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(ref, n); t1 /= (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-9)
    t2 = np.cross(n, t1)
    R = np.stack([t1, t2, n], axis=2)                              # columns [t1,t2,n]
    q = Rotation.from_matrix(R).as_quat()                          # (N,4) xyzw
    return np.concatenate([q[:, 3:4], q[:, :3]], axis=1)           # -> wxyz


# ----------------------------------------------------------------------------- avatar training
def train_avatar(args):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import tracking.behave_data as bd
    import tracking.reconstruct as rc
    import tracking.smpl_person as sp
    from tracking.render_compose import render_composite
    from gaussian_renderer import render, GaussianModel
    from utils.graphics_utils import BasicPointCloud
    from utils.loss_utils import ssim

    os.makedirs(f"{args.out}/frames", exist_ok=True)
    dataset, opt, pipe = sp._fork_params()
    dataset.white_background = False
    bg = torch.zeros(3, device="cuda")

    TS, PMASK, extent = bd.load_timesteps(args.conv_root, dataset)
    N = len(TS)
    frames = sp.resolve_frames(TS, args.smpl_root, args.behave_frames, args.n_select,
                               args.select_stride, args.smpl_source)
    valid = [(ti, frames[ti]) for ti in range(N) if frames[ti] and os.path.isdir(frames[ti])]
    assert valid, "no SMPL frames resolved"

    # --- static map (person masked out, fused over spread timesteps) ---
    static_tis = sorted(set(np.linspace(0, N - 1, min(args.n_static, N)).astype(int).tolist()))
    spts, scols, views_static = bd.fuse_region(TS, PMASK, static_tis, "static")
    static = rc.reconstruct_masked(views_static, spts, scols, opt, pipe, bg, extent,
                                   dataset.sh_degree, args.recon_iters, args.lambda_depth)
    rc.freeze(static)

    # --- per-frame posed body: verts (aligned) + normals -> quats ---
    faces = None
    VERT, QUAT = {}, {}
    for ti, frame in valid:
        v, _ = sp.get_person_verts(frame, args)
        v = sp.align_to_scene(v, t=args.align_t, s=args.align_s)
        if faces is None:
            faces = load_faces(args, len(v))
        VERT[ti] = v.astype(np.float32)
        QUAT[ti] = normals_to_quats(vertex_normals(v, faces)).astype(np.float32)
    ref = valid[len(valid) // 2][0]                                # a mid frame to seed the canonical

    # --- canonical surfels: one per body vertex, GREY init; appearance trainable, xyz/rot overridden ---
    g = GaussianModel(dataset.sh_degree)
    g.create_from_pcd(BasicPointCloud(points=VERT[ref].astype(np.float64),
                                      colors=np.full_like(VERT[ref], 0.6, dtype=np.float64),
                                      normals=np.zeros_like(VERT[ref])), spatial_lr_scale=extent)
    g.active_sh_degree = g.max_sh_degree
    if args.person_embed and os.path.exists(args.person_embed):    # constant "person" semantic (queryable)
        emb = torch.tensor(np.load(args.person_embed), dtype=torch.float, device="cuda").reshape(1, -1)
        with torch.no_grad():
            g._semantic.data = emb.repeat(g._semantic.shape[0], 1)
    # optimise APPEARANCE only (positions/rotations come from the mesh each frame)
    optimizer = torch.optim.Adam([
        {"params": [g._features_dc], "lr": 0.01, "name": "f_dc"},
        {"params": [g._features_rest], "lr": 0.01 / 20.0, "name": "f_rest"},
        {"params": [g._opacity], "lr": 0.05, "name": "opacity"},
        {"params": [g._scaling], "lr": 0.005, "name": "scaling"},
    ], eps=1e-15)

    def frame_tensors(ti):
        return (torch.tensor(VERT[ti], device="cuda"),
                torch.tensor(QUAT[ti], device="cuda"))

    print(f"training avatar: {g.get_xyz.shape[0]} surfels x {len(valid)} frames, {args.iters} iters")
    for it in range(1, args.iters + 1):
        ti = random.choice(valid)[0]
        cam = random.choice(TS[ti]["cams"])
        keep = PMASK[(ti, cam.image_name)]
        xyz, rot = frame_tensors(ti)
        pkg = render(cam, g, pipe, bg, means3D_override=xyz, rotations_override=rot)
        img, gt = pkg["render"], cam.original_image.cuda()
        Ll1 = (torch.abs(img - gt) * keep).sum() / (keep.sum() * 3 + 1e-8)
        loss = (1 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1 - ssim(img * keep, gt * keep))
        loss.backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if it % 500 == 0:
            print(f"  iter {it:5d} loss {loss.item():.4f}")

    # --- render the textured avatar in the map + person-only PSNR ---
    def person_psnr(ti, cam):
        xyz, rot = frame_tensors(ti)
        comp = render_composite(cam, [static, g], pipe, bg,
                                xyz_overrides=[None, xyz], rot_overrides=[None, rot])["render"]
        return sp._masked_psnr(comp, cam.original_image.cuda(), PMASK[(ti, cam.image_name)]), comp

    try:
        import imageio.v2 as imageio
    except ImportError:
        imageio = None
    per_psnr, vframes = [], []
    for ti, _ in valid:
        cam = TS[ti]["cams"][args.view]
        with torch.no_grad():
            ps, comp = person_psnr(ti, cam)
        per_psnr.append(ps)
        gt_np, comp_np = sp._to_np(cam.original_image.cuda()), sp._to_np(comp)
        if imageio is not None:
            imageio.imwrite(f"{args.out}/frames/ts{ti:03d}.png", (comp_np * 255).astype(np.uint8))
            vframes.append((np.concatenate([gt_np, comp_np], 1) * 255).astype(np.uint8))
    if imageio is not None and vframes:
        imageio.mimwrite(f"{args.out}/sequence.mp4", vframes, fps=args.fps, macro_block_size=None)

    metrics = {"smpl_source": args.smpl_source, "surfels": int(g.get_xyz.shape[0]),
               "frames": len(valid), "iters": args.iters,
               "person_psnr_mean": float(np.nanmean(per_psnr)),
               "config": {k: v for k, v in vars(args).items()}}
    json.dump(metrics, open(f"{args.out}/metrics.json", "w"), indent=2)
    print("METRICS:", json.dumps(metrics, indent=2))

    show = [t for t, _ in valid[::max(1, len(valid) // 4)]][:4]
    fig, ax = plt.subplots(len(show), 2, figsize=(8, 3 * len(show))); ax = np.atleast_2d(ax)
    for r, ti in enumerate(show):
        cam = TS[ti]["cams"][args.view]
        with torch.no_grad():
            _, comp = person_psnr(ti, cam)
        ax[r, 0].imshow(sp._to_np(cam.original_image.cuda())); ax[r, 0].set_ylabel(f"ts {ti}")
        ax[r, 1].imshow(sp._to_np(comp))
        for a in ax[r]: a.set_xticks([]); a.set_yticks([])
    for c, t in enumerate(["GT", "static + avatar"]): ax[0, c].set_title(t)
    plt.tight_layout(); plt.savefig(f"{args.out}/compare.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    g.save_ply(f"{args.out}/avatar_canonical.ply")
    print("saved ->", args.out)


def main():
    p = argparse.ArgumentParser(description="Optimised appearance surfel avatar bound to the body mesh")
    p.add_argument("--conv_root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--smpl_source", choices=["gt", "mamma"], default="mamma")
    p.add_argument("--smpl_root", required=True)
    p.add_argument("--smpl_model_dir", default=None, help="SMPL-X model dir (for mamma faces)")
    p.add_argument("--behave_frames", nargs="*", default=None)
    p.add_argument("--n_select", type=int, default=24)
    p.add_argument("--select_stride", type=int, default=None)
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--recon_iters", type=int, default=7000, help="static-map recon iters")
    p.add_argument("--n_static", type=int, default=4)
    p.add_argument("--lambda_depth", type=float, default=1.0)
    p.add_argument("--view", type=int, default=0)
    p.add_argument("--align_s", type=float, default=1.0)
    p.add_argument("--align_t", type=float, nargs=3, default=None)
    p.add_argument("--person_embed", default=None, help="npy of a constant per-surfel semantic (e.g. SigLIP2 'person')")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--force_smplx", action="store_true", help="(smpl_person compat) ignore exported mesh, pose params")
    args = p.parse_args()
    train_avatar(args)


if __name__ == "__main__":
    main()
