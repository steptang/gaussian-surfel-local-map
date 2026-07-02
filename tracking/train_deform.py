"""Build a deformable-object rig on converted BEHAVE scenes and evaluate it.

Pipeline (all from the fork's own modules -- notebooks just call this):
  1. load per-timestep scenes + person masks                 (tracking.behave_data)
  2. reconstruct the canonical person @ ref timestep         (tracking.reconstruct, keep=person)
  3. reconstruct the static map by fusing spread timesteps   (tracking.reconstruct, keep=non-person)
  4. train the deformation field (coarse pose + MLP Δpos/Δrot) with a local-rigidity prior
       - single-pose (default): canonical frozen, MLP does all motion
       - --multi_pose: jointly refine the canonical geometry with the deformation
  5. evaluate deform vs rigid (person-only + unified full-scene) on held-out timesteps
  6. save metrics.json, compare.png, per-timestep frames, a timesteps video, and PLYs

Run:  python -m tracking.train_deform --conv_root <converted_scene_dir> --out <output_dir>
"""
import os
import json
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from utils.loss_utils import ssim
from tracking import behave_data as bd
from tracking import reconstruct as rc
from tracking.deform import (DeformationField, knn_indices, local_rigidity_loss,
                             axis_angle_to_quat, quat_multiply)
from tracking.render_compose import render_composite


def _fork_params():
    from argparse import ArgumentParser
    p = ArgumentParser()
    lp, op, pp = ModelParams(p), OptimizationParams(p), PipelineParams(p)
    a = p.parse_args([])
    return lp.extract(a), op.extract(a), pp.extract(a)


def _warp(canonical, field, coarse_t, t, rigid, train_canonical):
    base = canonical.get_xyz if train_canonical else canonical.get_xyz.detach()
    rot0 = canonical.get_rotation if train_canonical else canonical.get_rotation.detach()
    if rigid:
        z = torch.zeros_like(base)
        return base + coarse_t, rot0, z, z
    dpos, daa = field(base.detach(), t)
    xyz = base + coarse_t + dpos
    rot = F.normalize(quat_multiply(axis_angle_to_quat(daa), rot0), dim=-1)
    return xyz, rot, dpos, daa


@torch.no_grad()
def _to_np(img):
    return img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def _psnr(img, gt, mk):
    mse = ((img - gt) ** 2 * mk).sum() / (mk.sum() * 3 + 1e-8)
    return -10.0 * float(np.log10(mse.item() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref_ts", type=int, default=0)
    ap.add_argument("--holdout_every", type=int, default=5)
    ap.add_argument("--recon_iters", type=int, default=7000)
    ap.add_argument("--deform_iters", type=int, default=5000)
    ap.add_argument("--lambda_depth", type=float, default=1.0)
    ap.add_argument("--lambda_reg", type=float, default=1e-3)
    ap.add_argument("--lambda_rot", type=float, default=1e-3)
    ap.add_argument("--lambda_rigid", type=float, default=100.0)
    ap.add_argument("--pos_freqs", type=int, default=6)
    ap.add_argument("--time_freqs", type=int, default=5)
    ap.add_argument("--n_static", type=int, default=4, help="# spread timesteps fused for the static map")
    ap.add_argument("--multi_pose", action="store_true",
                    help="Lever A: jointly co-adapt the canonical geometry with the deformation "
                         "(low LR; same surfel count). Else the canonical is frozen (single-pose).")
    ap.add_argument("--canonical_lr_scale", type=float, default=0.1)
    ap.add_argument("--densify_canonical", action="store_true",
                    help="Lever B: also GROW canonical coverage via densification during joint training "
                         "(fills surfaces occluded at the reference pose). Implies --multi_pose.")
    ap.add_argument("--densify_from", type=int, default=500)
    ap.add_argument("--densify_interval", type=int, default=200)
    ap.add_argument("--densify_until_frac", type=float, default=0.6,
                    help="stop densifying after this fraction of deform_iters (let it settle at the end)")
    args = ap.parse_args()
    if args.densify_canonical:
        args.multi_pose = True          # densification is only meaningful on a trainable canonical
    os.makedirs(f"{args.out}/frames", exist_ok=True)

    dataset, opt, pipe = _fork_params()
    dataset.white_background = False
    bg = torch.zeros(3, device="cuda")

    TS, PMASK, extent = bd.load_timesteps(args.conv_root, dataset)
    N = len(TS)
    assert N >= 3, f"need >=3 timesteps, got {N}"
    hold = [i for i in range(N) if i % args.holdout_every == 0 and i != args.ref_ts]
    train = [i for i in range(N) if i not in hold]
    print(f"{N} timesteps | train {len(train)} | holdout {len(hold)}")

    # --- canonical (person) @ ref timestep ---
    pts, cols, views_ref = bd.fuse_region(TS, PMASK, [args.ref_ts], "person")
    print("reconstructing canonical (person) @ ts", args.ref_ts)
    canonical = rc.reconstruct_masked(views_ref, pts, cols, opt, pipe, bg, extent,
                                      dataset.sh_degree, args.recon_iters, args.lambda_depth)

    # --- static map fused from spread timesteps (fills person-occlusion holes) ---
    static_tis = sorted(set(np.linspace(0, N - 1, args.n_static).astype(int).tolist()))
    spts, scols, views_static = bd.fuse_region(TS, PMASK, static_tis, "static")
    print("reconstructing static map from timesteps", static_tis)
    static = rc.reconstruct_masked(views_static, spts, scols, opt, pipe, bg, extent,
                                   dataset.sh_degree, args.recon_iters, args.lambda_depth)
    rc.freeze(static)

    # --- coarse pose + deformation field ---
    cents = bd.person_centroids(TS, PMASK)
    coarse = bd.coarse_translations(cents, args.ref_ts)
    C = canonical.get_xyz.mean(0).detach()
    S = float((canonical.get_xyz - C).norm(dim=1).max().item()) + 1e-6
    field = DeformationField(C, S, args.pos_freqs, args.time_freqs).cuda()
    mlp_opt = torch.optim.Adam(field.parameters(), lr=5e-4)

    if args.multi_pose:
        # Co-adapt the canonical GENTLY: positions drift slowly + colour adapts, but FREEZE
        # scale/rotation/opacity. Training those jointly with the deformation blows the surfels up
        # into streaks (scale explosion) and fights the MLP's Δrotation -> keep the reconstruction's solidity.
        for grp in canonical.optimizer.param_groups:
            if grp["name"] == "xyz":
                grp["lr"] *= args.canonical_lr_scale
            elif grp["name"] in ("scaling", "rotation", "opacity"):
                grp["lr"] = 0.0
    else:
        rc.freeze(canonical)

    nbr = knn_indices(canonical.get_xyz, k=8)
    densify_until = int(args.deform_iters * args.densify_until_frac)
    print(f"training deformation (multi_pose={args.multi_pose}, densify_canonical={args.densify_canonical})")
    for it in range(1, args.deform_iters + 1):
        ti = random.choice(train); t = TS[ti]["t"]
        xyz_def, rot_def, dpos, daa = _warp(canonical, field, coarse[ti], t, False, args.multi_pose)
        loss = 0.0
        pkgs = []
        for cam in TS[ti]["cams"]:
            pkg = render(cam, canonical, pipe, bg, means3D_override=xyz_def, rotations_override=rot_def)
            pkgs.append(pkg)
            img, dep = pkg["render"], pkg["surf_depth"]
            gt = cam.original_image.cuda(); mk = PMASK[(ti, cam.image_name)]
            Ll1 = (torch.abs(img - gt) * mk).sum() / (mk.sum() * 3 + 1e-8)
            loss = loss + 0.8 * Ll1 + 0.2 * (1.0 - ssim(img * mk, gt * mk))
            gd = cam.gt_depth.cuda(); vm = (gd > 0) & (mk > 0.5)
            if vm.any():
                loss = loss + args.lambda_depth * torch.abs(dep - gd)[vm].mean()
        loss = loss / len(TS[ti]["cams"])
        loss = loss + args.lambda_reg * (dpos ** 2).mean() + args.lambda_rot * (daa ** 2).mean()
        if nbr.shape[0] == dpos.shape[0]:
            loss = loss + args.lambda_rigid * local_rigidity_loss(dpos, nbr)
        loss.backward()
        # --- Lever B: grow canonical coverage via densification (from the deformed-render gradients) ---
        if args.densify_canonical and it < densify_until:
            with torch.no_grad():
                for pkg in pkgs:
                    vis, radii = pkg["visibility_filter"], pkg["radii"]
                    canonical.max_radii2D[vis] = torch.max(canonical.max_radii2D[vis], radii[vis])
                    canonical.add_densification_stats(pkg["viewspace_points"], vis)
                if it > args.densify_from and it % args.densify_interval == 0:
                    canonical.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, extent, 20)
                    nbr = knn_indices(canonical.get_xyz, k=8)   # neighbourhoods changed -> recompute
        mlp_opt.step(); mlp_opt.zero_grad(set_to_none=True)
        if args.multi_pose:
            canonical.optimizer.step(); canonical.optimizer.zero_grad(set_to_none=True)
        if it % 500 == 0:
            print(f"iter {it:5d} loss {loss.item():.4f} surfels {canonical.get_xyz.shape[0]}")

    # --- evaluate: person-only + unified, deform vs rigid ---
    @torch.no_grad()
    def person_psnr(ti, rigid):
        xyz, rot, _, _ = _warp(canonical, field, coarse[ti], TS[ti]["t"], rigid, False)
        vals = []
        for cam in TS[ti]["cams"]:
            img = render(cam, canonical, pipe, bg, means3D_override=xyz, rotations_override=rot)["render"]
            vals.append(_psnr(img, cam.original_image.cuda(), PMASK[(ti, cam.image_name)]))
        return float(np.mean(vals))

    @torch.no_grad()
    def unified_psnr(ti, rigid):
        xyz, rot, _, _ = _warp(canonical, field, coarse[ti], TS[ti]["t"], rigid, False)
        vals = []
        for cam in TS[ti]["cams"]:
            img = render_composite(cam, [static, canonical], pipe, bg,
                                   xyz_overrides=[None, xyz], rot_overrides=[None, rot])["render"]
            vals.append(_psnr(img, cam.original_image.cuda(), torch.ones_like(cam.original_image[:1].cuda())))
        return float(np.mean(vals))

    metrics = {
        "n_timesteps": N, "holdout": hold, "multi_pose": args.multi_pose,
        "person_deform": float(np.mean([person_psnr(i, False) for i in hold])),
        "person_rigid":  float(np.mean([person_psnr(i, True) for i in hold])),
        "unified_deform": float(np.mean([unified_psnr(i, False) for i in hold])),
        "unified_rigid":  float(np.mean([unified_psnr(i, True) for i in hold])),
    }
    json.dump(metrics, open(f"{args.out}/metrics.json", "w"), indent=2)
    print("METRICS:", json.dumps(metrics, indent=2))

    # --- compare grid on held-out timesteps (GT | deform | rigid, unified) ---
    @torch.no_grad()
    def unified_img(ti, rigid, cam):
        xyz, rot, _, _ = _warp(canonical, field, coarse[ti], TS[ti]["t"], rigid, False)
        return _to_np(render_composite(cam, [static, canonical], pipe, bg,
                                       xyz_overrides=[None, xyz], rot_overrides=[None, rot])["render"])
    show = hold[:4]
    fig, ax = plt.subplots(len(show), 3, figsize=(11, 3 * len(show))); ax = np.atleast_2d(ax)
    for r, ti in enumerate(show):
        cam = TS[ti]["cams"][0]
        ax[r, 0].imshow(_to_np(cam.original_image.cuda())); ax[r, 0].set_ylabel(f"ts {ti}")
        ax[r, 1].imshow(unified_img(ti, False, cam))
        ax[r, 2].imshow(unified_img(ti, True, cam))
        for a in ax[r]: a.set_xticks([]); a.set_yticks([])
    for c, ttl in enumerate(["GT (full)", "static + deform", "static + rigid"]): ax[0, c].set_title(ttl)
    plt.tight_layout(); plt.savefig(f"{args.out}/compare.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # --- per-timestep frames (for the notebook's slider) + a GT|deform video ---
    try:
        import imageio.v2 as imageio
    except ImportError:
        imageio = None
    vframes = []
    cam0 = TS[0]["cams"][0]
    for ti in range(N):
        cam = TS[ti]["cams"][0]
        gt = _to_np(cam.original_image.cuda())
        df = unified_img(ti, False, cam)
        imageio.imwrite(f"{args.out}/frames/ts{ti:03d}_gt.png", (gt * 255).astype(np.uint8)) if imageio else None
        imageio.imwrite(f"{args.out}/frames/ts{ti:03d}_deform.png", (df * 255).astype(np.uint8)) if imageio else None
        vframes.append((np.concatenate([gt, df], axis=1) * 255).astype(np.uint8))
    if imageio is not None:
        imageio.mimwrite(f"{args.out}/timesteps.mp4", vframes, fps=8, macro_block_size=None)

    canonical.save_ply(f"{args.out}/canonical.ply")
    static.save_ply(f"{args.out}/static.ply")
    print("saved outputs ->", args.out)


if __name__ == "__main__":
    main()
