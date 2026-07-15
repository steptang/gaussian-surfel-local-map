"""Part A / A.5 / B of the M10 plan: render a learning-based 3D *body model* moving through the
static surfel map, and extract a root trajectory for prediction.

Pipeline (mirrors ``train_deform`` but the mover is an SMPL/SMPL-X body, NOT a free-deform surfel
cluster):

  static map  = rc.reconstruct_masked(keep=non-person, fused over spread timesteps)   [as today]
  per frame t = pose an SMPL-X body -> world verts -> a small GaussianModel ("body")
  render(t)   = render_composite([static, body], cam)                                 [one pass]

Two SMPL sources (``--smpl_source``):
  * ``gt``    — BEHAVE's own SMPL(-H) fits (``person/fit02/person_fit.pkl``).  **Part A**: de-risk
                compositing with a known-good body, decoupled from estimation.
  * ``mamma`` — MAMMA's estimated SMPL-X (a per-frame ``*.npz``/``*.pkl`` written by running MAMMA on
                the 4 BEHAVE views).  **Part A.5**: swap GT -> estimated, same downstream path.

Outputs (under ``--out``): per-timestep composite frames, a GT|render ``.mp4``, per-frame person-only
PSNR (``metrics.json``), and ``root_traj.npy`` (world root/pelvis per timestep) consumed by Part B.

**Runs on Colab (A100).** The CUDA rasterizer submodule can't build on Mac; ``create_from_pcd`` calls
distCUDA2. Import-safe on Mac (torch/smplx pulled in lazily inside the functions that need a GPU).

--------------------------------------------------------------------------------------------------
THREE project-specific unknowns are marked `# TODO(verify)` — check them on the first Colab run:
  (1) BEHAVE SMPL pkl keys + model type (SMPL-H vs SMPL); MAMMA's export layout.
  (2) mapping converted ``timestep_i`` <-> the raw BEHAVE frame dir the SMPL came from.
  (3) world-frame alignment: BEHAVE SMPL should already sit in the Kinect world frame the converter
      renders in (=> identity), but eyeball the first frame and set ``--align_*`` if it floats/rotates.
--------------------------------------------------------------------------------------------------
"""
import os
import re
import glob
import json
import pickle
import argparse

import numpy as np


# ----------------------------------------------------------------------------- SMPL loading / posing
def find_behave_frames(seq_root):
    """Ordered raw BEHAVE frame dirs (``.../Date0X_Sub0Y_obj/t*.000``).

    Must line up 1:1, in order, with the converted ``timestep_*`` dirs `bd.load_timesteps` returns.
    The converter may subsample timesteps -> pass ``--behave_frames`` explicitly if this glob and the
    converted set disagree.  # TODO(verify) mapping (2).
    """
    return sorted(glob.glob(f"{seq_root}/t*.000"))


def resolve_frames(TS, smpl_root, override=None, n_select=24, stride=None):
    """Map each converted timestep dir -> its raw BEHAVE frame dir.

    The S1-3 converter (colab BEHAVE_deform_S1-3) selects
    ``SEL = sorted(t*.000)[::stride][:n_select]`` (default n_select=24, stride=len(all)//n_select)
    and writes each scene as ``timestep_{k:05d}`` where **k = index INTO SEL**. So a converted
    ``timestep_00007`` maps to ``SEL[7]`` — NOT raw frame ``t0007.000`` and NOT the 7th loaded TS.
    Pass ``--behave_frames`` to override, or ``--n_select``/``--select_stride`` if the convert used
    other values. Returns a list parallel to ``TS`` (None where k is out of range).
    """
    if override:
        return list(override)
    allf = find_behave_frames(smpl_root)                  # sorted t*.000
    st = stride or max(1, len(allf) // max(1, n_select))
    sel = allf[::st][:n_select]
    out = []
    for i, ts in enumerate(TS):
        m = re.search(r"timestep_(\d+)", os.path.basename(ts["dir"].rstrip("/")))
        k = int(m.group(1)) if m else i
        out.append(sel[k] if k < len(sel) else None)
    return out


def load_behave_smpl(frame_dir):
    """Read one BEHAVE person fit -> dict(pose, betas, trans, gender).

    BEHAVE ships SMPL-H fits at ``<frame>/person/fit02/person_fit.pkl``. Keys vary slightly across
    releases; handle the common ones + fall back gracefully.  # TODO(verify) keys/model-type (1).
    """
    for cand in (f"{frame_dir}/person/fit02/person_fit.pkl",
                 f"{frame_dir}/person/fit01/person_fit.pkl"):
        if os.path.exists(cand):
            with open(cand, "rb") as f:
                d = pickle.load(f, encoding="latin1")
            pose = np.asarray(d.get("pose", d.get("full_pose"))).reshape(-1)
            betas = np.asarray(d.get("betas"))[:10]
            trans = np.asarray(d.get("trans", d.get("transl", np.zeros(3)))).reshape(3)
            gender = str(d.get("gender", "male"))
            return {"pose": pose, "betas": betas, "trans": trans, "gender": gender}
    raise FileNotFoundError(f"no BEHAVE person fit under {frame_dir}")


def load_mamma_smpl(frame_dir):
    """Read one MAMMA per-frame SMPL-X export -> dict(pose, betas, trans, gender).

    MAMMA writes SMPL-X params per frame (exact filename set on first run).  # TODO(verify) layout (1).
    """
    cands = glob.glob(f"{frame_dir}/*.npz") + glob.glob(f"{frame_dir}/*mamma*.pkl")
    if not cands:
        raise FileNotFoundError(f"no MAMMA export under {frame_dir}")
    p = cands[0]
    d = np.load(p, allow_pickle=True) if p.endswith(".npz") else pickle.load(open(p, "rb"))
    get = (lambda k, default=None: d[k] if k in d else default)
    body = get("body_pose")
    globo = get("global_orient")
    pose = (np.concatenate([np.asarray(globo).reshape(-1), np.asarray(body).reshape(-1)])
            if body is not None else np.asarray(get("pose")).reshape(-1))
    return {"pose": pose, "betas": np.asarray(get("betas"))[:10],
            "trans": np.asarray(get("transl", np.zeros(3))).reshape(3),
            "gender": str(get("gender", "neutral")), "model_type": "smplx"}


def find_person_mesh(frame_dir):
    """BEHAVE's per-frame person mesh under ``<frame>/person/`` (no SMPL model files needed).

    Layout varies (fit02/ vs fit01/ vs a flat person.ply). Search recursively; prefer a fitted
    SMPL mesh (path contains 'fit') over the raw fused scan.
    """
    hits = glob.glob(f"{frame_dir}/person/**/person.ply", recursive=True)
    if not hits:
        return None
    fits = [h for h in hits if "fit" in os.path.dirname(h).lower()]
    return (fits or hits)[0]


def load_ply_verts(path):
    """(N,3) vertices from a PLY (BEHAVE's fitted person mesh)."""
    try:
        from plyfile import PlyData
        v = PlyData.read(path)["vertex"]
        return np.stack([np.asarray(v["x"]), np.asarray(v["y"]), np.asarray(v["z"])], 1).astype(np.float64)
    except Exception:
        import trimesh
        return np.asarray(trimesh.load(path, process=False).vertices, dtype=np.float64)


_SMPL_CACHE = {}


def build_smpl_model(model_dir, gender, model_type="smplh"):
    """Cached ``smplx`` body model (one per (type, gender))."""
    import smplx  # lazy: only needed on the GPU box
    key = (model_type, gender)
    if key not in _SMPL_CACHE:
        _SMPL_CACHE[key] = smplx.create(model_dir, model_type=model_type, gender=gender,
                                        use_pca=False, batch_size=1)
    return _SMPL_CACHE[key]


def pose_smpl(model, params, model_type="smplh"):
    """SMPL params -> (verts (N,3), pelvis (3)) numpy in the fit's world frame."""
    import torch
    n_body = 63 if model_type == "smplx" else 63          # 21 body joints * 3
    pose = torch.tensor(params["pose"], dtype=torch.float32).reshape(1, -1)
    go = pose[:, :3]
    body = pose[:, 3:3 + n_body]
    betas = torch.tensor(params["betas"], dtype=torch.float32).reshape(1, -1)
    trans = torch.tensor(params["trans"], dtype=torch.float32).reshape(1, 3)
    out = model(global_orient=go, body_pose=body, betas=betas, transl=trans)
    verts = out.vertices[0].detach().cpu().numpy()
    pelvis = out.joints[0, 0].detach().cpu().numpy()
    return verts, pelvis


def align_to_scene(pts, R=None, s=1.0, t=None):
    """Optional similarity transform fit-world -> scene-world. Identity by default (unknown (3))."""
    if R is None and t is None and s == 1.0:
        return pts
    R = np.eye(3) if R is None else np.asarray(R)
    t = np.zeros(3) if t is None else np.asarray(t)
    return (s * (R @ pts.T)).T + t


def get_person_verts(frame_dir, args):
    """(verts (N,3), pelvis (3)) in the fit's world frame.

    GT prefers BEHAVE's per-frame fitted mesh ``person/fit02/person.ply`` -> NO SMPL model files
    needed. Falls back to posing params via smplx (needs ``--smpl_model_dir``); MAMMA always poses
    via smplx. Pelvis = mesh centroid (mesh path) or SMPL joint-0 (smplx path) -> the Part-B root.
    """
    if args.smpl_source == "gt" and not args.force_smplx:
        mp = find_person_mesh(frame_dir)
        if mp is not None:
            v = load_ply_verts(mp)
            return v, v.mean(0)
    loader = load_mamma_smpl if args.smpl_source == "mamma" else load_behave_smpl
    params = loader(frame_dir)
    mtype = params.get("model_type", "smplx" if args.smpl_source == "mamma" else "smplh")
    assert args.smpl_model_dir, "posing via smplx needs --smpl_model_dir (or use BEHAVE's person.ply mesh)"
    return pose_smpl(build_smpl_model(args.smpl_model_dir, params["gender"], mtype), params, mtype)


# ----------------------------------------------------------------------------- body -> Gaussians
def body_gaussians(verts, sh_degree, extent, color=(0.75, 0.75, 0.78),
                   opacity=0.95, scale_mul=1.0):
    """Seed a solid GaussianModel from posed body vertices (one surfel per vertex).

    ``create_from_pcd`` auto-initialises the 32-d ``_semantic`` slot to zeros, so the body composites
    cleanly with the (semantic) static map via ``render_composite``. We then SOLIDIFY it (high opacity,
    NN-sized scales) since it's a fixed posed mesh, not a thing we optimise.
    """
    import torch
    from utils.graphics_utils import BasicPointCloud
    from gaussian_renderer import GaussianModel

    cols = np.tile(np.asarray(color, dtype=np.float32)[None], (len(verts), 1))
    g = GaussianModel(sh_degree)
    g.create_from_pcd(BasicPointCloud(points=verts.astype(np.float64),
                                      colors=cols.astype(np.float64),
                                      normals=np.zeros_like(verts)), spatial_lr_scale=extent)
    with torch.no_grad():
        g._opacity.data = g.inverse_opacity_activation(
            opacity * torch.ones_like(g.get_opacity)).detach()
        if scale_mul != 1.0:
            g._scaling.data = (g._scaling.data + float(np.log(scale_mul)))
    g.active_sh_degree = g.max_sh_degree
    return g


# ----------------------------------------------------------------------------- helpers
def _to_np(t):
    return t.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def _masked_psnr(render_t, gt_t, mask_t):
    """PSNR over the person mask only (person-region fidelity, cf. train_deform's person_psnr)."""
    import torch
    m = (mask_t > 0.5)
    if m.sum() == 0:
        return float("nan")
    mse = (((render_t - gt_t) ** 2) * m).sum() / (m.sum() * 3 + 1e-8)
    return float((-10.0 * torch.log10(mse + 1e-12)).item())


def _fork_params():
    """ModelParams/OptimizationParams/PipelineParams with fork defaults (mirrors train_deform)."""
    from arguments import ModelParams, OptimizationParams, PipelineParams
    p = argparse.ArgumentParser()
    lp, op, pp = ModelParams(p), OptimizationParams(p), PipelineParams(p)
    a = p.parse_args([])
    return lp.extract(a), op.extract(a), pp.extract(a)


# ----------------------------------------------------------------------------- Part A / A.5 render
def render_sequence(args):
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import tracking.behave_data as bd
    import tracking.reconstruct as rc
    from gaussian_renderer import render  # noqa: F401 (kept for parity / debugging)
    from tracking.render_compose import render_composite

    os.makedirs(f"{args.out}/frames", exist_ok=True)
    dataset, opt, pipe = _fork_params()
    dataset.white_background = False
    bg = torch.zeros(3, device="cuda")

    TS, PMASK, extent = bd.load_timesteps(args.conv_root, dataset)
    N = len(TS)
    assert N >= 2, f"need >=2 timesteps, got {N}"

    # --- static map fused from spread timesteps (fills person-occlusion holes) [as train_deform] ---
    static_tis = sorted(set(np.linspace(0, N - 1, min(args.n_static, N)).astype(int).tolist()))
    spts, scols, views_static = bd.fuse_region(TS, PMASK, static_tis, "static")
    print("reconstructing static map from timesteps", static_tis)
    static = rc.reconstruct_masked(views_static, spts, scols, opt, pipe, bg, extent,
                                   dataset.sh_degree, args.recon_iters, args.lambda_depth)
    rc.freeze(static)

    # --- SMPL source: one params dict per timestep ---
    frames = resolve_frames(TS, args.smpl_root, args.behave_frames, args.n_select, args.select_stride)
    assert len(frames) == N and all(f and os.path.isdir(f) for f in frames), \
        f"frame mapping failed (unknown (2)); resolved: {frames}"
    print("timestep -> raw frame mapping (verify (2)):")
    for ti in range(min(N, 4)):
        print(f"  {os.path.basename(TS[ti]['dir'])} -> {os.path.basename(frames[ti])}")

    try:
        import imageio.v2 as imageio
    except ImportError:
        imageio = None

    per_psnr, root_traj, vframes = [], [], []
    for ti in range(N):
        verts, pelvis = get_person_verts(frames[ti], args)
        verts = align_to_scene(verts, t=args.align_t, s=args.align_s)
        pelvis = align_to_scene(pelvis[None], t=args.align_t, s=args.align_s)[0]
        root_traj.append(pelvis)

        body = body_gaussians(verts, dataset.sh_degree, extent,
                              opacity=args.body_opacity, scale_mul=args.body_scale)

        cam = TS[ti]["cams"][args.view]
        with torch.no_grad():
            comp = render_composite(cam, [static, body], pipe, bg)["render"]
        gt = cam.original_image.cuda()
        pm = PMASK[(ti, cam.image_name)]
        per_psnr.append(_masked_psnr(comp, gt, pm))

        gt_np, comp_np = _to_np(gt), _to_np(comp)
        if imageio is not None:
            imageio.imwrite(f"{args.out}/frames/ts{ti:03d}.png", (comp_np * 255).astype(np.uint8))
            vframes.append((np.concatenate([gt_np, comp_np], 1) * 255).astype(np.uint8))
        print(f"ts {ti:03d} | person-PSNR {per_psnr[-1]:.2f}")

    if imageio is not None and vframes:
        imageio.mimwrite(f"{args.out}/sequence.mp4", vframes, fps=args.fps, macro_block_size=None)

    root_traj = np.stack(root_traj)
    np.save(f"{args.out}/root_traj.npy", root_traj)
    metrics = {
        "smpl_source": args.smpl_source,
        "n_timesteps": N,
        "person_psnr_mean": float(np.nanmean(per_psnr)),
        "person_psnr_per_ts": [None if np.isnan(x) else round(x, 3) for x in per_psnr],
        "static_surfels": int(static.get_xyz.shape[0]),
        "config": {k: v for k, v in vars(args).items()},
    }
    json.dump(metrics, open(f"{args.out}/metrics.json", "w"), indent=2)
    print("METRICS:", json.dumps(metrics, indent=2))

    # quick GT|render grid on a few frames
    show = list(range(0, N, max(1, N // 4)))[:4]
    fig, ax = plt.subplots(len(show), 2, figsize=(8, 3 * len(show))); ax = np.atleast_2d(ax)
    for r, ti in enumerate(show):
        cam = TS[ti]["cams"][args.view]
        verts, _ = get_person_verts(frames[ti], args)
        verts = align_to_scene(verts, t=args.align_t, s=args.align_s)
        body = body_gaussians(verts, dataset.sh_degree, extent, opacity=args.body_opacity, scale_mul=args.body_scale)
        with torch.no_grad():
            comp = _to_np(render_composite(cam, [static, body], pipe, bg)["render"])
        ax[r, 0].imshow(_to_np(cam.original_image.cuda())); ax[r, 0].set_ylabel(f"ts {ti}")
        ax[r, 1].imshow(comp)
        for a in ax[r]: a.set_xticks([]); a.set_yticks([])
    for c, ttl in enumerate(["GT", f"static + {args.smpl_source} body"]): ax[0, c].set_title(ttl)
    plt.tight_layout(); plt.savefig(f"{args.out}/compare.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    static.save_ply(f"{args.out}/static.ply")
    print("saved outputs ->", args.out)
    return root_traj


# ----------------------------------------------------------------------------- Part B: motion + predict
def motion_predict(root_traj, hist=8, horizon=8):
    """Constant-velocity baseline forecast over the root trajectory -> ADE/FDE (metres).

    Slides a window: from frames [i-hist, i] fit a mean velocity, predict [i+1, i+horizon], score
    vs the actual future. This is the beat-me baseline; a learned predictor (UPTor/Trajectron++)
    replaces the const-vel step later.  Circle-walk => non-zero curvature => const-vel IS beatable.
    """
    T = len(root_traj)
    ades, fdes = [], []
    for i in range(hist, T - horizon):
        vel = (root_traj[i] - root_traj[i - hist]) / hist
        steps = np.arange(1, horizon + 1)[:, None]
        pred = root_traj[i][None] + steps * vel[None]
        gt = root_traj[i + 1:i + 1 + horizon]
        err = np.linalg.norm(pred - gt, axis=1)
        ades.append(err.mean()); fdes.append(err[-1])
    return {"windows": len(ades),
            "const_vel_ADE": float(np.mean(ades)) if ades else float("nan"),
            "const_vel_FDE": float(np.mean(fdes)) if fdes else float("nan")}


def main():
    p = argparse.ArgumentParser(description="Render an SMPL body through the static map + motion baseline")
    p.add_argument("--conv_root", required=True, help="converted per-timestep BEHAVE scenes (timestep_*)")
    p.add_argument("--out", required=True)
    p.add_argument("--smpl_source", choices=["gt", "mamma"], default="gt")
    p.add_argument("--smpl_root", help="raw BEHAVE sequence root (for gt) or MAMMA export root (mamma)")
    p.add_argument("--behave_frames", nargs="*", default=None,
                   help="explicit per-timestep SMPL frame dirs (overrides the SEL reconstruction)")
    p.add_argument("--n_select", type=int, default=24,
                   help="converter's N_TIMESTEPS (SEL = sorted(t*.000)[::stride][:n_select])")
    p.add_argument("--select_stride", type=int, default=None,
                   help="converter's frame stride (default len(all)//n_select), to reproduce SEL")
    p.add_argument("--smpl_model_dir", default=None,
                   help="dir with SMPL/SMPL-H/SMPL-X model files (only needed to POSE params; GT uses "
                        "BEHAVE's person.ply mesh, so it's optional for --smpl_source gt)")
    p.add_argument("--force_smplx", action="store_true",
                   help="pose GT via smplx params even if a person.ply mesh exists (needs --smpl_model_dir)")
    p.add_argument("--view", type=int, default=0, help="which of the 4 cameras to render from")
    p.add_argument("--n_static", type=int, default=4)
    p.add_argument("--recon_iters", type=int, default=7000)
    p.add_argument("--lambda_depth", type=float, default=1.0)
    p.add_argument("--body_opacity", type=float, default=0.95)
    p.add_argument("--body_scale", type=float, default=1.0)
    p.add_argument("--align_s", type=float, default=1.0)
    p.add_argument("--align_t", type=float, nargs=3, default=None)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--pred_hist", type=int, default=8)
    p.add_argument("--pred_horizon", type=int, default=8)
    args = p.parse_args()

    root_traj = render_sequence(args)
    pred = motion_predict(root_traj, args.pred_hist, args.pred_horizon)
    json.dump(pred, open(f"{args.out}/prediction.json", "w"), indent=2)
    print("PREDICTION:", json.dumps(pred, indent=2))


if __name__ == "__main__":
    main()
