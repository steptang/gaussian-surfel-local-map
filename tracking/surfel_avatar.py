"""#2 (M10, v3) — CANONICAL + LBS surfel avatar bound to an SMPL(-X) body.

REDESIGN (2026-07-16) after studying the three surfel-human papers + what code exists:
  * EfficientHuman (arXiv 2504.20607)  — 2D surfels in a **canonical rest pose**, posed per frame by
    **LBS**; skinning = ``softmax(log(W_nearest) + MLP(PE(p_c)))``; a per-frame **pose-refine** Δθ.
  * SGIA (arXiv 2407.15212)            — 2DGS surfels + SMPL, **canonical everything** (appearance is
    frame-shared), LBS rotates the **tangent frame** (R' = A_rot·R), scale is **trainable** (frozen
    scale causes holes), progressive geometry→appearance schedule.
  * GSSHuman (TOG 2024)                — 4-RGB-D; **residual-rotation normals** (orient off a geometric
    prior, learn only ΔR), and **depth + normal supervision** on the surfels (neither LBS paper used
    depth — we have RGB-D, so this is a free win). None of the three ship code; densification we take
    from our own hbb1-2DGS base.

Why this replaces v1/v2 (one-surfel-per-vertex, shared colour over jittery world-frame meshes):
  v1/v2 read positions+normals off MAMMA's **per-frame world meshes** (jittery) and averaged a single
  shared appearance against them → wash-out blur; one surfel/vertex under-tiled → holes.
  Here appearance is optimised **once in a canonical rest pose** and *articulated* to each frame by LBS,
  so every frame agrees on one stable, per-surfel appearance (sharp); the canonical is **densely surface
  -sampled** and **adaptively densified** (no holes); scale is **trainable**; MAMMA jitter is absorbed by
  **temporal pose smoothing + a learned pose-refine Δθ**. Persistent canonical surfels (fixed IDs, one
  body posed by θ(t)) also give correspondence/motion for Part-B prediction, which per-frame recon throws
  away.

Recipe (all ON by default; --no_* to ablate):
  canonical rest-pose surfels (surface-sampled)  →  per-surfel SH/opacity/scale/ΔR (trainable, frame-shared)
  skinning W = softmax(log(W_nn) + MLP_skin(PE(x_c)))    [nearest-vertex base + learned residual]
  per frame t:  A_t = LBS transforms(θ_t + Δθ_t, β̄)      [β̄ = mean betas: one canonical shape]
                x'_k = (Σ_j W_kj A_t,j) x_k + transl_t     (position override)
                R'_k = orthonormal(A_rot) · R_canon(ΔR)     (rotation override — pose the tangent frame)
  losses:  masked L1+SSIM + silhouette(α vs mask) + depth-L1 + normal-consistency  (+ Δθ / skin regs)

NOT implemented (documented extension): SGIA's N_lb pose-conditioned **latent bones** for cloth (small
in-frame person → low priority) and its full **PBR/relighting** split (we keep radiance SH; the sharpness
win comes from canonical-space appearance, not PBR).

Run (Colab, GPU):
  python -m tracking.surfel_avatar --conv_root <converted-scenes> \
     --smpl_root <mamma-exports> --smpl_source mamma --smpl_model_dir <smplx_dir> \
     --out <dir> [--iters 6000] [--n_canon 60000] [--person_embed person.npy]

VERIFY on first run: (a) the posed body lands on the person in the compare grid (else --align_t/--body
frame convention); (b) surfel disks face the cameras (else the normal→quat / ΔR frame is off);
(c) densification grows, not prunes, the canonical count (else raise --densify_grad_threshold sign / lower
it, cf. the deform-rig Lever-B net-prune gotcha). MAMMA needs --smpl_model_dir now: the avatar poses
**params** via LBS (canonical), it no longer renders MAMMA's world-frame ``pred_vertices`` mesh directly.
"""
import os
import json
import argparse

import numpy as np


# ============================================================================= SMPL(-X) LBS machinery
# SMPL-X full-pose joint layout (axis-angle, 3 each): 0 global | 1..21 body | 22 jaw | 23 leye | 24 reye
# | 25..39 lhand | 40..54 rhand.  We assemble whatever the source provides, zero-pad the rest.
_SMPLX_OFF = {"global_orient": 0, "body_pose": 1, "jaw_pose": 22, "leye_pose": 23,
              "reye_pose": 24, "left_hand_pose": 25, "right_hand_pose": 40}


def load_smpl_params(frame_dir, source):
    """One frame -> dict with global_orient(3), body_pose, betas, transl, + optional hand/face poses.

    * mamma  — SMPL-X ``*.npz`` MAMMA writes (global_orient/body_pose/transl/betas + hands/jaw if present).
    * gt     — BEHAVE SMPL-H ``person_fit.pkl`` (full ``pose`` axis-angle + betas + trans).
    """
    import glob
    import pickle
    if source == "mamma":
        hits = glob.glob(f"{frame_dir}/*.npz")
        assert hits, f"no MAMMA .npz under {frame_dir}"
        d = np.load(hits[0], allow_pickle=True)
        get = lambda k: np.asarray(d[k]).reshape(-1) if k in d else None
        out = {"betas": (get("betas") if get("betas") is not None else np.zeros(10))[:10],
               "transl": (get("transl") if get("transl") is not None else np.zeros(3))[:3],
               "model_type": "smplx"}
        for k in ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose",
                  "left_hand_pose", "right_hand_pose"):
            v = get(k)
            if v is not None:
                out[k] = v
        if "global_orient" not in out and get("pose") is not None:      # some exports pack a flat pose
            p = get("pose"); out["global_orient"] = p[:3]; out["body_pose"] = p[3:66]
        return out
    # gt: BEHAVE SMPL-H flat pose
    for cand in (f"{frame_dir}/person/fit02/person_fit.pkl", f"{frame_dir}/person/fit01/person_fit.pkl"):
        if os.path.exists(cand):
            with open(cand, "rb") as f:
                d = pickle.load(f, encoding="latin1")
            pose = np.asarray(d.get("pose", d.get("full_pose"))).reshape(-1)
            return {"global_orient": pose[:3], "body_pose": pose[3:66],
                    "betas": np.asarray(d.get("betas"))[:10],
                    "transl": np.asarray(d.get("trans", d.get("transl", np.zeros(3)))).reshape(3),
                    "model_type": "smplh"}
    raise FileNotFoundError(f"no SMPL params under {frame_dir}")


def full_pose_vec(params, n_joints):
    """Assemble a (n_joints*3,) axis-angle vector from a params dict (zero-pad missing joints)."""
    fp = np.zeros(n_joints * 3, dtype=np.float32)
    for key, joint in _SMPLX_OFF.items():
        if key in params and params[key] is not None:
            v = np.asarray(params[key]).reshape(-1)
            s = joint * 3
            fp[s:s + min(len(v), n_joints * 3 - s)] = v[:n_joints * 3 - s]
    return fp


def canonical_shape(model, betas):
    """Rest-pose (zero-pose) shaped body -> (v_shaped (V,3), J_rest (Jn,3)) tensors on cuda."""
    import torch
    from smplx.lbs import blend_shapes, vertices2joints
    b = torch.tensor(betas, dtype=torch.float32, device="cuda")[None]
    shapedirs = model.shapedirs[..., :b.shape[1]]
    v_shaped = model.v_template[None].cuda() + blend_shapes(b, shapedirs.cuda())      # (1,V,3)
    J = vertices2joints(model.J_regressor.cuda(), v_shaped)                            # (1,Jn,3)
    return v_shaped[0], J[0]                       # drop batch dims: (V,3), (Jn,3) as documented


def frame_transforms(model, J_rest, full_pose):
    """Per-joint relative LBS transforms A (Jn,4,4) for one frame's full axis-angle pose (differentiable)."""
    import torch
    from smplx.lbs import batch_rodrigues, batch_rigid_transform
    rot = batch_rodrigues(full_pose.reshape(-1, 3)).view(1, -1, 3, 3)                  # (1,Jn,3,3)
    _, A = batch_rigid_transform(rot, J_rest[None], model.parents.long().cuda())
    return A[0]                                                                        # (Jn,4,4)


# ============================================================================= canonical surfel sampling
def sample_body_surface(v_shaped, faces, lbs_weights, n):
    """Densely sample the rest body surface -> (P(n,3), skin_weights(n,Jn), normals(n,3)) numpy.

    Barycentric sampling weighted by triangle area; skinning + normals are barycentric-interpolated from
    the face's 3 vertices. Denser than one-surfel-per-vertex -> closes the tiling holes at init.
    """
    v = v_shaped.detach().cpu().numpy()
    W = lbs_weights.detach().cpu().numpy()
    tri = v[faces]                                                                     # (F,3,3)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    fn = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(fn, axis=1) + 1e-12
    fidx = np.random.choice(len(faces), size=n, p=area / area.sum())
    r1, r2 = np.random.rand(n, 1), np.random.rand(n, 1)
    s = np.sqrt(r1)
    b0, b1, b2 = (1 - s), s * (1 - r2), s * r2                                         # barycentric
    bar = np.concatenate([b0, b1, b2], 1)                                              # (n,3)
    P = (bar[:, :, None] * tri[fidx]).sum(1)
    fv = faces[fidx]                                                                   # (n,3) vertex ids
    skin = (bar[:, :, None] * W[fv]).sum(1)                                            # (n,Jn)
    nrm = fn[fidx] / (np.linalg.norm(fn[fidx], axis=1, keepdims=True) + 1e-9)
    return P.astype(np.float32), skin.astype(np.float32), nrm.astype(np.float32)


def nearest_vertex_weights(xyz, v_shaped, lbs_weights):
    """Skinning base for arbitrary canonical points via nearest rest-body vertex (used after densify)."""
    from scipy.spatial import cKDTree
    v = v_shaped.detach().cpu().numpy()
    tree = cKDTree(v)
    _, idx = tree.query(xyz, k=1)
    return lbs_weights.detach().cpu().numpy()[idx]                                     # (P,Jn)


# ============================================================================= rotation helpers (wxyz)
def normals_to_quats(normals):
    """wxyz quats whose 3rd axis = the normal (2DGS disk lies in axes 1,2)."""
    from scipy.spatial.transform import Rotation
    n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
    ref = np.tile(np.array([1.0, 0.0, 0.0]), (len(n), 1))
    ref[np.abs((n * ref).sum(1)) > 0.9] = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(ref, n); t1 /= (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-9)
    t2 = np.cross(n, t1)
    R = np.stack([t1, t2, n], axis=2)
    q = Rotation.from_matrix(R).as_quat()                                             # xyzw
    return np.concatenate([q[:, 3:4], q[:, :3]], axis=1).astype(np.float32)           # wxyz


def orthonormalize(R):
    """Gram-Schmidt the (P,3,3) blended LBS linear part into a clean rotation."""
    import torch
    a0, a1 = R[:, :, 0], R[:, :, 1]
    b0 = a0 / (a0.norm(dim=-1, keepdim=True) + 1e-9)
    b1 = a1 - (b0 * a1).sum(-1, keepdim=True) * b0
    b1 = b1 / (b1.norm(dim=-1, keepdim=True) + 1e-9)
    b2 = torch.cross(b0, b1, dim=-1)
    return torch.stack([b0, b1, b2], dim=2)


def mat_to_quat(R):
    """(P,3,3) rotation -> (P,4) wxyz quaternion (branchless, sign via off-diagonals)."""
    import torch
    m = R
    q = torch.stack([1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2],
                     1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
                     1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2],
                     1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2]], dim=-1)
    q = torch.sqrt(torch.clamp(q, min=0.0)) / 2.0
    w = q[:, 0]
    x = torch.copysign(q[:, 1], m[:, 2, 1] - m[:, 1, 2])
    y = torch.copysign(q[:, 2], m[:, 0, 2] - m[:, 2, 0])
    z = torch.copysign(q[:, 3], m[:, 1, 0] - m[:, 0, 1])
    out = torch.stack([w, x, y, z], dim=-1)
    return out / (out.norm(dim=-1, keepdim=True) + 1e-9)


def positional_encoding(x, L):
    """(P,3) -> (P, 3+3*2*L) sin/cos positional encoding."""
    import torch
    feats = [x]
    for i in range(L):
        f = (2.0 ** i) * np.pi * x
        feats += [torch.sin(f), torch.cos(f)]
    return torch.cat(feats, dim=-1)


# ============================================================================= tiny refinement MLPs
def _mlp(inp, out, hidden=128, layers=3):
    import torch.nn as nn
    mods, d = [], inp
    for _ in range(layers - 1):
        mods += [nn.Linear(d, hidden), nn.ReLU()]; d = hidden
    lin = nn.Linear(d, out)
    nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)          # start as identity residual (Δ=0)
    mods += [lin]
    return nn.Sequential(*mods)


# ============================================================================= trainer
def train_avatar(args):
    import torch
    import torch.nn.functional as F
    import matplotlib.pyplot as plt
    import tracking.behave_data as bd
    import tracking.reconstruct as rc
    import tracking.smpl_person as sp
    from tracking.render_compose import render_composite
    from gaussian_renderer import render, GaussianModel
    from utils.graphics_utils import BasicPointCloud
    from utils.general_utils import build_rotation
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

    # --- static map (person masked out, fused over spread timesteps); PLY-cached across runs ---
    if args.static_ply and os.path.exists(args.static_ply):
        static = GaussianModel(dataset.sh_degree)
        static.load_ply(args.static_ply)                # frozen by construction: no optimizer attached
        print(f"static map loaded from cache {args.static_ply} ({static.get_xyz.shape[0]} surfels)")
    else:
        static_tis = sorted(set(np.linspace(0, N - 1, min(args.n_static, N)).astype(int).tolist()))
        spts, scols, views_static = bd.fuse_region(TS, PMASK, static_tis, "static")
        static = rc.reconstruct_masked(views_static, spts, scols, opt, pipe, bg, extent,
                                       dataset.sh_degree, args.recon_iters, args.lambda_depth)
        rc.freeze(static)
        if args.static_ply:
            static.save_ply(args.static_ply)
            print("static map cached ->", args.static_ply)

    # --- SMPL model + one canonical shape (mean betas) ---
    assert args.smpl_model_dir, "canonical LBS needs --smpl_model_dir (SMPL/SMPL-H/SMPL-X model files)"
    params = {}
    for ti, fr in valid:
        params[ti] = load_smpl_params(fr, args.smpl_source)
    mtype = params[valid[0][0]].get("model_type", "smplh")
    gender = str(params[valid[0][0]].get("gender", "neutral" if mtype == "smplx" else "male"))
    model = sp.build_smpl_model(args.smpl_model_dir, gender, mtype)
    betas_bar = np.mean([params[ti]["betas"] for ti, _ in valid], axis=0)
    v_shaped, J_rest = canonical_shape(model, betas_bar)
    faces = np.asarray(model.faces, dtype=np.int64)
    lbs_w = model.lbs_weights.cuda()
    n_joints = lbs_w.shape[1]

    # --- per-frame poses: assemble full-pose, temporally smooth (kill MAMMA jitter), to tensors ---
    order = [ti for ti, _ in valid]
    FP = np.stack([full_pose_vec(params[ti], n_joints) for ti in order])              # (F, Jn*3)
    TR = np.stack([np.asarray(params[ti]["transl"], np.float32).reshape(3) for ti in order])
    if args.smooth and args.smooth > 1 and len(order) >= args.smooth:
        k = args.smooth | 1                                                            # odd window
        pad = k // 2
        def movavg(a):
            ap = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
            ker = np.ones(k) / k
            return np.stack([np.convolve(ap[:, c], ker, "valid") for c in range(a.shape[1])], 1)
        FP, TR = movavg(FP), movavg(TR)
    # np.convolve promotes to float64 -> pin float32 (the MLPs/rasterizer are Float)
    FP_t = {ti: torch.tensor(FP[i], dtype=torch.float32, device="cuda") for i, ti in enumerate(order)}
    TR_t = {ti: torch.tensor(TR[i], dtype=torch.float32, device="cuda") for i, ti in enumerate(order)}
    body_slice = slice(3, 3 + 63)                                                      # 21 body joints

    # --- canonical surfels: dense surface samples, ΔR init from surface normal, scale TRAINABLE ---
    Pc, skin0, nrm = sample_body_surface(v_shaped, faces, lbs_w, args.n_canon)
    g = GaussianModel(dataset.sh_degree)
    g.create_from_pcd(BasicPointCloud(points=Pc.astype(np.float64),
                                      colors=np.full_like(Pc, 0.6, dtype=np.float64),
                                      normals=nrm.astype(np.float64)), spatial_lr_scale=extent)
    with torch.no_grad():
        g._rotation.data = torch.tensor(normals_to_quats(nrm), device="cuda")          # residual-rot prior
        g._opacity.data = g.inverse_opacity_activation(0.7 * torch.ones_like(g.get_opacity)).detach()
    g.active_sh_degree = g.max_sh_degree
    g.training_setup(opt)                                                              # xyz/scale/rot/SH/opac/sem
    if args.person_embed and os.path.exists(args.person_embed):
        emb = torch.tensor(np.load(args.person_embed), dtype=torch.float, device="cuda").reshape(1, -1)
        with torch.no_grad():
            g._semantic.data = emb.repeat(g._semantic.shape[0], 1)

    # skinning base (log NN weights) + learned residual MLP; pose-refine MLP (Δθ on the body joints)
    skin_base = torch.log(torch.tensor(skin0, device="cuda") + 1e-8)                   # (P,Jn)
    pe_dim = 3 + 3 * 2 * args.pe_freqs
    mlp_skin = _mlp(pe_dim, n_joints).cuda() if args.skin_mlp else None
    mlp_pose = _mlp(63, 63).cuda() if args.pose_refine else None
    extra_params = ([] if mlp_skin is None else list(mlp_skin.parameters())) + \
                   ([] if mlp_pose is None else list(mlp_pose.parameters()))
    extra_opt = torch.optim.Adam(extra_params, lr=args.mlp_lr) if extra_params else None

    def skin_weights():
        if mlp_skin is None:
            return torch.softmax(skin_base, dim=-1)
        return torch.softmax(skin_base + mlp_skin(positional_encoding(g.get_xyz.detach(), args.pe_freqs)), -1)

    def pose_and_A(ti):
        fp = FP_t[ti]
        if mlp_pose is not None:
            dtheta = mlp_pose(fp[body_slice][None])[0]
            fp = fp.clone(); fp[body_slice] = fp[body_slice] + dtheta
        return frame_transforms(model, J_rest, fp), TR_t[ti], fp

    def posed(ti):
        """Return (posed_xyz (P,3), posed_quat (P,4)) for the current canonical surfels at frame ti."""
        A, transl, _ = pose_and_A(ti)
        W = skin_weights()                                                             # (P,Jn)
        T = torch.einsum("pj,jmn->pmn", W, A)                                          # (P,4,4)
        xyz = g.get_xyz
        xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], -1)
        p = (T @ xyz_h[..., None])[:, :3, 0] + transl[None]
        Rb = orthonormalize(T[:, :3, :3])
        Rc = build_rotation(g.get_rotation)                                            # canonical ΔR frame
        return p, mat_to_quat(Rb @ Rc)

    # --- optimise appearance+geometry in canonical space, articulated to ALL frames ---
    print(f"avatar: {g.get_xyz.shape[0]} canonical surfels, {len(valid)} frames, {n_joints} joints, "
          f"{args.iters} iters | skin_mlp={bool(mlp_skin)} pose_refine={bool(mlp_pose)} smooth={args.smooth}")
    import random
    for it in range(1, args.iters + 1):
        g.update_learning_rate(it)
        ti = random.choice(order)
        cam = random.choice(TS[ti]["cams"])
        mask = PMASK[(ti, cam.image_name)]                                             # (1,H,W)
        xyz, quat = posed(ti)
        pkg = render(cam, g, pipe, bg, means3D_override=xyz, rotations_override=quat)
        img, alpha, dep = pkg["render"], pkg["rend_alpha"], pkg["surf_depth"]
        gt = cam.original_image.cuda()
        # photometric (person region) + silhouette (α vs mask, closes/kills holes+halo)
        Ll1 = (torch.abs(img - gt) * mask).sum() / (mask.sum() * 3 + 1e-8)
        loss = (1 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1 - ssim(img * mask, gt * mask))
        loss = loss + args.lambda_mask * torch.abs(alpha - mask).mean()
        # depth (RGB-D — our edge; both LBS papers omit it)
        if args.lambda_depth > 0 and getattr(cam, "gt_depth", None) is not None:
            gd = cam.gt_depth.cuda(); dvalid = (gd > 0) & (mask > 0.5)
            if dvalid.any():
                loss = loss + args.lambda_depth * torch.abs(dep - gd)[dvalid].mean()
        # 2DGS normal-consistency (warm up) + tiny depth-distortion
        if it > args.geom_warmup:
            ncons = (1 - (pkg["rend_normal"] * pkg["surf_normal"]).sum(0))[None]
            loss = loss + args.lambda_normal * ncons.mean() + args.lambda_dist * pkg["rend_dist"].mean()
        # regularisers: keep Δθ / skin-residual small
        if mlp_pose is not None:
            loss = loss + args.lambda_posereg * mlp_pose(FP_t[ti][body_slice][None]).pow(2).mean()
        if mlp_skin is not None:
            loss = loss + args.lambda_skinreg * mlp_skin(
                positional_encoding(g.get_xyz.detach(), args.pe_freqs)).pow(2).mean()
        loss.backward()

        with torch.no_grad():
            vis, radii = pkg["visibility_filter"], pkg["radii"]
            if it < args.densify_until:
                g.max_radii2D[vis] = torch.max(g.max_radii2D[vis], radii[vis])
                g.add_densification_stats(pkg["viewspace_points"], vis)
                if it > args.densify_from and it % args.densify_interval == 0:
                    size_thresh = 20 if it > opt.opacity_reset_interval else None
                    g.densify_and_prune(args.densify_grad_threshold, args.opacity_cull, extent, size_thresh)
                    # canonical topology changed -> resync skinning base + PE inputs to new surfels
                    skin_base = torch.log(torch.tensor(
                        nearest_vertex_weights(g.get_xyz.detach().cpu().numpy(), v_shaped, lbs_w),
                        device="cuda") + 1e-8)
                if it % opt.opacity_reset_interval == 0:
                    g.reset_opacity()
            g.optimizer.step(); g.optimizer.zero_grad(set_to_none=True)
            if extra_opt is not None:
                extra_opt.step(); extra_opt.zero_grad(set_to_none=True)
        if it % 500 == 0:
            print(f"  iter {it:5d} loss {loss.item():.4f} surfels {g.get_xyz.shape[0]}")

    # --- eval: composite avatar into the map + person-only PSNR, video, compare grid ---
    def person_eval(ti, cam):
        xyz, quat = posed(ti)
        comp = render_composite(cam, [static, g], pipe, bg,
                                xyz_overrides=[None, xyz], rot_overrides=[None, quat])["render"]
        return sp._masked_psnr(comp, cam.original_image.cuda(), PMASK[(ti, cam.image_name)]), comp

    try:
        import imageio.v2 as imageio
    except ImportError:
        imageio = None
    per_psnr, vframes = [], []
    for ti, _ in valid:
        cam = TS[ti]["cams"][args.view]
        with torch.no_grad():
            ps, comp = person_eval(ti, cam)
        per_psnr.append(ps)
        gt_np, comp_np = sp._to_np(cam.original_image.cuda()), sp._to_np(comp)
        if imageio is not None:
            imageio.imwrite(f"{args.out}/frames/ts{ti:03d}.png", (comp_np * 255).astype(np.uint8))
            vframes.append((np.concatenate([gt_np, comp_np], 1) * 255).astype(np.uint8))
    if imageio is not None and vframes:
        imageio.mimwrite(f"{args.out}/sequence.mp4", vframes, fps=args.fps, macro_block_size=None)

    metrics = {"smpl_source": args.smpl_source, "canonical_surfels": int(g.get_xyz.shape[0]),
               "n_joints": int(n_joints), "frames": len(valid), "iters": args.iters,
               "person_psnr_mean": float(np.nanmean(per_psnr)),
               "person_psnr_per_ts": [None if np.isnan(x) else round(x, 3) for x in per_psnr],
               "config": {k: v for k, v in vars(args).items()}}
    json.dump(metrics, open(f"{args.out}/metrics.json", "w"), indent=2)
    print("METRICS:", json.dumps({k: metrics[k] for k in
          ("canonical_surfels", "frames", "person_psnr_mean")}, indent=2))

    show = [t for t, _ in valid[::max(1, len(valid) // 4)]][:4]
    fig, ax = plt.subplots(len(show), 2, figsize=(8, 3 * len(show))); ax = np.atleast_2d(ax)
    for r, ti in enumerate(show):
        cam = TS[ti]["cams"][args.view]
        with torch.no_grad():
            _, comp = person_eval(ti, cam)
        ax[r, 0].imshow(sp._to_np(cam.original_image.cuda())); ax[r, 0].set_ylabel(f"ts {ti}")
        ax[r, 1].imshow(sp._to_np(comp))
        for a in ax[r]: a.set_xticks([]); a.set_yticks([])
    for c, t in enumerate(["GT", "static + canonical-LBS avatar"]): ax[0, c].set_title(t)
    plt.tight_layout(); plt.savefig(f"{args.out}/compare.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    g.save_ply(f"{args.out}/avatar_canonical.ply")
    print("saved ->", args.out)


def main():
    p = argparse.ArgumentParser(description="Canonical + LBS surfel avatar bound to an SMPL(-X) body")
    p.add_argument("--conv_root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--smpl_source", choices=["gt", "mamma"], default="mamma")
    p.add_argument("--smpl_root", required=True)
    p.add_argument("--smpl_model_dir", required=True, help="SMPL/SMPL-H/SMPL-X model dir (LBS needs it)")
    p.add_argument("--behave_frames", nargs="*", default=None)
    p.add_argument("--n_select", type=int, default=24)
    p.add_argument("--select_stride", type=int, default=None)
    # canonical / recipe
    p.add_argument("--n_canon", type=int, default=60000, help="# canonical surface-sampled surfels")
    p.add_argument("--iters", type=int, default=6000)
    p.add_argument("--pe_freqs", type=int, default=4, help="positional-encoding freqs for the skin MLP")
    p.add_argument("--smooth", type=int, default=5, help="temporal pose-smoothing window (odd; 0/1=off)")
    p.add_argument("--skin_mlp", dest="skin_mlp", action="store_true", default=True)
    p.add_argument("--no_skin_mlp", dest="skin_mlp", action="store_false")
    p.add_argument("--pose_refine", dest="pose_refine", action="store_true", default=True)
    p.add_argument("--no_pose_refine", dest="pose_refine", action="store_false")
    p.add_argument("--mlp_lr", type=float, default=1e-4)
    # losses
    p.add_argument("--lambda_mask", type=float, default=0.1)
    p.add_argument("--lambda_depth", type=float, default=0.5)
    p.add_argument("--lambda_normal", type=float, default=0.05)
    p.add_argument("--lambda_dist", type=float, default=100.0)
    p.add_argument("--lambda_posereg", type=float, default=1e-2)
    p.add_argument("--lambda_skinreg", type=float, default=1e-3)
    p.add_argument("--geom_warmup", type=int, default=1500, help="iters before normal/dist regs engage")
    # densification (exposed — cf. deform-rig Lever-B net-prune gotcha)
    p.add_argument("--densify_from", type=int, default=500)
    p.add_argument("--densify_until", type=int, default=4500)
    p.add_argument("--densify_interval", type=int, default=200)
    p.add_argument("--densify_grad_threshold", type=float, default=2e-4)
    p.add_argument("--opacity_cull", type=float, default=0.05)
    # static map / misc
    p.add_argument("--static_ply", default=None, help="static-map PLY cache: load if it exists, else "
                   "reconstruct and save there (name it by recon config, e.g. static_boxtiny_7k.ply)")
    p.add_argument("--recon_iters", type=int, default=7000)
    p.add_argument("--n_static", type=int, default=4)
    p.add_argument("--view", type=int, default=0)
    p.add_argument("--person_embed", default=None)
    p.add_argument("--fps", type=int, default=8)
    args = p.parse_args()
    train_avatar(args)


if __name__ == "__main__":
    main()
