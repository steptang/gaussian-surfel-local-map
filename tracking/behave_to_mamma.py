"""BEHAVE (4-Kinect RGB) <-> MAMMA glue for M10 Part A.5.

Two subcommands:
  * ``input``  — lay out BEHAVE's 4 views as MAMMA footage + write MAMMA's calibration YAML +
                 capture JSON, so MAMMA can estimate SMPL-X on our data.
  * ``output`` — convert MAMMA's ``ma_3d`` SMPL-X output into the per-frame ``.npz`` exports that
                 ``tracking.smpl_person --smpl_source mamma`` reads (keys: global_orient, body_pose,
                 betas, transl; model_type=smplx).

Frame selection MATCHES the S1-3 converter (``SEL = sorted(t*.000)[::stride][:n_select]``) so MAMMA
frame i lines up with converted ``timestep_{i}`` — keep ``--n_select``/``--stride`` identical.

MAMMA input contract (docs/YOUR-DATA.md):
  <root>/<seq>/images/cam_{NN}/{i:05d}.jpg            # cam_01..cam_04
  calib.yaml: cameras: {cam_01: {camera_model: pinhole, distortion_model: radtan,
                intrinsics:[fx,fy,cx,cy], distortion_coeffs:[k1,k2,p1,p2,k3],
                resolution:[W,H], translation:[tx,ty,tz], rotation_quaternion:[w,x,y,z]}}
  capture.json: {capture_root, calib, cam_fps, videos_subdir|images_subdir, cams:[...],
                 sequences:{"000":{name:<seq>}}}

The ``input`` adapter needs BEHAVE's reader (``xiexh20/behave-dataset`` on sys.path) for calibration,
same as the S1-3 notebook. ``output`` needs only numpy.

### VERIFY on the first MAMMA run (from ma_vis overlays — the body should sit ON the person):
  (E) EXTRINSIC convention. BEHAVE ``local2world`` = cam2world (X_world = R X_cam + t). Multiview calib
      files usually store **world2cam** (X_cam = R X_world + T). Default here = world2cam; flip with
      ``--extrinsics cam2world`` if the cameras end up mirrored/behind the subject.
  (Q) QUATERNION order. docs/YOUR-DATA.md shows ``[w, x, y, z]``; the shipped example calib looked like
      ``[x, y, z, w]``. Default = wxyz; switch with ``--quat_order xyzw``.
  (K) camera->cam_NN mapping: BEHAVE ``k{kid}`` -> ``cam_{kid+1:02d}`` (k0->cam_01).
"""
import os
import re
import glob
import json
import argparse

import numpy as np


# ----------------------------------------------------------------------------- shared
def select_frames(seq_dir, n_select=24, stride=None):
    """Reproduce the S1-3 converter's SEL so MAMMA frame i == converted timestep_i."""
    allf = sorted(glob.glob(f"{seq_dir}/t*.000"))
    assert allf, f"no t*.000 frames under {seq_dir}"
    st = stride or max(1, len(allf) // max(1, n_select))
    return allf[::st][:n_select]


def _R_to_quat(R, order="wxyz"):
    """(3,3) rotation -> unit quaternion in the requested order."""
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z]) / np.linalg.norm([w, x, y, z])
    return q.tolist() if order == "wxyz" else [q[1], q[2], q[3], q[0]]


# ----------------------------------------------------------------------------- INPUT adapter
def write_mamma_input(seq_dir, out_root, seq_name, kids, n_select, stride,
                      extrinsics="world2cam", quat_order="wxyz"):
    import cv2
    import yaml
    from data.kinect_transform import KinectTransform   # xiexh20/behave-dataset on sys.path

    SEL = select_frames(seq_dir, n_select, stride)
    seq_out = f"{out_root}/{seq_name}"
    os.makedirs(seq_out, exist_ok=True)
    kt = KinectTransform(seq_dir, kinect_count=len(kids))

    def intr(kid):
        kc = kt.intrinsics[kid]
        if hasattr(kc, "focal_dist"):
            (fx, fy), (cx, cy), (W, H) = kc.focal_dist, kc.center, kc.image_size
        else:
            K = np.asarray(kc.calibration_matrix); fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            W, H = kc.image_size
        return float(fx), float(fy), float(cx), float(cy), int(W), int(H)

    cameras = {}
    for kid in kids:
        fx, fy, cx, cy, W, H = intr(kid)
        R_c2w = np.asarray(kt.local2world_R[kid], float)                 # cam2world
        t_c2w = np.asarray(kt.local2world_t[kid], float).reshape(3)
        if extrinsics == "world2cam":
            R = R_c2w.T
            t = (-R @ t_c2w)
        else:                                                            # cam2world
            R, t = R_c2w, t_c2w
        cameras[f"cam_{kid + 1:02d}"] = dict(
            camera_model="pinhole", distortion_model="radtan",
            intrinsics=[fx, fy, cx, cy], distortion_coeffs=[0.0, 0.0, 0.0, 0.0],  # radtan wants 4 [k1,k2,p1,p2]
            resolution=[W, H], translation=t.tolist(),
            rotation_quaternion=_R_to_quat(R, quat_order))

    # images: k{kid}.color.jpg -> <seq>/cam_{kid+1:02d}/{i:05d}.jpg
    # MAMMA discovery globs <seq>/* for cam dirs that DIRECTLY hold images (no images/ level).
    for kid in kids:
        cdir = f"{seq_out}/cam_{kid + 1:02d}"; os.makedirs(cdir, exist_ok=True)
        for i, ft in enumerate(SEL):
            im = cv2.imread(f"{ft}/k{kid}.color.jpg")
            assert im is not None, f"missing {ft}/k{kid}.color.jpg"
            cv2.imwrite(f"{cdir}/{i:05d}.jpg", im)

    calib_path = f"{out_root}/{seq_name}.calib.yaml"
    yaml.safe_dump({"cameras": cameras}, open(calib_path, "w"), sort_keys=False)
    capture = {"capture_root": out_root, "calib": calib_path, "cam_fps": 30,
               "images_subdir": "images", "cams": [f"cam_{k + 1:02d}" for k in kids],
               "sequences": {"000": {"name": seq_name}}}
    cap_path = f"{out_root}/{seq_name}.capture.json"
    json.dump(capture, open(cap_path, "w"), indent=2)

    print(f"wrote {len(SEL)} frames x {len(kids)} cams -> {seq_out}/images")
    print(f"calib   -> {calib_path}   (extrinsics={extrinsics}, quat={quat_order}; VERIFY E/Q)")
    print(f"capture -> {cap_path}")
    print("\nRun MAMMA (in its env), e.g.:")
    print(f"  python -m inference run --cfg configs/examples/presets/quick.yaml \\\n"
          f"    --footage {out_root} --seq_name {seq_name} --calib {calib_path} --out-tag behave -v")
    return seq_out, calib_path


# ----------------------------------------------------------------------------- OUTPUT converter
def mamma_to_exports(mamma_out_dir, exports_root, quat=False):
    """MAMMA ma_3d SMPL-X output -> per-frame ``<exports_root>/frame_{i:05d}/mamma.npz``.

    ma_3d emits SMPL-X (global_orient, body_pose, betas, transl). The on-disk layout isn't documented,
    so this DISCOVERS the per-frame param files (``.npz``/``.pkl``/``.json`` with those keys) and
    normalises them. On the first run it PRINTS what it found — if nothing matches, paste me a
    ``find`` of ``mamma_out_dir`` and I'll pin the parser. (verify)
    """
    import pickle
    os.makedirs(exports_root, exist_ok=True)
    cands = sorted(glob.glob(f"{mamma_out_dir}/**/*.npz", recursive=True) +
                   glob.glob(f"{mamma_out_dir}/**/*.pkl", recursive=True) +
                   glob.glob(f"{mamma_out_dir}/**/*.json", recursive=True))
    print(f"found {len(cands)} candidate param files under {mamma_out_dir}; first few:")
    for p in cands[:5]:
        print("  ", p)

    def load(p):
        if p.endswith(".npz"):
            return dict(np.load(p, allow_pickle=True))
        if p.endswith(".pkl"):
            return pickle.load(open(p, "rb"))
        return json.load(open(p))

    KEYS = ("global_orient", "body_pose", "betas", "transl")
    n = 0
    for p in cands:
        try:
            d = load(p)
        except Exception:
            continue
        if not all(k in d for k in ("body_pose", "betas")):     # SMPL-X param file
            continue
        m = re.search(r"(\d{4,6})", os.path.basename(p))
        i = int(m.group(1)) if m else n
        out = {k: np.asarray(d[k]) for k in KEYS if k in d}
        out["model_type"] = "smplx"
        fd = f"{exports_root}/frame_{i:05d}"; os.makedirs(fd, exist_ok=True)
        np.savez(f"{fd}/mamma.npz", **out)
        n += 1
    print(f"wrote {n} per-frame exports -> {exports_root}/frame_*/mamma.npz"
          + ("" if n else "  (0 — inspect mamma_out_dir layout and adjust; see docstring)"))
    return n


def main():
    p = argparse.ArgumentParser(description="BEHAVE <-> MAMMA glue (Part A.5)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("input", help="BEHAVE 4-view -> MAMMA footage + calib + capture")
    pi.add_argument("--seq_dir", required=True, help="raw BEHAVE sequence (has t*.000/)")
    pi.add_argument("--out_root", required=True, help="MAMMA dataset root to write")
    pi.add_argument("--seq_name", required=True)
    pi.add_argument("--kids", type=int, nargs="+", default=[0, 1, 2, 3])
    pi.add_argument("--n_select", type=int, default=24)
    pi.add_argument("--stride", type=int, default=None)
    pi.add_argument("--extrinsics", choices=["world2cam", "cam2world"], default="world2cam")
    pi.add_argument("--quat_order", choices=["wxyz", "xyzw"], default="wxyz")

    po = sub.add_parser("output", help="MAMMA ma_3d SMPL-X -> per-frame .npz exports")
    po.add_argument("--mamma_out_dir", required=True, help="MAMMA output/ma_3d/<tag>/<seq> dir")
    po.add_argument("--exports_root", required=True, help="where smpl_person --smpl_root will point")

    a = p.parse_args()
    if a.cmd == "input":
        write_mamma_input(a.seq_dir, a.out_root, a.seq_name, a.kids, a.n_select, a.stride,
                          a.extrinsics, a.quat_order)
    else:
        mamma_to_exports(a.mamma_out_dir, a.exports_root)


if __name__ == "__main__":
    main()
