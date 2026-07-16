"""Part B: forecast the person's root trajectory and score it (ADE/FDE) per horizon.

Consumes ``root_traj.npy`` written by ``smpl_person`` (per-frame body centroid in world, (T,3)) — for
GT (Part A) or MAMMA-estimated (Part A.5) motion. Answers the "predictive, not just reconstructive"
point: given the past, forecast where the person goes.

Predictors:
  * ``const_velocity`` — extrapolate the mean velocity over the history window (the beat-me baseline).
  * ``const_turn``     — constant speed + constant yaw-rate fit over the history, integrated forward on
                         the ground plane (linear on the up axis). Beats const-velocity when the path
                         curves (a person walking an arc) — the whole point on a walking sequence.

Eval: slide a window; from history ``[i-hist, i]`` predict ``[i+1, i+horizon]``; score vs GT. Reports
per-horizon ADE/FDE for each predictor and saves a predicted-vs-GT ground-plane plot.

Run: ``python -m tracking.predict --root_traj <out>/root_traj.npy --out <out>``.
"""
import os
import json
import argparse

import numpy as np


def smooth_traj(traj, w):
    """Centred moving-average per axis (w frames). w<=1 is a no-op. Tames per-frame estimation jitter."""
    if w <= 1:
        return traj
    k = np.ones(w) / w
    pad = w // 2
    out = np.empty_like(traj)
    for i in range(traj.shape[1]):
        pp = np.pad(traj[:, i], pad, mode="edge")
        out[:, i] = np.convolve(pp, k, mode="same")[pad:pad + len(traj)]
    return out


def ground_axes(traj):
    """Pick the two ground-plane axes (highest variance) + the ~vertical axis (lowest variance).

    A walking person moves mostly in a plane, so the least-varying world axis is ~up. Returns
    ``(a, b, up)`` index triple.
    """
    var = traj.var(0)
    up = int(np.argmin(var))
    ab = [i for i in range(3) if i != up]
    return ab[0], ab[1], up


def const_velocity(hist, horizon, axes=None):
    """Extrapolate mean velocity over ``hist`` (which includes the current frame at [-1])."""
    v = (hist[-1] - hist[0]) / max(1, len(hist) - 1)
    steps = np.arange(1, horizon + 1)[:, None]
    return hist[-1][None] + steps * v[None]


def const_turn(hist, horizon, axes):
    """Constant speed + constant yaw-rate on the ground plane (linear on up); falls back to const-vel."""
    a, b, c = axes
    P = hist[:, [a, b]]
    d = np.diff(P, axis=0)
    if len(d) < 2 or np.linalg.norm(d, axis=1).mean() < 1e-6:
        return const_velocity(hist, horizon)
    head = np.arctan2(d[:, 1], d[:, 0])
    yaw = float(np.diff(np.unwrap(head)).mean())          # mean turn per step
    speed = float(np.linalg.norm(d, axis=1).mean())
    up = hist[:, c]
    up_v = (up[-1] - up[0]) / max(1, len(up) - 1)
    th, p, u = head[-1], P[-1].copy(), up[-1]
    out = []
    for _ in range(horizon):
        th += yaw
        p = p + speed * np.array([np.cos(th), np.sin(th)])
        u = u + up_v
        row = np.empty(3); row[a], row[b], row[c] = p[0], p[1], u
        out.append(row)
    return np.stack(out)


PREDICTORS = {"const_velocity": const_velocity, "const_turn": const_turn}


def evaluate(traj, hist=8, horizon=8):
    axes = ground_axes(traj)
    T = len(traj)
    res = {"n_frames": T, "hist": hist, "horizon": horizon}
    for name, fn in PREDICTORS.items():
        ade, fde, per_h = [], [], [[] for _ in range(horizon)]
        for i in range(hist, T - horizon):
            pred = fn(traj[i - hist:i + 1], horizon, axes)
            gt = traj[i + 1:i + 1 + horizon]
            err = np.linalg.norm(pred - gt, axis=1)
            ade.append(err.mean()); fde.append(err[-1])
            for h in range(horizon):
                per_h[h].append(err[h])
        res[name] = {
            "windows": len(ade),
            "ADE": float(np.mean(ade)) if ade else float("nan"),
            "FDE": float(np.mean(fde)) if fde else float("nan"),
            "ADE_per_horizon": [float(np.mean(x)) if x else float("nan") for x in per_h],
        }
    return res, axes


def plot(traj, axes, out_png, hist=8, horizon=8):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    a, b, _ = axes
    plt.figure(figsize=(6, 6))
    plt.plot(traj[:, a], traj[:, b], "k.-", lw=1, ms=3, label="GT path")
    i = max(hist, len(traj) // 2)
    if hist <= i < len(traj) - horizon:
        for name, fn in PREDICTORS.items():
            pr = fn(traj[i - hist:i + 1], horizon, axes)
            plt.plot(pr[:, a], pr[:, b], ".--", label=f"{name}")
        gt = traj[i + 1:i + 1 + horizon]
        plt.plot(gt[:, a], gt[:, b], "g.-", lw=2, label="GT future")
        plt.plot(traj[i, a], traj[i, b], "ro", label="forecast origin")
    plt.axis("equal"); plt.legend(); plt.title("root-trajectory forecast (ground plane)")
    plt.tight_layout(); plt.savefig(out_png, dpi=120); plt.close()


def main():
    p = argparse.ArgumentParser(description="Part B: root-trajectory forecasting (ADE/FDE)")
    p.add_argument("--root_traj", required=True, help="root_traj.npy from smpl_person (T,3)")
    p.add_argument("--out", required=True)
    p.add_argument("--hist", type=int, default=8)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--smooth", type=int, default=1, help="moving-average window over the trajectory (1=off)")
    a = p.parse_args()

    traj = np.load(a.root_traj)
    assert traj.ndim == 2 and traj.shape[1] == 3, f"expected (T,3), got {traj.shape}"
    traj = smooth_traj(traj, a.smooth)
    os.makedirs(a.out, exist_ok=True)
    res, axes = evaluate(traj, a.hist, a.horizon)
    res["smooth"] = a.smooth
    json.dump(res, open(f"{a.out}/prediction.json", "w"), indent=2)
    plot(traj, axes, f"{a.out}/trajectory.png", a.hist, a.horizon)
    print(json.dumps(res, indent=2))
    cv, ct = res["const_velocity"]["ADE"], res["const_turn"]["ADE"]
    if cv == cv and ct == ct:
        print(f"\nADE: const_velocity {cv:.3f} m  vs  const_turn {ct:.3f} m  "
              f"({'turn wins' if ct < cv else 'no gain from turn'})")


if __name__ == "__main__":
    main()
