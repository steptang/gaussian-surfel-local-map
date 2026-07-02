"""Per-object deformation rig for dynamic surfels.

A moving object is represented as: a frozen *canonical* GaussianModel (its geometry, built
once by masked reconstruction) + a per-timestep *coarse pose* (rigid translation, the M2
trajectory layer) + a learned *deformation field* D(x, t) -> (Δposition, Δrotation) that
re-poses individual surfels (the M3 articulation layer).

To render the object at time t we warp the canonical surfels (see ``warp``) and hand the
deformed positions/rotations to ``gaussian_renderer.render`` via its means3D/rotations
overrides -- no CUDA change and the canonical is never mutated.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def positional_encoding(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """NeRF-style [x, sin(2^i pi x), cos(2^i pi x) ...] along the last dim."""
    out = [x]
    for i in range(num_freqs):
        f = (2.0 ** i) * math.pi
        out += [torch.sin(f * x), torch.cos(f * x)]
    return torch.cat(out, dim=-1)


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two quaternions (..., 4) laid out (w, x, y, z)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def axis_angle_to_quat(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle -> (..., 4) unit quaternion. v -> 0 gives the identity."""
    angle = v.norm(dim=-1, keepdim=True)
    axis = v / (angle + 1e-8)
    return torch.cat([torch.cos(0.5 * angle), axis * torch.sin(0.5 * angle)], dim=-1)


class DeformationField(nn.Module):
    """MLP D(x, t) -> (Δposition[3], Δrotation[3] as axis-angle).

    Output layer is zero-initialised so the field starts at the identity (no deformation),
    letting the coarse pose explain the bulk motion before the MLP learns the residual.
    """

    def __init__(self, center: torch.Tensor, scale: float,
                 pos_freqs: int = 6, time_freqs: int = 5, width: int = 128, depth: int = 6):
        super().__init__()
        self.register_buffer("center", center.detach().clone())
        self.scale = float(scale)
        self.pos_freqs, self.time_freqs = pos_freqs, time_freqs
        din = 3 * (1 + 2 * pos_freqs) + 1 * (1 + 2 * time_freqs)
        layers, d = [], din
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers += [nn.Linear(width, 6)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, xyz: torch.Tensor, t: float):
        xn = (xyz - self.center) / self.scale
        tt = torch.full((xyz.shape[0], 1), float(t), device=xyz.device)
        feat = torch.cat([positional_encoding(xn, self.pos_freqs),
                          positional_encoding(tt, self.time_freqs)], dim=-1)
        out = self.net(feat)
        return out[:, :3], out[:, 3:]          # dpos, daa


def warp(canonical, field: DeformationField, coarse_translation: torch.Tensor,
         t: float, rigid: bool = False):
    """Deform the canonical surfels to time ``t``.

    Args:
        canonical: the frozen canonical GaussianModel.
        field: the DeformationField (M3). Ignored when ``rigid``.
        coarse_translation: (3,) rigid translation for this timestep (M2 coarse pose).
        rigid: if True, apply only the coarse pose (baseline; no articulation).

    Returns (deformed_xyz (N,3), deformed_rotation (N,4), dpos (N,3), daa (N,3)); dpos/daa
    are zero when ``rigid``.
    """
    xyz0 = canonical.get_xyz.detach()
    rot0 = canonical.get_rotation.detach()
    if rigid:
        z = torch.zeros_like(xyz0)
        return xyz0 + coarse_translation, rot0, z, z
    dpos, daa = field(xyz0, t)
    xyz = xyz0 + coarse_translation + dpos
    rot = F.normalize(quat_multiply(axis_angle_to_quat(daa), rot0), dim=-1)
    return xyz, rot, dpos, daa


def knn_indices(xyz: torch.Tensor, k: int = 8) -> torch.Tensor:
    """(N, 3) -> (N, k) indices of the k nearest canonical neighbours (self excluded).

    Computed once on the (frozen or slowly-changing) canonical positions and reused as the
    neighbourhood for the local-rigidity loss.
    """
    from sklearn.neighbors import NearestNeighbors
    x = xyz.detach().cpu().numpy()
    idx = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x, return_distance=False)[:, 1:]
    return torch.tensor(idx, device=xyz.device)


def local_rigidity_loss(delta: torch.Tensor, nbr_idx: torch.Tensor) -> torch.Tensor:
    """As-rigid-as-possible-lite: neighbouring surfels should share a similar displacement.

    Penalising ||Δ_i - Δ_j||^2 over each surfel's neighbours makes the deformation *locally
    coherent* (limbs move as semi-rigid pieces) so the object articulates without smearing/
    tearing -- decoupled from the magnitude penalty on Δ itself.
    """
    return ((delta[nbr_idx] - delta[:, None, :]) ** 2).mean()
