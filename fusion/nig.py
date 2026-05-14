"""Closed-form NIG conjugate update for per-surfel friction fusion (Path A).

Implements the Normal-Inverse-Gamma posterior update from PhysGS
(arXiv:2511.18570, Eq 12-14). Used as (a) the Path A baseline fusion driver,
(b) the distillation target the Path B MLP imitates, and (c) the fallback
when the MLP's confidence is low.

A surfel's friction belief is parametrised by (tau, kappa, alpha, beta)
where tau is E[mu], kappa is the effective sample count for mu, and
(alpha, beta) parametrise the inverse-gamma over sigma^2.

For a weighted observation (value, omega) with omega in [0, 1]:

    tau'   = (kappa * tau + omega * value) / (kappa + omega)              [Eq 12]
    kappa' = kappa + omega                                                  [Eq 12]
    alpha' = alpha + omega / 2                                              [Eq 13]
    beta'  = beta + (omega * kappa * (value - tau)^2) / (2 * (kappa + omega))  [Eq 13]

Posterior moments (Eq 14):

    E[mu]      = tau
    E[sigma^2] = beta / (alpha - 1)            for alpha > 1
    Var[mu]    = beta / (kappa * (alpha - 1))  for alpha > 1

All functions operate elementwise on numpy arrays via broadcasting, so the
same module serves single-surfel test cases and the (N,)-batched call in
fuse_surfels.py.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def closed_form_update(tau, kappa, alpha, beta, value, confidence):
    """Apply PhysGS Eq 12-13 elementwise.

    Returns (tau', kappa', alpha', beta') as float64 arrays broadcasted to
    the common shape of the inputs. A zero-confidence observation is an
    identity update (kappa, alpha, beta unchanged; tau unchanged when
    kappa > 0, else preserved).
    """
    tau = np.asarray(tau, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)

    new_kappa = kappa + confidence
    # Guard against new_kappa == 0 (only when both kappa and confidence are 0).
    safe_kappa = np.where(new_kappa > 0, new_kappa, 1.0)
    new_tau = np.where(new_kappa > 0,
                       (kappa * tau + confidence * value) / safe_kappa,
                       tau)
    new_alpha = alpha + 0.5 * confidence
    new_beta = beta + (confidence * kappa * (value - tau) ** 2) / (2.0 * safe_kappa)
    return new_tau, new_kappa, new_alpha, new_beta


def posterior_mean_var(tau, kappa, alpha, beta):
    """Return (E[mu], E[sigma^2], Var[mu]) per PhysGS Eq 14.

    NaN is returned for E[sigma^2] and Var[mu] where alpha <= 1 (moments
    are undefined for those parameter regimes).
    """
    tau = np.asarray(tau, dtype=np.float64)
    kappa = np.asarray(kappa, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)

    e_mu = tau
    with np.errstate(divide="ignore", invalid="ignore"):
        e_sigma2 = np.where(alpha > 1.0, beta / (alpha - 1.0), np.nan)
        var_mu = np.where(alpha > 1.0, beta / (kappa * (alpha - 1.0)), np.nan)
    return e_mu, e_sigma2, var_mu


def pool_buckets(
    buckets: Sequence[tuple[float, float, float, float]],
    weights: Sequence[float],
) -> tuple[float, float, float, float]:
    """Weighted per-parameter pool of NIG hyperparameters.

    Used for the material-marginal fallback in lookup.py: when an exact
    (material, friction_level) bucket is missing, pool the buckets sharing
    the material (weighted by per-bucket sample count) to get a slightly
    weaker "this material at unknown friction level" prior.

    Per-parameter weighted average — not a sufficient-statistic pool that
    would sum kappa/alpha as effective counts. The pooled prior is
    intentionally not stronger than the strongest contributing bucket,
    since marginalising over the unknown friction level is genuinely more
    uncertain than any single conditional.
    """
    if len(buckets) != len(weights):
        raise ValueError("buckets and weights must be the same length")
    if len(buckets) == 0:
        raise ValueError("at least one bucket required")
    w = np.asarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(w)) or w.sum() <= 0:
        raise ValueError("weights must be finite with positive sum")
    w = w / w.sum()
    arr = np.asarray(buckets, dtype=np.float64)  # (B, 4)
    pooled = (w[:, None] * arr).sum(axis=0)
    return tuple(float(x) for x in pooled)
