"""Tests for closed-form NIG conjugate updates (fusion/nig.py).

Verifies PhysGS Eq 12-14 invariants:

1. identity_zero_confidence — omega=0 leaves (tau, kappa, alpha, beta) unchanged.
2. first_observation_from_uninformed — with kappa=0 a single observation moves
   tau to the observed value exactly, beta unchanged.
3. many_observations_converge — N high-confidence observations at value x
   drive tau toward x, with kappa and alpha growing linearly in N.
4. beta_grows_with_disagreement — an observation far from prior tau increases beta;
   one at tau leaves beta unchanged.
5. posterior_moments_basic — Eq 14 yields documented closed forms.
6. moments_nan_for_low_alpha — alpha <= 1 returns NaN for sigma^2 and Var[mu].
7. pool_identical_buckets_is_identity — pooling identical buckets returns that bucket.
8. pool_weights_bias — weights skew the per-parameter average.
9. broadcast_arrays — closed_form_update works elementwise over arrays.

Run from repo root with: pytest tests/test_fusion_nig.py -v
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion.nig import closed_form_update, pool_buckets, posterior_mean_var


def test_identity_zero_confidence():
    tau, kappa, alpha, beta = 0.5, 1.0, 2.0, 0.1
    t, k, a, b = closed_form_update(tau, kappa, alpha, beta, value=0.9, confidence=0.0)
    assert math.isclose(t, tau)
    assert math.isclose(k, kappa)
    assert math.isclose(a, alpha)
    assert math.isclose(b, beta)


def test_first_observation_from_uninformed():
    # kappa=0 means the prior asserts nothing about tau, so the first
    # observation should set tau exactly to the observed value. beta picks
    # up no correction because the (value - tau) term is multiplied by kappa.
    t, k, a, b = closed_form_update(
        tau=0.0, kappa=0.0, alpha=2.0, beta=0.1,
        value=0.7, confidence=1.0,
    )
    assert math.isclose(t, 0.7, abs_tol=1e-12)
    assert math.isclose(k, 1.0)
    assert math.isclose(a, 2.5)
    assert math.isclose(b, 0.1, abs_tol=1e-12)


def test_many_observations_converge():
    # 50 observations of x=0.4 with confidence=1 should drive tau toward 0.4
    # and add 50 to kappa, 25 to alpha.
    tau, kappa, alpha, beta = 0.5, 1.0, 2.0, 0.1
    x, omega = 0.4, 1.0
    N = 50
    for _ in range(N):
        tau, kappa, alpha, beta = closed_form_update(tau, kappa, alpha, beta, x, omega)
    # Closed form for kappa and alpha after N constant-confidence updates is exact:
    assert math.isclose(kappa, 1.0 + N * omega)
    assert math.isclose(alpha, 2.0 + 0.5 * N * omega)
    # tau converges toward x; the running mean after N obs with prior weight 1 is
    # (1 * 0.5 + N * 0.4) / (1 + N).
    expected_tau = (1.0 * 0.5 + N * 0.4) / (1.0 + N)
    assert math.isclose(float(tau), expected_tau, abs_tol=1e-10)
    assert abs(float(tau) - x) < 0.01


def test_beta_grows_with_disagreement_zero_when_agree():
    # Observation at prior tau adds nothing to beta (squared-deviation is 0).
    t, k, a, b = closed_form_update(
        tau=0.5, kappa=10.0, alpha=2.0, beta=0.1,
        value=0.5, confidence=1.0,
    )
    assert math.isclose(b, 0.1, abs_tol=1e-12)

    # Observation 0.3 away from tau should add a positive correction.
    _, _, _, b_far = closed_form_update(
        tau=0.5, kappa=10.0, alpha=2.0, beta=0.1,
        value=0.8, confidence=1.0,
    )
    expected_extra = (1.0 * 10.0 * 0.3 ** 2) / (2.0 * 11.0)
    assert math.isclose(b_far, 0.1 + expected_extra, abs_tol=1e-12)


def test_posterior_moments_basic():
    e_mu, e_sigma2, var_mu = posterior_mean_var(tau=0.5, kappa=4.0, alpha=3.0, beta=0.2)
    assert math.isclose(float(e_mu), 0.5)
    assert math.isclose(float(e_sigma2), 0.2 / (3.0 - 1.0))
    assert math.isclose(float(var_mu), 0.2 / (4.0 * (3.0 - 1.0)))


def test_moments_nan_for_low_alpha():
    e_mu, e_sigma2, var_mu = posterior_mean_var(tau=0.5, kappa=1.0, alpha=0.9, beta=0.1)
    assert math.isclose(float(e_mu), 0.5)
    assert math.isnan(float(e_sigma2))
    assert math.isnan(float(var_mu))


def test_pool_identical_buckets_is_identity():
    bucket = (0.4, 100.0, 50.5, 1.2)
    out = pool_buckets([bucket, bucket, bucket], weights=[1.0, 1.0, 1.0])
    for a, b in zip(out, bucket):
        assert math.isclose(a, b, abs_tol=1e-12)


def test_pool_weights_bias():
    # Two buckets, one weighted 9x the other => pooled tau ~= the heavy one.
    heavy = (0.9, 200.0, 100.0, 2.0)
    light = (0.1, 10.0, 5.0, 0.1)
    tau_p, kappa_p, alpha_p, beta_p = pool_buckets([heavy, light], weights=[9.0, 1.0])
    assert math.isclose(tau_p, 0.9 * 0.9 + 0.1 * 0.1)
    assert math.isclose(kappa_p, 0.9 * 200.0 + 0.1 * 10.0)
    assert math.isclose(alpha_p, 0.9 * 100.0 + 0.1 * 5.0)
    assert math.isclose(beta_p, 0.9 * 2.0 + 0.1 * 0.1)


def test_pool_rejects_bad_input():
    with pytest.raises(ValueError):
        pool_buckets([], weights=[])
    with pytest.raises(ValueError):
        pool_buckets([(0.5, 1.0, 2.0, 0.1)], weights=[1.0, 2.0])
    with pytest.raises(ValueError):
        pool_buckets([(0.5, 1.0, 2.0, 0.1)], weights=[0.0])


def test_broadcast_arrays():
    # Per-surfel batched update: priors as (N,) arrays, observation as a
    # scalar applied to every surfel. Result should be (N,) and match the
    # elementwise application.
    N = 4
    tau = np.array([0.1, 0.2, 0.3, 0.4])
    kappa = np.array([1.0, 2.0, 3.0, 4.0])
    alpha = np.array([2.0, 2.0, 2.0, 2.0])
    beta = np.array([0.1, 0.1, 0.1, 0.1])

    t, k, a, b = closed_form_update(tau, kappa, alpha, beta, value=0.5, confidence=1.0)
    assert t.shape == (N,) and k.shape == (N,) and a.shape == (N,) and b.shape == (N,)
    # Spot-check one element against the scalar path.
    t0_scalar, _, _, _ = closed_form_update(tau[0], kappa[0], alpha[0], beta[0], 0.5, 1.0)
    assert math.isclose(float(t[0]), float(t0_scalar))
