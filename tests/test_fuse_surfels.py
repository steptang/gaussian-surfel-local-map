"""Tests for the Path A per-surfel fusion driver (fusion/fuse_surfels.py).

These tests exercise the pure-Python aggregation logic without CUDA:
synthetic (K, H, W) contrib buffers + a fake region map + a tiny in-memory
priors table. The full fuse_surfels() orchestrator is integration-tested
manually on a real scene; here we verify the math is right.

1. single_surfel_full_region_matches_closed_form — one surfel owns every
   pixel of one region; its posterior moves toward obs_value exactly as a
   single conjugate update with omega = vlm_confidence.
2. omega_normalised_by_region_size — a surfel covering only half a region
   gets omega = 0.5 * vlm_confidence, independent of how big the region is.
3. two_equal_surfels_get_equal_updates — when two surfels split a region
   evenly, both posteriors move identically.
4. surfels_outside_region_unchanged — surfels that don't contribute stay
   at the prior.
5. sentinel_ids_are_ignored — -1 slots in contrib_ids never cause writes.
6. missing_region_id_is_noop — region_id absent from region_map returns 0.
7. apply_view_lookup_stats — lookup source ("exact"/"material_marginal"/
   "global") is counted correctly per region.
8. parse_region_record_accepts_friction_alias — both `friction` and
   `friction_level` field names parse identically; vlm_confidence defaults
   to 1.0 when absent.
9. from_global_prior_initialises_all_surfels — initial state is the
   advertised broadcast.

Run from repo root with: pytest tests/test_fuse_surfels.py -v
"""

import math
import os
import sys
from collections import Counter

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion.fuse_surfels import (
    SurfelPhysicalState,
    _parse_region_record,
    apply_region_update,
    apply_view_update,
)
from fusion.lookup import PriorTable
from fusion.nig import closed_form_update


# --- Fixtures -------------------------------------------------------------

def _state(n: int = 4, tau: float = 0.5, kappa: float = 1.0,
           alpha: float = 2.0, beta: float = 0.1) -> SurfelPhysicalState:
    return SurfelPhysicalState.from_global_prior(n, tau, kappa, alpha, beta)


def _priors_with_one_bucket():
    """Tiny PriorTable with one exact bucket so 'rocks'+'medium' hits exact."""
    data = {
        "schema_version": 1,
        "model": "test",
        "property": "friction_coefficient",
        "materials": ["rocks"],
        "friction_levels": ["medium"],
        "prior_hyperparams": {"tau": 0.5, "kappa": 1.0, "alpha": 2.0, "beta": 0.1},
        "buckets": {
            "rocks|medium": {
                "material": "rocks", "friction": "medium",
                "tau": 0.7, "kappa": 100.0, "alpha": 50.0, "beta": 1.0, "n": 100,
            },
        },
    }
    return PriorTable(data)


# --- apply_region_update --------------------------------------------------

def test_single_surfel_full_region_matches_closed_form():
    # Region of 4 pixels, surfel 0 owns every pixel with full weight.
    # omega = vlm_conf * sum(weight) / num_pixels = 0.8 * (4 * 1.0) / 4 = 0.8.
    H, W, K = 2, 2, 4
    region_map = np.zeros((H, W), dtype=np.int32)   # whole map = region 0
    cids = np.full((K, H, W), -1, dtype=np.int32)
    cids[0, :, :] = 0
    cw = np.zeros((K, H, W), dtype=np.float32)
    cw[0, :, :] = 1.0

    state = _state(n=2)
    n = apply_region_update(state, region_id=0, obs_value=0.9,
                             vlm_confidence=0.8,
                             region_map=region_map,
                             contrib_ids=cids,
                             contrib_weights=cw)
    assert n == 1
    expected = closed_form_update(0.5, 1.0, 2.0, 0.1, value=0.9, confidence=0.8)
    assert math.isclose(state.tau[0], float(expected[0]), abs_tol=1e-10)
    assert math.isclose(state.kappa[0], float(expected[1]), abs_tol=1e-10)
    assert math.isclose(state.alpha[0], float(expected[2]), abs_tol=1e-10)
    assert math.isclose(state.beta[0], float(expected[3]), abs_tol=1e-10)
    # Untouched surfel stays at the prior.
    assert math.isclose(state.tau[1], 0.5)
    assert math.isclose(state.kappa[1], 1.0)


def test_omega_normalised_by_region_size():
    # Surfel 0 covers 2 of 4 pixels with weight 1; the other 2 pixels of the
    # region are empty (no contribution recorded). omega should be
    # 0.8 * (2 * 1.0) / 4 = 0.4.
    H, W, K = 2, 2, 4
    region_map = np.zeros((H, W), dtype=np.int32)
    cids = np.full((K, H, W), -1, dtype=np.int32)
    cw = np.zeros((K, H, W), dtype=np.float32)
    cids[0, 0, 0] = 0; cw[0, 0, 0] = 1.0
    cids[0, 0, 1] = 0; cw[0, 0, 1] = 1.0

    state = _state(n=1)
    n = apply_region_update(state, region_id=0, obs_value=0.9,
                             vlm_confidence=0.8,
                             region_map=region_map,
                             contrib_ids=cids, contrib_weights=cw)
    assert n == 1
    expected = closed_form_update(0.5, 1.0, 2.0, 0.1, value=0.9, confidence=0.4)
    assert math.isclose(state.tau[0], float(expected[0]), abs_tol=1e-10)
    assert math.isclose(state.kappa[0], float(expected[1]), abs_tol=1e-10)


def test_two_equal_surfels_get_equal_updates():
    # Surfels 0 and 1 each contribute 0.5 weight to every pixel of a
    # 4-pixel region. Each has total weight 2.0, omega = 0.8 * 2 / 4 = 0.4.
    H, W, K = 2, 2, 4
    region_map = np.zeros((H, W), dtype=np.int32)
    cids = np.full((K, H, W), -1, dtype=np.int32)
    cw = np.zeros((K, H, W), dtype=np.float32)
    cids[0, :, :] = 0; cw[0, :, :] = 0.5
    cids[1, :, :] = 1; cw[1, :, :] = 0.5

    state = _state(n=2)
    n = apply_region_update(state, region_id=0, obs_value=0.9,
                             vlm_confidence=0.8,
                             region_map=region_map,
                             contrib_ids=cids, contrib_weights=cw)
    assert n == 2
    assert math.isclose(state.tau[0], state.tau[1], abs_tol=1e-12)
    assert math.isclose(state.kappa[0], state.kappa[1], abs_tol=1e-12)
    assert math.isclose(state.alpha[0], state.alpha[1], abs_tol=1e-12)
    assert math.isclose(state.beta[0], state.beta[1], abs_tol=1e-12)


def test_surfels_outside_region_unchanged():
    # Region 1 contains only surfel 0 (at the single pixel where region==1).
    # Surfels 2, 3 are recorded but in pixels belonging to region 0, so the
    # region-1 update must leave them at the prior.
    H, W, K = 2, 2, 2
    region_map = np.array([[0, 0], [0, 1]], dtype=np.int32)
    cids = np.full((K, H, W), -1, dtype=np.int32)
    cw = np.zeros((K, H, W), dtype=np.float32)
    cids[0, 0, 0] = 2; cw[0, 0, 0] = 1.0   # surfel 2 in region 0
    cids[0, 0, 1] = 3; cw[0, 0, 1] = 1.0   # surfel 3 in region 0
    cids[0, 1, 1] = 0; cw[0, 1, 1] = 1.0   # surfel 0 in region 1

    state = _state(n=4)
    n = apply_region_update(state, region_id=1, obs_value=0.9,
                             vlm_confidence=1.0,
                             region_map=region_map,
                             contrib_ids=cids, contrib_weights=cw)
    assert n == 1
    # surfels 1, 2, 3 untouched
    for i in (1, 2, 3):
        assert math.isclose(state.tau[i], 0.5)
        assert math.isclose(state.kappa[i], 1.0)


def test_sentinel_ids_are_ignored():
    # All slots are -1 except one. bincount must not interpret -1 as a
    # surfel index (would corrupt state via negative indexing).
    H, W, K = 1, 1, 4
    region_map = np.zeros((H, W), dtype=np.int32)
    cids = np.full((K, H, W), -1, dtype=np.int32)
    cw = np.zeros((K, H, W), dtype=np.float32)
    cids[0, 0, 0] = 1; cw[0, 0, 0] = 1.0

    state = _state(n=3)
    n = apply_region_update(state, region_id=0, obs_value=0.9,
                             vlm_confidence=1.0,
                             region_map=region_map,
                             contrib_ids=cids, contrib_weights=cw)
    assert n == 1
    # only surfel 1 moved; 0 and 2 stayed at prior
    assert math.isclose(state.tau[0], 0.5)
    assert math.isclose(state.tau[2], 0.5)
    assert state.tau[1] != 0.5


def test_missing_region_id_is_noop():
    H, W, K = 2, 2, 2
    region_map = np.zeros((H, W), dtype=np.int32)
    cids = np.full((K, H, W), -1, dtype=np.int32)
    cw = np.zeros((K, H, W), dtype=np.float32)

    state = _state(n=2)
    n = apply_region_update(state, region_id=42, obs_value=0.9,
                             vlm_confidence=1.0,
                             region_map=region_map,
                             contrib_ids=cids, contrib_weights=cw)
    assert n == 0
    assert math.isclose(state.tau[0], 0.5)
    assert math.isclose(state.tau[1], 0.5)


# --- apply_view_update ----------------------------------------------------

def test_apply_view_lookup_stats():
    # Two regions: one hits the exact bucket, one falls through to global.
    H, W, K = 1, 2, 1
    region_map = np.array([[1, 2]], dtype=np.int32)
    cids = np.array([[[0, 1]]], dtype=np.int32)        # (K=1, 1, 2)
    cw = np.array([[[1.0, 1.0]]], dtype=np.float32)

    priors = _priors_with_one_bucket()
    records = [
        {"region_id": 1, "material": "rocks", "friction_level": "medium",
         "vlm_confidence": 0.9},
        {"region_id": 2, "material": "snow", "friction_level": "low",
         "vlm_confidence": 0.5},
    ]
    state = _state(n=2)
    stats = Counter()
    n = apply_view_update(state, records, region_map, cids, cw,
                          priors, lookup_stats=stats)
    assert n == 2
    assert stats["exact"] == 1
    assert stats["global"] == 1


def test_apply_view_skips_malformed_record(capsys):
    # A record missing the friction field is skipped with a log message;
    # the other record still applies.
    H, W, K = 1, 2, 1
    region_map = np.array([[1, 2]], dtype=np.int32)
    cids = np.array([[[0, 1]]], dtype=np.int32)
    cw = np.array([[[1.0, 1.0]]], dtype=np.float32)
    priors = _priors_with_one_bucket()
    records = [
        {"region_id": 1, "material": "rocks", "friction": "medium",
         "vlm_confidence": 1.0},
        {"region_id": 2, "material": "snow"},  # malformed: no friction
    ]
    state = _state(n=2)
    stats = Counter()
    n = apply_view_update(state, records, region_map, cids, cw,
                          priors, lookup_stats=stats)
    assert n == 1
    assert stats["exact"] == 1
    captured = capsys.readouterr()
    assert "skipping malformed region record" in captured.out


# --- parse_region_record --------------------------------------------------

def test_parse_region_record_accepts_friction_alias():
    rid, m, lvl, c = _parse_region_record(
        {"region_id": 5, "material": "Rocks", "friction": "Medium",
         "vlm_confidence": 0.7}
    )
    assert rid == 5
    assert m == "rocks"        # lowercased
    assert lvl == "medium"
    assert math.isclose(c, 0.7)

    rid, m, lvl, c = _parse_region_record(
        {"region_id": 6, "material": "wood", "friction_level": "high"}
    )
    assert lvl == "high"
    assert math.isclose(c, 1.0)   # defaults to 1.0 when absent


def test_parse_region_record_rejects_missing_friction():
    with pytest.raises(KeyError):
        _parse_region_record({"region_id": 1, "material": "rocks"})


# --- SurfelPhysicalState --------------------------------------------------

def test_from_global_prior_initialises_all_surfels():
    s = SurfelPhysicalState.from_global_prior(5, 0.4, 2.0, 3.0, 0.5)
    assert s.tau.shape == (5,)
    assert (s.tau == 0.4).all()
    assert (s.kappa == 2.0).all()
    assert (s.alpha == 3.0).all()
    assert (s.beta == 0.5).all()
    assert s.tau.dtype == np.float64
