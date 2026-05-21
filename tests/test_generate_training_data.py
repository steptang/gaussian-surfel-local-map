"""Tests for the Path B fusion-MLP training-data generator.

Covers:
1. distillation_schema_complete       — every required field present, shapes match.
2. distillation_target_matches_closed_form — saved target = vectorised closed-form.
3. distillation_local_equals_asymptotic — for distillation, the two targets agree.
4. distillation_features_zero         — geometric/semantic columns are zero-padded.
5. distillation_has_features_all_false.
6. sample_prior_states_in_physical_range — tau in (0,1.5], kappa/alpha/beta > 0.
7. sample_observations_mixture_sane    — values in expected range, confidences in [0,1].
8. build_rotation_np_matches_quat_identity — w=1 quat -> identity matrix.
9. extract_geometric_features_shape_and_finite — (N, 8), no NaN/Inf, normals unit.
10. emit_region_pairs_records_pre_and_post   — pre_state stored is the actual pre,
    target_local is closed-form-after-this-update, in-place state advances.
11. emit_region_pairs_target_asymptotic_uses_cache — target_asymptotic reads from
    the precomputed asymptotic_state, not from the live state.
12. emit_region_pairs_caps_at_max_per_region.
13. save_dataset_round_trip — torch.save -> torch.load preserves all fields.

Run from repo root with: pytest tests/test_generate_training_data.py -v
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion.fuse_surfels import SurfelPhysicalState
from fusion.generate_training_data import (
    DEFAULT_GEOMETRIC_DIM,
    DEFAULT_SEMANTIC_DIM,
    SCHEMA_VERSION,
    _build_rotation_np,
    _ScenePairAccumulator,
    emit_region_pairs,
    extract_geometric_features,
    generate_distillation_pairs,
    sample_observations,
    sample_prior_states,
    save_dataset,
)
from fusion.lookup import PriorTable
from fusion.nig import closed_form_update


def _synthetic_priors_dict():
    return {
        "schema_version": 1,
        "model": "test",
        "property": "friction_coefficient",
        "materials": ["rocks", "grass"],
        "friction_levels": ["low", "medium", "high"],
        "prior_hyperparams": {"tau": 0.5, "kappa": 1.0, "alpha": 2.0, "beta": 0.1},
        "buckets": {
            "rocks|medium": {"material": "rocks", "friction": "medium",
                             "tau": 0.5, "kappa": 100.0, "alpha": 50.0, "beta": 1.0, "n": 100},
            "grass|high":   {"material": "grass", "friction": "high",
                             "tau": 0.8, "kappa": 80.0, "alpha": 40.0, "beta": 0.8, "n": 80},
        },
    }


def _priors_json(tmp_path: Path) -> str:
    p = tmp_path / "priors.json"
    p.write_text(json.dumps(_synthetic_priors_dict()))
    return str(p)


# --- distillation ---------------------------------------------------------

def test_distillation_schema_complete(tmp_path):
    ds = generate_distillation_pairs(n=128, priors_path=_priors_json(tmp_path), seed=0)
    assert ds["schema_version"] == SCHEMA_VERSION
    assert ds["kind"] == "distillation"
    assert ds["n_samples"] == 128
    assert ds["prior_state"].shape == (128, 4)
    assert ds["observation"].shape == (128, 2)
    assert ds["geometric_features"].shape == (128, DEFAULT_GEOMETRIC_DIM)
    assert ds["semantic_features"].shape == (128, DEFAULT_SEMANTIC_DIM)
    assert ds["target_local"].shape == (128, 4)
    assert ds["target_asymptotic"].shape == (128, 4)
    assert ds["has_features"].shape == (128,)
    assert "feature_meta" in ds


def test_distillation_target_matches_closed_form(tmp_path):
    ds = generate_distillation_pairs(n=64, priors_path=_priors_json(tmp_path), seed=1)
    prior = ds["prior_state"].astype(np.float64)
    obs = ds["observation"].astype(np.float64)
    expected = closed_form_update(
        prior[:, 0], prior[:, 1], prior[:, 2], prior[:, 3],
        obs[:, 0], obs[:, 1],
    )
    expected_stack = np.stack(expected, axis=1).astype(np.float32)
    np.testing.assert_allclose(ds["target_local"], expected_stack, atol=1e-5)


def test_distillation_local_equals_asymptotic(tmp_path):
    ds = generate_distillation_pairs(n=32, priors_path=_priors_json(tmp_path), seed=2)
    np.testing.assert_array_equal(ds["target_local"], ds["target_asymptotic"])


def test_distillation_features_zero(tmp_path):
    ds = generate_distillation_pairs(n=16, priors_path=_priors_json(tmp_path), seed=3)
    assert (ds["geometric_features"] == 0).all()
    assert (ds["semantic_features"] == 0).all()
    assert (ds["has_features"] == False).all()


# --- sampling -------------------------------------------------------------

def test_sample_prior_states_in_physical_range(tmp_path):
    priors = PriorTable.from_path(_priors_json(tmp_path))
    rng = np.random.default_rng(0)
    states = sample_prior_states(1000, priors, rng)
    assert states.shape == (1000, 4)
    # tau clipped to (0, 1.5]; kappa, alpha, beta positive; alpha > 1 for
    # finite posterior variance.
    assert ((states[:, 0] > 0.0) & (states[:, 0] <= 1.5)).all()
    assert (states[:, 1] > 0.0).all()
    assert (states[:, 2] > 1.0).all()
    assert (states[:, 3] > 0.0).all()


def test_sample_observations_in_range(tmp_path):
    priors = PriorTable.from_path(_priors_json(tmp_path))
    rng = np.random.default_rng(0)
    states = sample_prior_states(500, priors, rng)
    obs = sample_observations(500, states, priors, rng)
    assert obs.shape == (500, 2)
    assert ((obs[:, 0] > 0.0) & (obs[:, 0] <= 1.5)).all()
    assert ((obs[:, 1] >= 0.0) & (obs[:, 1] <= 1.0)).all()


# --- numpy rotation port --------------------------------------------------

def test_build_rotation_np_identity_quat():
    # quat (1, 0, 0, 0) is identity in (w, x, y, z) convention.
    R = _build_rotation_np(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(R[0], np.eye(3), atol=1e-6)


def test_build_rotation_np_unit_quat_orthogonal():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(10, 4)).astype(np.float32)
    R = _build_rotation_np(q)
    # R R^T should be I for every sample.
    rrt = R @ R.transpose(0, 2, 1)
    eye = np.broadcast_to(np.eye(3, dtype=np.float32), (10, 3, 3))
    np.testing.assert_allclose(rrt, eye, atol=1e-4)


# --- geometric features ---------------------------------------------------

def test_extract_geometric_features_shape_and_finite():
    N = 8
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(N, 3)).astype(np.float32)
    scaling = np.abs(rng.normal(size=(N, 2))).astype(np.float32) + 0.01
    rotation = rng.normal(size=(N, 4)).astype(np.float32)
    opacity = np.abs(rng.normal(size=(N, 1))).astype(np.float32) + 0.01
    feats = extract_geometric_features(xyz, scaling, rotation, opacity)
    assert feats.shape == (N, 8)
    assert np.isfinite(feats).all()
    # Normals (first 3 columns) should be unit-length.
    norms = np.linalg.norm(feats[:, :3], axis=1)
    np.testing.assert_allclose(norms, np.ones(N), atol=1e-3)


# --- scene-pair emission --------------------------------------------------

def test_emit_region_pairs_records_pre_and_post():
    # Setup: one region covering all 4 pixels, surfel 0 owns it entirely.
    H, W, K = 2, 2, 1
    region_map = np.zeros((H, W), dtype=np.int32)
    cids = np.zeros((K, H, W), dtype=np.int32)
    cw = np.ones((K, H, W), dtype=np.float32)

    state = SurfelPhysicalState.from_global_prior(2, 0.5, 1.0, 2.0, 0.1)
    asymptotic = SurfelPhysicalState.from_global_prior(2, 0.99, 99.0, 99.0, 9.9)
    geom = np.full((2, 8), 7.0, dtype=np.float32)
    sem = np.full((2, 16), 3.0, dtype=np.float32)
    acc = _ScenePairAccumulator.empty()
    rng = np.random.default_rng(0)

    n = emit_region_pairs(
        state=state, asymptotic=asymptotic, region_id=0,
        obs_value=0.9, vlm_confidence=1.0,
        region_map=region_map, contrib_ids=cids, contrib_weights=cw,
        geometric_features=geom, semantic_features=sem,
        acc=acc, max_per_region=10, rng=rng,
    )
    assert n == 1
    assert len(acc) == 1
    # pre_state = prior (before update)
    np.testing.assert_allclose(acc.prior_state[0], [0.5, 1.0, 2.0, 0.1], atol=1e-5)
    # target_local = closed_form_update(prior, value=0.9, omega=1.0)
    expected = closed_form_update(0.5, 1.0, 2.0, 0.1, value=0.9, confidence=1.0)
    np.testing.assert_allclose(acc.target_local[0],
                                [float(e) for e in expected], atol=1e-5)
    # target_asymptotic = the cached asymptotic state for that surfel
    np.testing.assert_allclose(acc.target_asymptotic[0], [0.99, 99.0, 99.0, 9.9], atol=1e-5)
    # In-place state advanced to the post-update values.
    np.testing.assert_allclose(state.tau[0], float(expected[0]), atol=1e-5)
    # The other surfel untouched.
    assert math.isclose(state.tau[1], 0.5)


def test_emit_region_pairs_target_asymptotic_uses_cache():
    # The asymptotic target must come from the precomputed cache, not the
    # live state. Set state to a recognisable value, asymptotic to a
    # different recognisable value, then confirm the emitted target
    # references the cache.
    H, W, K = 1, 1, 1
    region_map = np.zeros((H, W), dtype=np.int32)
    cids = np.zeros((K, H, W), dtype=np.int32)
    cw = np.ones((K, H, W), dtype=np.float32)

    state = SurfelPhysicalState.from_global_prior(1, 0.5, 1.0, 2.0, 0.1)
    asymptotic = SurfelPhysicalState.from_global_prior(1, 0.123, 4.56, 7.89, 1.011)
    acc = _ScenePairAccumulator.empty()
    rng = np.random.default_rng(0)
    emit_region_pairs(
        state=state, asymptotic=asymptotic, region_id=0,
        obs_value=0.7, vlm_confidence=0.5,
        region_map=region_map, contrib_ids=cids, contrib_weights=cw,
        geometric_features=np.zeros((1, 8), dtype=np.float32),
        semantic_features=np.zeros((1, 16), dtype=np.float32),
        acc=acc, max_per_region=1, rng=rng,
    )
    np.testing.assert_allclose(acc.target_asymptotic[0],
                                [0.123, 4.56, 7.89, 1.011], atol=1e-5)


def test_emit_region_pairs_caps_at_max_per_region():
    # 6 surfels in the region but max_per_region=3.
    H, W, K = 1, 6, 1
    region_map = np.zeros((H, W), dtype=np.int32)
    cids = np.array([[[0, 1, 2, 3, 4, 5]]], dtype=np.int32)
    cw = np.ones((K, H, W), dtype=np.float32)
    state = SurfelPhysicalState.from_global_prior(6, 0.5, 1.0, 2.0, 0.1)
    asymptotic = SurfelPhysicalState.from_global_prior(6, 0.5, 1.0, 2.0, 0.1)
    acc = _ScenePairAccumulator.empty()
    rng = np.random.default_rng(0)
    n = emit_region_pairs(
        state=state, asymptotic=asymptotic, region_id=0,
        obs_value=0.9, vlm_confidence=1.0,
        region_map=region_map, contrib_ids=cids, contrib_weights=cw,
        geometric_features=np.zeros((6, 8), dtype=np.float32),
        semantic_features=np.zeros((6, 16), dtype=np.float32),
        acc=acc, max_per_region=3, rng=rng,
    )
    assert n == 3
    assert len(acc) == 3
    # All 6 surfels' state has still been updated in place, regardless
    # of subsampling for emission.
    for sid in range(6):
        assert state.tau[sid] != 0.5


# --- save/load round-trip ------------------------------------------------

def test_save_dataset_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    ds = generate_distillation_pairs(n=8, priors_path=_priors_json(tmp_path), seed=4)
    out_path = tmp_path / "ds.pt"
    save_dataset(ds, str(out_path))
    loaded = torch.load(str(out_path), weights_only=False)
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["kind"] == "distillation"
    assert loaded["n_samples"] == 8
    # arrays come back as torch tensors with the same values
    np.testing.assert_allclose(loaded["prior_state"].numpy(), ds["prior_state"], atol=1e-6)
    np.testing.assert_allclose(loaded["target_local"].numpy(), ds["target_local"], atol=1e-6)
