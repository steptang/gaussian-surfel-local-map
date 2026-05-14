"""Tests for the prior-lookup fallback hierarchy (fusion/lookup.py).

Verifies the three-tier lookup contract:

1. exact_bucket_hit — "<material>|<level>" present => returned verbatim with source="exact".
2. material_marginal_pool — material present at other friction_level(s) => weighted pool,
   source="material_marginal". n-weighting biases toward the larger bucket.
3. global_fallback — material absent everywhere => prior_hyperparams, source="global".
4. empty_buckets_falls_through — buckets={} always returns the global prior.
5. from_path_round_trip — PriorTable.from_path parses an on-disk priors.json correctly.

Run from repo root with: pytest tests/test_fusion_lookup.py -v
"""

import json
import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion.lookup import PriorTable


def _synthetic_priors():
    """Mirror the priors.json schema documented in the task brief."""
    return {
        "schema_version": 1,
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "property": "friction_coefficient",
        "materials": ["concrete", "rocks", "wood"],
        "friction_levels": ["low", "medium", "high"],
        "prior_hyperparams": {"tau": 0.5, "kappa": 1.0, "alpha": 2.0, "beta": 0.1},
        "buckets": {
            "rocks|medium": {
                "material": "rocks", "friction": "medium",
                "tau": 0.478, "kappa": 451.0, "alpha": 227.5, "beta": 2.85, "n": 450,
            },
            "rocks|high": {
                "material": "rocks", "friction": "high",
                "tau": 0.62, "kappa": 51.0, "alpha": 26.0, "beta": 0.4, "n": 50,
            },
            "concrete|high": {
                "material": "concrete", "friction": "high",
                "tau": 0.81, "kappa": 101.0, "alpha": 51.0, "beta": 0.5, "n": 100,
            },
        },
    }


def test_exact_bucket_hit():
    table = PriorTable(_synthetic_priors())
    r = table.lookup("rocks", "medium")
    assert r.source == "exact"
    assert r.detail == "exact:rocks|medium"
    assert math.isclose(r.prior.tau, 0.478)
    assert math.isclose(r.prior.kappa, 451.0)
    assert math.isclose(r.prior.alpha, 227.5)
    assert math.isclose(r.prior.beta, 2.85)


def test_material_marginal_pool_n_weighted():
    # rocks|low isn't in the table; the pool over (rocks|medium, rocks|high) is
    # n-weighted: weights 450 and 50, normalised to 0.9 and 0.1.
    table = PriorTable(_synthetic_priors())
    r = table.lookup("rocks", "low")
    assert r.source == "material_marginal"
    assert "rocks" in r.detail and "2 buckets" in r.detail

    expected_tau = 0.9 * 0.478 + 0.1 * 0.62
    expected_kappa = 0.9 * 451.0 + 0.1 * 51.0
    assert math.isclose(r.prior.tau, expected_tau, abs_tol=1e-9)
    assert math.isclose(r.prior.kappa, expected_kappa, abs_tol=1e-9)


def test_global_fallback_unknown_material():
    table = PriorTable(_synthetic_priors())
    r = table.lookup("snow", "low")
    assert r.source == "global"
    assert r.detail == "global"
    assert math.isclose(r.prior.tau, 0.5)
    assert math.isclose(r.prior.kappa, 1.0)
    assert math.isclose(r.prior.alpha, 2.0)
    assert math.isclose(r.prior.beta, 0.1)


def test_empty_buckets_falls_through():
    data = _synthetic_priors()
    data["buckets"] = {}
    table = PriorTable(data)
    r = table.lookup("rocks", "medium")
    assert r.source == "global"


def test_from_path_round_trip(tmp_path: Path):
    data = _synthetic_priors()
    p = tmp_path / "priors.json"
    p.write_text(json.dumps(data))
    table = PriorTable.from_path(p)
    r = table.lookup("concrete", "high")
    assert r.source == "exact"
    assert math.isclose(r.prior.tau, 0.81)
