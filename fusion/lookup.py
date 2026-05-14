"""Prior lookup with fallback hierarchy for per-surfel NIG fusion.

Given a priors.json produced by the shared preprocessing repo's
build_priors.py and a (material, friction_level) query, return the NIG
prior to use as the starting point for fusion (Path A or Path B).

Hierarchy:
    1. Exact bucket hit: "<material>|<level>" present in buckets.
    2. Material-marginal: no exact match but the material appears in some
       bucket — pool across friction levels (weighted by per-bucket n).
    3. Global fallback: priors.json:prior_hyperparams (weak global prior).

Callers receive a LookupResult tagged with which path was taken so the
paper-eval pass can report exact-bucket hit rate ("X% of surfels got
exact-bucket priors").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .nig import pool_buckets


Source = Literal["exact", "material_marginal", "global"]


@dataclass(frozen=True)
class NIGPrior:
    tau: float
    kappa: float
    alpha: float
    beta: float


@dataclass(frozen=True)
class LookupResult:
    prior: NIGPrior
    source: Source
    detail: str   # e.g. "exact:rocks|medium" or "material_marginal:rocks(2 buckets)"


class PriorTable:
    """Parsed priors.json wrapped for repeated lookups."""

    def __init__(self, data: dict):
        gh = data["prior_hyperparams"]
        self._global = NIGPrior(
            tau=float(gh["tau"]),
            kappa=float(gh["kappa"]),
            alpha=float(gh["alpha"]),
            beta=float(gh["beta"]),
        )
        self._buckets: dict[str, dict] = data.get("buckets", {})
        self._by_material: dict[str, list[dict]] = {}
        for bucket in self._buckets.values():
            self._by_material.setdefault(bucket["material"], []).append(bucket)

    @classmethod
    def from_path(cls, path: str | Path) -> "PriorTable":
        with open(path, "r") as f:
            return cls(json.load(f))

    def lookup(self, material: str, friction_level: str) -> LookupResult:
        key = f"{material}|{friction_level}"
        if key in self._buckets:
            b = self._buckets[key]
            return LookupResult(
                prior=NIGPrior(
                    tau=float(b["tau"]),
                    kappa=float(b["kappa"]),
                    alpha=float(b["alpha"]),
                    beta=float(b["beta"]),
                ),
                source="exact",
                detail=f"exact:{key}",
            )

        candidates = self._by_material.get(material, [])
        if candidates:
            tuples = [
                (float(b["tau"]), float(b["kappa"]), float(b["alpha"]), float(b["beta"]))
                for b in candidates
            ]
            # n-weighting biases the pool toward buckets backed by more
            # calibration data; falls back to 1.0 if a bucket somehow
            # lacks n.
            weights = [float(b.get("n", 1.0)) for b in candidates]
            tau_p, kappa_p, alpha_p, beta_p = pool_buckets(tuples, weights)
            return LookupResult(
                prior=NIGPrior(tau=tau_p, kappa=kappa_p, alpha=alpha_p, beta=beta_p),
                source="material_marginal",
                detail=f"material_marginal:{material}({len(candidates)} buckets)",
            )

        return LookupResult(prior=self._global, source="global", detail="global")
