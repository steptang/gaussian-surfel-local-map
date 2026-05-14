"""Per-surfel physical-property posterior fusion.

Submodules:
    nig    : closed-form NIG conjugate update (PhysGS Eq 12-14) — Path A
             baseline, distillation target for Path B, low-confidence fallback.
    lookup : (material, friction_level) -> NIG prior from priors.json with
             material-marginal and global-prior fallbacks.
"""
