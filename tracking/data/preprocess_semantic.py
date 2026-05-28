"""Run SAM3 + SigLIP2 preprocessing on every per-timestep scene directory.

This is a thin orchestration shim over the existing preprocess/sam3_masks.py
and preprocess/siglip2_embeddings.py CLI scripts. We invoke them via
subprocess so each timestep's preprocessing runs in a fresh Python
process -- matching how a user would run them by hand, and isolating any
model-state leaks between timesteps.

For each scene dir <timestep_dir>:

    <timestep_dir>/images/                     <- written by write_scene.py
    <timestep_dir>/sam3/cam_XX_regions.png     <- SAM3 output (after step 1)
    <timestep_dir>/sam3/cam_XX_meta.json
    <timestep_dir>/sam3/cam_XX_embeds.npy      <- SigLIP2 output (after step 2)

Concept lists, confidence, and SigLIP variant are passed through to the
underlying scripts. We support both --concepts (file) and --concept_list
(inline) forms.

The default `text_encoder_variant` in the parent's ModelParams is
"siglip2-base-patch16-512"; we expose `--variant` here with the same
default so SAM3 region-map dims and the embeds K_target line up with
what the training code expects.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SAM3_SCRIPT = os.path.join(REPO_ROOT, "preprocess", "sam3_masks.py")
SIGLIP_SCRIPT = os.path.join(REPO_ROOT, "preprocess", "siglip2_embeddings.py")


@dataclass(frozen=True)
class SemanticOptions:
    """Knobs passed straight through to the underlying CLIs."""
    concepts_file: str | None = None
    concept_list: str | None = None
    sam_confidence: float = 0.5
    sam_iou_dedup: float = 0.7
    siglip_variant: str = "google/siglip2-base-patch16-512"
    siglip_batch_size: int = 16
    overwrite: bool = False
    sam_dir_name: str = "sam3"      # subdir under each timestep_*/ where outputs go


def _check_scripts_exist():
    for p in (SAM3_SCRIPT, SIGLIP_SCRIPT):
        if not os.path.exists(p):
            raise FileNotFoundError(f"preprocessing script missing: {p}")


def _run(cmd: list[str], description: str) -> None:
    """Spawn cmd; surface non-zero exit codes with the captured stderr tail."""
    print(f"[preprocess_semantic] {description}: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{description} failed (exit {proc.returncode}); "
            f"cmd: {' '.join(cmd)}"
        )


def preprocess_one_timestep(timestep_dir: str, options: SemanticOptions) -> None:
    """Run SAM3 then SigLIP2 on one timestep's images/ folder."""
    _check_scripts_exist()
    images_dir = os.path.join(timestep_dir, "images")
    sam_dir = os.path.join(timestep_dir, options.sam_dir_name)
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"no images/ under {timestep_dir}")

    # -- 1. SAM3 region maps ---------------------------------------------
    sam_cmd = [
        sys.executable, SAM3_SCRIPT,
        "--input_dir", images_dir,
        "--output_dir", sam_dir,
        "--confidence", str(options.sam_confidence),
        "--iou_dedup", str(options.sam_iou_dedup),
    ]
    if options.concepts_file:
        sam_cmd += ["--concepts", options.concepts_file]
    if options.concept_list:
        sam_cmd += ["--concept_list", options.concept_list]
    if not (options.concepts_file or options.concept_list):
        raise ValueError(
            "SemanticOptions: must set concepts_file or concept_list "
            "(SAM3 needs a concept vocabulary)"
        )
    if options.overwrite:
        sam_cmd += ["--overwrite"]
    _run(sam_cmd, f"SAM3 on {os.path.basename(timestep_dir)}")

    # -- 2. SigLIP2 per-region embeddings --------------------------------
    siglip_cmd = [
        sys.executable, SIGLIP_SCRIPT,
        "--input_dir", images_dir,
        "--regions_dir", sam_dir,
        "--output_dir", sam_dir,
        "--variant", options.siglip_variant,
        "--batch_size", str(options.siglip_batch_size),
    ]
    if options.overwrite:
        siglip_cmd += ["--overwrite"]
    _run(siglip_cmd, f"SigLIP2 on {os.path.basename(timestep_dir)}")


def preprocess_timesteps(timestep_dirs: Iterable[str],
                          options: SemanticOptions) -> None:
    """Run SAM3 + SigLIP2 across many timesteps, sequentially."""
    for d in timestep_dirs:
        preprocess_one_timestep(d, options)
