"""CLI: Stage A end-to-end.

Reads a DMV-format scene (h5 + poses_bounds.npy), writes Blender-style
per-timestep scene directories, and optionally runs SAM3 + SigLIP2
preprocessing on each one. Output is consumed by Stage B
(tracking.driver.run_per_timestep).

Example -- prepare every timestep with semantic preprocessing:

    python -m tracking.data.build_dataset \\
        --h5 path/to/scene.h5 --poses path/to/poses_bounds.npy \\
        --work-root path/to/work/scene \\
        --concept-list "person,table,chair,floor,wall" \\
        --variant google/siglip2-base-patch16-512

Single-timestep verification mode (matches Stephanie's planned silent-
failure check on LLFF conversion + sparse-view reconstruction):

    python -m tracking.data.build_dataset \\
        --h5 ... --poses ... --work-root ... \\
        --timesteps 0 \\
        --concept-list "person,floor,wall"
"""

from __future__ import annotations

import argparse
import sys

from .dmv_loader import DMVScene, select_timesteps
from .preprocess_semantic import SemanticOptions, preprocess_timesteps
from .write_scene import WriteOptions, write_timesteps


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage A: DMV scene -> per-timestep Blender-style scenes (+ optional SAM3/SigLIP2 preprocessing)"
    )
    p.add_argument("--h5", required=True, help="path to the DMV h5 file")
    p.add_argument("--poses", required=True, help="path to poses_bounds.npy (LLFF, n_cams x 17)")
    p.add_argument("--work-root", required=True,
                   help="output directory; per-timestep subdirs will be created under it")

    # Timestep selection (mutually informational; precedence handled in select_timesteps).
    p.add_argument("--timesteps", nargs="+", type=int, default=None,
                   help="explicit list of timesteps to prepare (overrides --stride / --count)")
    p.add_argument("--stride", type=int, default=None,
                   help="subsample every Nth timestep (default: all)")
    p.add_argument("--count", type=int, default=None,
                   help="prepare the first N timesteps (default: all)")

    # Writer options
    p.add_argument("--max-image-side", type=int, default=None,
                   help="optional resize cap on the long image side; default no resize")
    p.add_argument("--no-masks", action="store_true",
                   help="skip writing per-camera fg/bg mask PNGs")

    # Semantic preprocessing
    p.add_argument("--skip-semantic", action="store_true",
                   help="skip SAM3 + SigLIP2 preprocessing entirely (use for the single-timestep "
                        "verification path where you only want to inspect reconstruction quality)")
    p.add_argument("--concepts", default=None,
                   help="file with one concept per line (SAM3 vocabulary)")
    p.add_argument("--concept-list", default=None,
                   help="inline comma-separated concepts; overrides --concepts")
    p.add_argument("--sam-confidence", type=float, default=0.5)
    p.add_argument("--sam-iou-dedup", type=float, default=0.7)
    p.add_argument("--variant", default="google/siglip2-base-patch16-512",
                   help="SigLIP2 model id; MUST match the K_target the training code expects")
    p.add_argument("--siglip-batch-size", type=int, default=16)
    p.add_argument("--overwrite-semantic", action="store_true",
                   help="re-run SAM3/SigLIP2 even if outputs already exist")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    with DMVScene(args.h5, args.poses) as scene:
        timesteps = select_timesteps(
            scene.timesteps(),
            explicit=args.timesteps,
            stride=args.stride,
            count=args.count,
        )
        print(f"[build_dataset] scene: n_cams={scene.meta.n_cams}, "
              f"n_timesteps={scene.meta.n_timesteps}, image=({scene.meta.height}, {scene.meta.width})")
        print(f"[build_dataset] preparing {len(timesteps)} timesteps: {timesteps[:8]}"
              f"{'...' if len(timesteps) > 8 else ''}")

        wopts = WriteOptions(
            work_root=args.work_root,
            write_masks=not args.no_masks,
            max_image_side=args.max_image_side,
        )
        dirs = write_timesteps(scene, timesteps, wopts)
        print(f"[build_dataset] wrote {len(dirs)} timestep dirs under {args.work_root}")

    if args.skip_semantic:
        print("[build_dataset] --skip-semantic set; SAM3/SigLIP2 not run.")
        return 0
    if not (args.concepts or args.concept_list):
        print("[build_dataset] no --concepts / --concept-list given; "
              "skipping semantic preprocessing. Pass --skip-semantic to silence this.")
        return 0

    sopts = SemanticOptions(
        concepts_file=args.concepts,
        concept_list=args.concept_list,
        sam_confidence=args.sam_confidence,
        sam_iou_dedup=args.sam_iou_dedup,
        siglip_variant=args.variant,
        siglip_batch_size=args.siglip_batch_size,
        overwrite=args.overwrite_semantic,
    )
    preprocess_timesteps(dirs, sopts)
    print(f"[build_dataset] SAM3 + SigLIP2 done for {len(dirs)} timesteps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
