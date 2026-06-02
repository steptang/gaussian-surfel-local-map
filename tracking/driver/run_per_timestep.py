"""CLI: Stage B -- run the existing 2DGS reconstruction per selected timestep.

Iterates over the per-timestep scene dirs that Stage A produced and
launches train.py once per timestep via subprocess. Subprocess isolation
is deliberate: each timestep starts with fresh CUDA / Python state, no
leakage of densification buffers or accumulated radii from the previous
run, and a crash on one timestep doesn't take down the loop.

This driver does NOT modify train.py or the rasterizer. It only invokes
the same script a user would run by hand:

    python train.py -s <work_root>/timestep_NNNNN -m <out_root>/timestep_NNNNN \\
        --iterations N  --lambda_semantic L  --K_target K --sam_dir sam3 ...

Single-timestep verification:

    python -m tracking.driver.run_per_timestep \\
        --work-root work/scene --out-root out/scene \\
        --timesteps 0 --iterations 3000 --render

The ``--render`` flag invokes tracking.driver.render_timestep.py after
each reconstruction so the result can be inspected immediately.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Iterable, Sequence


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TRAIN_SCRIPT = os.path.join(REPO_ROOT, "train.py")


def _list_timestep_dirs(work_root: str) -> list[str]:
    """Return work_root/timestep_* directories in sorted timestep order."""
    if not os.path.isdir(work_root):
        raise FileNotFoundError(f"work_root not found: {work_root}")
    out = []
    for name in sorted(os.listdir(work_root)):
        if name.startswith("timestep_"):
            full = os.path.join(work_root, name)
            if os.path.isdir(full) and os.path.exists(os.path.join(full, "transforms_train.json")):
                out.append(full)
    return out


def _parse_timestep_indices(dirs: list[str]) -> list[int]:
    """Extract the integer timestep from each timestep_NNNNN dir name."""
    out = []
    for d in dirs:
        name = os.path.basename(d)
        try:
            out.append(int(name.split("_", 1)[1]))
        except (IndexError, ValueError):
            raise ValueError(f"can't parse timestep index from {name}")
    return out


def _filter_timesteps(all_dirs: list[str], explicit: list[int] | None) -> list[str]:
    if explicit is None:
        return all_dirs
    idx = set(explicit)
    have = _parse_timestep_indices(all_dirs)
    kept = [d for d, i in zip(all_dirs, have) if i in idx]
    missing = idx - set(have)
    if missing:
        raise ValueError(f"requested timesteps not present in work_root: {sorted(missing)}")
    return kept


def _train_one(
    timestep_dir: str,
    out_dir: str,
    iterations: int,
    checkpoint_iterations: Sequence[int],
    extra_train_args: Sequence[str],
    semantic_args: Sequence[str],
) -> None:
    """Run train.py on one timestep. Raises on non-zero exit.

    checkpoint_iterations is forwarded as the *last* positional
    arguments of the cmd line. With argparse + ``nargs='+'`` the
    forwarded values can be swallowed by a preceding flag if anything
    else after them looks like a non-flag token; placing them last
    keeps the parser unambiguous.
    """
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "-s", timestep_dir,
        "-m", out_dir,
        "--iterations", str(iterations),
    ]
    cmd += list(semantic_args) + list(extra_train_args)
    # Append checkpoint_iterations *last* so the nargs='+' consumer
    # doesn't lose the trailing values to a later flag.
    if checkpoint_iterations:
        cmd += ["--checkpoint_iterations"] + [str(it) for it in checkpoint_iterations]
    print(f"[run_per_timestep] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"train.py failed (exit {proc.returncode}) for {timestep_dir}")


def _expected_semantic_artifacts(timestep_dir: str, sam_dir: str) -> list[str]:
    """Per-camera SAM3 + SigLIP artifacts that train.py's reader looks for.

    For every cam_XX.png under <timestep_dir>/images/ the Blender reader
    expects matching <timestep_dir>/<sam_dir>/cam_XX_regions.png and
    cam_XX_embeds.npy. Missing artifacts cause the loss to silently
    degrade to 0 for that view, producing trained models whose
    semantic features are random noise -- discoverable only when
    downstream text query returns garbage. The preflight check below
    rejects this before any train.py invocation.
    """
    images_dir = os.path.join(timestep_dir, "images")
    sam_full = os.path.join(timestep_dir, sam_dir)
    expected: list[str] = []
    if not os.path.isdir(images_dir):
        return expected
    for name in sorted(os.listdir(images_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        expected.append(os.path.join(sam_full, f"{stem}_regions.png"))
        expected.append(os.path.join(sam_full, f"{stem}_embeds.npy"))
    return expected


def _preflight_semantic_artifacts(
    selected_dirs: Sequence[str], sam_dir: str
) -> None:
    """Fail fast if any selected timestep is missing its SAM3+SigLIP
    artifacts. Surfaces a Stage A completeness problem at the start
    of Stage B rather than after hours of (silently-degraded) training.
    """
    summary: list[tuple[str, int, int]] = []   # (timestep_dir, n_missing, n_expected)
    examples: list[str] = []
    for d in selected_dirs:
        expected = _expected_semantic_artifacts(d, sam_dir)
        if not expected:
            continue
        missing = [p for p in expected if not os.path.exists(p)]
        if missing:
            summary.append((d, len(missing), len(expected)))
            # Stash up to 3 example paths per dir, so the user sees the
            # actual missing filenames not just a count.
            for p in missing[:3]:
                examples.append(p)
    if not summary:
        return
    lines = [
        f"Stage A semantic artifacts missing for {len(summary)} timestep(s); "
        "did Stage A's SAM3 / SigLIP2 preprocessing complete successfully?",
    ]
    for d, nm, ne in summary[:5]:
        lines.append(f"  {os.path.basename(d)}: {nm}/{ne} per-camera files missing")
    if len(summary) > 5:
        lines.append(f"  ... and {len(summary) - 5} more timestep(s)")
    lines.append("example missing paths:")
    for p in examples[:6]:
        lines.append(f"  {p}")
    lines.append(
        "to skip semantic supervision intentionally, re-run with --lambda-semantic 0; "
        "to keep it, fix Stage A and re-run."
    )
    raise SystemExit("\n".join(lines))


def _render_one(out_dir: str, iteration: int) -> None:
    """Invoke the existing render.py to spit out a quick visual."""
    render_script = os.path.join(REPO_ROOT, "render.py")
    cmd = [sys.executable, render_script, "-m", out_dir, "--iteration", str(iteration),
           "--skip_test", "--skip_mesh"]
    print(f"[run_per_timestep] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        print(f"[run_per_timestep] WARN: render.py exit {proc.returncode} for {out_dir}; "
              "training output is still on disk", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage B: invoke train.py on each selected per-timestep scene dir"
    )
    p.add_argument("--work-root", required=True,
                   help="Stage A output (contains timestep_* subdirectories)")
    p.add_argument("--out-root", required=True,
                   help="output root; per-timestep training output goes under <out_root>/timestep_NNNNN")
    p.add_argument("--timesteps", nargs="+", type=int, default=None,
                   help="explicit timestep indices to reconstruct (default: every timestep in work_root)")

    p.add_argument("--iterations", type=int, default=30_000,
                   help="full reconstruction iterations per timestep (default 30000; "
                        "use a lower value like 3000 for smoke tests)")
    p.add_argument("--checkpoint-iterations", nargs="+", type=int, default=None,
                   help="iterations at which to save chkpnt*.pth (forwarded to train.py "
                        "as --checkpoint_iterations). Defaults to [--iterations], so the "
                        "final checkpoint is always saved -- matching the static pipeline's "
                        "artifact set so text_query.py / cross_view_label_conflict.py / etc. "
                        "work without dynamics-specific patches.")

    # Semantic-loss knobs (must mirror values used by Stage A's SAM3/SigLIP run).
    p.add_argument("--lambda-semantic", type=float, default=0.5,
                   help="weight of the semantic loss (default 0.5; set 0 to disable)")
    p.add_argument("--K-target", type=int, default=768,
                   help="SigLIP2 embedding dim; must match Stage A's --variant choice")
    p.add_argument("--sam-dir", default="sam3",
                   help="subdir under each timestep_*/ where SAM3 outputs live")
    p.add_argument("--skip-preflight", action="store_true",
                   help="skip the SAM3 + SigLIP2 artifact preflight check (default: check, "
                        "fail fast if Stage A is incomplete). Use only if you know what you're doing.")

    # Verification helper.
    p.add_argument("--render", action="store_true",
                   help="after each reconstruction, invoke render.py for inspection")

    # Pass-through for any extra train.py args.
    p.add_argument("--train-arg", action="append", default=[],
                   help="extra argument forwarded to train.py (repeatable). For multi-value "
                        "flags like --checkpoint_iterations, prefer the dedicated driver flag "
                        "to avoid argparse quoting bugs.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not os.path.exists(TRAIN_SCRIPT):
        raise FileNotFoundError(f"train.py not at expected location: {TRAIN_SCRIPT}")

    all_dirs = _list_timestep_dirs(args.work_root)
    if not all_dirs:
        raise SystemExit(f"no timestep_* dirs under {args.work_root}; run Stage A first")
    selected = _filter_timesteps(all_dirs, args.timesteps)
    indices = _parse_timestep_indices(selected)

    # Resolve checkpoint_iterations. Default to [--iterations] so Stage B's
    # output directory contains the same artifacts as the static pipeline
    # (point_cloud + cfg_args + chkpnt*.pth). text_query.py and similar
    # downstream tools depend on chkpnt*.pth; failing to produce it by
    # default has bitten this pipeline before.
    if args.checkpoint_iterations is None:
        checkpoint_iterations = [args.iterations]
    else:
        checkpoint_iterations = list(args.checkpoint_iterations)

    print(f"[run_per_timestep] reconstructing {len(selected)} timesteps: "
          f"{indices[:8]}{'...' if len(indices) > 8 else ''}")
    print(f"[run_per_timestep] iterations per timestep: {args.iterations}")
    print(f"[run_per_timestep] checkpoint_iterations: {checkpoint_iterations}")

    # Preflight: catch missing SAM3/SigLIP2 artifacts before any train.py
    # invocation. With semantic loss enabled but no artifacts on disk, the
    # loss silently degrades to 0 and the trained model carries random
    # semantic features -- a failure only discoverable downstream.
    if args.lambda_semantic > 0 and not args.skip_preflight:
        _preflight_semantic_artifacts(selected, args.sam_dir)

    semantic_args: list[str] = []
    if args.lambda_semantic > 0:
        semantic_args += [
            "--lambda_semantic", str(args.lambda_semantic),
            "--K_target", str(args.K_target),
            "--sam_dir", args.sam_dir,
        ]
    else:
        semantic_args += ["--lambda_semantic", "0.0"]

    for t_idx, src in zip(indices, selected):
        out_dir = os.path.join(args.out_root, f"timestep_{t_idx:05d}")
        os.makedirs(out_dir, exist_ok=True)
        _train_one(
            timestep_dir=src,
            out_dir=out_dir,
            iterations=args.iterations,
            checkpoint_iterations=checkpoint_iterations,
            extra_train_args=args.train_arg,
            semantic_args=semantic_args,
        )
        if args.render:
            _render_one(out_dir, args.iterations)

    print(f"[run_per_timestep] done. outputs under {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
