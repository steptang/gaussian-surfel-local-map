"""CLI: thin wrapper around the existing render.py for a single timestep.

This is the verification path Stephanie planned for: after reconstructing
a single timestep, render it from every training viewpoint and inspect
the result for mirrored / garbage geometry (the LLFF-conversion silent
failure) and for sparse-view (~10 views) artifacts.

Usage:

    python -m tracking.driver.render_timestep \\
        --model-path out/scene/timestep_00000 --iteration 3000

This forwards to ``render.py -m <model-path> --iteration <iter> --skip_test
--skip_mesh`` which is enough for an eyeball check; bypasses the mesh
extraction (slow, irrelevant for the conversion sanity check). Note:
this exists as a separate script mainly so single-timestep runs have a
single obvious entry point in tracking/driver/.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RENDER_SCRIPT = os.path.join(REPO_ROOT, "render.py")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Render one trained per-timestep model from all training views."
    )
    p.add_argument("--model-path", "-m", required=True,
                   help="path to <out_root>/timestep_NNNNN/")
    p.add_argument("--iteration", type=int, default=-1,
                   help="trained-model iteration to render; -1 = latest")
    p.add_argument("--skip-test", action="store_true", default=True,
                   help="skip rendering the test split (default true; DMV writer leaves test empty)")
    p.add_argument("--skip-mesh", action="store_true", default=True,
                   help="skip mesh extraction (slow, irrelevant for the LLFF sanity check)")
    p.add_argument("--mesh", action="store_true",
                   help="opt back into mesh extraction (overrides --skip-mesh default)")
    p.add_argument("--render-arg", action="append", default=[],
                   help="extra argument forwarded to render.py (repeatable)")
    args = p.parse_args(argv)

    if not os.path.exists(RENDER_SCRIPT):
        raise FileNotFoundError(f"render.py not at expected location: {RENDER_SCRIPT}")

    cmd = [sys.executable, RENDER_SCRIPT, "-m", args.model_path,
           "--iteration", str(args.iteration)]
    if args.skip_test:
        cmd += ["--skip_test"]
    if args.skip_mesh and not args.mesh:
        cmd += ["--skip_mesh"]
    cmd += list(args.render_arg)
    print(f"[render_timestep] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
