"""Tests for the Stage B driver's preflight and subprocess error-reporting.

Both targets are critical because they fail silently in obvious ways:

* preflight_semantic_artifacts: without this, training proceeds with
  missing SAM3/SigLIP files and the loss silently degrades to 0 (no
  semantic supervision). The trained model carries random semantic
  features, and the bug is only discovered when downstream text
  query returns garbage.

* preprocess_semantic._run stderr tail: without this, the wrapper's
  RuntimeError says only ``SAM3 ... failed (exit 1); cmd: ...`` and
  the actual cause (e.g., a missing tokenizer asset deep in SAM3's
  traceback) is visible only by scrolling through hundreds of lines
  of progress output.

Run from repo root with: pytest tests/test_driver_preflight.py -v
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.driver.run_per_timestep import (
    _expected_semantic_artifacts,
    _preflight_semantic_artifacts,
)
from tracking.data.preprocess_semantic import _run


def _populate_timestep(root: Path, t: int, cam_names: list[str],
                       sam_dir: str = "sam3", with_artifacts: bool = True) -> Path:
    """Build a fake Stage A timestep dir with empty image PNGs and (optionally)
    the matching SAM3/SigLIP outputs.
    """
    d = root / f"timestep_{t:05d}"
    (d / "images").mkdir(parents=True)
    (d / sam_dir).mkdir(parents=True)
    for cam in cam_names:
        (d / "images" / f"{cam}.png").write_bytes(b"")
        if with_artifacts:
            (d / sam_dir / f"{cam}_regions.png").write_bytes(b"")
            (d / sam_dir / f"{cam}_embeds.npy").write_bytes(b"")
    return d


def test_expected_artifacts_lists_per_camera_pair(tmp_path):
    d = _populate_timestep(tmp_path, 0, ["cam_00", "cam_01", "cam_02"],
                            with_artifacts=False)
    out = _expected_semantic_artifacts(str(d), sam_dir="sam3")
    # 2 expected files per camera (regions.png + embeds.npy).
    assert len(out) == 6
    names = [os.path.basename(p) for p in out]
    assert "cam_00_regions.png" in names
    assert "cam_00_embeds.npy" in names
    assert "cam_02_embeds.npy" in names


def test_expected_artifacts_handles_no_images_dir(tmp_path):
    d = tmp_path / "timestep_00000"
    d.mkdir()
    assert _expected_semantic_artifacts(str(d), sam_dir="sam3") == []


def test_preflight_passes_when_all_artifacts_present(tmp_path):
    dirs = [
        str(_populate_timestep(tmp_path, t, ["cam_00", "cam_01"],
                                with_artifacts=True))
        for t in range(3)
    ]
    # Must not raise.
    _preflight_semantic_artifacts(dirs, sam_dir="sam3")


def test_preflight_raises_with_summary_when_artifacts_missing(tmp_path):
    dirs = []
    dirs.append(str(_populate_timestep(tmp_path, 0, ["cam_00", "cam_01"],
                                         with_artifacts=True)))
    # Two broken timesteps -- one fully empty, one half-populated.
    broken_1 = _populate_timestep(tmp_path, 1, ["cam_00", "cam_01"],
                                    with_artifacts=False)
    dirs.append(str(broken_1))
    half = _populate_timestep(tmp_path, 2, ["cam_00", "cam_01"],
                                with_artifacts=False)
    (half / "sam3" / "cam_00_regions.png").write_bytes(b"")
    (half / "sam3" / "cam_00_embeds.npy").write_bytes(b"")
    dirs.append(str(half))

    with pytest.raises(SystemExit) as excinfo:
        _preflight_semantic_artifacts(dirs, sam_dir="sam3")
    msg = str(excinfo.value)
    # Summary must mention BOTH bad timesteps + at least one concrete
    # missing path so the user can find them.
    assert "2 timestep(s)" in msg
    assert "timestep_00001" in msg
    assert "timestep_00002" in msg
    # And the actionable hint must be present.
    assert "--lambda-semantic 0" in msg


def test_run_includes_stderr_tail_on_failure():
    """_run must surface the last lines of the subprocess' stderr in
    the RuntimeError so the failure cause is visible without scrolling.
    """
    # Tiny Python script that writes a recognisable traceback to stderr
    # and exits non-zero. Run via the same interpreter for portability.
    script = textwrap.dedent("""
        import sys
        print("normal progress line A", file=sys.stderr)
        print("normal progress line B", file=sys.stderr)
        print("Traceback (most recent call last):", file=sys.stderr)
        print('  File "fake.py", line 1, in <module>', file=sys.stderr)
        print('    raise FileNotFoundError("/path/to/missing.bin")', file=sys.stderr)
        print("FileNotFoundError: /path/to/missing.bin", file=sys.stderr)
        sys.exit(1)
    """)
    cmd = [sys.executable, "-c", script]
    with pytest.raises(RuntimeError) as excinfo:
        _run(cmd, description="synthetic-failure")
    msg = str(excinfo.value)
    # Tail must include the actual error, not just the exit code.
    assert "FileNotFoundError: /path/to/missing.bin" in msg, (
        "stderr tail not included in RuntimeError; got:\n" + msg
    )
    assert "exit 1" in msg


def test_run_does_not_crash_when_subprocess_succeeds():
    cmd = [sys.executable, "-c", "print('ok')"]
    _run(cmd, description="ok-cmd")     # must not raise
