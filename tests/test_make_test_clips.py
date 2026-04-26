"""Exercise the synthetic clip generator at a small resolution so CI can run it.

Only validates that each generator writes a non-trivial mp4 with the requested
duration — not visual fidelity (that's subjective and out of scope).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

# Import the script as a module without touching sys.path globally
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_test_clips.py"
_spec = importlib.util.spec_from_file_location("make_test_clips", _SCRIPT)
mkt = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(mkt)  # type: ignore[union-attr]


@pytest.mark.parametrize("name", ["fire", "smoke", "falldown", "weapon"])
def test_clip_generator_produces_mp4(tmp_path, name):
    gen = mkt.CLIP_GEN[name]
    out = tmp_path / f"{name}_test.mp4"
    # tiny clip: 160x120, 10 fps, 1 s → 10 frames
    gen(out, size=(160, 120), fps=10, seconds=1)

    assert out.exists() and out.stat().st_size > 1024, "mp4 should exist and be non-trivial"

    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened(), f"could not re-open {out}"
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    # mp4 writers occasionally drop the last frame; allow 8-10.
    assert 8 <= n <= 10, f"expected ~10 frames in {out}, got {n}"
