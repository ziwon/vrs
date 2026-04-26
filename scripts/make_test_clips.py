"""Generate 4 synthetic mp4 clips — one per watch-policy event.

HONEST CAVEAT: these clips use programmatic OpenCV drawing (flickering noise
patches for fire, expanding Gaussian blobs for smoke, stick figures for
falldown, silhouette + dark rectangle for weapon). They are for PIPELINE
PLUMBING tests only — they verify that your GPU can decode FHD video, run
YOLOE, run Cosmos-Reason2-2B, and write annotated outputs without errors.

YOLOE's zero-shot detector is trained on real-world imagery and will usually
NOT fire on these synthetic clips. That is expected. For ACCURACY testing,
use real datasets:

  * fire / smoke   — FireNet, D-Fire, NIST Fire Eval
  * falldown       — Le2i Falldown, UP-Fall Detection, Charfi HQ-Fall
  * weapon         — curate from a law-enforcement training set; most public
                     weapon datasets are violence-adjacent and need care

Usage:
    uv run scripts/make_test_clips.py --out runs/test_clips
    uv run scripts/make_test_clips.py --out runs/test_clips \\
        --size 1920x1080 --fps 30 --seconds 15 --which fire,smoke
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import cv2
import numpy as np

FHD = (1920, 1080)
FPS = 30
SECONDS = 15
BG_COLOR = (40, 35, 30)  # dark indoor BGR


def _bg(shape, tint=BG_COLOR):
    """Dark indoor background with subtle vignette — looks like a CCTV still."""
    h, w = shape
    img = np.full((h, w, 3), tint, dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    d = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / np.sqrt((w / 2) ** 2 + (h / 2) ** 2)
    v = np.clip(1.0 - 0.35 * d, 0.0, 1.0).astype(np.float32)
    return (img.astype(np.float32) * v[..., None]).astype(np.uint8)


def _writer(path: Path, fps: int, w: int, h: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not wr.isOpened():
        raise RuntimeError(f"could not open VideoWriter at {path}")
    return wr


# ──────────────────────────────────────────────────────────────────────
# fire — animated flickering orange/yellow noise patch
# ──────────────────────────────────────────────────────────────────────


def gen_fire(path: Path, size=FHD, fps=FPS, seconds=SECONDS) -> None:
    w, h = size
    wr = _writer(path, fps, w, h)
    fire_w, fire_h = 280, 420
    cx, cy = w // 2, int(h * 0.75)
    rng = np.random.default_rng(0)
    grad = np.linspace(1.0, 0.0, fire_h)[:, None].astype(np.float32)
    try:
        for i in range(fps * seconds):
            frame = _bg((h, w))
            heat = rng.random((fire_h, fire_w), dtype=np.float32) * grad
            r = np.clip(heat * 255.0 + 80, 0, 255).astype(np.uint8)
            g = np.clip(heat * 180.0, 0, 255).astype(np.uint8)
            b = np.clip(heat * 40.0, 0, 255).astype(np.uint8)
            fire = np.stack([b, g, r], axis=-1)
            flicker = 0.75 + 0.2 * math.sin(i * 0.7) + 0.1 * math.sin(i * 1.3)
            fire = np.clip(fire.astype(np.float32) * flicker, 0, 255).astype(np.uint8)

            y1 = cy - fire_h
            x1 = cx - fire_w // 2
            alpha = (heat > 0.2).astype(np.float32)[..., None]

            # Clip the synthetic patch against the output frame so tiny test
            # resolutions still produce a valid video instead of slicing past
            # the image bounds.
            clip_x1 = max(0, x1)
            clip_y1 = max(0, y1)
            clip_x2 = min(w, x1 + fire_w)
            clip_y2 = min(h, y1 + fire_h)
            if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
                wr.write(frame)
                continue

            src_x1 = clip_x1 - x1
            src_y1 = clip_y1 - y1
            src_x2 = src_x1 + (clip_x2 - clip_x1)
            src_y2 = src_y1 + (clip_y2 - clip_y1)

            roi = frame[clip_y1:clip_y2, clip_x1:clip_x2]
            fire_crop = fire[src_y1:src_y2, src_x1:src_x2]
            alpha_crop = alpha[src_y1:src_y2, src_x1:src_x2]
            frame[clip_y1:clip_y2, clip_x1:clip_x2] = (
                alpha_crop * fire_crop + (1 - alpha_crop) * roi
            ).astype(np.uint8)
            wr.write(frame)
    finally:
        wr.release()


# ──────────────────────────────────────────────────────────────────────
# smoke — expanding Gaussian blob with noise
# ──────────────────────────────────────────────────────────────────────


def gen_smoke(path: Path, size=FHD, fps=FPS, seconds=SECONDS) -> None:
    w, h = size
    wr = _writer(path, fps, w, h)
    cx, cy = w // 2, int(h * 0.55)
    rng = np.random.default_rng(1)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    smoke_color = np.array([130, 130, 130], dtype=np.float32)  # gray BGR
    try:
        n = fps * seconds
        for i in range(n):
            frame = _bg((h, w))
            t = i / max(n - 1, 1)
            radius = 150 + 600 * t
            alpha = np.exp(-d2 / (2 * radius * radius)) * 0.7
            noise = (rng.random((h, w), dtype=np.float32) - 0.5) * 0.2
            alpha = np.clip(alpha + noise * alpha, 0.0, 1.0).astype(np.float32)
            blended = alpha[..., None] * smoke_color + (1 - alpha[..., None]) * frame.astype(
                np.float32
            )
            wr.write(blended.astype(np.uint8))
    finally:
        wr.release()


# ──────────────────────────────────────────────────────────────────────
# falldown — stick figure rotating from upright to horizontal
# ──────────────────────────────────────────────────────────────────────


def gen_falldown(path: Path, size=FHD, fps=FPS, seconds=SECONDS) -> None:
    w, h = size
    wr = _writer(path, fps, w, h)
    try:
        n = fps * seconds
        for i in range(n):
            frame = _bg((h, w))
            t = i / max(n - 1, 1)
            if t < 0.35:
                angle = 0.0  # standing still
            elif t < 0.55:
                angle = (t - 0.35) / 0.20 * (math.pi / 2)  # falling
            else:
                angle = math.pi / 2  # on the floor

            base = (w // 2, int(h * 0.82))
            person_h = 320
            dx = int(person_h * math.sin(angle))
            dy = int(person_h * math.cos(angle))
            head = (base[0] + dx, base[1] - dy)

            # torso
            cv2.line(frame, base, head, (210, 195, 180), 40)
            # head
            cv2.circle(frame, head, 38, (205, 185, 160), -1)
            # arm (for silhouette variety)
            arm_angle = angle + 0.6
            arm_end = (
                head[0] + int(140 * math.sin(arm_angle)),
                head[1] - int(140 * math.cos(arm_angle)) + 30,
            )
            cv2.line(frame, head, arm_end, (205, 185, 160), 18)
            # ground reference
            cv2.line(frame, (0, int(h * 0.82)), (w, int(h * 0.82)), (80, 70, 60), 2)
            wr.write(frame)
    finally:
        wr.release()


# ──────────────────────────────────────────────────────────────────────
# weapon — silhouette raising a dark rectangular object
# ──────────────────────────────────────────────────────────────────────


def gen_weapon(path: Path, size=FHD, fps=FPS, seconds=SECONDS) -> None:
    w, h = size
    wr = _writer(path, fps, w, h)
    try:
        n = fps * seconds
        for i in range(n):
            frame = _bg((h, w))
            t = i / max(n - 1, 1)
            raise_t = min(max((t - 0.3) / 0.4, 0.0), 1.0)  # arm raises between 30-70% of clip

            body_top = (w // 2, int(h * 0.45))
            feet = (w // 2, int(h * 0.9))
            cv2.line(frame, feet, body_top, (190, 175, 155), 55)
            cv2.circle(frame, (body_top[0], body_top[1] - 55), 42, (185, 165, 140), -1)

            # arm angle: down → forward horizontal
            arm_angle = math.pi * (0.15 + 0.35 * raise_t)  # sweep from ~27° to ~90°
            arm_len = 180
            arm_end = (
                body_top[0] + int(arm_len * math.sin(arm_angle)),
                body_top[1] - int(arm_len * math.cos(arm_angle)),
            )
            cv2.line(frame, body_top, arm_end, (190, 175, 155), 26)

            # dark rectangular "object" in hand
            obj_w, obj_h = 130, 30
            ax, ay = arm_end
            cv2.rectangle(
                frame,
                (ax - 10, ay - obj_h // 2),
                (ax - 10 + obj_w, ay + obj_h // 2),
                (25, 22, 22),
                -1,
            )
            wr.write(frame)
    finally:
        wr.release()


CLIP_GEN = {
    "fire": gen_fire,
    "smoke": gen_smoke,
    "falldown": gen_falldown,
    "weapon": gen_weapon,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate VRS plumbing-test clips")
    ap.add_argument("--out", default="runs/test_clips")
    ap.add_argument("--size", default="1920x1080", help="WxH, default 1920x1080")
    ap.add_argument("--fps", type=int, default=FPS)
    ap.add_argument("--seconds", type=int, default=SECONDS)
    ap.add_argument(
        "--which",
        default="all",
        help="'all' or a comma list of: fire,smoke,falldown,weapon",
    )
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    keys = (
        list(CLIP_GEN.keys()) if args.which == "all" else [s.strip() for s in args.which.split(",")]
    )
    for name in keys:
        fn = CLIP_GEN.get(name)
        if fn is None:
            logging.warning("unknown clip name: %r", name)
            continue
        path = out / f"{name}_test.mp4"
        print(f"[gen] {name:<10} → {path}   ({w}x{h} @ {args.fps} fps, {args.seconds}s)")
        fn(path, size=(w, h), fps=args.fps, seconds=args.seconds)

    print(
        "\nDone. Next step — smoke-test the single-stream pipeline on your RTX 5080:\n"
        f"  uv run scripts/run_mp4.py \\\n"
        f"      --video  {out}/fire_test.mp4 \\\n"
        f"      --config configs/default.yaml \\\n"
        f"      --policy configs/policies/safety.yaml \\\n"
        f"      --out    runs/demo_fire\n"
        "\nOr run the full benchmark harness:\n"
        f"  uv run scripts/bench.py --clips {out} --out runs/bench"
    )


if __name__ == "__main__":
    main()
