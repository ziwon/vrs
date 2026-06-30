"""Dump same-frame YOLOE PyTorch raw output as .f32 + metadata JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from vrs.policy import load_watch_policy


def main() -> None:
    args = build_arg_parser().parse_args()
    frame = read_frame(
        args.source,
        frame_index=args.frame_index,
        raw_rgb_width=args.raw_rgb_width,
        raw_rgb_height=args.raw_rgb_height,
    )
    image, letterbox = letterbox_bgr(frame, size=args.imgsz, pad_value=args.pad_value)

    import torch
    from ultralytics import YOLOE

    policy = load_watch_policy(args.policy)
    prompts = policy.yoloe_vocabulary()
    model = YOLOE(args.weights)
    model.set_classes(prompts, model.get_text_pe(prompts))
    model.model.eval()
    if args.half:
        model.model.half()

    tensor = torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0)
    tensor /= 255.0
    if args.input_prefix:
        input_array = tensor.numpy().astype(np.float32, copy=False)
        write_dump(
            prefix=Path(args.input_prefix),
            array=input_array,
            metadata={
                "schema_version": "vrs.deepstream.yoloe_raw_tensor.v1",
                "runtime": "pytorch-preprocess",
                "source": args.source,
                "frame_index": args.frame_index,
                "binding": "images",
                "imgsz": args.imgsz,
                "pad_value": args.pad_value,
                "dtype": "float32",
                "dims": list(input_array.shape),
                "letterbox": letterbox,
            },
        )
    if args.half:
        tensor = tensor.half()
    if args.device:
        tensor = tensor.to(args.device)
        model.model.to(args.device)
    with torch.no_grad():
        output = select_detection_output(model.model(tensor))
    array = output.detach().float().cpu().numpy().astype(np.float32, copy=False)

    write_dump(
        prefix=Path(args.out_prefix),
        array=array,
        metadata={
            "schema_version": "vrs.deepstream.yoloe_raw_tensor.v1",
            "runtime": "pytorch-ultralytics",
            "source": args.source,
            "frame_index": args.frame_index,
            "weights": args.weights,
            "policy": args.policy,
            "imgsz": args.imgsz,
            "pad_value": args.pad_value,
            "dtype": "float32",
            "dims": list(array.shape),
            "class_count": len(prompts),
            "prompts": prompts,
            "letterbox": letterbox,
        },
    )
    print(f"written {args.out_prefix}.json and {args.out_prefix}.f32")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Dump YOLOE PyTorch raw detection tensor")
    ap.add_argument("--source", required=True)
    ap.add_argument("--frame-index", type=int, default=0)
    ap.add_argument("--raw-rgb-width", type=int, help="read --source as raw HWC RGB with width")
    ap.add_argument("--raw-rgb-height", type=int, help="read --source as raw HWC RGB with height")
    ap.add_argument("--weights", default="yoloe-11s-seg.pt")
    ap.add_argument("--policy", default="configs/policies/safety.yaml")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument(
        "--input-prefix",
        help="optional prefix for dumping the preprocessed model input tensor as float32",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--pad-value", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    return ap


def read_frame(
    source: str,
    *,
    frame_index: int,
    raw_rgb_width: int | None = None,
    raw_rgb_height: int | None = None,
) -> np.ndarray:
    if raw_rgb_width is not None or raw_rgb_height is not None:
        if raw_rgb_width is None or raw_rgb_height is None:
            raise ValueError("--raw-rgb-width and --raw-rgb-height must be provided together")
        return read_raw_rgb_frame(source, width=raw_rgb_width, height=raw_rgb_height)

    path = Path(source)
    if path.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png", ".webp"}:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {source}")
        return image

    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {source}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, image = cap.read()
    cap.release()
    if not ok or image is None:
        raise RuntimeError(f"failed to read frame {frame_index} from {source}")
    return image


def read_raw_rgb_frame(source: str, *, width: int, height: int) -> np.ndarray:
    data = np.fromfile(source, dtype=np.uint8)
    expected = width * height * 3
    if data.size != expected:
        raise ValueError(f"{source} has {data.size} bytes, expected {expected}")
    rgb = data.reshape(height, width, 3)
    return rgb[:, :, ::-1].copy()


def letterbox_bgr(image: np.ndarray, *, size: int, pad_value: int) -> tuple[np.ndarray, dict]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, dtype=image.dtype)
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return canvas, {
        "source_width": width,
        "source_height": height,
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "resized_width": resized_width,
        "resized_height": resized_height,
    }


def select_detection_output(output):
    if isinstance(output, (list, tuple)):
        first = output[0]
        if isinstance(first, (list, tuple)):
            return first[0]
        return first
    return output


def write_dump(*, prefix: Path, array: np.ndarray, metadata: dict) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    bin_path = prefix.with_suffix(".f32")
    json_path = prefix.with_suffix(".json")
    array.tofile(bin_path)
    payload = dict(metadata)
    payload["binary"] = bin_path.name
    payload["volume"] = int(array.size)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
