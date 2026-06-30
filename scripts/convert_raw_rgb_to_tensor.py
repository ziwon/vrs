"""Convert a raw HWC RGB uint8 frame into a CHW float32 TensorRT input dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    args = build_arg_parser().parse_args()
    tensor = raw_rgb_to_chw_float32(Path(args.input), width=args.width, height=args.height)
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    bin_path = prefix.with_suffix(".f32")
    json_path = prefix.with_suffix(".json")
    tensor.tofile(bin_path)
    metadata = {
        "schema_version": "vrs.deepstream.yoloe_raw_tensor.v1",
        "runtime": args.runtime,
        "source": args.input,
        "binding": args.binding,
        "dtype": "float32",
        "binary": bin_path.name,
        "dims": list(tensor.shape),
        "volume": int(tensor.size),
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"written {json_path} and {bin_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Convert raw RGB frame to CHW float32 tensor")
    ap.add_argument("--input", required=True, help="raw HWC RGB uint8 file")
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--binding", default="images")
    ap.add_argument("--runtime", default="raw-rgb-preprocess")
    return ap


def raw_rgb_to_chw_float32(path: Path, *, width: int, height: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    expected = width * height * 3
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} bytes, expected {expected}")
    image = data.reshape(height, width, 3)
    return (image.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


if __name__ == "__main__":
    main()
