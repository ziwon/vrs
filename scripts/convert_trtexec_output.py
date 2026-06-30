"""Convert trtexec --exportOutput JSON into a raw tensor dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    tensor = select_tensor(payload, args.tensor)
    dims = parse_dims(tensor["dimensions"])
    array = np.asarray(tensor["values"], dtype=np.float32)
    expected = int(np.prod(dims))
    if array.size != expected:
        raise ValueError(f"{args.tensor} has {array.size} values, expected {expected}")
    array = array.reshape(tuple(dims))

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    bin_path = prefix.with_suffix(".f32")
    json_path = prefix.with_suffix(".json")
    array.tofile(bin_path)
    metadata = {
        "schema_version": "vrs.deepstream.yoloe_raw_tensor.v1",
        "runtime": "trtexec",
        "source": args.input,
        "tensor": args.tensor,
        "dtype": "float32",
        "binary": bin_path.name,
        "dims": dims,
        "volume": int(array.size),
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"written {json_path} and {bin_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Convert trtexec exported output to .f32 dump")
    ap.add_argument("--input", required=True, help="trtexec --exportOutput JSON")
    ap.add_argument("--tensor", default="output0", help="tensor name to extract")
    ap.add_argument("--out-prefix", required=True)
    return ap


def select_tensor(payload: object, name: str) -> dict:
    if not isinstance(payload, list):
        raise ValueError("trtexec output must be a JSON array")
    for item in payload:
        if isinstance(item, dict) and item.get("name") == name:
            return item
    raise ValueError(f"tensor not found: {name}")


def parse_dims(value: str) -> list[int]:
    dims = [int(part) for part in value.split("x") if part]
    if not dims:
        raise ValueError(f"invalid dimensions: {value!r}")
    return dims


if __name__ == "__main__":
    main()
