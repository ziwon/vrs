"""Compare YOLOE raw tensor dumps produced by PyTorch or DeepStream."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    args = build_arg_parser().parse_args()
    left = load_dump(Path(args.left))
    right = load_dump(Path(args.right))
    left_array = normalize_yoloe_output(left["array"])
    right_array = normalize_yoloe_output(right["array"])
    class_count = infer_class_count(left["metadata"], right["metadata"])
    anchor_filter = build_anchor_filter(
        left_array,
        right_array,
        score_start=args.score_start,
        score_end=args.score_end,
        min_score=args.min_score,
        class_count=class_count,
    )
    if anchor_filter is not None:
        left_array = left_array[:, anchor_filter]
        right_array = right_array[:, anchor_filter]
    if args.channels:
        left_array = left_array[: args.channels, :]
        right_array = right_array[: args.channels, :]
    report = compare_arrays(left_array, right_array)
    report.update(
        {
            "schema_version": "vrs.eval.yoloe_raw_tensor_compare.v1",
            "left": metadata_summary(left["metadata"]),
            "right": metadata_summary(right["metadata"]),
            "left_shape": list(left_array.shape),
            "right_shape": list(right_array.shape),
            "channels_compared": args.channels,
            "anchor_filter": anchor_filter_summary(args, anchor_filter, class_count=class_count),
        }
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"written {out}: max_abs={report['max_abs_delta']:.6g} "
        f"mean_abs={report['mean_abs_delta']:.6g} cosine={report['cosine_similarity']:.6g}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compare YOLOE raw .f32 tensor dumps")
    ap.add_argument("--left", required=True, help="left dump metadata JSON")
    ap.add_argument("--right", required=True, help="right dump metadata JSON")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--channels",
        type=int,
        help="compare only the first N channels after normalizing to [channels, anchors]",
    )
    ap.add_argument(
        "--min-score",
        type=float,
        help="compare only anchors whose max class score in either tensor is at least this value",
    )
    ap.add_argument(
        "--score-start",
        type=int,
        default=4,
        help="first class-score channel for --min-score filtering",
    )
    ap.add_argument(
        "--score-end",
        type=int,
        help="exclusive class-score channel end for --min-score filtering; defaults to 4 + class_count",
    )
    return ap


def load_dump(metadata_path: Path) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    binary = Path(metadata["binary"])
    if not binary.is_absolute():
        binary = metadata_path.parent / binary
    dims = metadata.get("dims") or metadata.get("shape")
    if not dims:
        raise ValueError(f"dump metadata has no dims/shape: {metadata_path}")
    array = np.fromfile(binary, dtype=np.float32)
    expected = int(np.prod(dims))
    if array.size != expected:
        raise ValueError(f"{binary} has {array.size} floats, expected {expected}")
    return {"metadata": metadata, "array": array.reshape(tuple(int(v) for v in dims))}


def normalize_yoloe_output(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"expected [channels, anchors] or [anchors, channels], got {array.shape}")
    if array.shape[0] > array.shape[1]:
        array = array.T
    return np.asarray(array, dtype=np.float32)


def compare_arrays(left: np.ndarray, right: np.ndarray) -> dict:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    delta = right - left
    abs_delta = np.abs(delta)
    left_flat = left.reshape(-1).astype(np.float64)
    right_flat = right.reshape(-1).astype(np.float64)
    denom = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    cosine = float(np.dot(left_flat, right_flat) / denom) if denom else None
    return {
        "count": int(left.size),
        "max_abs_delta": float(abs_delta.max(initial=0.0)),
        "mean_abs_delta": float(abs_delta.mean()) if abs_delta.size else 0.0,
        "p99_abs_delta": float(np.quantile(abs_delta, 0.99)) if abs_delta.size else 0.0,
        "mean_signed_delta": float(delta.mean()) if delta.size else 0.0,
        "cosine_similarity": cosine,
        "per_channel": per_channel_stats(left, right),
        "top_abs_deltas": top_abs_deltas(left, right, limit=10),
    }


def build_anchor_filter(
    left: np.ndarray,
    right: np.ndarray,
    *,
    score_start: int,
    score_end: int | None,
    min_score: float | None,
    class_count: int | None,
) -> np.ndarray | None:
    if min_score is None:
        return None
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch before anchor filtering: {left.shape} != {right.shape}")
    score_end = score_end if score_end is not None else default_score_end(score_start, class_count)
    if score_start < 0 or score_end <= score_start or score_end > left.shape[0]:
        raise ValueError(
            f"invalid score channel range [{score_start}, {score_end}) for shape {left.shape}"
        )
    left_scores = left[score_start:score_end, :].max(axis=0)
    right_scores = right[score_start:score_end, :].max(axis=0)
    mask = (left_scores >= min_score) | (right_scores >= min_score)
    if not mask.any():
        raise ValueError(f"no anchors matched --min-score {min_score}")
    return mask


def infer_class_count(left_metadata: dict, right_metadata: dict) -> int | None:
    left_class_count = left_metadata.get("class_count")
    right_class_count = right_metadata.get("class_count")
    if left_class_count is None:
        return int(right_class_count) if right_class_count is not None else None
    if right_class_count is None:
        return int(left_class_count)
    return min(int(left_class_count), int(right_class_count))


def default_score_end(score_start: int, class_count: int | None) -> int:
    if class_count is None:
        return score_start + 1
    return score_start + class_count


def anchor_filter_summary(
    args: argparse.Namespace,
    anchor_filter: np.ndarray | None,
    *,
    class_count: int | None,
) -> dict | None:
    if anchor_filter is None:
        return None
    score_end = (
        args.score_end
        if args.score_end is not None
        else default_score_end(args.score_start, class_count)
    )
    return {
        "min_score": args.min_score,
        "score_start": args.score_start,
        "score_end": score_end,
        "anchors_selected": int(anchor_filter.sum()),
    }


def per_channel_stats(left: np.ndarray, right: np.ndarray) -> list[dict]:
    delta = right - left
    abs_delta = np.abs(delta)
    rows = []
    for channel in range(left.shape[0]):
        channel_abs = abs_delta[channel]
        rows.append(
            {
                "channel": channel,
                "max_abs_delta": float(channel_abs.max(initial=0.0)),
                "mean_abs_delta": float(channel_abs.mean()) if channel_abs.size else 0.0,
                "p99_abs_delta": float(np.quantile(channel_abs, 0.99)) if channel_abs.size else 0.0,
                "left_mean": float(left[channel].mean()) if left[channel].size else 0.0,
                "right_mean": float(right[channel].mean()) if right[channel].size else 0.0,
                "mean_signed_delta": float(delta[channel].mean()) if delta[channel].size else 0.0,
            }
        )
    return rows


def top_abs_deltas(left: np.ndarray, right: np.ndarray, *, limit: int) -> list[dict]:
    abs_delta = np.abs(right - left)
    flat = abs_delta.reshape(-1)
    if flat.size == 0:
        return []
    limit = min(limit, flat.size)
    indexes = np.argpartition(flat, -limit)[-limit:]
    indexes = indexes[np.argsort(flat[indexes])[::-1]]
    rows = []
    for flat_index in indexes:
        channel, anchor = np.unravel_index(flat_index, abs_delta.shape)
        left_value = float(left[channel, anchor])
        right_value = float(right[channel, anchor])
        rows.append(
            {
                "channel": int(channel),
                "anchor": int(anchor),
                "left": left_value,
                "right": right_value,
                "signed_delta": right_value - left_value,
                "abs_delta": float(abs_delta[channel, anchor]),
            }
        )
    return rows


def metadata_summary(metadata: dict) -> dict:
    return {
        "runtime": metadata.get("runtime"),
        "dims": metadata.get("dims") or metadata.get("shape"),
        "binary": metadata.get("binary"),
        "class_count": metadata.get("class_count"),
    }


if __name__ == "__main__":
    main()
