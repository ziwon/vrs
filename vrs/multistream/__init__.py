"""Multi-stream cascade.

Run N RTSP / mp4 sources concurrently on a single GPU, sharing one YOLOE
detector and one Cosmos-Reason2-2B verifier across all streams.

Topology:
    DecoderThread x N  ->  FrameQueue  ->  DetectorWorker (batched YOLOE)
                                                  |
                                                  v  CandidateAlert + (sid, frame)
                                          CandidateQueue (bounded, drop-policy)
                                                  |
                                                  v
                                          VerifierWorker (VLM)
                                                  |
                                                  v
                                          SinkWorker x N  ->  jsonl + mp4

Lightweight re-exports only — ``MultiStreamPipeline`` / ``build_multistream_pipeline``
pull heavy deps (cv2, ultralytics, transformers) on first access via ``__getattr__``,
so unit tests can touch ``queues`` / ``workers`` without those deps installed.
"""

from .queues import BoundedQueue, DropPolicy

__all__ = [
    "BoundedQueue",
    "DropPolicy",
    "MultiStreamPipeline",
    "build_multistream_pipeline",
]


def __getattr__(name):  # PEP 562 — lazy pipeline import
    if name in ("MultiStreamPipeline", "build_multistream_pipeline"):
        from .pipeline import MultiStreamPipeline, build_multistream_pipeline

        return {
            "MultiStreamPipeline": MultiStreamPipeline,
            "build_multistream_pipeline": build_multistream_pipeline,
        }[name]
    raise AttributeError(f"module 'vrs.multistream' has no attribute {name!r}")
