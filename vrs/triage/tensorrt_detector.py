"""TensorRT-exported YOLOE detector.

**Status note.** This is a skeleton: it implements the ``Detector`` Protocol
and passes a pre-exported TensorRT ``.engine`` path through Ultralytics'
``YOLO`` wrapper, which natively dispatches to TRT when the weights file
has the ``.engine`` extension. The ``set_classes`` / text-PE path is the
same as the ultralytics backend, so the zero-cost open-vocab property
carries over. What is *not* verified in this repo's CI:

* the actual TRT engine export (either via Ultralytics ``model.export(
  format="engine")`` or via NVIDIA TAO's YOLOE workflow),
* whether the exported engine preserves ``set_classes`` re-parameterization —
  in practice the text PE must be baked in at export time, so the export
  step should happen *after* the watch policy is finalized,
* Blackwell vs Ada vs Hopper engine compatibility (TRT engines are
  GPU-architecture-specific).

Pair this with a short smoke run before a deployment flips to it — exactly
the same discipline the vLLM backend uses on the slow path.

Two practical export paths, both produce a ``.engine`` this class can
load:

1. **Ultralytics export** (simplest, works today)::

       from ultralytics import YOLOE
       m = YOLOE("yoloe-11l-seg.pt")
       m.set_classes(prompts, m.get_text_pe(prompts))
       m.export(format="engine", imgsz=640, half=True, device=0)

2. **NVIDIA TAO** (when a TAO-native YOLOE recipe / custom training is
   in play — TAO's strength is fine-tuning + quant-aware export, not
   plain inference-only export). See TAO's object-detection notebooks for
   the ``tao-deploy`` flow. The resulting ``.engine`` drops in identically.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..policy import WatchPolicy
from ..schemas import Detection, Frame
from .yoloe_detector import YOLOEConfig

logger = logging.getLogger(__name__)


class TensorRTYOLOEDetector:
    """``Detector`` backed by a pre-exported TensorRT engine.

    Accepts the same ``YOLOEConfig`` as the ultralytics backend so
    config-level tuning (``imgsz``, ``conf_floor``, ``iou``) is identical.
    ``cfg.model`` must point at an ``.engine`` file.
    """

    def __init__(self, cfg: YOLOEConfig, policy: WatchPolicy):
        engine_path = Path(cfg.model)
        if engine_path.suffix != ".engine":
            raise ValueError(
                f"tensorrt backend expects a .engine file, got {cfg.model!r}. "
                "Export the YOLOE weights first — see module docstring for the "
                "two supported paths."
            )
        if not engine_path.exists():
            raise FileNotFoundError(
                f"TRT engine not found: {engine_path}. TRT engines are GPU-"
                "architecture-specific; make sure this engine was built on "
                "hardware matching the deployment target."
            )

        # Import ultralytics *after* path validation so the precondition
        # errors above can be exercised on hosts without ultralytics
        # installed (unit tests care about the validation branches).
        from ultralytics import YOLO

        self.cfg = cfg
        self.policy = policy
        # Ultralytics' YOLO dispatches to TRT when given an .engine path —
        # same predict() surface as the ultralytics backend.
        self.model = YOLO(str(engine_path), task="detect")

        # We don't call set_classes here because the prompt → class mapping
        # was baked in at export time (see docstring). We assume the export
        # happened against the same watch policy this class is built with;
        # a mismatch will produce wrong class_name strings but won't crash.
        prompts = policy.yoloe_vocabulary()
        self._prompt_to_event = [policy.event_for_prompt_index(i) for i in range(len(prompts))]
        self._min_score = {it.name: max(it.min_score, cfg.conf_floor) for it in policy}

    # ---- detection API -------------------------------------------

    def __call__(self, frame: Frame) -> list[Detection]:
        return self.batch([frame])[0]

    def batch(self, frames: list[Frame]) -> list[list[Detection]]:
        if not frames:
            return []
        images = [f.image for f in frames]
        results = self.model.predict(
            images,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            conf=self.cfg.conf_floor,
            iou=self.cfg.iou,
            # .engine files ignore half=/dtype — precision is baked in at export.
            verbose=False,
        )

        out: list[list[Detection]] = []
        for r in results:
            if r is None or r.boxes is None or len(r.boxes) == 0:
                out.append([])
                continue
            xyxy = r.boxes.xyxy.detach().cpu().numpy()
            confs = r.boxes.conf.detach().cpu().numpy()
            cls_idx = r.boxes.cls.detach().cpu().numpy().astype(int)

            dets: list[Detection] = []
            for box, conf, ci in zip(xyxy, confs, cls_idx, strict=True):
                if ci < 0 or ci >= len(self._prompt_to_event):
                    continue
                event_name = self._prompt_to_event[ci]
                if conf < self._min_score.get(event_name, self.cfg.conf_floor):
                    continue
                dets.append(
                    Detection(
                        class_name=event_name,
                        score=float(conf),
                        xyxy=tuple(float(v) for v in box),
                        raw_label=self.policy.yoloe_vocabulary()[ci],
                    )
                )
            out.append(dets)
        while len(out) < len(frames):
            out.append([])
        return out
