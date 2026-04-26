"""YOLOE open-vocabulary detector — the fast path.

Wraps Ultralytics YOLOE so it accepts a list of noun-phrase prompts (from the
WatchPolicy) and returns ``Detection`` records keyed back to event names.

Why YOLOE here:
  * Real-time (~161 FPS on a T4) so the cascade has spare budget.
  * Open-vocabulary via text prompts — adding a custom event class is a YAML
    edit, never a re-train and never a prompt-bank rescore.
  * Returns bounding boxes, which the slow-path verifier consumes as spatial
    grounding (much stronger than frame-level scores).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..policy import WatchPolicy
from ..schemas import Detection, Frame


@dataclass
class YOLOEConfig:
    model: str = "yoloe-11l-seg.pt"  # Ultralytics handles the download
    device: str = "cuda"
    imgsz: int = 640
    conf_floor: float = 0.20
    iou: float = 0.50
    half: bool = True


class YOLOEDetector:
    def __init__(self, cfg: YOLOEConfig, policy: WatchPolicy):
        from ultralytics import YOLOE

        self.cfg = cfg
        self.policy = policy
        self.model = YOLOE(cfg.model)

        # Load text prompts as the open-vocab classes.
        # Ultralytics YOLOE pre-computes text embeddings here and re-parameterizes
        # them into the detection head, so per-frame inference has zero text cost.
        prompts = policy.yoloe_vocabulary()
        self.model.set_classes(prompts, self.model.get_text_pe(prompts))

        # event lookup, mirrored from the policy for O(1) prompt-idx → event-name
        self._prompt_to_event = [policy.event_for_prompt_index(i) for i in range(len(prompts))]

        # per-class score floors from the policy (at least cfg.conf_floor)
        self._min_score = {it.name: max(it.min_score, cfg.conf_floor) for it in policy}

    # ---- main api ---------------------------------------------------

    def __call__(self, frame: Frame) -> list[Detection]:
        """Run YOLOE on one frame; return per-policy-event detections."""
        return self.batch([frame])[0]

    def batch(self, frames: list[Frame]) -> list[list[Detection]]:
        """Run YOLOE on N frames in a single forward pass.

        Multi-stream throughput improves measurably (~2-3x at batch=4) because
        the kernel launch overhead amortizes across the batch. Returns a list
        of per-frame detection lists, in input order.
        """
        if not frames:
            return []
        images = [f.image for f in frames]
        results = self.model.predict(
            images,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            conf=self.cfg.conf_floor,
            iou=self.cfg.iou,
            half=self.cfg.half,
            verbose=False,
        )
        # Ultralytics may return fewer Results than inputs on degenerate frames;
        # pad with empty lists so per-frame indexing stays correct.
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
        # pad if Ultralytics returned a shorter list (defensive)
        while len(out) < len(frames):
            out.append([])
        return out
