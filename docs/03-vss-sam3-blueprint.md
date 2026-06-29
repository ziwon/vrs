# VRS + VSS Blueprint + SAM3 Integration Development Plan

Status date: 2026-05-22.

## Overview

This document defines the phased development roadmap for evolving VRS (Video
Reasoning System) from a lightweight edge-oriented AI event detection pipeline
into a distributed semantic video intelligence platform.

The target direction is vendor-neutral. NVIDIA Video Search and Summarization
(VSS), VAST VSS/Foundation Stack patterns, and Meta SAM 3 are references for
architecture and capability planning, not mandatory implementation dependencies.
VRS should preserve a lightweight core while allowing enterprise deployments to
add segmentation workers, platform storage, semantic retrieval, incident Q&A,
and operator feedback loops.

References:

- NVIDIA VSS Blueprint: <https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization>
- VAST VSS Foundation Stack: <https://github.com/vast-data/vss-blueprint>
- Meta SAM 3 Research Repository: <https://github.com/facebookresearch/sam3>

The objective is to evolve VRS into a platform capable of:

- Real-time RTSP/video ingestion.
- Multi-stage AI verification.
- Segmentation-aware evidence extraction.
- Event indexing and semantic retrieval.
- Long-video summarization.
- Incident investigation and Q&A.
- Operator feedback-driven policy optimization.
- Enterprise-scale observability and storage.

## 1. Planning Decisions

These decisions constrain the roadmap and should be treated as architectural
requirements until superseded by measured implementation evidence.

### 1.1 SAM3 Runtime Compatibility Gate

SAM3 integration must be implemented as an optional isolated GPU worker or
sidecar service, not as an in-process dependency of VRS Core. VRS Core must keep
its current runtime compatibility profile, while SAM3 workers may use their own
Python, PyTorch, CUDA, checkpoint-access, model-cache, and container
requirements.

This keeps the edge pipeline deployable on existing VRS environments while
allowing SAM3/SAM3.1 to evolve on a separate release cadence.

### 1.2 Platform Contract Layer

Before introducing Kafka, object storage, relational indexes, or vector databases, VRS must define
canonical contracts for:

- Events.
- Clips.
- Masks.
- Tracks.
- Candidate alerts.
- Verified alerts.
- Object references.
- Schema versions.
- Retention and deletion policies.
- Idempotency and retry semantics.

Infrastructure components should implement these contracts. They should not
define them implicitly.

Initial versioned contracts now live under `contracts/schemas/` with Python
adapters in `vrs.contracts`: `detection.v1`, `candidate_alert.v1`,
`verified_alert.v1`, `evidence_ref.v1`, `stream.v1`, and
`object_manifest.v1`. Existing dataclasses remain the in-process compatibility
surface, and local JSONL remains the audit/export fallback while object storage
becomes the canonical evidence and manifest store.

### 1.3 SAM3 Evaluation Gate

SAM3 adoption is accepted only if it improves measurable quality or evidence
value under defined latency and VRAM limits. Evaluation should include:

- False-positive delta.
- Verifier flip rate.
- Mask quality when labels exist.
- Evidence usefulness for operator review.
- p50/p95 latency.
- Peak VRAM.
- Queue impact and backpressure behavior.
- Failure fallback behavior.

SAM3 may still be valuable as an evidence enhancement even when alert precision
improvement is modest, but that decision must be explicit and measured.

### 1.4 VLM Runtime Matrix

VLMs must be categorized as:

| Category | Meaning |
| --- | --- |
| Already explored | Smoke-tested or benchmarked in this repository. |
| Candidate for evaluation | Plausible model/backend that needs comparable eval data. |
| Production default TBD | No production default is declared until quality, runtime, and cost evidence supports it. |

The roadmap must not imply a production default until runtime, quality, and
operational evidence supports that decision.

### 1.5 Repository Structure Note

The proposed package layout in this document is a long-term target split, not an
immediate rewrite. Any migration should map from current modules to target
modules incrementally and preserve stable public APIs during the transition.

### 1.6 VSS Positioning

The platform direction is vendor-neutral. NVIDIA VSS and VAST VSS/Foundation
Stack patterns are references for layered architecture, video reasoning workflows,
and enterprise storage/search patterns. VRS should not hard-code itself to one
vendor's runtime, data platform, or deployment model.

### 1.7 Canonical Storage Direction

After Phase 3, object storage should become the canonical durable source for
evidence assets and metadata manifests. PostgreSQL or another relational store
may be used as a query index, operational projection, or API cache, but the
system should be able to rebuild those indexes from object storage and event
contracts. JSONL remains as an immutable audit/export fallback and lightweight
edge-mode output.

## 2. Architectural Vision

### 2.1 Core Philosophy

VRS follows a layered evolution model:

| Layer | Goal |
| --- | --- |
| VRS Core | Lightweight realtime event detection. |
| VRS Platform | Distributed event processing and durable evidence management. |
| VRS Enterprise | Semantic video intelligence, investigation, and feedback optimization. |

The architecture intentionally separates:

- Fast detection.
- Optional segmentation refinement.
- Temporal reasoning.
- Policy decisioning.
- Semantic indexing.
- Human feedback optimization.

This allows the platform to scale from:

- Single edge appliance.
- Multi-camera NVR.
- Small on-prem cluster.
- City-scale AI CCTV infrastructure.
- Enterprise VSS/VLM platform.

### 2.2 Separation of Concerns

VRS should avoid collapsing all intelligence into one model call. Each layer has
a different latency, cost, and reliability profile:

| Layer | Primary Role | Failure Mode |
| --- | --- | --- |
| Reader and sampler | Deliver bounded frame/clip inputs. | Drop or reconnect without corrupting state. |
| Fast detector | High-recall candidate generation. | Missed candidate or noisy candidate. |
| Event state engine | Temporal smoothing and policy windows. | Duplicate or stale candidate. |
| SAM3 worker | Optional mask/track evidence refinement. | Missing evidence, not blocked core alerting. |
| VLM verifier | Semantic validation and explanation. | Reject, pass-through, or retry based on policy. |
| Platform storage | Durable metadata and evidence assets. | Idempotent retry and audit preservation. |
| Search and Q&A | Operator investigation and retrieval. | Degraded search, not alert-path failure. |

## 3. Target Architecture

### 3.1 Core Edge Path

```text
RTSP/mp4
  +-- Reader
      +-- FrameSampler / ClipBuffer
          +-- YOLOE fast detector
              +-- EventStateQueue
                  +-- CandidateAlert
                      +-- optional SAM3 worker request
                      |   +-- mask / refined bbox / track evidence
                      +-- VLM verifier
                          +-- VerifiedAlert
                              +-- JSONL + thumbnail overlay + metrics
```

The core path must continue to work when the SAM3 worker is disabled,
unavailable, or overloaded.

### 3.2 SAM3 Worker Path

```text
CandidateAlert or bbox prompt
  +-- SAM3 GPU worker / sidecar service
      +-- image refinement job
      |   +-- mask + refined bbox + mask summary
      +-- short-clip tracking job
          +-- object track + mask trajectory + motion features
```

SAM3 worker outputs are evidence attachments. The VRS Core alert contract should
store references to those outputs rather than requiring SAM3-specific in-memory
objects.

### 3.3 Platform Extension

```text
VerifiedAlert
    +-- Platform contract adapter
        +-- Event bus
        +-- Object storage for clips / thumbnails / masks / metadata manifests
        +-- Optional relational query index
        +-- Vector index for embeddings
        +-- FastAPI backend
        +-- Web UI
        +-- Prometheus / Grafana
        +-- Semantic search / incident Q&A / summarization
```

Infrastructure choices remain deployment-specific. The architecture requires
contract compatibility, not a single mandatory broker, database, or object store.

## 4. VRS Core Development

### 4.1 Objectives

The VRS Core layer focuses on:

- Low latency.
- GPU-efficient inference.
- Realtime alerting.
- Minimal operational complexity.
- Local-first operation without enterprise services.

Primary target:

- Edge GPU systems.
- Small clusters.
- On-prem CCTV deployments.

### 4.2 RTSP / MP4 Ingestion

Components:

| Component | Role |
| --- | --- |
| Reader | Decode RTSP/mp4. |
| FrameSampler | FPS reduction and cadence control. |
| ClipBuffer | Rolling short-term evidence memory. |
| MetadataExtractor | Timestamp, camera, and source metadata. |

Features:

- RTSP reconnect.
- Multi-stream support.
- Hardware decode with NVDEC where available.
- Backpressure handling.
- Frame dropping policy.
- Bounded memory behavior.

Tech stack candidates:

| Area | Technology |
| --- | --- |
| Decode | FFmpeg / PyAV / OpenCV / GStreamer. |
| GPU decode | NVIDIA NVDEC. |
| Buffering | asyncio queues or worker queues. |

### 4.3 YOLOE Fast Detector

Responsibilities:

- Fast object/event candidate generation.
- Low-cost filtering stage.
- Bounding box extraction.

Design principles:

- High recall first.
- Precision improved by event state, VLM verification, and optional SAM3 evidence.
- GPU batching where it improves throughput without hurting realtime behavior.
- TensorRT optimization only after class/prompt mapping is stable.

Output contract draft:

```json
{
  "schema_version": "detection.v1",
  "camera_id": "cam-01",
  "source_id": "rtsp-front-gate",
  "timestamp": "2026-05-22T12:00:00Z",
  "frame_ref": "object://frames/cam-01/000001.jpg",
  "detections": [
    {
      "class": "person",
      "bbox_xyxy": [10, 20, 100, 220],
      "confidence": 0.82,
      "detector": "yoloe"
    }
  ]
}
```

### 4.4 DeepStream Runtime Option

DeepStream is an optional NVIDIA GPU deployment path for production-grade RTSP
ingest, batching, decode, inference, tracking, and message publishing. It is not
required for VRS Core and should not become the only supported runtime.

Use DeepStream when a deployment needs:

- High-density RTSP ingest.
- NVDEC-first decode.
- GStreamer-based media pipeline control.
- TensorRT-optimized detector execution.
- Built-in batching and stream multiplexing.
- Tracker integration.
- Message broker integration through `nvmsgbroker`.

DeepStream should integrate with VRS through the same contracts used by other
runtimes. Its outputs should be normalized into VRS detection, candidate alert,
track, and evidence schemas before entering the EventStateQueue or Platform
Contract Layer.

Recommended mapping:

| DeepStream component | VRS integration role |
| --- | --- |
| `nvurisrcbin` / RTSP source bins | Production stream ingestion. |
| `nvstreammux` | Multi-stream batching and synchronization. |
| `nvinfer` | TensorRT detector execution. |
| `nvtracker` | Optional tracker evidence source. |
| `nvmsgconv` / `nvmsgbroker` | Contract-normalized event publishing. |
| Custom probe/plugin | Convert DeepStream metadata into VRS schemas. |

Design constraints:

- Keep DeepStream as a runtime adapter, not an architecture dependency.
- Preserve the non-DeepStream Python/OpenCV/PyAV path for portability and tests.
- Normalize metadata before durable storage or verifier prompts.
- Treat DeepStream tracker output as optional evidence unless policy explicitly
  requires it.
- Keep SAM3 as a separate worker/service even when DeepStream is used for ingest
  and detection.

Evaluation criteria:

| Metric | Reason |
| --- | --- |
| Stream density per GPU | Confirm deployment value versus the default runtime. |
| End-to-end latency | Protect realtime alert behavior. |
| Detector parity | Ensure TensorRT/DeepStream output matches expected classes/prompts. |
| Queue drops | Detect overload and backpressure issues. |
| Broker delivery reliability | Validate event publication behavior. |
| Operational complexity | Decide whether DeepStream is worth the added deployment surface. |

## 5. Platform Contract Layer

The contract layer is a required milestone before durable enterprise services are
introduced. It should be versioned, testable, and serializable.

### 5.1 Canonical Identifiers

Recommended identifiers:

| Identifier | Purpose |
| --- | --- |
| `event_id` | Stable logical incident or candidate event ID. |
| `camera_id` | Operator-facing camera identity. |
| `source_id` | Ingest source identity, such as RTSP URI alias or file ID. |
| `clip_id` | Durable evidence clip identity. |
| `mask_id` | Durable segmentation mask identity. |
| `track_id` | Temporal object track identity. |
| `alert_id` | Candidate or verified alert identity. |
| `policy_id` | Policy/rule version that produced the decision. |

### 5.2 Evidence Object References

Contracts should reference evidence objects by durable URI-like references, not
inline large payloads.

```json
{
  "clip_ref": "object://clips/cam-01/2026/05/22/clip-123.mp4",
  "thumbnail_ref": "object://thumbs/cam-01/alert-456.jpg",
  "mask_ref": "object://masks/cam-01/mask-789.png",
  "track_ref": "object://tracks/cam-01/track-101.json"
}
```

### 5.3 CandidateAlert Contract

```json
{
  "schema_version": "candidate_alert.v1",
  "alert_id": "alert-456",
  "event_id": "event-123",
  "camera_id": "cam-01",
  "event_type": "restricted_area_person",
  "confidence": 0.91,
  "bbox_xyxy": [10, 20, 100, 220],
  "clip_ref": "object://clips/cam-01/clip-123.mp4",
  "mask_ref": null,
  "track_ref": null,
  "policy_id": "restricted-area.v3",
  "created_at": "2026-05-22T12:00:03Z"
}
```

### 5.4 VerifiedAlert Contract

```json
{
  "schema_version": "verified_alert.v1",
  "alert_id": "alert-456",
  "event_id": "event-123",
  "camera_id": "cam-01",
  "verified": true,
  "risk_level": "high",
  "summary": "A person is climbing over the restricted-area fence at night.",
  "policy_match": ["restricted-area.v3"],
  "evidence_refs": {
    "clip_ref": "object://clips/cam-01/clip-123.mp4",
    "thumbnail_ref": "object://thumbs/cam-01/alert-456.jpg",
    "mask_ref": "object://masks/cam-01/mask-789.png",
    "track_ref": "object://tracks/cam-01/track-101.json"
  },
  "verifier": {
    "backend": "openai-compatible",
    "model": "qwen-vl-served"
  },
  "created_at": "2026-05-22T12:00:08Z"
}
```

### 5.5 Idempotency and Retention

Before adding a broker or durable database, define:

- Idempotency keys for repeated candidate, verifier, and storage writes.
- Retry semantics for GPU workers and storage adapters.
- Retention rules for clips, masks, thumbnails, logs, and embeddings.
- Deletion behavior for privacy and governance requests.
- Schema migration policy for historical alerts.

## 6. SAM3 Integration Plan

### 6.1 Why SAM3

SAM3 integration is an optional evidence and segmentation enhancement layer. It
is not a required dependency for VRS Core alerting.

Potential capabilities:

| Capability | Potential benefit | Validation requirement |
| --- | --- | --- |
| Pixel segmentation | Better object localization. | Mask quality or operator evidence score. |
| Mask trajectory | Temporal evidence. | Track continuity and verifier usefulness. |
| Occlusion handling | Better short-window evidence. | Measured on labeled or reviewed clips. |
| Fine object boundary | Better forensic thumbnails. | Operator review score or downstream metric. |
| Region-focused crops | Better VLM grounding. | Verifier flip-rate and latency measurement. |

### 6.2 SAM3 Image Refiner

Pipeline position:

```text
YOLOE bbox
   +-- SAM3 image refinement worker
        +-- refined bbox + segmentation mask + mask summary
```

Inputs:

- RGB frame or frame reference.
- YOLOE bbox prompt.
- Optional text prompt or class label.
- Camera/source metadata.

Outputs:

```json
{
  "schema_version": "mask_evidence.v1",
  "mask_id": "mask-789",
  "source_alert_id": "alert-456",
  "mask_ref": "object://masks/cam-01/mask-789.png",
  "refined_bbox_xyxy": [12, 22, 98, 218],
  "mask_area": 2048,
  "shape_features": {
    "aspect_ratio": 0.42,
    "fill_ratio": 0.68
  },
  "worker": {
    "name": "sam3-image-worker",
    "model": "sam3"
  }
}
```

Fallback rule: if SAM3 refinement times out, fails, or is unavailable, VRS should
continue with the detector bbox and record the missing evidence status.

### 6.3 SAM3.1 Short Clip Tracker

Purpose: track segmented objects across short temporal windows without blocking
core alert generation.

Pipeline:

```text
CandidateAlert
    +-- Clip extraction
         +-- SAM3.1 tracking worker
              +-- trajectory evidence
```

Output:

```json
{
  "schema_version": "track_evidence.v1",
  "track_id": "trk-01",
  "source_alert_id": "alert-456",
  "trajectory": [
    {
      "timestamp": "2026-05-22T12:00:01Z",
      "bbox_xyxy": [10, 20, 100, 220],
      "mask_ref": "object://masks/cam-01/mask-001.png"
    }
  ],
  "motion_features": {
    "duration_s": 4.2,
    "path_length_px": 180.0
  }
}
```

Use cases:

- Intrusion tracking.
- Fall detection verification.
- Loitering analysis.
- Multi-frame evidence generation.

### 6.4 SAM3 Evaluation Gate

A SAM3 integration should not graduate from experimental to enabled-by-default
until it has a measured acceptance report.

Required measurements:

| Metric | Reason |
| --- | --- |
| False-positive delta | Confirm whether SAM3 reduces noisy alerts. |
| Verifier flip rate | Measure whether masks/tracks change VLM decisions. |
| Mask quality | Validate segmentation when labels exist. |
| Operator evidence score | Capture forensic/review usefulness. |
| p50/p95 latency | Protect realtime pipeline behavior. |
| Peak VRAM | Size worker deployment. |
| Queue depth and drops | Detect backpressure impact. |
| Timeout/fallback rate | Ensure core alerting remains robust. |

## 7. Event State Engine

### 7.1 EventStateQueue

The EventStateQueue acts as:

- Temporal memory.
- Debounce logic.
- Multi-frame aggregation engine.
- Policy windowing engine.

Responsibilities:

| Feature | Purpose |
| --- | --- |
| Temporal smoothing | Reduce noisy alerts. |
| Multi-frame consensus | Improve confidence. |
| Policy windowing | Enforce event duration rules. |
| State expiration | Bound memory. |
| Evidence collection | Select frames/clips for verifier and SAM3 workers. |

### 7.2 CandidateAlert Generation

Candidate alerts should be created only after policy and temporal thresholds are
met. Candidate alerts may be enriched by SAM3, but should not require SAM3.

Key design rules:

- Preserve original detector evidence.
- Record policy version and thresholds used.
- Keep alert generation idempotent.
- Attach optional evidence refs after enrichment.
- Expire stale state deterministically.

## 8. VLM Verification Layer

### 8.1 Purpose

The VLM stage performs:

- Semantic verification.
- Scene understanding.
- Contextual reasoning.
- Policy validation.
- Human-readable explanation.

Example:

YOLOE:

> "person detected"

VLM:

> "A person is climbing over a restricted-area fence at night."

### 8.2 Runtime Matrix Categories

Candidate VLMs should be tracked in the runtime matrix, not presented as a fixed
production shortlist.

| Category | Models/backends | Current interpretation |
| --- | --- | --- |
| Already explored | Qwen3-VL-2B, Qwen2.5-VL-7B, Cosmos baseline paths where benchmarked. | Use existing smoke and benchmark docs as evidence. |
| Candidate for evaluation | Larger Qwen served VLMs, InternVL, LLaVA-NeXT, Florence-2, Cosmos-Reason variants. | Run comparable quality, latency, and cost evals before adoption. |
| Production default TBD | None. | Production default requires real-dataset quality and runtime evidence. |

### 8.3 Prompt Strategy

Inputs:

- Keyframes.
- Segmentation overlays when available.
- Object trajectories when available.
- Camera metadata.
- Policy context.
- Prior event state when useful.

Output:

```json
{
  "schema_version": "vlm_verdict.v1",
  "verified": true,
  "summary": "...",
  "risk_level": "high",
  "policy_match": ["restricted-area.v3"],
  "evidence_used": ["keyframes", "mask_overlay", "trajectory"]
}
```

### 8.4 Verification Guardrails

- Structured JSON output should remain mandatory for automated decisions.
- Malformed output must follow an explicit failure policy.
- Prompts should be versioned with policy IDs.
- Latency and error rates should be reported per backend/model.
- SAM3 evidence should be evaluated against the same prompt and model baseline.

## 9. VRS Platform Layer

### 9.1 Scope

The platform layer turns local verified alerts into durable, queryable,
operational evidence. It should be added only after the Platform Contract Layer
is defined.

### 9.2 Event Bus

Candidate transports:

| Technology | Role | Notes |
| --- | --- | --- |
| Kafka | Distributed event backbone. | Best for durable, replayable event streams. |
| Redis Streams | Lightweight work queue. | Useful for GPU worker queues or smaller deployments. |
| In-process queue | Edge mode. | Keeps single-node deployment simple. |

Recommended separation:

| Type | Transport |
| --- | --- |
| Alert/events | Kafka or equivalent durable stream. |
| GPU tasks | Redis Streams or queue service. |
| Metrics | Prometheus/OpenTelemetry. |
| Metadata writes | Object metadata manifest writer plus optional relational projection. |

### 9.3 Storage Layer

After Phase 3, object storage becomes the canonical durable source for evidence
assets and metadata manifests. Relational databases are optional projections for
query performance and API ergonomics.

| Storage | Purpose | Canonical role |
| --- | --- | --- |
| Object storage | Clips, thumbnails, masks, tracks, metadata manifests, exports. | Canonical evidence and metadata store. |
| PostgreSQL or equivalent | Structured query index, alert state projection, policy version lookup. | Rebuildable projection, not the only durable source. |
| JSONL | Immutable edge logs, audit fallback, export. | Fallback and audit/export format. |
| Parquet | Analytics export. | Derived analytical format. |

Object storage may be SeaweedFS, S3-compatible storage, VAST, or another deployment
choice. The contract should depend on object references, not a vendor-specific
client.

### 9.4 Vector Search Layer

Candidate engines:

| Engine | Strength |
| --- | --- |
| Milvus | Large-scale vector infrastructure. |
| Qdrant | Simpler dedicated vector deployment. |
| pgvector | Operational simplicity with PostgreSQL. |

Indexed assets:

- Event embeddings.
- Region/crop embeddings.
- Caption embeddings.
- Incident summaries.
- Operator feedback vectors.

Vector indexes are derived from canonical metadata/evidence stores. They should
be rebuildable.

## 10. Semantic Video Search

### 10.1 Segment Indexing

Workflow:

```text
Video or clip evidence
   +-- Scene / temporal segmentation
        +-- Caption and embedding generation
             +-- Vector indexing
                  +-- Search result with canonical evidence refs
```

Metadata:

```json
{
  "schema_version": "segment_embedding.v1",
  "camera_id": "cam-01",
  "time_range": {
    "start": "2026-05-22T12:00:00Z",
    "end": "2026-05-22T12:00:10Z"
  },
  "caption": "Person near restricted gate.",
  "embedding_ref": "vector://events/event-123",
  "clip_ref": "object://clips/cam-01/clip-123.mp4"
}
```

### 10.2 Semantic Search Examples

Query:

> "Person leaving bag unattended"

Results:

- Matching incidents.
- Ranked clips.
- Temporal summaries.
- Trajectory evidence.
- Linked canonical alert and evidence records.

## 11. Long Video Summarization

### 11.1 Goals

Provide:

- Shift summaries.
- Incident timelines.
- Operator briefing.
- Daily security digest.

### 11.2 Multi-Stage Summarization

Stage 1: event extraction.

Stage 2: temporal clustering.

Stage 3: evidence retrieval and grounding.

Stage 4: VLM/LLM narrative generation.

Example:

```text
08:14 - Person entered restricted area
08:16 - Object left unattended
08:19 - Security personnel arrived
```

Summaries should cite underlying alert IDs and clip refs so operators can audit
the generated narrative.

## 12. Incident Q&A

### 12.1 Retrieval-Augmented Video Reasoning

Pipeline:

```text
User question
   +-- Metadata and vector search
        +-- Relevant alerts / clips / masks / tracks
             +-- VLM or LLM reasoning
                  +-- Answer with evidence refs
```

Example questions:

- "Who entered after midnight?"
- "Show forklift near loading dock."
- "Was the person carrying a backpack?"
- "When did smoke first appear?"

### 12.2 Answer Requirements

Answers should include:

- Time range.
- Camera/source ID.
- Evidence refs.
- Confidence or uncertainty.
- Policy/alert linkage when applicable.

## 13. Operator Feedback Optimization

### 13.1 Human-in-the-Loop

Operators can:

- Confirm alerts.
- Reject false positives.
- Adjust severity.
- Add semantic tags.
- Rate evidence usefulness.
- Mark missing or misleading summaries.

### 13.2 Optimization Targets

| Area | Optimization |
| --- | --- |
| Thresholds | Detection tuning. |
| Policies | Context adaptation. |
| Prompts | VLM accuracy. |
| Search ranking | Better retrieval. |
| Evidence generation | Better SAM3/thumbnail/clip selection. |

### 13.3 Governance Requirements

Feedback must be auditable. Any automatic policy or threshold update should
record:

- Old value.
- New value.
- Reason.
- Supporting metrics.
- Operator or system actor.
- Rollback path.

## 14. Web Platform

### 14.1 FastAPI Backend

Responsibilities:

- API gateway.
- Event and incident management.
- Search APIs.
- Evidence retrieval APIs.
- Authentication and authorization.
- Policy APIs.
- Feedback APIs.

### 14.2 Web UI

Features:

| Feature | Description |
| --- | --- |
| Live dashboard | Active alerts and camera health. |
| Semantic search | Natural language retrieval. |
| Timeline explorer | Incident navigation. |
| Mask overlays | SAM visualization when evidence exists. |
| Evidence viewer | Multi-frame and clip review. |
| Feedback console | Operator labeling and correction. |
| Audit view | Alert, policy, and evidence traceability. |

## 15. Observability

### 15.1 Metrics

Prometheus/OpenTelemetry targets:

- FPS.
- GPU utilization.
- Queue latency.
- Queue drops.
- Detection latency.
- SAM3 worker latency.
- SAM3 timeout/fallback rate.
- VLM latency.
- Alert rate.
- False positive rate.
- Storage write failures.
- Search latency.

### 15.2 Grafana Dashboards

Dashboard categories:

| Dashboard | Purpose |
| --- | --- |
| GPU monitoring | Inference health. |
| Pipeline latency | Bottleneck analysis. |
| Camera health | Stream status. |
| Worker health | SAM3/VLM task latency and failures. |
| Search analytics | Query behavior. |
| Storage health | Metadata/object write and read behavior. |

## 16. Scalability Strategy

### 16.1 Deployment Tiers

| Tier | Deployment |
| --- | --- |
| Edge | Single GPU appliance with JSONL/object files. |
| NVIDIA edge | Optional DeepStream adapter for high-density RTSP ingest. |
| Small cluster | Docker Compose with optional worker services. |
| Medium scale | k3s with object storage, optional relational index, and vector service. |
| Enterprise | Kubernetes with durable event bus and observability stack. |

### 16.2 GPU Allocation Strategy

| Workload | GPU Priority | Notes |
| --- | --- | --- |
| YOLOE | High throughput. | Realtime path. |
| SAM3 | Burst refinement. | Optional worker, bounded queue. |
| VLM | Heavy reasoning. | Slow path, policy-controlled. |
| Embeddings | Background batch. | Rebuildable derived index. |

### 16.3 Backpressure Rules

- Core detection should degrade by frame dropping before unbounded queue growth.
- SAM3 enrichment should degrade by timeout and missing-evidence status.
- VLM verification should use explicit pass-through/reject/retry policy.
- Platform writes should be idempotent and retryable.
- Search indexes should be rebuildable from canonical storage.

## 17. Security & Governance

### 17.1 Security Goals

- Immutable audit logs.
- Role-based access.
- Event traceability.
- Evidence retention policies.
- Privacy-preserving exports.
- Tamper-evident alert records.

### 17.2 Enterprise Extensions

Potential future integration:

- TPM-backed attestation.
- Confidential GPU computing.
- Signed model pipelines.
- Secure evidence export.
- Model and prompt provenance.

## 18. Recommended Development Phases

### Phase 1 - VRS Core

Deliverables:

- RTSP ingest.
- YOLOE detection.
- Event queue.
- JSONL alerts.
- Thumbnail overlays.
- Metrics for detector/verifier latency and queue behavior.
- Optional DeepStream runtime adapter design for NVIDIA-heavy deployments.

Goal: realtime edge AI detection MVP.

Acceptance criteria:

- Core pipeline runs without enterprise services.
- Alert JSONL schema is versioned.
- Existing test and pre-commit suites pass.
- Runtime matrix records validated environments.

### Phase 2 - SAM3 Worker Integration

Deliverables:

- Optional SAM3 image refinement worker.
- Optional SAM3.1 short-clip tracking worker.
- Mask overlays.
- Trajectory evidence.
- Timeout and fallback behavior.
- Evaluation report for SAM3 quality, latency, and VRAM.

Goal: improve evidence quality and reduce false positives without breaking VRS
Core compatibility.

Acceptance criteria:

- VRS Core does not import SAM3 directly.
- SAM3 worker can be disabled without changing alert behavior.
- SAM3 outputs attach through contract refs.
- Evaluation gate passes before default enablement.

### Phase 3 - Platform Contract and Storage Layer

Deliverables:

- Canonical event/clip/mask/track/alert schemas.
- Object storage evidence and metadata-manifest store.
- Optional PostgreSQL or equivalent query projection.
- JSONL audit/export fallback.
- Idempotent storage adapters.
- Initial FastAPI backend.

Goal: durable distributed event processing platform.

Acceptance criteria:

- Object storage is canonical for platform deployments; relational indexes are rebuildable projections.
- JSONL remains supported for edge mode and audit/export fallback.
- Storage writes are idempotent.
- Evidence refs are resolvable from API responses.

### Phase 4 - Semantic Intelligence

Deliverables:

- Vector search.
- Long-video indexing.
- Summarization.
- Incident Q&A.
- Evidence-grounded answers.

Goal: enterprise semantic video understanding.

Acceptance criteria:

- Search indexes are rebuildable from canonical storage.
- Answers cite alert IDs and evidence refs.
- Summaries preserve time ranges and camera/source IDs.
- Retrieval quality is measured with representative queries.

### Phase 5 - Feedback Optimization

Deliverables:

- Operator feedback loop.
- Policy optimization.
- Retrieval tuning.
- Adaptive prompting.
- Evidence usefulness scoring.

Goal: self-improving AI surveillance platform with auditable changes.

Acceptance criteria:

- Feedback records are versioned and auditable.
- Automated tuning has caps, cooldowns, and rollback.
- Prompt and policy changes are tied to measurable improvements.

## 19. Long-Term Repository Structure

This is a long-term target split, not an immediate rewrite.

```text
vrs/
|-- core/              # shared pipeline contracts and orchestration
|-- ingest/            # readers, samplers, clip buffers
|-- detectors/         # YOLOE and future detector backends
|-- sam/               # SAM3 worker clients and contract adapters
|-- vlm/               # verifier backends and prompt orchestration
|-- pipelines/         # edge, multistream, and platform pipelines
|-- storage/           # object/relational-projection/JSONL adapters
|-- vector/            # embedding and vector-index adapters
|-- backend/           # FastAPI service
|-- webui/             # operator UI
|-- infra/             # compose/k8s deployment assets
|-- observability/     # metrics and dashboards
|-- schemas/           # versioned platform contracts
+-- docs/
```

Current modules should migrate incrementally. For example:

| Current area | Long-term area |
| --- | --- |
| `vrs/triage` | `vrs/detectors` and `vrs/core`. |
| `vrs/runtime` | `vrs/vlm`. |
| `vrs/verifier` | `vrs/vlm` and `vrs/core`. |
| `vrs/sinks` | `vrs/storage`. |
| `vrs/multistream` | `vrs/pipelines` and `vrs/ingest`. |
| `vrs/observability` | `vrs/observability`. |

## 20. Final Strategic Positioning

VRS should not evolve into merely:

- Another detector pipeline.
- Another VLM wrapper.
- Another vector search demo.

Instead, the strategic direction is:

> "A vendor-neutral distributed semantic video reasoning platform optimized for
> realtime operational intelligence."

The combination of:

- YOLOE realtime recall.
- Optional SAM3 segmentation evidence.
- VLM semantic reasoning.
- Canonical evidence contracts.
- Durable metadata and object storage.
- Vector retrieval.
- Human feedback optimization.

creates a differentiated architecture positioned between:

- AI CCTV platforms.
- Video Search Systems (VSS).
- Multimodal RAG systems.
- Operational AI copilots.

for enterprise and public-sector deployments.
