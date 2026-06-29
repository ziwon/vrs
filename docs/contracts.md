# VRS Runtime Contracts

VRS now has explicit versioned contracts under `contracts/schemas/` plus Python
adapters in `vrs.contracts`.

The existing dataclasses in `vrs.schemas` remain the in-process compatibility
surface:

- `Detection`
- `CandidateAlert`
- `VerifiedAlert`
- `Frame`

Use `to_contract()` or the adapter functions when publishing across a process,
bus, object store, or DeepStream boundary. Use existing `to_json()` for the
current local `alerts.jsonl` audit/export shape.

## Contract Ownership

DeepStream data-plane workers should publish:

- `detection.v1`
- track metadata carried on `detection.v1.track_id` until a dedicated
  `track.v1` contract is added
- `evidence_ref.v1`
- object manifests for exported frames, clips, thumbnails, and metadata

VRS event-state and policy code promotes detections into:

- `candidate_alert.v1`

VRS verifier workers promote candidate alerts into:

- `verified_alert.v1`

Verifier output includes a stable alert id plus a separate
`verification_id`, `verification_attempt`, `verdict_version`, model id/version,
and prompt id/version fields. Use these fields for audit, replay, and retry
tracking; do not encode mutable verifier verdicts into `alert_id`.

`stream.v1` describes RTSP, file, or generated stream sources. Object storage is
the canonical durable location for evidence assets and metadata manifests.
Relational stores may index these contracts, but should be rebuildable from the
contracts and object storage.

## Current Compatibility Boundary

The local JSONL sink still writes the legacy `VerifiedAlert.to_json()` record so
existing web UI, audit signing, tests, and run artifacts continue to work. The
canonical contract adapters are additive and are the expected shape for the next
transport and object-storage milestones.

Single-stream and multistream runtime sinks now write one
`manifests/<manifest_id>.json` per verified alert plus an append-friendly
`object_manifest.index.jsonl`. Each per-alert manifest is an
`object_manifest.v1` document containing the `verified_alert.v1` record and
local `evidence_ref.v1` references for emitted thumbnails. Set
`sink.write_manifest: false` to disable this during specialized tests; keep it
enabled for runtime artifacts that need to cross process, bus, or object-storage
boundaries.

## Transport And Storage Interfaces

`vrs.transport` defines the service-free event bus boundary:

- `EventMessage`
- `EventTransport`
- `InMemoryEventTransport` for unit tests
- `RedisStreamsConfig` for edge-mode stream naming
- `KafkaConfig` for production topic naming

Redis and Kafka clients are not required by unit tests and are not implemented
yet. Future adapters should implement `EventTransport`.

`vrs.storage` defines the object-store boundary:

- `ObjectStore`
- `StoredObject`
- `LocalObjectStore`
- `S3CompatibleConfig`
- `S3CompatibleObjectStore`

The runtime manifest sink can use any `ObjectStore`. `LocalObjectStore` covers
dev/test/local edge paths. `S3CompatibleObjectStore` uses the same protocol for
SeaweedFS or external S3-compatible storage and returns `s3://bucket/key` URIs
described by `S3CompatibleConfig`.

## DeepStream Boundary

`vrs.deepstream.adapter` contains the first metadata adapter for the DeepStream
data plane. It accepts dependency-free `DeepStreamDetectionMetadata` records and
emits `detection.v1` with `source_runtime: deepstream`. A real DeepStream worker
should map `NvDsFrameMeta` and `NvDsObjectMeta` into this adapter, then publish
the result through an `EventTransport`. VLM verification remains outside
DeepStream.

`python -m vrs.deepstream.worker` is the first runnable service boundary. It
converts DeepStream-exported metadata JSON/JSONL into canonical `detection.v1`
JSONL for parity evaluation and downstream transport wiring.

## Static Control-Plane Rendering

`vrs.control.static_assignment` is the first control-plane primitive. It accepts
stream manifest-style inputs, emits `stream.v1` contracts, assigns streams to
DeepStream worker IDs deterministically, and renders worker configs containing
transport and object-store settings. It is static config rendering, not a
scheduler or operator.

`vrs.control.registry` adds the next in-memory primitives: a stream registry,
worker health records, and queue-pressure summaries derived from the existing
`MultiStreamPipeline.queue_stats()` shape. These records are meant to feed a
future control plane without requiring a database or cluster scheduler today.
