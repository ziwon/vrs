"""Canonical object manifest writer for runtime artifacts.

The sink writes one ``object_manifest.v1`` JSON document per verified alert and
an append-friendly ``object_manifest.index.jsonl``. This avoids rewriting a
growing run-level manifest on every alert while keeping local files easy to
rebuild into an object-store index later.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..contracts import object_manifest_v1, stable_id
from ..schemas import VerifiedAlert
from ..storage import LocalObjectStore, ObjectStore, object_store_from_env


class ObjectManifestSink:
    """Writes an object_manifest.v1 snapshot and local append index."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        stream_id: str | None = None,
        run_id: str | None = None,
        manifest_dir: str = "manifests",
        index_name: str = "object_manifest.index.jsonl",
        store: ObjectStore | None = None,
        use_env_store: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.store = store or self._default_store(use_env_store)
        self.stream_id = stream_id
        self.run_id = run_id or self.root_dir.name
        self.manifest_dir = manifest_dir.strip("/\\") or "manifests"
        self.index_name = index_name
        self.index_path = self.root_dir / self.index_name
        self._indexed_manifest_ids = self._load_existing_index()

    def write(self, alert: VerifiedAlert) -> dict[str, Any]:
        evidence_refs = self._alert_evidence_refs(alert)
        record = alert.to_contract(stream_id=self.stream_id, evidence_refs=evidence_refs)
        alert_id = str(record["alert_id"])
        event_id = str(record["event_id"])
        manifest_id = stable_id("manifest", self.run_id, self.stream_id, event_id, alert_id)
        manifest_key = f"{self.manifest_dir}/{manifest_id}.json"
        manifest = object_manifest_v1(
            manifest_id=manifest_id,
            run_id=self.run_id,
            stream_id=self.stream_id,
            event_id=event_id,
            alert_id=alert_id,
            evidence_refs=evidence_refs,
            records=[record],
            created_at=str(record["created_at"]),
            metadata={
                "storage": self._storage_kind(),
                "manifest_strategy": "per-alert-with-jsonl-index",
                "limitations": "clip/frame refs are included only when upstream sinks provide them",
            },
        )
        manifest_obj = self.store.put_json(manifest_key, manifest)
        self._append_index_once(
            manifest_id=manifest_id,
            manifest_obj=manifest_obj,
            event_id=event_id,
            alert_id=alert_id,
            evidence_refs=evidence_refs,
        )
        return manifest

    def _alert_evidence_refs(self, alert: VerifiedAlert) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        if alert.thumbnail_path:
            refs.append(self._file_ref(alert.thumbnail_path, kind="thumbnail"))
        return refs

    def _file_ref(self, relative_path: str, *, kind: str) -> dict[str, Any]:
        rel = Path(relative_path)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"evidence path must be relative to run root: {relative_path!r}")
        obj = self.store.ref_for_key(rel.as_posix())
        return obj.to_evidence_ref(
            kind=kind,
            metadata={"relative_path": rel.as_posix(), "storage": self._storage_kind()},
        )

    def _append_index_once(
        self,
        *,
        manifest_id: str,
        manifest_obj: Any,
        event_id: str,
        alert_id: str,
        evidence_refs: list[dict[str, Any]],
    ) -> None:
        if manifest_id in self._indexed_manifest_ids:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "schema_version": "object_manifest_index.v1",
            "manifest_id": manifest_id,
            "run_id": self.run_id,
            "stream_id": self.stream_id,
            "event_id": event_id,
            "alert_id": alert_id,
            "manifest_ref": manifest_obj.to_evidence_ref(
                kind="metadata_manifest",
                metadata={"relative_path": manifest_obj.key, "storage": self._storage_kind()},
            ),
            "evidence_refs": evidence_refs,
        }
        with self.index_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._indexed_manifest_ids.add(manifest_id)

    def _load_existing_index(self) -> set[str]:
        if not self.index_path.exists():
            return set()
        out = set()
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            manifest_id = payload.get("manifest_id")
            if manifest_id:
                out.add(str(manifest_id))
        return out

    def _storage_kind(self) -> str:
        if isinstance(self.store, LocalObjectStore):
            return "local-filesystem"
        return "object-store"

    def _default_store(self, use_env_store: bool) -> ObjectStore:
        if use_env_store and _has_object_store_env():
            return object_store_from_env()
        return LocalObjectStore(self.root_dir)


def _has_object_store_env() -> bool:
    return any(
        name in os.environ
        for name in (
            "VRS_OBJECT_STORE",
            "VRS_OBJECT_STORE_ROOT",
            "VRS_RUNS_ROOT",
            "VRS_RUNS_DIR",
        )
    )
