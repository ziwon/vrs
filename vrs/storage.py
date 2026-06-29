"""Object storage abstractions for evidence and metadata manifests."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .contracts import evidence_ref_v1


@dataclass(frozen=True)
class StoredObject:
    uri: str
    key: str
    media_type: str
    size_bytes: int | None = None
    sha256: str | None = None
    created_at: str | None = None

    def to_evidence_ref(
        self, *, kind: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return evidence_ref_v1(
            uri=self.uri,
            kind=kind,
            media_type=self.media_type,
            size_bytes=self.size_bytes,
            sha256=self.sha256,
            created_at=self.created_at,
            metadata=metadata,
        )


class ObjectStore(Protocol):
    def put_bytes(self, key: str, data: bytes, *, media_type: str | None = None) -> StoredObject:
        """Store bytes under a key and return a resolvable object reference."""

    def put_json(self, key: str, payload: dict[str, Any]) -> StoredObject:
        """Store JSON under a key and return a resolvable object reference."""

    def ref_for_key(self, key: str, *, media_type: str | None = None) -> StoredObject:
        """Return a reference for an existing object key."""


@dataclass(frozen=True)
class S3CompatibleConfig:
    """Configuration for S3-compatible object stores such as SeaweedFS."""

    bucket: str
    endpoint_url: str | None = None
    region: str | None = None
    prefix: str = ""
    force_path_style: bool = True
    access_key_id: str | None = None
    secret_access_key: str | None = None

    def uri_for_key(self, key: str) -> str:
        prefix = self.prefix.strip("/")
        full_key = f"{prefix}/{key.lstrip('/')}" if prefix else key.lstrip("/")
        return f"s3://{self.bucket}/{full_key}"


class LocalObjectStore:
    """Filesystem-backed object store for dev, tests, and edge mode."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, key: str, data: bytes, *, media_type: str | None = None) -> StoredObject:
        rel = self._safe_key(key)
        path = self.root_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_bytes(data)
        os.replace(tmp_path, path)
        return self.ref_for_key(rel.as_posix(), media_type=media_type)

    def put_json(self, key: str, payload: dict[str, Any]) -> StoredObject:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
        return self.put_bytes(key, data, media_type="application/json")

    def ref_for_key(self, key: str, *, media_type: str | None = None) -> StoredObject:
        rel = self._safe_key(key)
        path = (self.root_dir / rel).resolve()
        stat = path.stat() if path.exists() else None
        guessed_type = (
            media_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        )
        return StoredObject(
            uri=path.as_uri(),
            key=rel.as_posix(),
            media_type=guessed_type,
            size_bytes=stat.st_size if stat is not None else None,
            sha256=_sha256_file(path) if stat is not None and path.is_file() else None,
            created_at=(
                datetime.fromtimestamp(stat.st_mtime, UTC).isoformat().replace("+00:00", "Z")
                if stat is not None
                else None
            ),
        )

    @staticmethod
    def _safe_key(key: str) -> Path:
        rel = Path(key)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"object key must be relative: {key!r}")
        return rel


class S3CompatibleObjectStore:
    """S3-compatible object store implementation for SeaweedFS and external S3.

    ``boto3`` is imported lazily so local-only tests and development paths do
    not require network/storage dependencies to be initialized.
    """

    def __init__(self, config: S3CompatibleConfig, *, client: Any | None = None):
        self.config = config
        self.client = client or self._build_client(config)

    def put_bytes(self, key: str, data: bytes, *, media_type: str | None = None) -> StoredObject:
        safe_key = self._safe_key(key)
        full_key = self._full_key(safe_key)
        guessed_type = media_type or mimetypes.guess_type(safe_key)[0] or "application/octet-stream"
        self.client.put_object(
            Bucket=self.config.bucket,
            Key=full_key,
            Body=data,
            ContentType=guessed_type,
        )
        return StoredObject(
            uri=self.config.uri_for_key(safe_key),
            key=full_key,
            media_type=guessed_type,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            created_at=now_iso(),
        )

    def put_json(self, key: str, payload: dict[str, Any]) -> StoredObject:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
        return self.put_bytes(key, data, media_type="application/json")

    def ref_for_key(self, key: str, *, media_type: str | None = None) -> StoredObject:
        safe_key = self._safe_key(key)
        full_key = self._full_key(safe_key)
        guessed_type = media_type or mimetypes.guess_type(safe_key)[0] or "application/octet-stream"
        return StoredObject(
            uri=self.config.uri_for_key(safe_key),
            key=full_key,
            media_type=guessed_type,
        )

    def _full_key(self, key: str) -> str:
        prefix = self.config.prefix.strip("/")
        return f"{prefix}/{key}" if prefix else key

    @staticmethod
    def _safe_key(key: str) -> str:
        rel = LocalObjectStore._safe_key(key)
        return rel.as_posix()

    @staticmethod
    def _build_client(config: S3CompatibleConfig) -> Any:
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover - exercised only without optional dep.
            raise RuntimeError(
                "boto3 is required for S3/SeaweedFS object storage. "
                "Install vrs with the storage dependency or use LocalObjectStore."
            ) from exc

        s3_config = Config(
            s3={"addressing_style": "path" if config.force_path_style else "auto"}
        )
        return boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            region_name=config.region,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            config=s3_config,
        )


def object_store_from_env(prefix: str = "VRS_OBJECT_STORE") -> ObjectStore:
    """Create the configured object store from environment variables."""

    mode = os.getenv(prefix, os.getenv("VRS_OBJECT_STORE", "local-pvc")).lower()
    if mode in {"local", "local-pvc", "filesystem"}:
        root_dir = os.getenv("VRS_OBJECT_STORE_ROOT", os.getenv("VRS_RUNS_DIR", "/data/runs"))
        return LocalObjectStore(root_dir)
    if mode in {"seaweedfs", "s3", "external"}:
        bucket = os.environ["VRS_OBJECT_STORE_BUCKET"]
        return S3CompatibleObjectStore(
            S3CompatibleConfig(
                bucket=bucket,
                endpoint_url=os.getenv("VRS_OBJECT_STORE_ENDPOINT"),
                region=os.getenv("VRS_OBJECT_STORE_REGION"),
                prefix=os.getenv("VRS_OBJECT_STORE_PREFIX", ""),
                force_path_style=os.getenv("VRS_OBJECT_STORE_FORCE_PATH_STYLE", "true").lower()
                in {"1", "true", "yes"},
                access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
        )
    raise ValueError(f"unsupported VRS_OBJECT_STORE mode: {mode!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
