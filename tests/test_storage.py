import json
from pathlib import Path

import pytest

from vrs.storage import (
    LocalObjectStore,
    S3CompatibleConfig,
    S3CompatibleObjectStore,
    object_store_from_env,
)


def test_local_object_store_writes_json_and_returns_file_uri(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)

    obj = store.put_json("manifests/object_manifest.json", {"schema_version": "object_manifest.v1"})

    path = tmp_path / "manifests" / "object_manifest.json"
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == "object_manifest.v1"
    assert obj.uri == path.resolve().as_uri()
    assert obj.key == "manifests/object_manifest.json"
    assert obj.media_type == "application/json"
    assert obj.size_bytes == path.stat().st_size
    assert obj.sha256 is not None
    assert obj.created_at is not None
    assert obj.to_evidence_ref(kind="metadata_manifest")["created_at"] == obj.created_at


def test_local_object_store_rejects_path_escape(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)

    with pytest.raises(ValueError, match="relative"):
        store.put_bytes("../escape", b"nope")


def test_s3_compatible_config_builds_uri_without_client_dependency() -> None:
    cfg = S3CompatibleConfig(
        bucket="vrs-evidence",
        endpoint_url="http://seaweedfs:8333",
        prefix="edge/site-a",
    )

    assert cfg.uri_for_key("runs/1/object_manifest.json") == (
        "s3://vrs-evidence/edge/site-a/runs/1/object_manifest.json"
    )
    assert cfg.endpoint_url == "http://seaweedfs:8333"


def test_s3_compatible_object_store_puts_object_with_injected_client() -> None:
    class Client:
        def __init__(self) -> None:
            self.calls = []

        def put_object(self, **kwargs):
            self.calls.append(kwargs)

    client = Client()
    store = S3CompatibleObjectStore(
        S3CompatibleConfig(bucket="vrs-evidence", prefix="edge/site-a"),
        client=client,
    )

    obj = store.put_json("manifests/object_manifest.json", {"schema_version": "object_manifest.v1"})

    assert client.calls == [
        {
            "Bucket": "vrs-evidence",
            "Key": "edge/site-a/manifests/object_manifest.json",
            "Body": b'{\n  "schema_version": "object_manifest.v1"\n}\n',
            "ContentType": "application/json",
        }
    ]
    assert obj.uri == "s3://vrs-evidence/edge/site-a/manifests/object_manifest.json"
    assert obj.key == "edge/site-a/manifests/object_manifest.json"
    assert obj.size_bytes == len(client.calls[0]["Body"])
    assert obj.sha256 is not None


def test_object_store_from_env_builds_seaweedfs_store(monkeypatch) -> None:
    monkeypatch.setenv("VRS_OBJECT_STORE", "seaweedfs")
    monkeypatch.setenv("VRS_OBJECT_STORE_ENDPOINT", "http://seaweedfs:8333")
    monkeypatch.setenv("VRS_OBJECT_STORE_BUCKET", "vrs-evidence")
    monkeypatch.setenv("VRS_OBJECT_STORE_PREFIX", "edge/site-a")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "vrs")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    store = object_store_from_env()

    assert isinstance(store, S3CompatibleObjectStore)
    assert store.config.bucket == "vrs-evidence"
    assert store.config.endpoint_url == "http://seaweedfs:8333"
    assert store.config.prefix == "edge/site-a"
