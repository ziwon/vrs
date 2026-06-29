import importlib
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

CHART_DIR = Path("charts/vrs")


def test_helm_chart_has_required_profiles_and_components() -> None:
    assert (CHART_DIR / "Chart.yaml").exists()
    for name in (
        "values.yaml",
        "values-dev.yaml",
        "values-edge.yaml",
        "values-prod.yaml",
        "values-kind.yaml",
    ):
        values = yaml.safe_load((CHART_DIR / name).read_text(encoding="utf-8"))
        assert values["profile"]

    templates = {path.name for path in (CHART_DIR / "templates").glob("*.yaml")}
    assert {
        "api-deployment.yaml",
        "metadata-adapter-deployment.yaml",
        "verifier-worker-deployment.yaml",
        "redis.yaml",
        "object-storage.yaml",
        "metrics-service.yaml",
    }.issubset(templates)


def test_helm_profiles_keep_gpu_roles_explicit() -> None:
    default = yaml.safe_load((CHART_DIR / "values.yaml").read_text(encoding="utf-8"))
    edge = yaml.safe_load((CHART_DIR / "values-edge.yaml").read_text(encoding="utf-8"))
    prod = yaml.safe_load((CHART_DIR / "values-prod.yaml").read_text(encoding="utf-8"))
    kind = yaml.safe_load((CHART_DIR / "values-kind.yaml").read_text(encoding="utf-8"))

    assert "vrs.deepstream.worker" in default["deepstreamWorker"]["command"]
    assert default["verifierWorker"]["enabled"] is False
    assert edge["deepstreamWorker"]["gpuRole"] == "deepstream"
    assert edge["verifierWorker"]["gpuRole"] == "verifier"
    assert edge["verifierWorker"]["enabled"] is False
    assert prod["objectStorage"]["mode"] == "seaweedfs"
    assert prod["deepstreamWorker"]["image"]["repository"] == "vrs-deepstream"
    assert prod["deepstreamWorker"]["command"] == ["/opt/vrs/bin/vrs-deepstream-worker"]
    assert "--pipeline" in prod["deepstreamWorker"]["args"]
    assert prod["metrics"]["serviceMonitor"]["enabled"] is True
    assert prod["sampleMetadata"]["enabled"] is False
    assert kind["deepstreamWorker"]["gpuRole"] == "none"
    assert kind["deepstreamWorker"]["resources"]["limits"] == {}
    assert kind["sampleMetadata"]["enabled"] is True


def test_enabled_chart_commands_reference_importable_modules() -> None:
    values = yaml.safe_load((CHART_DIR / "values.yaml").read_text(encoding="utf-8"))
    if values["deepstreamWorker"]["enabled"]:
        command = values["deepstreamWorker"]["command"]
        module = command[command.index("-m") + 1]
        importlib.import_module(module)
    assert values["verifierWorker"]["enabled"] is False


def test_helm_template_renders_storage_mounts_and_sample_metadata() -> None:
    if shutil.which("helm") is None:
        pytest.skip("helm binary is not installed")

    rendered = subprocess.run(
        ["helm", "template", "test", str(CHART_DIR)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    docs = [doc for doc in yaml.safe_load_all(rendered) if isinstance(doc, dict)]

    api = _deployment(docs, "test-vrs-api")
    adapter = _deployment(docs, "test-vrs-metadata-adapter")
    redis = _deployment(docs, "test-vrs-redis")

    assert _container(api, "api")["volumeMounts"][0]["mountPath"] == "/data"
    assert any(
        vol.get("persistentVolumeClaim", {}).get("claimName") == "test-vrs-evidence"
        for vol in api["spec"]["template"]["spec"]["volumes"]
    )
    adapter_container = _container(adapter, "metadata-adapter")
    assert adapter["spec"]["template"]["metadata"]["labels"]["vrs.ai/gpu-role"] == "deepstream"
    assert adapter_container["resources"]["limits"]["nvidia.com/gpu"] == 1
    adapter_mounts = adapter_container["volumeMounts"]
    assert any(mount["mountPath"] == "/data" for mount in adapter_mounts)
    assert any(mount["mountPath"] == "/data/deepstream/metadata.jsonl" for mount in adapter_mounts)
    assert any(
        vol["name"] == "sample-metadata" for vol in adapter["spec"]["template"]["spec"]["volumes"]
    )
    assert _container(redis, "redis")["volumeMounts"][0]["mountPath"] == "/data"
    assert any(
        doc["kind"] == "PersistentVolumeClaim" and doc["metadata"]["name"] == "test-vrs-redis"
        for doc in docs
    )
    assert not any(
        doc["kind"] == "Deployment" and doc["metadata"]["name"] == "test-vrs-verifier"
        for doc in docs
    )


def test_helm_template_prod_renders_seaweedfs_storage() -> None:
    if shutil.which("helm") is None:
        pytest.skip("helm binary is not installed")

    rendered = subprocess.run(
        ["helm", "template", "test", str(CHART_DIR), "-f", str(CHART_DIR / "values-prod.yaml")],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    docs = [doc for doc in yaml.safe_load_all(rendered) if isinstance(doc, dict)]

    seaweedfs = _deployment(docs, "test-vrs-seaweedfs")
    api = _deployment(docs, "test-vrs-api")
    adapter = _deployment(docs, "test-vrs-metadata-adapter")
    adapter_container = _container(adapter, "metadata-adapter")
    api_container = _container(api, "api")

    assert _container(seaweedfs, "seaweedfs")["volumeMounts"][0]["mountPath"] == "/data"
    assert any(
        doc["kind"] == "PersistentVolumeClaim" and doc["metadata"]["name"] == "test-vrs-seaweedfs"
        for doc in docs
    )
    assert any(
        doc["kind"] == "Secret" and doc["metadata"]["name"] == "test-vrs-seaweedfs" for doc in docs
    )
    secret = next(
        doc
        for doc in docs
        if doc["kind"] == "Secret" and doc["metadata"]["name"] == "test-vrs-seaweedfs"
    )
    assert "accessKey" in secret["stringData"]
    assert "secretKey" in secret["stringData"]
    adapter_env = _env_by_name(adapter_container)
    api_env = _env_by_name(api_container)
    assert adapter_container["image"] == "vrs-deepstream:ds8"
    assert adapter_container["command"] == ["/opt/vrs/bin/vrs-deepstream-worker"]
    assert "--pipeline" in adapter_container["args"]
    assert adapter_env["VRS_OBJECT_STORE"]["value"] == "seaweedfs"
    assert adapter_env["VRS_OBJECT_STORE_ENDPOINT"]["value"] == "http://test-vrs-seaweedfs:8333"
    assert adapter_env["VRS_OBJECT_STORE_BUCKET"]["value"] == "vrs-evidence"
    assert "valueFrom" in adapter_env["AWS_ACCESS_KEY_ID"]
    assert api_env["VRS_OBJECT_STORE"]["value"] == "seaweedfs"
    assert not any(
        vol.get("emptyDir") == {}
        for vol in adapter["spec"]["template"]["spec"].get("volumes") or []
    )
    assert not any(
        doc["kind"] == "ConfigMap" and doc["metadata"]["name"] == "test-vrs-sample-metadata"
        for doc in docs
    )
    assert any(
        doc["kind"] == "ServiceMonitor" and doc["metadata"]["name"] == "test-vrs" for doc in docs
    )


def test_helm_template_kind_renders_without_gpu_request() -> None:
    if shutil.which("helm") is None:
        pytest.skip("helm binary is not installed")

    rendered = subprocess.run(
        ["helm", "template", "test", str(CHART_DIR), "-f", str(CHART_DIR / "values-kind.yaml")],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    docs = [doc for doc in yaml.safe_load_all(rendered) if isinstance(doc, dict)]
    adapter = _deployment(docs, "test-vrs-metadata-adapter")
    adapter_container = _container(adapter, "metadata-adapter")

    assert adapter["spec"]["template"]["metadata"]["labels"]["vrs.ai/gpu-role"] == "none"
    assert "nvidia.com/gpu" not in adapter_container.get("resources", {}).get("limits", {})
    assert any(
        doc["kind"] == "ConfigMap" and doc["metadata"]["name"] == "test-vrs-sample-metadata"
        for doc in docs
    )


def _deployment(docs: list[dict], name: str) -> dict:
    return next(
        doc for doc in docs if doc.get("kind") == "Deployment" and doc["metadata"]["name"] == name
    )


def _container(deployment: dict, name: str) -> dict:
    return next(
        item
        for item in deployment["spec"]["template"]["spec"]["containers"]
        if item["name"] == name
    )


def _env_by_name(container: dict) -> dict:
    return {item["name"]: item for item in container.get("env", [])}
