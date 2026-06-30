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
        "values-k3s-gpu.yaml",
        "values-prod.yaml",
        "values-kind.yaml",
    ):
        values = yaml.safe_load((CHART_DIR / name).read_text(encoding="utf-8"))
        assert values["profile"]

    templates = {path.name for path in (CHART_DIR / "templates").glob("*.yaml")}
    assert {
        "api-deployment.yaml",
        "console-configmap.yaml",
        "console-deployment.yaml",
        "console-service.yaml",
        "deepstream-worker-deployment.yaml",
        "verifier-worker-deployment.yaml",
        "redis.yaml",
        "object-storage.yaml",
        "metrics-service.yaml",
    }.issubset(templates)


def test_helm_profiles_keep_gpu_roles_explicit() -> None:
    default = yaml.safe_load((CHART_DIR / "values.yaml").read_text(encoding="utf-8"))
    edge = yaml.safe_load((CHART_DIR / "values-edge.yaml").read_text(encoding="utf-8"))
    k3s_gpu = yaml.safe_load((CHART_DIR / "values-k3s-gpu.yaml").read_text(encoding="utf-8"))
    prod = yaml.safe_load((CHART_DIR / "values-prod.yaml").read_text(encoding="utf-8"))
    kind = yaml.safe_load((CHART_DIR / "values-kind.yaml").read_text(encoding="utf-8"))

    assert "vrs.deepstream.worker" in default["deepstreamWorker"]["command"]
    assert default["console"]["enabled"] is True
    assert default["console"]["image"]["repository"] == "vrs-console"
    assert default["verifierWorker"]["enabled"] is False
    assert edge["deepstreamWorker"]["gpuRole"] == "deepstream"
    assert edge["verifierWorker"]["gpuRole"] == "verifier"
    assert edge["verifierWorker"]["enabled"] is False
    assert k3s_gpu["deepstreamWorker"]["image"]["repository"] == "vrs-deepstream"
    assert k3s_gpu["deepstreamWorker"]["command"] == ["/opt/vrs/bin/vrs-deepstream-worker"]
    assert k3s_gpu["deepstreamWorker"]["publisher"]["enabled"] is True
    assert k3s_gpu["objectStorage"]["mode"] == "seaweedfs"
    assert k3s_gpu["objectStorage"]["seaweedfs"]["enabled"] is True
    assert k3s_gpu["sampleMetadata"]["enabled"] is False
    assert prod["objectStorage"]["mode"] == "seaweedfs"
    assert prod["console"]["replicas"] == 2
    assert prod["deepstreamWorker"]["image"]["repository"] == "vrs-deepstream"
    assert prod["deepstreamWorker"]["command"] == ["/opt/vrs/bin/vrs-deepstream-worker"]
    assert "--pipeline" in prod["deepstreamWorker"]["args"]
    assert prod["deepstreamWorker"]["publisher"]["enabled"] is True
    pipeline = prod["deepstreamWorker"]["args"][
        prod["deepstreamWorker"]["args"].index("--pipeline") + 1
    ]
    assert "width=640 height=640 enable-padding=1" in pipeline
    assert "nvdspreprocess" in pipeline
    assert "preprocess-yoloe-safety.txt" in pipeline
    assert "nvinfer input-tensor-meta=true" in pipeline
    assert "pgie-yoloe-safety-preprocess.txt" in pipeline
    assert "vrsmeta" in pipeline
    assert "output-path=/tmp/vrs/deepstream_detections.jsonl" in pipeline
    assert "pgie.txt" not in pipeline
    assert "--disable-probe" in prod["deepstreamWorker"]["args"]
    assert any(
        mount["mountPath"] == "/models" for mount in prod["deepstreamWorker"]["volumeMounts"]
    )
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
    console = _deployment(docs, "test-vrs-console")
    console_config = _configmap(docs, "test-vrs-console")
    worker = _deployment(docs, "test-vrs-deepstream-worker")
    redis = _deployment(docs, "test-vrs-redis")

    console_container = _container(console, "console")
    assert console_container["image"] == "vrs-console:latest"
    assert console["spec"]["template"]["metadata"]["labels"]["vrs.ai/gpu-role"] == "none"
    assert any(
        mount["mountPath"] == "/etc/nginx/conf.d/default.conf"
        for mount in console_container["volumeMounts"]
    )
    assert "http://test-vrs-api:8000/api/" in console_config["data"]["default.conf"]
    assert 'apiBaseUrl: ""' in console_config["data"]["config.js"]
    assert _service(docs, "test-vrs-console")["spec"]["ports"][0]["port"] == 80
    assert _service(docs, "test-vrs-console")["spec"]["type"] == "ClusterIP"
    assert _service(docs, "test-vrs-api")["spec"]["type"] == "ClusterIP"

    api_container = _container(api, "api")
    api_env = _env_by_name(api_container)
    assert api_container["volumeMounts"][0]["mountPath"] == "/data"
    assert api_env["VRS_RUNS_ROOT"]["value"] == "/data/runs"
    assert api_env["VRS_POLICY_PATH"]["value"] == "/etc/vrs/policy.yaml"
    assert "VRS_RUNS_DIR" not in api_env
    assert any(
        vol.get("persistentVolumeClaim", {}).get("claimName") == "test-vrs-evidence"
        for vol in api["spec"]["template"]["spec"]["volumes"]
    )
    worker_container = _container(worker, "deepstream-worker")
    assert worker["spec"]["template"]["metadata"]["labels"]["vrs.ai/gpu-role"] == "deepstream"
    assert worker_container["resources"]["limits"]["nvidia.com/gpu"] == 1
    worker_mounts = worker_container["volumeMounts"]
    assert any(mount["mountPath"] == "/data" for mount in worker_mounts)
    assert any(mount["mountPath"] == "/data/deepstream/metadata.jsonl" for mount in worker_mounts)
    assert any(
        vol["name"] == "sample-metadata" for vol in worker["spec"]["template"]["spec"]["volumes"]
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


def test_helm_template_renders_platform_storage_and_service_overrides(tmp_path: Path) -> None:
    if shutil.which("helm") is None:
        pytest.skip("helm binary is not installed")

    platform_values = tmp_path / "platform.yaml"
    platform_values.write_text(
        """
storageClassName: local-path
api:
  service:
    type: NodePort
    annotations:
      example.com/exposure: internal
    nodePort: 30080
console:
  service:
    type: LoadBalancer
    annotations:
      example.com/exposure: edge
redis:
  persistence:
    storageClassName: fast-local
""",
        encoding="utf-8",
    )

    rendered = subprocess.run(
        [
            "helm",
            "template",
            "test",
            str(CHART_DIR),
            "-f",
            str(platform_values),
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    docs = [doc for doc in yaml.safe_load_all(rendered) if isinstance(doc, dict)]

    api_service = _service(docs, "test-vrs-api")
    console_service = _service(docs, "test-vrs-console")
    evidence_pvc = _pvc(docs, "test-vrs-evidence")
    redis_pvc = _pvc(docs, "test-vrs-redis")

    assert api_service["spec"]["type"] == "NodePort"
    assert api_service["metadata"]["annotations"]["example.com/exposure"] == "internal"
    assert api_service["spec"]["ports"][0]["nodePort"] == 30080
    assert console_service["spec"]["type"] == "LoadBalancer"
    assert console_service["metadata"]["annotations"]["example.com/exposure"] == "edge"
    assert evidence_pvc["spec"]["storageClassName"] == "local-path"
    assert redis_pvc["spec"]["storageClassName"] == "fast-local"


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
    console = _deployment(docs, "test-vrs-console")
    console_config = _configmap(docs, "test-vrs-console")
    api = _deployment(docs, "test-vrs-api")
    worker = _deployment(docs, "test-vrs-deepstream-worker")
    worker_container = _container(worker, "deepstream-worker")
    publisher_container = _container(worker, "detection-publisher")
    api_container = _container(api, "api")

    assert _container(console, "console")["image"] == "vrs-console:latest"
    assert "http://test-vrs-api:8000/api/" in console_config["data"]["default.conf"]
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
    worker_env = _env_by_name(worker_container)
    api_env = _env_by_name(api_container)
    assert worker_container["image"] == "vrs-deepstream:ds8"
    assert worker_container["command"] == ["/opt/vrs/bin/vrs-deepstream-worker"]
    assert "--pipeline" in worker_container["args"]
    rendered_pipeline = worker_container["args"][worker_container["args"].index("--pipeline") + 1]
    assert "width=640 height=640 enable-padding=1" in rendered_pipeline
    assert "nvdspreprocess config-file=/opt/vrs/share/deepstream/configs/preprocess-yoloe-safety.txt" in rendered_pipeline
    assert "nvinfer input-tensor-meta=true" in rendered_pipeline
    assert "/opt/vrs/share/deepstream/configs/pgie-yoloe-safety-preprocess.txt" in rendered_pipeline
    assert "vrsmeta" in rendered_pipeline
    assert "detector-id=ds8-nvinfer-preprocess" in rendered_pipeline
    assert "output-path=/tmp/vrs/deepstream_detections.jsonl" in rendered_pipeline
    assert "/opt/vrs/share/deepstream/configs/yoloe-safety-labels.txt" in rendered_pipeline
    assert "--disable-probe" in worker_container["args"]
    assert any(mount["mountPath"] == "/models" for mount in worker_container["volumeMounts"])
    assert any(
        vol.get("persistentVolumeClaim", {}).get("claimName") == "vrs-deepstream-models"
        for vol in worker["spec"]["template"]["spec"]["volumes"]
    )
    assert publisher_container["command"] == ["python", "-m", "vrs.deepstream.jsonl_bridge"]
    assert "redis://test-vrs-redis:6379/0" in publisher_container["args"]
    assert any(mount["mountPath"] == "/tmp/vrs" for mount in publisher_container["volumeMounts"])
    assert worker_env["VRS_OBJECT_STORE"]["value"] == "seaweedfs"
    assert worker_env["VRS_OBJECT_STORE_ENDPOINT"]["value"] == "http://test-vrs-seaweedfs:8333"
    assert worker_env["VRS_OBJECT_STORE_BUCKET"]["value"] == "vrs-evidence"
    assert "valueFrom" in worker_env["AWS_ACCESS_KEY_ID"]
    assert api_env["VRS_RUNS_ROOT"]["value"] == "/data/runs"
    assert api_env["VRS_POLICY_PATH"]["value"] == "/etc/vrs/policy.yaml"
    assert api_env["VRS_OBJECT_STORE"]["value"] == "seaweedfs"
    assert "VRS_RUNS_DIR" not in api_env
    empty_dirs = [
        vol["name"]
        for vol in worker["spec"]["template"]["spec"].get("volumes") or []
        if vol.get("emptyDir") == {}
    ]
    assert empty_dirs == ["deepstream-output"]
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
    console = _deployment(docs, "test-vrs-console")
    worker = _deployment(docs, "test-vrs-deepstream-worker")
    worker_container = _container(worker, "deepstream-worker")

    assert _container(console, "console")["image"] == "vrs-console:latest"
    assert worker["spec"]["template"]["metadata"]["labels"]["vrs.ai/gpu-role"] == "none"
    assert "nvidia.com/gpu" not in worker_container.get("resources", {}).get("limits", {})
    assert any(
        doc["kind"] == "ConfigMap" and doc["metadata"]["name"] == "test-vrs-sample-metadata"
        for doc in docs
    )


def test_helm_template_k3s_gpu_renders_native_deepstream_worker(tmp_path: Path) -> None:
    if shutil.which("helm") is None:
        pytest.skip("helm binary is not installed")

    scheduling_values = tmp_path / "scheduling.yaml"
    scheduling_values.write_text(
        """
deepstreamWorker:
  runtimeClassName: nvidia
  nodeSelector:
    vrs.ai/gpu-node: "true"
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
""",
        encoding="utf-8",
    )

    rendered = subprocess.run(
        [
            "helm",
            "template",
            "test",
            str(CHART_DIR),
            "-f",
            str(CHART_DIR / "values-k3s-gpu.yaml"),
            "-f",
            str(scheduling_values),
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    docs = [doc for doc in yaml.safe_load_all(rendered) if isinstance(doc, dict)]
    worker = _deployment(docs, "test-vrs-deepstream-worker")
    worker_spec = worker["spec"]["template"]["spec"]
    worker_container = _container(worker, "deepstream-worker")
    publisher_container = _container(worker, "detection-publisher")

    assert worker_spec["runtimeClassName"] == "nvidia"
    assert worker_spec["nodeSelector"] == {"vrs.ai/gpu-node": "true"}
    assert worker_spec["tolerations"][0]["key"] == "nvidia.com/gpu"
    assert worker_container["image"] == "vrs-deepstream:ds8"
    assert worker_container["command"] == ["/opt/vrs/bin/vrs-deepstream-worker"]
    assert worker_container["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert "--pipeline" in worker_container["args"]
    rendered_pipeline = worker_container["args"][worker_container["args"].index("--pipeline") + 1]
    assert "stream-id=k3s-gpu-smoke" in rendered_pipeline
    assert "output-path=/tmp/vrs/deepstream_detections.jsonl" in rendered_pipeline
    assert publisher_container["command"] == ["python", "-m", "vrs.deepstream.jsonl_bridge"]
    assert any(
        vol.get("hostPath", {}).get("path") == "/data/vrs"
        for vol in worker_spec["volumes"]
    )
    assert not any(
        doc["kind"] == "ConfigMap" and doc["metadata"]["name"] == "test-vrs-sample-metadata"
        for doc in docs
    )


def _deployment(docs: list[dict], name: str) -> dict:
    return next(
        doc for doc in docs if doc.get("kind") == "Deployment" and doc["metadata"]["name"] == name
    )


def _service(docs: list[dict], name: str) -> dict:
    return next(
        doc for doc in docs if doc.get("kind") == "Service" and doc["metadata"]["name"] == name
    )


def _configmap(docs: list[dict], name: str) -> dict:
    return next(
        doc for doc in docs if doc.get("kind") == "ConfigMap" and doc["metadata"]["name"] == name
    )


def _pvc(docs: list[dict], name: str) -> dict:
    return next(
        doc
        for doc in docs
        if doc.get("kind") == "PersistentVolumeClaim" and doc["metadata"]["name"] == name
    )


def _container(deployment: dict, name: str) -> dict:
    return next(
        item
        for item in deployment["spec"]["template"]["spec"]["containers"]
        if item["name"] == name
    )


def _env_by_name(container: dict) -> dict:
    return {item["name"]: item for item in container.get("env", [])}
