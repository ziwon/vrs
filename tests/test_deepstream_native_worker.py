from pathlib import Path

ROOT = Path("native/deepstream")


def test_native_deepstream_worker_sources_are_present() -> None:
    assert (ROOT / "CMakeLists.txt").exists()
    assert (ROOT / "src" / "main.cpp").exists()
    assert Path("Dockerfile.deepstream").exists()


def test_native_deepstream_worker_reads_nvds_metadata_and_emits_detection_contract() -> None:
    source = (ROOT / "src" / "main.cpp").read_text(encoding="utf-8")

    assert "gst_buffer_get_nvds_batch_meta" in source
    assert "NvDsFrameMeta" in source
    assert "NvDsObjectMeta" in source
    assert "schema_version" in source
    assert "detection.v1" in source
    assert "source_runtime" in source
    assert "deepstream" in source
    assert "candidate_alert.v1" not in source
    assert "verified_alert.v1" not in source


def test_deepstream_dockerfile_targets_ds8_container() -> None:
    dockerfile = Path("Dockerfile.deepstream").read_text(encoding="utf-8")

    assert "nvcr.io/nvidia/deepstream:8.0-triton-multiarch" in dockerfile
    assert "cmake --build build/deepstream" in dockerfile
    assert "vrs-deepstream-worker" in dockerfile
