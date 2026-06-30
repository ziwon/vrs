from pathlib import Path

ROOT = Path("native/deepstream")


def test_native_deepstream_worker_sources_are_present() -> None:
    assert (ROOT / "CMakeLists.txt").exists()
    assert (ROOT / "src" / "main.cpp").exists()
    assert (ROOT / "src" / "metadata_core.cpp").exists()
    assert (ROOT / "src" / "metadata_core.hpp").exists()
    assert (ROOT / "src" / "gst_vrsmeta.cpp").exists()
    assert (ROOT / "tests" / "test_metadata_core.cpp").exists()
    assert Path("Dockerfile.deepstream").exists()


def test_native_deepstream_worker_reads_nvds_metadata_and_emits_detection_contract() -> None:
    source = (ROOT / "src" / "main.cpp").read_text(encoding="utf-8")
    metadata_core = (ROOT / "src" / "metadata_core.cpp").read_text(encoding="utf-8")

    assert "gst_buffer_get_nvds_meta" in source
    assert "NVDS_BATCH_GST_META" in source
    assert "NvDsFrameMeta" in source
    assert "NvDsObjectMeta" in source
    assert "detection_jsonl" in source
    assert "schema_version" in metadata_core
    assert "detection.v1" in metadata_core
    assert "source_runtime" in metadata_core
    assert "deepstream" in metadata_core
    assert "--bbox-offset-y" in source
    assert "bbox_scale_x" in source
    assert "candidate_alert.v1" not in source
    assert "verified_alert.v1" not in source
    assert "candidate_alert.v1" not in metadata_core
    assert "verified_alert.v1" not in metadata_core


def test_deepstream_dockerfile_targets_ds8_container() -> None:
    dockerfile = Path("Dockerfile.deepstream").read_text(encoding="utf-8")

    assert "nvcr.io/nvidia/deepstream:8.0-triton-multiarch" in dockerfile
    assert "cmake --build build/deepstream" in dockerfile
    assert "vrs-deepstream-worker" in dockerfile
    assert "configs/deepstream" in dockerfile
    assert "libgstvrsmeta.so" in dockerfile
    assert "GST_PLUGIN_PATH" in dockerfile


def test_native_metadata_core_has_cmake_test_target() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "vrs_deepstream_metadata_core" in cmake
    assert "vrs-deepstream-metadata-core-test" in cmake


def test_vrsmeta_plugin_skeleton_is_registered() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    source = (ROOT / "src" / "gst_vrsmeta.cpp").read_text(encoding="utf-8")

    assert "gstreamer-base-1.0" in cmake
    assert "add_library(gstvrsmeta SHARED" in cmake
    assert "nvds_meta" in cmake
    assert "nvdsgst_meta" in cmake
    assert "lib/gstreamer-1.0" in cmake
    assert "G_DEFINE_TYPE" in source
    assert "GST_PLUGIN_DEFINE" in source
    assert "gst_element_register(plugin, \"vrsmeta\"" in source
    assert "gst_base_transform_set_in_place" in source
    assert "GstBaseTransform" in source


def test_vrsmeta_plugin_exports_deepstream_metadata() -> None:
    source = (ROOT / "src" / "gst_vrsmeta.cpp").read_text(encoding="utf-8")

    assert "gst_buffer_get_nvds_meta" in source
    assert "NVDS_BATCH_GST_META" in source
    assert "NvDsFrameMeta" in source
    assert "NvDsObjectMeta" in source
    assert "output-path" in source
    assert "output-mode" in source
    assert "bbox-scale-x" in source
    assert "bbox-offset-y" in source
    assert "detection_jsonl" in source
    assert "GST_FLOW_ERROR" in source
