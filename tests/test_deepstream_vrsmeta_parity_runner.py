import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path("scripts/run_deepstream_vrsmeta_parity.py")
_SPEC = importlib.util.spec_from_file_location("run_deepstream_vrsmeta_parity", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

VideoInfo = _MODULE.VideoInfo
compare_command = _MODULE.compare_command
deepstream_pipeline = _MODULE.deepstream_pipeline
source_decode_chain = _MODULE.source_decode_chain


def test_deepstream_pipeline_uses_vrsmeta_and_disable_probe_output() -> None:
    pipeline = deepstream_pipeline(
        source="/clips/120.mp4",
        output="/out/deepstream_120.jsonl",
        stream_id="fire120",
        detector_id="ds8-yoloe-vrsmeta",
        mux_width=640,
        mux_height=640,
        pgie_config="/opt/vrs/share/deepstream/configs/pgie-yoloe-safety.txt",
        labels="/opt/vrs/share/deepstream/configs/yoloe-safety-labels.txt",
    )

    assert "qtdemux ! h264parse ! nvv4l2decoder" in pipeline
    assert "width=640 height=640 enable-padding=1" in pipeline
    assert "nvinfer config-file-path=/opt/vrs/share/deepstream/configs/pgie-yoloe-safety.txt" in pipeline
    assert "nvtracker" in pipeline
    assert "vrsmeta stream-id=fire120 detector-id=ds8-yoloe-vrsmeta" in pipeline
    assert "output-path=/out/deepstream_120.jsonl" in pipeline


def test_source_decode_chain_supports_mp4_and_avi() -> None:
    assert source_decode_chain("/clips/a.mp4").endswith("qtdemux ! h264parse ! nvv4l2decoder")
    assert "avidemux ! decodebin" in source_decode_chain("/clips/a.avi")


def test_compare_command_enables_auto_letterbox() -> None:
    cmd = compare_command(
        python_jsonl=Path("python.jsonl"),
        candidate_jsonl=Path("ds.jsonl"),
        out=Path("parity.json"),
        policy="configs/policies/safety.yaml",
        video=VideoInfo(width=640, height=360, fps="24/1", frames=132, duration_s=5.5),
        mux_width=640,
        mux_height=640,
        time_tolerance_s=0.08,
        iou_threshold=0.5,
    )

    assert "scripts/compare_aligned_detector_parity.py" in cmd
    assert "--auto-candidate-letterbox" in cmd
    assert cmd[cmd.index("--source-width") + 1] == "640"
    assert cmd[cmd.index("--source-height") + 1] == "360"
    assert cmd[cmd.index("--candidate-mux-width") + 1] == "640"
    assert cmd[cmd.index("--candidate-mux-height") + 1] == "640"
