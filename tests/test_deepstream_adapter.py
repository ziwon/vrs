from vrs.deepstream import DeepStreamDetectionMetadata, detection_from_deepstream


def test_deepstream_metadata_adapts_to_detection_contract() -> None:
    evidence = {
        "schema_version": "evidence_ref.v1",
        "uri": "s3://vrs/run-1/cam-1/frame.jpg",
        "kind": "frame",
        "media_type": "image/jpeg",
        "created_at": "2026-06-30T00:00:00Z",
    }
    record = detection_from_deepstream(
        DeepStreamDetectionMetadata(
            stream_id="cam-1",
            clip_id="clip-a",
            frame_index=12,
            pts_s=3.0,
            class_name="fire",
            confidence=0.91,
            bbox_xyxy=(1, 2, 30, 40),
            raw_label="fire",
            track_id=7,
            detector_id="deepstream-yoloe-trt",
            evidence_refs=[evidence],
        )
    )

    assert record["schema_version"] == "detection.v1"
    assert record["record_type"] == "detection"
    assert record["source_runtime"] == "deepstream"
    assert record["stream_id"] == "cam-1"
    assert record["clip_id"] == "clip-a"
    assert record["frame_index"] == 12
    assert record["pts_s"] == 3.0
    assert record["bbox_xyxy"] == [1.0, 2.0, 30.0, 40.0]
    assert record["track_id"] == 7
    assert record["detector_id"] == "deepstream-yoloe-trt"
    assert record["evidence_refs"] == [evidence]
