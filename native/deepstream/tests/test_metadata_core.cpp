#include "metadata_core.hpp"

#include <cassert>
#include <string>

int main() {
  vrs::deepstream::RuntimeConfig config;
  config.stream_id = "cam-01";
  config.source_id = "source-a";
  config.detector_id = "ds8-yoloe";
  config.bbox.offset_y = -80.0;
  config.bbox.scale_x = 0.5;
  config.bbox.scale_y = 0.5;

  vrs::deepstream::RawObject object;
  object.frame_index = 42;
  object.pts_s = 1.25;
  object.class_id = 1;
  object.confidence = 0.875;
  object.left = 20.0;
  object.top = 140.0;
  object.width = 80.0;
  object.height = 80.0;
  object.track_id = "7";

  vrs::deepstream::LabelMap labels = {{1, "billowing smoke"}};
  const std::string row = vrs::deepstream::detection_jsonl(config, object, labels);

  assert(row.find("\"schema_version\":\"detection.v1\"") != std::string::npos);
  assert(row.find("\"class_name\":\"billowing smoke\"") != std::string::npos);
  assert(row.find("\"bbox_xyxy\":[10.000000,30.000000,50.000000,70.000000]") != std::string::npos);
  assert(row.find("\"track_id\":7") != std::string::npos);
  assert(row.find("\"source_id\":\"source-a\"") != std::string::npos);
  assert(row.find("\"detector_id\":\"ds8-yoloe\"") != std::string::npos);
  return 0;
}
