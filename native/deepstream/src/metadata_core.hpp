#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace vrs::deepstream {

using LabelMap = std::unordered_map<int, std::string>;

struct BboxTransform {
  double scale_x = 1.0;
  double scale_y = 1.0;
  double offset_x = 0.0;
  double offset_y = 0.0;
};

struct RuntimeConfig {
  std::string stream_id = "deepstream-stream";
  std::string source_id;
  std::string detector_id = "deepstream-nvinfer";
  BboxTransform bbox;
};

struct RawObject {
  int frame_index = 0;
  double pts_s = 0.0;
  int class_id = 0;
  std::string object_label;
  double confidence = 0.0;
  double left = 0.0;
  double top = 0.0;
  double width = 0.0;
  double height = 0.0;
  std::string track_id;
};

struct BboxXyxy {
  double left = 0.0;
  double top = 0.0;
  double right = 0.0;
  double bottom = 0.0;
};

LabelMap load_labels(const std::string &path);
std::string class_name_for(const RawObject &object, const LabelMap &labels);
BboxXyxy transform_bbox(const RawObject &object, const BboxTransform &transform);
std::string detection_id_for(
    const RuntimeConfig &config,
    const RawObject &object,
    const std::string &class_name,
    const BboxXyxy &bbox);
std::string detection_jsonl(
    const RuntimeConfig &config,
    const RawObject &object,
    const LabelMap &labels);

}  // namespace vrs::deepstream
