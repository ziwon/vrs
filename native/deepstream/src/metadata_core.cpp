#include "metadata_core.hpp"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vrs::deepstream {
namespace {

std::string json_escape(const std::string &value) {
  std::ostringstream out;
  for (unsigned char c : value) {
    switch (c) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c);
        } else {
          out << c;
        }
    }
  }
  return out.str();
}

uint64_t fnv1a64(const std::string &value) {
  uint64_t hash = 1469598103934665603ULL;
  for (unsigned char c : value) {
    hash ^= static_cast<uint64_t>(c);
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string stable_id(const std::string &prefix, const std::vector<std::string> &parts) {
  std::ostringstream joined;
  for (const auto &part : parts) {
    joined << part << "|";
  }
  std::ostringstream out;
  out << prefix << "_" << std::hex << std::setw(16) << std::setfill('0') << fnv1a64(joined.str());
  return out.str();
}

}  // namespace

LabelMap load_labels(const std::string &path) {
  LabelMap labels;
  if (path.empty()) {
    return labels;
  }
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open labels file: " + path);
  }
  std::string line;
  int idx = 0;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.empty()) {
      labels[idx] = line;
    }
    ++idx;
  }
  return labels;
}

std::string class_name_for(const RawObject &object, const LabelMap &labels) {
  if (!object.object_label.empty()) {
    return object.object_label;
  }
  auto found = labels.find(object.class_id);
  if (found != labels.end()) {
    return found->second;
  }
  return "class_" + std::to_string(object.class_id);
}

BboxXyxy transform_bbox(const RawObject &object, const BboxTransform &transform) {
  return {
      (object.left + transform.offset_x) * transform.scale_x,
      (object.top + transform.offset_y) * transform.scale_y,
      (object.left + object.width + transform.offset_x) * transform.scale_x,
      (object.top + object.height + transform.offset_y) * transform.scale_y,
  };
}

std::string detection_id_for(
    const RuntimeConfig &config,
    const RawObject &object,
    const std::string &class_name,
    const BboxXyxy &bbox) {
  return stable_id(
      "det",
      {
          "deepstream",
          config.stream_id,
          config.source_id,
          std::to_string(object.frame_index),
          std::to_string(object.pts_s),
          class_name,
          object.track_id,
          std::to_string(bbox.left) + "," + std::to_string(bbox.top) + "," +
              std::to_string(bbox.right) + "," + std::to_string(bbox.bottom),
      });
}

std::string detection_jsonl(
    const RuntimeConfig &config,
    const RawObject &object,
    const LabelMap &labels) {
  const std::string class_name = class_name_for(object, labels);
  const std::string raw_label = object.object_label.empty() ? class_name : object.object_label;
  const BboxXyxy bbox = transform_bbox(object, config.bbox);
  const std::string detection_id = detection_id_for(config, object, class_name, bbox);

  std::ostringstream out;
  out << std::fixed << std::setprecision(6);
  out << "{";
  out << "\"schema_version\":\"detection.v1\",";
  out << "\"record_type\":\"detection\",";
  out << "\"detection_id\":\"" << detection_id << "\",";
  out << "\"idempotency_key\":\"" << detection_id << "\",";
  out << "\"class_name\":\"" << json_escape(class_name) << "\",";
  out << "\"score\":" << object.confidence << ",";
  out << "\"bbox_xyxy\":[" << bbox.left << "," << bbox.top << "," << bbox.right << ","
      << bbox.bottom << "],";
  out << "\"raw_label\":\"" << json_escape(raw_label) << "\",";
  if (object.track_id.empty()) {
    out << "\"track_id\":null,";
  } else {
    out << "\"track_id\":" << object.track_id << ",";
  }
  out << "\"source_runtime\":\"deepstream\",";
  out << "\"observed_at\":\"stream:" << object.pts_s << "\",";
  out << "\"evidence_refs\":[],";
  out << "\"stream_id\":\"" << json_escape(config.stream_id) << "\",";
  if (!config.source_id.empty()) {
    out << "\"source_id\":\"" << json_escape(config.source_id) << "\",";
  }
  out << "\"frame_index\":" << object.frame_index << ",";
  out << "\"pts_s\":" << object.pts_s << ",";
  out << "\"detector_id\":\"" << json_escape(config.detector_id) << "\"";
  out << "}\n";
  return out.str();
}

}  // namespace vrs::deepstream
