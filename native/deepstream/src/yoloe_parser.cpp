#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {

struct Candidate {
  NvDsInferObjectDetectionInfo object;
  float x1;
  float y1;
  float x2;
  float y2;
};

float clamp(float value, float lo, float hi) {
  return std::max(lo, std::min(value, hi));
}

float iou(const Candidate &a, const Candidate &b) {
  const float xx1 = std::max(a.x1, b.x1);
  const float yy1 = std::max(a.y1, b.y1);
  const float xx2 = std::min(a.x2, b.x2);
  const float yy2 = std::min(a.y2, b.y2);
  const float w = std::max(0.0f, xx2 - xx1);
  const float h = std::max(0.0f, yy2 - yy1);
  const float inter = w * h;
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  const float denom = area_a + area_b - inter;
  return denom > 0.0f ? inter / denom : 0.0f;
}

unsigned int volume(const NvDsInferDims &dims) {
  unsigned int out = 1;
  for (unsigned int i = 0; i < dims.numDims; ++i) {
    out *= static_cast<unsigned int>(dims.d[i]);
  }
  return out;
}

const NvDsInferLayerInfo *find_detection_layer(
    std::vector<NvDsInferLayerInfo> const &layers,
    unsigned int class_count) {
  const NvDsInferLayerInfo *best = nullptr;
  int best_anchors = 0;
  for (const auto &layer : layers) {
    const unsigned int layer_volume = volume(layer.inferDims);
    if (layer_volume == 0) {
      continue;
    }
    if (layer.inferDims.numDims >= 2) {
      const int d0 = layer.inferDims.d[0];
      const int d1 = layer.inferDims.d[1];
      const int d2 = layer.inferDims.numDims >= 3 ? layer.inferDims.d[2] : 0;
      const bool channel_first_2d = d0 >= static_cast<int>(class_count + 4) && d1 > 1000;
      const bool channel_last_2d = d1 >= static_cast<int>(class_count + 4) && d0 > 1000;
      const bool batched_channel_first =
          d0 == 1 && d1 >= static_cast<int>(class_count + 4) && d2 > 1000;
      const bool looks_like_yolo =
          channel_first_2d || channel_last_2d || batched_channel_first;
      const int anchors = batched_channel_first ? d2 : (channel_first_2d ? d1 : d0);
      if (looks_like_yolo && anchors > best_anchors) {
        best = &layer;
        best_anchors = anchors;
      }
    }
  }
  return best;
}

float threshold_for_class(
    NvDsInferParseDetectionParams const &params,
    unsigned int class_id) {
  if (class_id < params.perClassPreclusterThreshold.size()) {
    return params.perClassPreclusterThreshold[class_id];
  }
  return params.perClassPreclusterThreshold.empty() ? 0.25f : params.perClassPreclusterThreshold[0];
}

void dump_raw_layer_once(
    const NvDsInferLayerInfo &layer,
    unsigned int class_count,
    int channels,
    int anchors,
    bool channel_first) {
  static bool dumped = false;
  if (dumped) {
    return;
  }
  const char *prefix = std::getenv("VRS_YOLOE_RAW_DUMP");
  if (prefix == nullptr || prefix[0] == '\0') {
    return;
  }
  dumped = true;

  const unsigned int layer_volume = volume(layer.inferDims);
  const auto *data = static_cast<const float *>(layer.buffer);
  const std::string base(prefix);
  const std::string bin_path = base + ".f32";
  const std::string json_path = base + ".json";
  const std::string bin_name = std::filesystem::path(bin_path).filename().string();

  std::ofstream bin(bin_path, std::ios::out | std::ios::binary | std::ios::trunc);
  if (!bin) {
    std::cerr << "NvDsInferParseCustomYoloE: failed to open raw dump " << bin_path << "\n";
    return;
  }
  bin.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(
      layer_volume * sizeof(float)));
  bin.close();

  std::ofstream meta(json_path, std::ios::out | std::ios::trunc);
  if (!meta) {
    std::cerr << "NvDsInferParseCustomYoloE: failed to open raw dump metadata " << json_path
              << "\n";
    return;
  }
  meta << "{\n"
       << "  \"schema_version\": \"vrs.deepstream.yoloe_raw_tensor.v1\",\n"
       << "  \"runtime\": \"deepstream-nvinfer\",\n"
       << "  \"layer_name\": \"" << (layer.layerName ? layer.layerName : "") << "\",\n"
       << "  \"dtype\": \"float32\",\n"
       << "  \"binary\": \"" << bin_name << "\",\n"
       << "  \"runtime_binary_path\": \"" << bin_path << "\",\n"
       << "  \"dims\": [";
  for (unsigned int i = 0; i < layer.inferDims.numDims; ++i) {
    if (i != 0) {
      meta << ", ";
    }
    meta << layer.inferDims.d[i];
  }
  meta << "],\n"
       << "  \"volume\": " << layer_volume << ",\n"
       << "  \"class_count\": " << class_count << ",\n"
       << "  \"channels\": " << channels << ",\n"
       << "  \"anchors\": " << anchors << ",\n"
       << "  \"channel_first\": " << (channel_first ? "true" : "false") << "\n"
       << "}\n";
}

}  // namespace

extern "C" bool NvDsInferParseCustomYoloE(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList) {
  static bool debug_once = false;
  const bool debug = std::getenv("VRS_YOLOE_PARSER_DEBUG") != nullptr && !debug_once;
  if (debug) {
    debug_once = true;
    std::cerr << "NvDsInferParseCustomYoloE: layers=" << outputLayersInfo.size()
              << " classes=" << detectionParams.numClassesConfigured << "\n";
    for (std::size_t i = 0; i < outputLayersInfo.size(); ++i) {
      const auto &layer = outputLayersInfo[i];
      std::cerr << "  layer[" << i << "] name=" << (layer.layerName ? layer.layerName : "")
                << " dims=";
      for (unsigned int d = 0; d < layer.inferDims.numDims; ++d) {
        if (d != 0) {
          std::cerr << "x";
        }
        std::cerr << layer.inferDims.d[d];
      }
      std::cerr << "\n";
    }
  }
  objectList.clear();
  const unsigned int class_count = detectionParams.numClassesConfigured;
  if (class_count == 0) {
    return false;
  }

  const NvDsInferLayerInfo *layer = find_detection_layer(outputLayersInfo, class_count);
  if (layer == nullptr || layer->buffer == nullptr || layer->inferDims.numDims < 2) {
    if (debug) {
      std::cerr << "NvDsInferParseCustomYoloE: no detection layer selected\n";
    }
    return false;
  }

  const auto *data = static_cast<const float *>(layer->buffer);
  int channels = layer->inferDims.d[0];
  int anchors = layer->inferDims.d[1];
  bool channel_first = true;
  int offset = 0;
  if (layer->inferDims.numDims >= 3 && layer->inferDims.d[0] == 1) {
    channels = layer->inferDims.d[1];
    anchors = layer->inferDims.d[2];
    offset = 0;
  }
  if (anchors < static_cast<int>(class_count + 4) && channels > 100) {
    std::swap(channels, anchors);
    channel_first = false;
  }
  if (channels < static_cast<int>(class_count + 4) || anchors <= 0) {
    if (debug) {
      std::cerr << "NvDsInferParseCustomYoloE: invalid selected dims channels=" << channels
                << " anchors=" << anchors << "\n";
    }
    return false;
  }

  dump_raw_layer_once(*layer, class_count, channels, anchors, channel_first);

  auto at = [&](int channel, int anchor) -> float {
    if (channel_first) {
      return data[offset + channel * anchors + anchor];
    }
    return data[offset + anchor * channels + channel];
  };

  std::vector<Candidate> candidates;
  candidates.reserve(1024);
  const float net_w = static_cast<float>(networkInfo.width);
  const float net_h = static_cast<float>(networkInfo.height);
  float global_best_score = 0.0f;

  for (int anchor = 0; anchor < anchors; ++anchor) {
    unsigned int best_class = 0;
    float best_score = 0.0f;
    for (unsigned int cls = 0; cls < class_count; ++cls) {
      const float score = at(4 + static_cast<int>(cls), anchor);
      if (score > best_score) {
        best_score = score;
        best_class = cls;
      }
      global_best_score = std::max(global_best_score, score);
    }
    if (best_score < threshold_for_class(detectionParams, best_class)) {
      continue;
    }

    const float cx = at(0, anchor);
    const float cy = at(1, anchor);
    const float w = at(2, anchor);
    const float h = at(3, anchor);
    const float x1 = clamp(cx - w * 0.5f, 0.0f, net_w);
    const float y1 = clamp(cy - h * 0.5f, 0.0f, net_h);
    const float x2 = clamp(cx + w * 0.5f, 0.0f, net_w);
    const float y2 = clamp(cy + h * 0.5f, 0.0f, net_h);
    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    Candidate candidate;
    candidate.x1 = x1;
    candidate.y1 = y1;
    candidate.x2 = x2;
    candidate.y2 = y2;
    candidate.object.classId = best_class;
    candidate.object.left = x1;
    candidate.object.top = y1;
    candidate.object.width = x2 - x1;
    candidate.object.height = y2 - y1;
    candidate.object.detectionConfidence = best_score;
    candidates.push_back(candidate);
  }
  if (debug) {
    std::cerr << "NvDsInferParseCustomYoloE: selected=" << (layer->layerName ? layer->layerName : "")
              << " channels=" << channels << " anchors=" << anchors
              << " global_best_score=" << global_best_score
              << " pre_nms_candidates=" << candidates.size() << "\n";
  }

  std::sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
    return a.object.detectionConfidence > b.object.detectionConfidence;
  });

  constexpr float kNmsIou = 0.50f;
  constexpr std::size_t kMaxObjects = 300;
  std::vector<Candidate> kept;
  kept.reserve(std::min(candidates.size(), kMaxObjects));
  for (const auto &candidate : candidates) {
    bool suppress = false;
    for (const auto &existing : kept) {
      if (candidate.object.classId == existing.object.classId && iou(candidate, existing) > kNmsIou) {
        suppress = true;
        break;
      }
    }
    if (!suppress) {
      kept.push_back(candidate);
      objectList.push_back(candidate.object);
      if (kept.size() >= kMaxObjects) {
        break;
      }
    }
  }
  return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloE);
