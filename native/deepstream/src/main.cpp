#include <gst/gst.h>

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" {
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
}

namespace {

std::atomic_bool g_stop(false);

struct Options {
  std::string pipeline;
  std::string out_path;
  std::string stream_id = "deepstream-stream";
  std::string source_id;
  std::string detector_id = "deepstream-nvinfer";
  std::string probe_element = "sink";
  std::string probe_pad = "sink";
  std::string labels_path;
  bool append = false;
};

struct Context {
  Options options;
  std::ofstream out;
  std::unordered_map<int, std::string> labels;
};

void print_usage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0 << " --pipeline PIPELINE --out detections.jsonl [options]\n\n"
      << "Options:\n"
      << "  --stream-id ID          stream_id written to detection.v1 records\n"
      << "  --source-id ID          optional source_id written to detection.v1 records\n"
      << "  --detector-id ID        detector_id written to detection.v1 records\n"
      << "  --probe-element NAME    named GStreamer element to probe (default: sink)\n"
      << "  --probe-pad NAME        pad name on probe element (default: sink)\n"
      << "  --labels FILE           newline-delimited class labels fallback\n"
      << "  --append                append to output JSONL instead of truncating\n"
      << "  --print-example-pipeline\n";
}

std::string example_pipeline() {
  return "filesrc location=/data/vrs/sample.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! "
         "m.sink_0 nvstreammux name=m batch-size=1 width=1280 height=720 live-source=0 ! "
         "nvinfer config-file-path=/etc/vrs/deepstream/pgie.txt ! "
         "nvtracker ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so ! "
         "fakesink name=sink sync=false";
}

Options parse_args(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto require_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(name + " requires a value");
      }
      return argv[++i];
    };
    if (arg == "--pipeline") {
      options.pipeline = require_value(arg);
    } else if (arg == "--out") {
      options.out_path = require_value(arg);
    } else if (arg == "--stream-id") {
      options.stream_id = require_value(arg);
    } else if (arg == "--source-id") {
      options.source_id = require_value(arg);
    } else if (arg == "--detector-id") {
      options.detector_id = require_value(arg);
    } else if (arg == "--probe-element") {
      options.probe_element = require_value(arg);
    } else if (arg == "--probe-pad") {
      options.probe_pad = require_value(arg);
    } else if (arg == "--labels") {
      options.labels_path = require_value(arg);
    } else if (arg == "--append") {
      options.append = true;
    } else if (arg == "--print-example-pipeline") {
      std::cout << example_pipeline() << "\n";
      std::exit(0);
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + arg);
    }
  }
  if (options.pipeline.empty()) {
    throw std::invalid_argument("--pipeline is required");
  }
  if (options.out_path.empty()) {
    throw std::invalid_argument("--out is required");
  }
  return options;
}

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

std::unordered_map<int, std::string> load_labels(const std::string &path) {
  std::unordered_map<int, std::string> labels;
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

double pts_seconds(const NvDsFrameMeta *frame_meta) {
  if (frame_meta == nullptr || frame_meta->buf_pts == GST_CLOCK_TIME_NONE) {
    return 0.0;
  }
  return static_cast<double>(frame_meta->buf_pts) / static_cast<double>(GST_SECOND);
}

std::string class_name_for(const NvDsObjectMeta *obj_meta, const Context *ctx) {
  if (obj_meta->obj_label[0] != '\0') {
    return obj_meta->obj_label;
  }
  auto found = ctx->labels.find(static_cast<int>(obj_meta->class_id));
  if (found != ctx->labels.end()) {
    return found->second;
  }
  return "class_" + std::to_string(obj_meta->class_id);
}

void write_detection(const NvDsFrameMeta *frame_meta, const NvDsObjectMeta *obj_meta, Context *ctx) {
  const double pts_s = pts_seconds(frame_meta);
  const auto &rect = obj_meta->rect_params;
  const double left = static_cast<double>(rect.left);
  const double top = static_cast<double>(rect.top);
  const double right = left + static_cast<double>(rect.width);
  const double bottom = top + static_cast<double>(rect.height);
  const std::string class_name = class_name_for(obj_meta, ctx);
  const std::string raw_label = obj_meta->obj_label[0] != '\0' ? obj_meta->obj_label : class_name;
  const std::string track_id =
      obj_meta->object_id == UNTRACKED_OBJECT_ID ? "" : std::to_string(obj_meta->object_id);
  const std::string detection_id = stable_id(
      "det",
      {
          "deepstream",
          ctx->options.stream_id,
          ctx->options.source_id,
          std::to_string(frame_meta->frame_num),
          std::to_string(pts_s),
          class_name,
          track_id,
          std::to_string(left) + "," + std::to_string(top) + "," + std::to_string(right) + "," +
              std::to_string(bottom),
      });

  ctx->out << std::fixed << std::setprecision(6);
  ctx->out << "{";
  ctx->out << "\"schema_version\":\"detection.v1\",";
  ctx->out << "\"record_type\":\"detection\",";
  ctx->out << "\"detection_id\":\"" << detection_id << "\",";
  ctx->out << "\"idempotency_key\":\"" << detection_id << "\",";
  ctx->out << "\"class_name\":\"" << json_escape(class_name) << "\",";
  ctx->out << "\"score\":" << static_cast<double>(obj_meta->confidence) << ",";
  ctx->out << "\"bbox_xyxy\":[" << left << "," << top << "," << right << "," << bottom << "],";
  ctx->out << "\"raw_label\":\"" << json_escape(raw_label) << "\",";
  if (track_id.empty()) {
    ctx->out << "\"track_id\":null,";
  } else {
    ctx->out << "\"track_id\":" << track_id << ",";
  }
  ctx->out << "\"source_runtime\":\"deepstream\",";
  ctx->out << "\"observed_at\":\"stream:" << pts_s << "\",";
  ctx->out << "\"evidence_refs\":[],";
  ctx->out << "\"stream_id\":\"" << json_escape(ctx->options.stream_id) << "\",";
  if (!ctx->options.source_id.empty()) {
    ctx->out << "\"source_id\":\"" << json_escape(ctx->options.source_id) << "\",";
  }
  ctx->out << "\"frame_index\":" << frame_meta->frame_num << ",";
  ctx->out << "\"pts_s\":" << pts_s << ",";
  ctx->out << "\"detector_id\":\"" << json_escape(ctx->options.detector_id) << "\"";
  ctx->out << "}\n";
}

GstPadProbeReturn metadata_probe(GstPad *, GstPadProbeInfo *info, gpointer user_data) {
  auto *ctx = static_cast<Context *>(user_data);
  auto *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (buffer == nullptr) {
    return GST_PAD_PROBE_OK;
  }
  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buffer);
  if (batch_meta == nullptr) {
    return GST_PAD_PROBE_OK;
  }
  for (NvDsMetaList *frame_node = batch_meta->frame_meta_list; frame_node != nullptr;
       frame_node = frame_node->next) {
    auto *frame_meta = static_cast<NvDsFrameMeta *>(frame_node->data);
    if (frame_meta == nullptr) {
      continue;
    }
    for (NvDsMetaList *obj_node = frame_meta->obj_meta_list; obj_node != nullptr;
         obj_node = obj_node->next) {
      auto *obj_meta = static_cast<NvDsObjectMeta *>(obj_node->data);
      if (obj_meta != nullptr) {
        write_detection(frame_meta, obj_meta, ctx);
      }
    }
  }
  ctx->out.flush();
  return GST_PAD_PROBE_OK;
}

void handle_signal(int) { g_stop.store(true); }

void install_probe(GstElement *pipeline, Context *ctx) {
  GstElement *element = gst_bin_get_by_name(GST_BIN(pipeline), ctx->options.probe_element.c_str());
  if (element == nullptr) {
    throw std::runtime_error("probe element not found: " + ctx->options.probe_element);
  }
  GstPad *pad = gst_element_get_static_pad(element, ctx->options.probe_pad.c_str());
  gst_object_unref(element);
  if (pad == nullptr) {
    throw std::runtime_error("probe pad not found: " + ctx->options.probe_element + "." +
                             ctx->options.probe_pad);
  }
  gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, metadata_probe, ctx, nullptr);
  gst_object_unref(pad);
}

int run(const Options &options) {
  Context ctx;
  ctx.options = options;
  ctx.labels = load_labels(options.labels_path);
  const std::filesystem::path out_path(options.out_path);
  if (out_path.has_parent_path()) {
    std::filesystem::create_directories(out_path.parent_path());
  }
  ctx.out.open(
      options.out_path,
      std::ios::out | (options.append ? std::ios::app : std::ios::trunc));
  if (!ctx.out) {
    throw std::runtime_error("failed to open output file: " + options.out_path);
  }

  GError *error = nullptr;
  GstElement *pipeline = gst_parse_launch(options.pipeline.c_str(), &error);
  if (pipeline == nullptr) {
    std::string msg = error != nullptr ? error->message : "unknown pipeline parse error";
    if (error != nullptr) {
      g_error_free(error);
    }
    throw std::runtime_error("failed to create pipeline: " + msg);
  }
  install_probe(pipeline, &ctx);

  GstBus *bus = gst_element_get_bus(pipeline);
  gst_element_set_state(pipeline, GST_STATE_PLAYING);

  bool done = false;
  while (!done && !g_stop.load()) {
    GstMessage *msg = gst_bus_timed_pop_filtered(
        bus,
        250 * GST_MSECOND,
        static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS | GST_MESSAGE_STATE_CHANGED));
    if (msg == nullptr) {
      continue;
    }
    switch (GST_MESSAGE_TYPE(msg)) {
      case GST_MESSAGE_ERROR: {
        GError *err = nullptr;
        gchar *debug = nullptr;
        gst_message_parse_error(msg, &err, &debug);
        std::cerr << "GStreamer error: " << (err ? err->message : "unknown") << "\n";
        if (debug != nullptr) {
          std::cerr << "Debug details: " << debug << "\n";
        }
        if (err != nullptr) {
          g_error_free(err);
        }
        if (debug != nullptr) {
          g_free(debug);
        }
        done = true;
        break;
      }
      case GST_MESSAGE_EOS:
        done = true;
        break;
      default:
        break;
    }
    gst_message_unref(msg);
  }

  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(bus);
  gst_object_unref(pipeline);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);
  gst_init(&argc, &argv);
  try {
    return run(parse_args(argc, argv));
  } catch (const std::exception &exc) {
    std::cerr << "vrs-deepstream-worker: " << exc.what() << "\n";
    print_usage(argv[0]);
    return 2;
  }
}
