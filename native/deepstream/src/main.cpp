#include <gst/gst.h>

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "metadata_core.hpp"

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
  bool disable_probe = false;
  bool append = false;
  double bbox_scale_x = 1.0;
  double bbox_scale_y = 1.0;
  double bbox_offset_x = 0.0;
  double bbox_offset_y = 0.0;
};

struct Context {
  Options options;
  std::ofstream out;
  vrs::deepstream::LabelMap labels;
};

void print_usage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0
      << " --pipeline PIPELINE [--out detections.jsonl | --disable-probe] [options]\n\n"
      << "Options:\n"
      << "  --stream-id ID          stream_id written to detection.v1 records\n"
      << "  --source-id ID          optional source_id written to detection.v1 records\n"
      << "  --detector-id ID        detector_id written to detection.v1 records\n"
      << "  --probe-element NAME    named GStreamer element to probe (default: sink)\n"
      << "  --probe-pad NAME        pad name on probe element (default: sink)\n"
      << "  --labels FILE           newline-delimited class labels fallback\n"
      << "  --bbox-scale-x VALUE    scale object bbox x coordinates before writing\n"
      << "  --bbox-scale-y VALUE    scale object bbox y coordinates before writing\n"
      << "  --bbox-offset-x VALUE   add to object bbox x coordinates before scaling\n"
      << "  --bbox-offset-y VALUE   add to object bbox y coordinates before scaling\n"
      << "  --disable-probe         run pipeline without worker-owned metadata export\n"
      << "  --append                append to output JSONL instead of truncating\n"
      << "  --print-example-pipeline\n";
}

std::string example_pipeline() {
  // The muxer is square and matches the detector input (640x640) with
  // enable-padding=1 so any source aspect ratio is letterboxed once, preserving
  // geometry. A non-square muxer (e.g. 1280x720) stretches non-16:9 sources and
  // breaks detector parity with the Ultralytics letterbox path. See
  // docs/benchmarks/deepstream-ds8-yoloe-validation-2026-06-30.md.
  return "filesrc location=/data/vrs/sample.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! "
         "m.sink_0 nvstreammux name=m batch-size=1 width=640 height=640 enable-padding=1 "
         "live-source=0 ! "
         "nvinfer config-file-path=/opt/vrs/share/deepstream/configs/pgie-yoloe-safety.txt ! "
         "nvtracker ll-lib-file=/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so ! "
         "vrsmeta stream-id=deepstream-stream detector-id=ds8-yoloe "
         "labels=/opt/vrs/share/deepstream/configs/yoloe-safety-labels.txt "
         "output-path=/tmp/vrs/deepstream_detections.jsonl ! "
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
    } else if (arg == "--bbox-scale-x") {
      options.bbox_scale_x = std::stod(require_value(arg));
    } else if (arg == "--bbox-scale-y") {
      options.bbox_scale_y = std::stod(require_value(arg));
    } else if (arg == "--bbox-offset-x") {
      options.bbox_offset_x = std::stod(require_value(arg));
    } else if (arg == "--bbox-offset-y") {
      options.bbox_offset_y = std::stod(require_value(arg));
    } else if (arg == "--disable-probe") {
      options.disable_probe = true;
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
  if (!options.disable_probe && options.out_path.empty()) {
    throw std::invalid_argument("--out is required");
  }
  return options;
}

double pts_seconds(const NvDsFrameMeta *frame_meta) {
  if (frame_meta == nullptr || frame_meta->buf_pts == GST_CLOCK_TIME_NONE) {
    return 0.0;
  }
  return static_cast<double>(frame_meta->buf_pts) / static_cast<double>(GST_SECOND);
}

NvDsBatchMeta *get_batch_meta(GstBuffer *buffer) {
  NvDsMeta *meta = gst_buffer_get_nvds_meta(buffer);
  if (meta == nullptr || meta->meta_type != NVDS_BATCH_GST_META) {
    return nullptr;
  }
  return static_cast<NvDsBatchMeta *>(meta->meta_data);
}

vrs::deepstream::RuntimeConfig runtime_config_from(const Options &options) {
  vrs::deepstream::RuntimeConfig config;
  config.stream_id = options.stream_id;
  config.source_id = options.source_id;
  config.detector_id = options.detector_id;
  config.bbox.scale_x = options.bbox_scale_x;
  config.bbox.scale_y = options.bbox_scale_y;
  config.bbox.offset_x = options.bbox_offset_x;
  config.bbox.offset_y = options.bbox_offset_y;
  return config;
}

void write_detection(const NvDsFrameMeta *frame_meta, const NvDsObjectMeta *obj_meta, Context *ctx) {
  const auto &rect = obj_meta->rect_params;
  vrs::deepstream::RawObject object;
  object.frame_index = static_cast<int>(frame_meta->frame_num);
  object.pts_s = pts_seconds(frame_meta);
  object.class_id = static_cast<int>(obj_meta->class_id);
  object.object_label = obj_meta->obj_label[0] != '\0' ? obj_meta->obj_label : "";
  object.confidence = static_cast<double>(obj_meta->confidence);
  object.left = static_cast<double>(rect.left);
  object.top = static_cast<double>(rect.top);
  object.width = static_cast<double>(rect.width);
  object.height = static_cast<double>(rect.height);
  object.track_id =
      obj_meta->object_id == UNTRACKED_OBJECT_ID ? "" : std::to_string(obj_meta->object_id);

  ctx->out << vrs::deepstream::detection_jsonl(
      runtime_config_from(ctx->options),
      object,
      ctx->labels);
}

GstPadProbeReturn metadata_probe(GstPad *, GstPadProbeInfo *info, gpointer user_data) {
  auto *ctx = static_cast<Context *>(user_data);
  auto *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (buffer == nullptr) {
    return GST_PAD_PROBE_OK;
  }
  NvDsBatchMeta *batch_meta = get_batch_meta(buffer);
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
  if (!options.disable_probe) {
    ctx.labels = vrs::deepstream::load_labels(options.labels_path);
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
  if (!options.disable_probe) {
    install_probe(pipeline, &ctx);
  }

  GstBus *bus = gst_element_get_bus(pipeline);
  const GstStateChangeReturn state_change = gst_element_set_state(pipeline, GST_STATE_PLAYING);
  if (state_change == GST_STATE_CHANGE_FAILURE) {
    gst_object_unref(bus);
    gst_object_unref(pipeline);
    throw std::runtime_error("failed to set pipeline to PLAYING");
  }

  bool done = false;
  bool had_error = false;
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
        had_error = true;
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
  return had_error ? 1 : 0;
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
