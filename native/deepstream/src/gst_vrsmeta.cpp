#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>

#include <fstream>
#include <exception>
#include <memory>
#include <string>

#include "metadata_core.hpp"

extern "C" {
#include "gstnvdsmeta.h"
#include "nvdsmeta.h"
}

typedef struct _GstVrsMeta {
  GstBaseTransform parent;
  gchar *stream_id;
  gchar *source_id;
  gchar *detector_id;
  gchar *labels_path;
  gchar *output_mode;
  gchar *output_path;
  gboolean append;
  gdouble bbox_scale_x;
  gdouble bbox_scale_y;
  gdouble bbox_offset_x;
  gdouble bbox_offset_y;
  gboolean debug;
  vrs::deepstream::LabelMap *labels;
  std::ofstream *out;
} GstVrsMeta;

typedef struct _GstVrsMetaClass {
  GstBaseTransformClass parent_class;
} GstVrsMetaClass;

#define GST_TYPE_VRSMETA (gst_vrsmeta_get_type())
GType gst_vrsmeta_get_type();

G_DEFINE_TYPE(GstVrsMeta, gst_vrsmeta, GST_TYPE_BASE_TRANSFORM)

enum {
  PROP_0,
  PROP_STREAM_ID,
  PROP_SOURCE_ID,
  PROP_DETECTOR_ID,
  PROP_LABELS,
  PROP_OUTPUT_MODE,
  PROP_OUTPUT_PATH,
  PROP_APPEND,
  PROP_BBOX_SCALE_X,
  PROP_BBOX_SCALE_Y,
  PROP_BBOX_OFFSET_X,
  PROP_BBOX_OFFSET_Y,
  PROP_DEBUG,
};

namespace {

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

vrs::deepstream::RuntimeConfig runtime_config_from(const GstVrsMeta *self) {
  vrs::deepstream::RuntimeConfig config;
  config.stream_id = self->stream_id != nullptr ? self->stream_id : "deepstream-stream";
  config.source_id = self->source_id != nullptr ? self->source_id : "";
  config.detector_id = self->detector_id != nullptr ? self->detector_id : "deepstream-nvinfer";
  config.bbox.scale_x = self->bbox_scale_x;
  config.bbox.scale_y = self->bbox_scale_y;
  config.bbox.offset_x = self->bbox_offset_x;
  config.bbox.offset_y = self->bbox_offset_y;
  return config;
}

bool ensure_labels_loaded(GstVrsMeta *self) {
  if (self->labels != nullptr) {
    return true;
  }
  try {
    self->labels = new vrs::deepstream::LabelMap(
        vrs::deepstream::load_labels(self->labels_path != nullptr ? self->labels_path : ""));
  } catch (const std::exception &exc) {
    GST_ELEMENT_ERROR(
        self,
        RESOURCE,
        READ,
        ("failed to load vrsmeta labels"),
        ("%s", exc.what()));
    return false;
  }
  return true;
}

bool ensure_output_open(GstVrsMeta *self) {
  if (self->output_path == nullptr || self->output_path[0] == '\0') {
    return true;
  }
  if (self->out != nullptr && self->out->is_open()) {
    return true;
  }
  self->out = new std::ofstream(
      self->output_path,
      std::ios::out | (self->append ? std::ios::app : std::ios::trunc));
  if (!*self->out) {
    GST_ELEMENT_ERROR(
        self,
        RESOURCE,
        OPEN_WRITE,
        ("failed to open vrsmeta output"),
        ("%s", self->output_path));
    return false;
  }
  return true;
}

bool write_object(
    GstVrsMeta *self,
    const NvDsFrameMeta *frame_meta,
    const NvDsObjectMeta *obj_meta,
    const vrs::deepstream::RuntimeConfig &config) {
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

  (*self->out) << vrs::deepstream::detection_jsonl(config, object, *self->labels);
  return static_cast<bool>(*self->out);
}

}  // namespace

static GstFlowReturn gst_vrsmeta_transform_ip(GstBaseTransform *base, GstBuffer *buffer) {
  auto *self = reinterpret_cast<GstVrsMeta *>(base);
  if (self->output_path == nullptr || self->output_path[0] == '\0') {
    return GST_FLOW_OK;
  }
  if (g_strcmp0(self->output_mode, "jsonl") != 0) {
    GST_ELEMENT_ERROR(
        self,
        RESOURCE,
        SETTINGS,
        ("unsupported vrsmeta output-mode"),
        ("%s", self->output_mode != nullptr ? self->output_mode : ""));
    return GST_FLOW_ERROR;
  }
  if (!ensure_labels_loaded(self) || !ensure_output_open(self)) {
    return GST_FLOW_ERROR;
  }

  NvDsBatchMeta *batch_meta = get_batch_meta(buffer);
  if (batch_meta == nullptr) {
    return GST_FLOW_OK;
  }

  const vrs::deepstream::RuntimeConfig config = runtime_config_from(self);
  for (NvDsMetaList *frame_node = batch_meta->frame_meta_list; frame_node != nullptr;
       frame_node = frame_node->next) {
    auto *frame_meta = static_cast<NvDsFrameMeta *>(frame_node->data);
    if (frame_meta == nullptr) {
      continue;
    }
    for (NvDsMetaList *obj_node = frame_meta->obj_meta_list; obj_node != nullptr;
         obj_node = obj_node->next) {
      auto *obj_meta = static_cast<NvDsObjectMeta *>(obj_node->data);
      if (obj_meta != nullptr && !write_object(self, frame_meta, obj_meta, config)) {
        GST_ELEMENT_ERROR(
            self,
            RESOURCE,
            WRITE,
            ("failed to write vrsmeta output"),
            ("%s", self->output_path));
        return GST_FLOW_ERROR;
      }
    }
  }
  self->out->flush();
  return GST_FLOW_OK;
}

static void gst_vrsmeta_set_property(
    GObject *object,
    guint prop_id,
    const GValue *value,
    GParamSpec *pspec) {
  auto *self = reinterpret_cast<GstVrsMeta *>(object);
  switch (prop_id) {
    case PROP_STREAM_ID:
      g_free(self->stream_id);
      self->stream_id = g_value_dup_string(value);
      break;
    case PROP_SOURCE_ID:
      g_free(self->source_id);
      self->source_id = g_value_dup_string(value);
      break;
    case PROP_DETECTOR_ID:
      g_free(self->detector_id);
      self->detector_id = g_value_dup_string(value);
      break;
    case PROP_LABELS:
      g_free(self->labels_path);
      self->labels_path = g_value_dup_string(value);
      delete self->labels;
      self->labels = nullptr;
      break;
    case PROP_OUTPUT_MODE:
      g_free(self->output_mode);
      self->output_mode = g_value_dup_string(value);
      break;
    case PROP_OUTPUT_PATH:
      g_free(self->output_path);
      self->output_path = g_value_dup_string(value);
      delete self->out;
      self->out = nullptr;
      break;
    case PROP_APPEND:
      self->append = g_value_get_boolean(value);
      break;
    case PROP_BBOX_SCALE_X:
      self->bbox_scale_x = g_value_get_double(value);
      break;
    case PROP_BBOX_SCALE_Y:
      self->bbox_scale_y = g_value_get_double(value);
      break;
    case PROP_BBOX_OFFSET_X:
      self->bbox_offset_x = g_value_get_double(value);
      break;
    case PROP_BBOX_OFFSET_Y:
      self->bbox_offset_y = g_value_get_double(value);
      break;
    case PROP_DEBUG:
      self->debug = g_value_get_boolean(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
  }
}

static void gst_vrsmeta_get_property(
    GObject *object,
    guint prop_id,
    GValue *value,
    GParamSpec *pspec) {
  auto *self = reinterpret_cast<GstVrsMeta *>(object);
  switch (prop_id) {
    case PROP_STREAM_ID:
      g_value_set_string(value, self->stream_id);
      break;
    case PROP_SOURCE_ID:
      g_value_set_string(value, self->source_id);
      break;
    case PROP_DETECTOR_ID:
      g_value_set_string(value, self->detector_id);
      break;
    case PROP_LABELS:
      g_value_set_string(value, self->labels_path);
      break;
    case PROP_OUTPUT_MODE:
      g_value_set_string(value, self->output_mode);
      break;
    case PROP_OUTPUT_PATH:
      g_value_set_string(value, self->output_path);
      break;
    case PROP_APPEND:
      g_value_set_boolean(value, self->append);
      break;
    case PROP_BBOX_SCALE_X:
      g_value_set_double(value, self->bbox_scale_x);
      break;
    case PROP_BBOX_SCALE_Y:
      g_value_set_double(value, self->bbox_scale_y);
      break;
    case PROP_BBOX_OFFSET_X:
      g_value_set_double(value, self->bbox_offset_x);
      break;
    case PROP_BBOX_OFFSET_Y:
      g_value_set_double(value, self->bbox_offset_y);
      break;
    case PROP_DEBUG:
      g_value_set_boolean(value, self->debug);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
  }
}

static void gst_vrsmeta_finalize(GObject *object) {
  auto *self = reinterpret_cast<GstVrsMeta *>(object);
  g_free(self->stream_id);
  g_free(self->source_id);
  g_free(self->detector_id);
  g_free(self->labels_path);
  g_free(self->output_mode);
  g_free(self->output_path);
  delete self->labels;
  delete self->out;
  G_OBJECT_CLASS(gst_vrsmeta_parent_class)->finalize(object);
}

static void gst_vrsmeta_class_init(GstVrsMetaClass *klass) {
  auto *object_class = G_OBJECT_CLASS(klass);
  auto *element_class = GST_ELEMENT_CLASS(klass);
  object_class->set_property = gst_vrsmeta_set_property;
  object_class->get_property = gst_vrsmeta_get_property;
  object_class->finalize = gst_vrsmeta_finalize;

  gst_element_class_set_static_metadata(
      element_class,
      "VRS metadata exporter",
      "Filter/Metadata",
      "Exports DeepStream object metadata as VRS detection.v1 JSONL",
      "VRS");

  g_object_class_install_property(
      object_class,
      PROP_STREAM_ID,
      g_param_spec_string("stream-id", "Stream ID", "VRS stream_id", "deepstream-stream",
                          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_SOURCE_ID,
      g_param_spec_string("source-id", "Source ID", "Optional VRS source_id", "",
                          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_DETECTOR_ID,
      g_param_spec_string("detector-id", "Detector ID", "VRS detector_id", "deepstream-nvinfer",
                          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_LABELS,
      g_param_spec_string("labels", "Labels path", "Newline-delimited label file", "",
                          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_OUTPUT_MODE,
      g_param_spec_string("output-mode", "Output mode", "Output mode: jsonl", "jsonl",
                          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_OUTPUT_PATH,
      g_param_spec_string("output-path", "Output path", "JSONL output path", "",
                          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_APPEND,
      g_param_spec_boolean("append", "Append", "Append to output-path", FALSE,
                           static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_BBOX_SCALE_X,
      g_param_spec_double("bbox-scale-x", "BBox scale X", "Scale bbox X coordinates", -G_MAXDOUBLE,
                          G_MAXDOUBLE, 1.0,
                          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_BBOX_SCALE_Y,
      g_param_spec_double("bbox-scale-y", "BBox scale Y", "Scale bbox Y coordinates", -G_MAXDOUBLE,
                          G_MAXDOUBLE, 1.0,
                          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_BBOX_OFFSET_X,
      g_param_spec_double("bbox-offset-x", "BBox offset X", "Offset bbox X coordinates",
                          -G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
                          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_BBOX_OFFSET_Y,
      g_param_spec_double("bbox-offset-y", "BBox offset Y", "Offset bbox Y coordinates",
                          -G_MAXDOUBLE, G_MAXDOUBLE, 0.0,
                          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_DEBUG,
      g_param_spec_boolean("debug", "Debug", "Enable debug logging", FALSE,
                           static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  GstCaps *caps = gst_caps_new_any();
  gst_element_class_add_pad_template(
      element_class,
      gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, caps));
  gst_element_class_add_pad_template(
      element_class,
      gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps));
  gst_caps_unref(caps);

  auto *base_class = GST_BASE_TRANSFORM_CLASS(klass);
  base_class->transform_ip = GST_DEBUG_FUNCPTR(gst_vrsmeta_transform_ip);
}

static void gst_vrsmeta_init(GstVrsMeta *self) {
  self->stream_id = g_strdup("deepstream-stream");
  self->source_id = g_strdup("");
  self->detector_id = g_strdup("deepstream-nvinfer");
  self->labels_path = g_strdup("");
  self->output_mode = g_strdup("jsonl");
  self->output_path = g_strdup("");
  self->append = FALSE;
  self->bbox_scale_x = 1.0;
  self->bbox_scale_y = 1.0;
  self->bbox_offset_x = 0.0;
  self->bbox_offset_y = 0.0;
  self->debug = FALSE;
  self->labels = nullptr;
  self->out = nullptr;
  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
  gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(self), FALSE);
}

static gboolean plugin_init(GstPlugin *plugin) {
  return gst_element_register(plugin, "vrsmeta", GST_RANK_NONE, GST_TYPE_VRSMETA);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    vrsmeta,
    "VRS DeepStream metadata plugin",
    plugin_init,
    "0.3.0",
    "Apache-2.0",
    "vrs",
    "https://github.com/ziwon/vrs")
