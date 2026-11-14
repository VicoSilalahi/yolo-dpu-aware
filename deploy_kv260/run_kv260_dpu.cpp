// -----------------------------------------------------------------------------
// run_kv260_dpu.cpp
//
// A small, readable C++ runner for the DPU-compiled YOLOv8-seg .xmodel.
// It uses the Vitis AI Runtime (VART) plus XIR to load the graph, send one
// letterboxed image through the DPU, decode the ten raw output tensors, and
// write an annotated JPEG.
//
// Output tensors expected from the compiled graph:
//   for each stride (80, 40, 20):
//     cv2 = box distribution, 64 channels (4 sides x 16 DFL bins)
//     cv3 = class logits,      nc channels
//     cv4 = mask coefficients, 32 channels
//   proto = mask prototypes,   32 x 160 x 160
//
// Build:  make
// Run:    ./run_kv260_dpu <model.xmodel> <image.jpg> [options]
//
// The DPU API used here is the Vitis AI 2.x/3.x VART API:
//   xir::Graph::deserialize -> find subgraph with device=="DPU"
//   vart::Runner::create_runner(subgraph, "run")
//   vart::CpuFlatTensorBuffer over host memory
//   runner->execute_async(...) + runner->wait(job_id, -1)
// -----------------------------------------------------------------------------

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

// xir/util/any.hpp must come first: the 2.5 graph header uses xir::any without
// including it itself, and it requires C++17.
#include <xir/util/any.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>
#include <vart/runner.hpp>
#include <vart/experimental/runner_helper.hpp>

// =============================================================================
// Small utilities
// =============================================================================

namespace {

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

std::vector<std::string> split_csv(const std::string& text) {
  std::vector<std::string> items;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    const auto first = item.find_first_not_of(" \t");
    const auto last = item.find_last_not_of(" \t");
    if (first != std::string::npos) {
      items.push_back(item.substr(first, last - first + 1));
    }
  }
  return items;
}

// COCO-80 labels, used for the base model when --classes / CLASSES is omitted.
const std::vector<std::string>& coco_class_names() {
  static const std::vector<std::string> names = {
      "person",       "bicycle",      "car",          "motorbike",
      "aeroplane",    "bus",          "train",        "truck",
      "boat",         "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",       "bird",         "cat",
      "dog",          "horse",        "sheep",        "cow",
      "elephant",     "bear",         "zebra",        "giraffe",
      "backpack",     "umbrella",     "handbag",      "tie",
      "suitcase",     "frisbee",      "skis",         "snowboard",
      "sports ball",  "kite",         "baseball bat", "baseball glove",
      "skateboard",   "surfboard",    "tennis racket", "bottle",
      "wine glass",   "cup",          "fork",         "knife",
      "spoon",        "bowl",         "banana",       "apple",
      "sandwich",     "orange",       "broccoli",     "carrot",
      "hot dog",      "pizza",        "donut",        "cake",
      "chair",        "sofa",         "pottedplant",  "bed",
      "diningtable",  "toilet",       "tvmonitor",    "laptop",
      "mouse",        "remote",       "keyboard",     "cell phone",
      "microwave",    "oven",         "toaster",      "sink",
      "refrigerator", "book",         "clock",        "vase",
      "scissors",     "teddy bear",   "hair drier",   "toothbrush",
  };
  return names;
}

std::string shape_to_string(const std::vector<std::int32_t>& shape) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) out << ", ";
    out << shape[i];
  }
  out << "]";
  return out.str();
}

bool is_integer(const xir::Tensor* tensor) {
  const auto& dtype = tensor->get_data_type();
  return dtype.type == xir::DataType::INT || dtype.type == xir::DataType::XINT;
}

int fix_point_of(const xir::Tensor* tensor) {
  if (!tensor->has_attr("fix_point")) return -1;
  return tensor->get_attr<int>("fix_point");
}

std::string tensor_format_name(vart::Runner::TensorFormat format) {
  return format == vart::Runner::TensorFormat::NHWC ? "NHWC" : "NCHW";
}

// Host memory for the DPU. A page-aligned buffer keeps the DMA path happy and
// avoids subtle failures that can happen with ordinary small allocations.
class AlignedBuffer {
 public:
  explicit AlignedBuffer(std::size_t bytes) {
    const std::size_t rounded = (bytes + 4095u) & ~std::size_t(4095u);
    void* pointer = nullptr;
    if (posix_memalign(&pointer, 4096, rounded) != 0) {
      fail("posix_memalign failed for " + std::to_string(bytes) + " bytes");
    }
    data_ = static_cast<std::int8_t*>(pointer);
    std::memset(data_, 0, rounded);
  }

  ~AlignedBuffer() { std::free(data_); }

  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;

  std::int8_t* data() { return data_; }

 private:
  std::int8_t* data_ = nullptr;
};

// Nearest-rank percentile. Used only for the benchmark summary.
double percentile(std::vector<double> samples, double fraction) {
  if (samples.empty()) return 0.0;
  std::sort(samples.begin(), samples.end());
  const std::size_t index = static_cast<std::size_t>(
      std::ceil(fraction * static_cast<double>(samples.size())) - 1.0);
  return samples[std::min(index, samples.size() - 1)];
}

float sigmoid(float value) {
  const float clipped = std::max(-50.0f, std::min(50.0f, value));
  return 1.0f / (1.0f + std::exp(-clipped));
}

}  // namespace

// =============================================================================
// Command line
// =============================================================================

struct Options {
  std::string model;
  std::string image;
  std::string output = "result_cpp.jpg";
  std::string classes_csv;
  int warmup = 2;
  int bench = 10;
  float conf = 0.25f;
  float iou = 0.45f;
  bool draw = true;
  bool inspect = false;
};

void print_usage(const char* program) {
  std::cout
      << "Usage: " << program << " <model.xmodel> <image> [options]\n\n"
      << "Options:\n"
      << "  -o, --output <path>    annotated image path (default result_cpp.jpg)\n"
      << "  -w, --warmup <n>       warmup runs before timing (default 2)\n"
      << "  -b, --bench <n>        timed runs (default 10)\n"
      << "  -c, --conf <float>     confidence threshold (default 0.25)\n"
      << "  -i, --iou <float>      NMS IoU threshold (default 0.45)\n"
      << "      --classes <a,b,c>  class names (default CLASSES env or class_0...)\n"
      << "      --no-draw          run inference without writing an image\n"
      << "      --inspect          print the DPU input/output layout and exit\n"
      << "  -h, --help             show this message\n";
}

Options parse_args(int argc, char** argv) {
  Options options;
  std::vector<std::string> positional;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) fail(std::string("missing value for ") + name);
      return argv[++i];
    };

    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (arg == "-o" || arg == "--output") {
      options.output = next("--output");
    } else if (arg == "-w" || arg == "--warmup") {
      options.warmup = std::stoi(next("--warmup"));
    } else if (arg == "-b" || arg == "--bench") {
      options.bench = std::stoi(next("--bench"));
    } else if (arg == "-c" || arg == "--conf") {
      options.conf = std::stof(next("--conf"));
    } else if (arg == "-i" || arg == "--iou") {
      options.iou = std::stof(next("--iou"));
    } else if (arg == "--classes") {
      options.classes_csv = next("--classes");
    } else if (arg == "--no-draw") {
      options.draw = false;
    } else if (arg == "--inspect") {
      options.inspect = true;
    } else if (!arg.empty() && arg[0] == '-') {
      fail("unknown option: " + arg);
    } else {
      positional.push_back(arg);
    }
  }

  if (positional.empty() || (positional.size() < 2 && !options.inspect)) {
    print_usage(argv[0]);
    fail("model and image arguments are required");
  }
  options.model = positional[0];
  if (positional.size() >= 2) options.image = positional[1];
  if (positional.size() >= 3) options.output = positional[2];

  if (options.classes_csv.empty()) {
    const char* from_env = std::getenv("CLASSES");
    if (from_env != nullptr) options.classes_csv = from_env;
  }
  if (options.bench < 1) options.bench = 1;
  if (options.warmup < 0) options.warmup = 0;

  return options;
}

// =============================================================================
// Preprocessing
// =============================================================================

struct Letterbox {
  cv::Mat image;  // BGR, exactly target_width x target_height
  double ratio = 1.0;
  int pad_left = 0;
  int pad_top = 0;
};

Letterbox letterbox(const cv::Mat& source, int target_width, int target_height) {
  Letterbox result;
  result.ratio = std::min(static_cast<double>(target_width) / source.cols,
                          static_cast<double>(target_height) / source.rows);

  const int resized_width =
      static_cast<int>(std::round(source.cols * result.ratio));
  const int resized_height =
      static_cast<int>(std::round(source.rows * result.ratio));

  cv::Mat resized;
  cv::resize(source, resized, cv::Size(resized_width, resized_height), 0, 0,
             cv::INTER_LINEAR);

  result.pad_left = (target_width - resized_width) / 2;
  result.pad_top = (target_height - resized_height) / 2;
  const int pad_right = target_width - resized_width - result.pad_left;
  const int pad_bottom = target_height - resized_height - result.pad_top;

  cv::copyMakeBorder(resized, result.image, result.pad_top, pad_bottom,
                     result.pad_left, pad_right, cv::BORDER_CONSTANT,
                     cv::Scalar(114, 114, 114));
  return result;
}

// BGR -> RGB, then an integer LUT that performs x / 255 * 2^fix_point directly.
// This is both faster and bit-exact compared with float math on the A53 cores.
std::vector<std::int8_t> quantize_bgr_to_rgb(const cv::Mat& bgr, int fix_point) {
  if (fix_point < 0 || fix_point > 7) {
    fail("unsupported input fix_point: " + std::to_string(fix_point));
  }

  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

  cv::Mat lookup(1, 256, CV_8U);
  const double scale = static_cast<double>(1 << fix_point) / 255.0;
  for (int value = 0; value < 256; ++value) {
    const int quantized =
        std::max(0, std::min(127, static_cast<int>(std::lround(value * scale))));
    lookup.at<uchar>(0, value) = static_cast<uchar>(quantized);
  }

  cv::Mat quantized;
  cv::LUT(rgb, lookup, quantized);
  if (!quantized.isContinuous()) quantized = quantized.clone();

  std::vector<std::int8_t> result(quantized.total() * quantized.channels());
  std::memcpy(result.data(), quantized.data, result.size());
  return result;
}

// Convert the interleaved HWC image into the layout expected by the runner.
std::vector<std::int8_t> to_runner_layout(const std::vector<std::int8_t>& hwc,
                                          int height, int width, int channels,
                                          vart::Runner::TensorFormat format) {
  if (format == vart::Runner::TensorFormat::NHWC) {
    return hwc;
  }

  std::vector<std::int8_t> chw(hwc.size());
  const int plane = height * width;
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < plane; ++i) {
      chw[c * plane + i] = hwc[i * channels + c];
    }
  }
  return chw;
}

// =============================================================================
// Output tensor decoding
// =============================================================================

struct OutputRef {
  const xir::Tensor* tensor = nullptr;
  const std::int8_t* data = nullptr;
  int fix_point = 0;
};

struct Detection {
  float x1 = 0, y1 = 0, x2 = 0, y2 = 0;  // in letterboxed input coordinates
  float score = 0;
  int cls = 0;
  std::vector<float> coefficients;        // 32 mask coefficients
};

struct TensorDims {
  int height = 0;
  int width = 0;
  int channels = 0;
};

TensorDims output_dims(const OutputRef& ref, vart::Runner::TensorFormat format) {
  const auto shape = ref.tensor->get_shape();
  if (shape.size() != 4) {
    fail("expected a 4-D output tensor, got " + shape_to_string(shape));
  }
  TensorDims dims;
  if (format == vart::Runner::TensorFormat::NHWC) {
    dims.height = shape[1];
    dims.width = shape[2];
    dims.channels = shape[3];
  } else {
    dims.channels = shape[1];
    dims.height = shape[2];
    dims.width = shape[3];
  }
  return dims;
}

// Read one element from an NCHW or NHWC int8 tensor and dequantize it.
inline float value_at(const OutputRef& ref, const TensorDims& dims, int y, int x,
                      int c, vart::Runner::TensorFormat format) {
  const std::size_t index =
      format == vart::Runner::TensorFormat::NHWC
          ? (static_cast<std::size_t>(y) * dims.width + x) * dims.channels + c
          : (static_cast<std::size_t>(c) * dims.height + y) * dims.width + x;
  return static_cast<float>(ref.data[index]) / static_cast<float>(1 << ref.fix_point);
}

// Split an output tensor name into kind (box/class/mask) and stride index.
// Example:
//   ...__Segment_22__Sequential_cv3__ModuleList_1__Conv2d_2__ret_305_fix
//   -> kind = "class", index = 1
bool parse_head_name(const std::string& name, std::string& kind, int& index) {
  const std::string marker = "ModuleList_";
  const auto marker_pos = name.find(marker);
  if (marker_pos == std::string::npos) return false;

  const std::size_t digits_begin = marker_pos + marker.size();
  std::size_t digits_end = digits_begin;
  while (digits_end < name.size() &&
         std::isdigit(static_cast<unsigned char>(name[digits_end]))) {
    ++digits_end;
  }
  if (digits_end == digits_begin) return false;
  index = std::stoi(name.substr(digits_begin, digits_end - digits_begin));

  if (name.find("Sequential_cv2") != std::string::npos) {
    kind = "box";
  } else if (name.find("Sequential_cv3") != std::string::npos) {
    kind = "class";
  } else if (name.find("Sequential_cv4") != std::string::npos) {
    kind = "mask";
  } else {
    return false;
  }
  return true;
}

// YOLOv8 regression head: 4 sides x 16 bins, softmax over the bins, then the
// expected bin index gives the distance from the anchor point to each side.
void dfl_distances(const OutputRef& box, const TensorDims& dims, int y, int x,
                   vart::Runner::TensorFormat format, float dist[4]) {
  constexpr int kBins = 16;
  for (int side = 0; side < 4; ++side) {
    float values[kBins];
    float maximum = -std::numeric_limits<float>::infinity();
    for (int bin = 0; bin < kBins; ++bin) {
      values[bin] =
          value_at(box, dims, y, x, side * kBins + bin, format);
      maximum = std::max(maximum, values[bin]);
    }

    float sum = 0.0f;
    for (int bin = 0; bin < kBins; ++bin) {
      values[bin] = std::exp(values[bin] - maximum);
      sum += values[bin];
    }

    float expected = 0.0f;
    for (int bin = 0; bin < kBins; ++bin) {
      expected += (values[bin] / sum) * static_cast<float>(bin);
    }
    dist[side] = expected;
  }
}

std::vector<Detection> decode_detections(
    const std::map<int, OutputRef>& boxes,
    const std::map<int, OutputRef>& classes,
    const std::map<int, OutputRef>& masks,
    vart::Runner::TensorFormat format, int input_width, int input_height,
    float confidence_threshold) {
  std::vector<Detection> detections;

  for (const auto& entry : boxes) {
    const int stride_index = entry.first;
    const auto class_it = classes.find(stride_index);
    const auto mask_it = masks.find(stride_index);
    if (class_it == classes.end() || mask_it == masks.end()) continue;

    const OutputRef& box = entry.second;
    const OutputRef& cls = class_it->second;
    const OutputRef& mask = mask_it->second;

    const TensorDims box_dims = output_dims(box, format);
    const TensorDims cls_dims = output_dims(cls, format);
    const TensorDims mask_dims = output_dims(mask, format);

    const float stride_x =
        static_cast<float>(input_width) / static_cast<float>(box_dims.width);
    const float stride_y =
        static_cast<float>(input_height) / static_cast<float>(box_dims.height);

    std::cout << "  stride " << stride_index << ": " << box_dims.height << "x"
              << box_dims.width << ", stride " << stride_x << "x" << stride_y
              << ", classes " << cls_dims.channels << "\n";

    for (int y = 0; y < box_dims.height; ++y) {
      for (int x = 0; x < box_dims.width; ++x) {
        float distance[4];
        dfl_distances(box, box_dims, y, x, format, distance);

        const float anchor_x = static_cast<float>(x) + 0.5f;
        const float anchor_y = static_cast<float>(y) + 0.5f;

        int best_class = 0;
        float best_score = -1.0f;
        for (int c = 0; c < cls_dims.channels; ++c) {
          const float score =
              sigmoid(value_at(cls, cls_dims, y, x, c, format));
          if (score > best_score) {
            best_score = score;
            best_class = c;
          }
        }
        if (best_score < confidence_threshold) continue;

        Detection detection;
        detection.x1 = (anchor_x - distance[0]) * stride_x;
        detection.y1 = (anchor_y - distance[1]) * stride_y;
        detection.x2 = (anchor_x + distance[2]) * stride_x;
        detection.y2 = (anchor_y + distance[3]) * stride_y;
        detection.score = best_score;
        detection.cls = best_class;
        detection.coefficients.resize(mask_dims.channels);
        for (int c = 0; c < mask_dims.channels; ++c) {
          detection.coefficients[c] =
              value_at(mask, mask_dims, y, x, c, format);
        }
        detections.push_back(std::move(detection));
      }
    }
  }

  return detections;
}

// =============================================================================
// NMS and drawing
// =============================================================================

float intersection_over_union(const Detection& a, const Detection& b) {
  const float x1 = std::max(a.x1, b.x1);
  const float y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2);
  const float y2 = std::min(a.y2, b.y2);
  const float intersection =
      std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  return intersection / (area_a + area_b - intersection + 1e-6f);
}

std::vector<int> class_aware_nms(const std::vector<Detection>& detections,
                                 float iou_threshold) {
  std::map<int, std::vector<int>> per_class;
  for (int i = 0; i < static_cast<int>(detections.size()); ++i) {
    per_class[detections[i].cls].push_back(i);
  }

  std::vector<int> kept;
  for (auto& entry : per_class) {
    auto& indices = entry.second;
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
      return detections[a].score > detections[b].score;
    });

    std::vector<int> suppressed(indices.size(), 0);
    for (std::size_t i = 0; i < indices.size(); ++i) {
      if (suppressed[i]) continue;
      kept.push_back(indices[i]);
      for (std::size_t j = i + 1; j < indices.size(); ++j) {
        if (!suppressed[j] &&
            intersection_over_union(detections[indices[i]], detections[indices[j]]) >
                iou_threshold) {
          suppressed[j] = 1;
        }
      }
    }
  }

  std::sort(kept.begin(), kept.end(), [&](int a, int b) {
    return detections[a].score > detections[b].score;
  });
  if (kept.size() > 100) kept.resize(100);
  return kept;
}

cv::Scalar color_for_class(int class_id) {
  static const cv::Scalar palette[] = {
      {60, 76, 231},   {74, 195, 139},  {43, 57, 192},   {34, 126, 230},
      {103, 219, 249}, {202, 131, 34},  {127, 61, 52},   {92, 163, 214},
      {153, 60, 198},  {58, 136, 61},   {220, 198, 55},  {78, 12, 128},
      {134, 227, 106}, {29, 107, 193},  {202, 73, 168},  {113, 176, 24},
  };
  constexpr int kPaletteSize = sizeof(palette) / sizeof(palette[0]);
  return palette[class_id % kPaletteSize];
}

// Compute the mask only inside the box crop, blend it, and draw the box/label.
void draw_detections(cv::Mat& image, const std::vector<Detection>& detections,
                     const std::vector<int>& kept, const OutputRef& proto,
                     vart::Runner::TensorFormat format, int input_width,
                     int input_height, double ratio, int pad_left, int pad_top,
                     const std::vector<std::string>& class_names) {
  const TensorDims proto_dims = output_dims(proto, format);
  const int original_width = image.cols;
  const int original_height = image.rows;

  for (int index : kept) {
    const Detection& detection = detections[index];

    const float x1 = std::max(0.0f, std::min(detection.x1, input_width - 1.0f));
    const float y1 = std::max(0.0f, std::min(detection.y1, input_height - 1.0f));
    const float x2 = std::max(0.0f, std::min(detection.x2, input_width - 1.0f));
    const float y2 = std::max(0.0f, std::min(detection.y2, input_height - 1.0f));
    if (x2 <= x1 || y2 <= y1) continue;

    // Map the letterboxed box back onto the original image.
    int ox1 = static_cast<int>(std::round((x1 - pad_left) / ratio));
    int oy1 = static_cast<int>(std::round((y1 - pad_top) / ratio));
    int ox2 = static_cast<int>(std::round((x2 - pad_left) / ratio));
    int oy2 = static_cast<int>(std::round((y2 - pad_top) / ratio));
    ox1 = std::max(0, std::min(original_width - 1, ox1));
    oy1 = std::max(0, std::min(original_height - 1, oy1));
    ox2 = std::max(ox1 + 1, std::min(original_width, ox2));
    oy2 = std::max(oy1 + 1, std::min(original_height, oy2));

    // Crop the prototype mask in proto space (usually 160x160).
    int mx1 = static_cast<int>(std::floor(x1 * proto_dims.width / input_width));
    int my1 = static_cast<int>(std::floor(y1 * proto_dims.height / input_height));
    int mx2 = static_cast<int>(std::ceil(x2 * proto_dims.width / input_width));
    int my2 = static_cast<int>(std::ceil(y2 * proto_dims.height / input_height));
    mx1 = std::max(0, std::min(proto_dims.width - 1, mx1));
    my1 = std::max(0, std::min(proto_dims.height - 1, my1));
    mx2 = std::max(mx1 + 1, std::min(proto_dims.width, mx2));
    my2 = std::max(my1 + 1, std::min(proto_dims.height, my2));

    cv::Mat mask_crop(my2 - my1, mx2 - mx1, CV_32F);
    for (int y = 0; y < mask_crop.rows; ++y) {
      for (int x = 0; x < mask_crop.cols; ++x) {
        float dot = 0.0f;
        for (int c = 0; c < proto_dims.channels; ++c) {
          dot += detection.coefficients[c] *
                 value_at(proto, proto_dims, my1 + y, mx1 + x, c, format);
        }
        mask_crop.at<float>(y, x) = sigmoid(dot);
      }
    }

    cv::Mat resized_mask;
    cv::resize(mask_crop, resized_mask, cv::Size(ox2 - ox1, oy2 - oy1), 0, 0,
               cv::INTER_LINEAR);

    const cv::Scalar color = color_for_class(detection.cls);
    cv::Mat region = image(cv::Rect(ox1, oy1, ox2 - ox1, oy2 - oy1));
    for (int y = 0; y < region.rows; ++y) {
      for (int x = 0; x < region.cols; ++x) {
        if (resized_mask.at<float>(y, x) <= 0.5f) continue;
        cv::Vec3b& pixel = region.at<cv::Vec3b>(y, x);
        for (int c = 0; c < 3; ++c) {
          pixel[c] = static_cast<uchar>(pixel[c] * 0.45 + color[c] * 0.55);
        }
      }
    }

    cv::rectangle(image, cv::Point(ox1, oy1), cv::Point(ox2, oy2), color, 2);

    std::ostringstream label_stream;
    label_stream << class_names[detection.cls] << " " << std::fixed
                 << std::setprecision(2) << detection.score;
    const std::string label = label_stream.str();

    int baseline = 0;
    const cv::Size text =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    const int label_top = std::max(0, oy1 - text.height - 6);
    cv::rectangle(image, cv::Point(ox1, label_top),
                  cv::Point(ox1 + text.width + 4, oy1), color, cv::FILLED);
    cv::putText(image, label, cv::Point(ox1 + 2, std::max(12, oy1 - 4)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1,
                cv::LINE_AA);
  }
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
  try {
    const Options options = parse_args(argc, argv);

    std::cout << "Model : " << options.model << "\n";
    if (!options.image.empty()) std::cout << "Image : " << options.image << "\n";

    // ------------------------------------------------------------------
    // 1. Load the graph and locate the DPU subgraph.
    // ------------------------------------------------------------------
    auto graph = xir::Graph::deserialize(options.model);
    xir::Subgraph* dpu_subgraph = nullptr;
    for (xir::Subgraph* candidate :
         graph->get_root_subgraph()->children_topological_sort()) {
      if (candidate->has_attr("device") &&
          candidate->get_attr<std::string>("device") == "DPU") {
        dpu_subgraph = candidate;
        break;
      }
    }
    if (dpu_subgraph == nullptr) {
      fail("no subgraph with device == \"DPU\" found in " + options.model);
    }

    // --inspect works without a DPU device: it only reads the XIR metadata.
    if (options.inspect) {
      std::cout << "\nDPU subgraph: " << dpu_subgraph->get_name() << "\nInputs:\n";
      for (const xir::Tensor* tensor : dpu_subgraph->get_input_tensors()) {
        std::cout << "  " << tensor->get_name()
                  << " shape=" << shape_to_string(tensor->get_shape())
                  << " " << tensor->get_data_type().to_string()
                  << " fix_point=" << fix_point_of(tensor) << "\n";
      }
      std::cout << "Outputs:\n";
      for (const xir::Tensor* tensor : dpu_subgraph->get_output_tensors()) {
        std::cout << "  " << tensor->get_name()
                  << " shape=" << shape_to_string(tensor->get_shape())
                  << " " << tensor->get_data_type().to_string()
                  << " fix_point=" << fix_point_of(tensor) << "\n";
      }
      return 0;
    }

    auto runner = vart::Runner::create_runner(dpu_subgraph, "run");
    if (!runner) fail("vart::Runner::create_runner returned null");

    const auto input_tensors = runner->get_input_tensors();
    const auto output_tensors = runner->get_output_tensors();
    if (input_tensors.size() != 1) {
      fail("expected exactly one input tensor, got " +
           std::to_string(input_tensors.size()));
    }

    const xir::Tensor* input_tensor = input_tensors[0];
    const auto input_shape = input_tensor->get_shape();
    if (!is_integer(input_tensor)) {
      fail("this runner expects an INT8 input tensor, got " +
           input_tensor->get_data_type().to_string());
    }
    const int input_fix_point = fix_point_of(input_tensor);
    if (input_fix_point < 0) fail("input tensor has no fix_point attribute");

    const auto format = runner->get_tensor_format();
    if (input_shape.size() != 4) {
      fail("expected a 4-D input tensor, got " + shape_to_string(input_shape));
    }

    const int input_height =
        format == vart::Runner::TensorFormat::NHWC ? input_shape[1] : input_shape[2];
    const int input_width =
        format == vart::Runner::TensorFormat::NHWC ? input_shape[2] : input_shape[3];
    const int input_channels =
        format == vart::Runner::TensorFormat::NHWC ? input_shape[3] : input_shape[1];
    if (input_channels != 3) {
      fail("expected a 3-channel input tensor, got " +
           std::to_string(input_channels));
    }

    std::cout << "\nInput : " << shape_to_string(input_shape) << " "
              << input_tensor->get_data_type().to_string()
              << " fix_point=" << input_fix_point
              << " format=" << tensor_format_name(format) << "\n";

    std::cout << "Outputs (" << output_tensors.size() << "):\n";
    for (std::size_t i = 0; i < output_tensors.size(); ++i) {
      std::cout << "  [" << i << "] " << output_tensors[i]->get_name()
                << "\n      shape=" << shape_to_string(output_tensors[i]->get_shape())
                << " " << output_tensors[i]->get_data_type().to_string()
                << " fix_point=" << fix_point_of(output_tensors[i]) << "\n";
    }

    // ------------------------------------------------------------------
    // 2. Allocate host buffers and wrap them as VART tensor buffers.
    // ------------------------------------------------------------------
    AlignedBuffer input_storage(input_tensor->get_element_num());
    vart::CpuFlatTensorBuffer input_buffer(input_storage.data(), input_tensor);

    std::vector<std::unique_ptr<AlignedBuffer>> output_storage;
    std::vector<std::unique_ptr<vart::CpuFlatTensorBuffer>> output_buffers;
    output_storage.reserve(output_tensors.size());
    output_buffers.reserve(output_tensors.size());
    for (const xir::Tensor* tensor : output_tensors) {
      output_storage.emplace_back(
          std::make_unique<AlignedBuffer>(tensor->get_element_num()));
      output_buffers.emplace_back(std::make_unique<vart::CpuFlatTensorBuffer>(
          output_storage.back()->data(), tensor));
    }

    std::vector<vart::TensorBuffer*> input_pointers{&input_buffer};
    std::vector<vart::TensorBuffer*> output_pointers;
    output_pointers.reserve(output_buffers.size());
    for (auto& buffer : output_buffers) output_pointers.push_back(buffer.get());

    // ------------------------------------------------------------------
    // 3. Preprocess the image once; the benchmark reuses this buffer.
    // ------------------------------------------------------------------
    cv::Mat source = cv::imread(options.image, cv::IMREAD_COLOR);
    if (source.empty()) fail("cannot read image: " + options.image);

    const auto preprocess_begin = std::chrono::steady_clock::now();
    const Letterbox letterboxed = letterbox(source, input_width, input_height);
    const auto interleaved = quantize_bgr_to_rgb(letterboxed.image, input_fix_point);
    const auto input_data = to_runner_layout(interleaved, input_height, input_width,
                                             input_channels, format);
    if (input_data.size() != static_cast<std::size_t>(input_tensor->get_element_num())) {
      fail("input buffer size mismatch");
    }
    std::memcpy(input_storage.data(), input_data.data(), input_data.size());
    const double preprocess_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                  preprocess_begin)
            .count();
    std::cout << "\nPreprocess: " << std::fixed << std::setprecision(2)
              << preprocess_ms << " ms\n";

    // ------------------------------------------------------------------
    // 4. Warm up, then time execute_async + wait.
    // ------------------------------------------------------------------
    const auto run_once = [&]() {
      const auto job = runner->execute_async(input_pointers, output_pointers);
      if (job.second != 0) {
        fail("execute_async failed with status " + std::to_string(job.second));
      }
      const int status = runner->wait(job.first, -1);
      if (status != 0) {
        fail("runner->wait failed with status " + std::to_string(status));
      }
    };

    for (int i = 0; i < options.warmup; ++i) run_once();

    std::vector<double> latency_ms;
    latency_ms.reserve(options.bench);
    for (int i = 0; i < options.bench; ++i) {
      const auto begin = std::chrono::steady_clock::now();
      run_once();
      latency_ms.push_back(
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - begin)
              .count());
    }

    double total_ms = 0.0;
    for (double value : latency_ms) total_ms += value;
    const double mean_ms = total_ms / static_cast<double>(latency_ms.size());

    std::cout << "DPU latency (" << options.bench << " runs): "
              << "mean=" << mean_ms << " ms, "
              << "p50=" << percentile(latency_ms, 0.50) << " ms, "
              << "p90=" << percentile(latency_ms, 0.90) << " ms, "
              << "fps=" << (mean_ms > 0.0 ? 1000.0 / mean_ms : 0.0) << "\n";

    // ------------------------------------------------------------------
    // 5. Report the raw outputs and collect the ten head tensors.
    // ------------------------------------------------------------------
    std::map<int, OutputRef> box_tensors;
    std::map<int, OutputRef> class_tensors;
    std::map<int, OutputRef> mask_tensors;
    OutputRef proto_tensor;
    bool has_proto = false;

    std::cout << "\nOutput ranges:\n";
    for (std::size_t i = 0; i < output_tensors.size(); ++i) {
      const xir::Tensor* tensor = output_tensors[i];
      const std::int8_t* data = output_storage[i]->data();
      const std::size_t count = tensor->get_element_num();

      std::int8_t minimum = 127;
      std::int8_t maximum = -128;
      for (std::size_t j = 0; j < count; ++j) {
        minimum = std::min(minimum, data[j]);
        maximum = std::max(maximum, data[j]);
      }

      OutputRef ref;
      ref.tensor = tensor;
      ref.data = data;
      ref.fix_point = fix_point_of(tensor) >= 0 ? fix_point_of(tensor) : 0;

      const std::string name = tensor->get_name();
      std::cout << "  [" << i << "] min=" << static_cast<int>(minimum)
                << " max=" << static_cast<int>(maximum) << "\n";

      if (name.find("Proto_proto") != std::string::npos) {
        proto_tensor = ref;
        has_proto = true;
        continue;
      }

      std::string kind;
      int stride_index = 0;
      if (!parse_head_name(name, kind, stride_index)) {
        std::cout << "      (not a YOLO head tensor, ignored)\n";
        continue;
      }
      if (kind == "box") {
        box_tensors[stride_index] = ref;
      } else if (kind == "class") {
        class_tensors[stride_index] = ref;
      } else if (kind == "mask") {
        mask_tensors[stride_index] = ref;
      }
    }

    if (box_tensors.empty() || class_tensors.empty() || mask_tensors.empty()) {
      fail("could not find the box/class/mask head tensors in the model");
    }
    if (!has_proto) fail("could not find the proto mask tensor in the model");

    // The mask branch is coeff[32] dot proto[32]; reject a mismatched pair
    // before indexing coefficients in draw_detections().
    const int proto_channels = output_dims(proto_tensor, format).channels;
    for (const auto& entry : mask_tensors) {
      if (output_dims(entry.second, format).channels != proto_channels) {
        fail("mask coefficient channels do not match the proto tensor");
      }
    }

    // ------------------------------------------------------------------
    // 6. Decode, NMS, annotate.
    // ------------------------------------------------------------------
    int class_count = 0;
    for (const auto& entry : class_tensors) {
      class_count = output_dims(entry.second, format).channels;
      break;
    }
    if (class_count <= 0) fail("class output has no channels");

    std::vector<std::string> class_names = split_csv(options.classes_csv);
    if (class_names.size() != static_cast<std::size_t>(class_count)) {
      if (class_count == 80) {
        class_names = coco_class_names();
      } else {
        class_names.clear();
        for (int i = 0; i < class_count; ++i) {
          class_names.push_back("class_" + std::to_string(i));
        }
      }
    }

    std::cout << "\nDecoding " << class_count << " classes:";
    for (const auto& name : class_names) std::cout << " " << name;
    std::cout << "\n";

    const std::vector<Detection> detections = decode_detections(
        box_tensors, class_tensors, mask_tensors, format, input_width,
        input_height, options.conf);
    std::cout << "Candidates above conf=" << options.conf << ": "
              << detections.size() << "\n";

    const std::vector<int> kept = class_aware_nms(detections, options.iou);
    std::cout << "Detections after NMS: " << kept.size() << "\n";

    for (int index : kept) {
      const Detection& detection = detections[index];
      const int ox1 = static_cast<int>(
          std::round((detection.x1 - letterboxed.pad_left) / letterboxed.ratio));
      const int oy1 = static_cast<int>(
          std::round((detection.y1 - letterboxed.pad_top) / letterboxed.ratio));
      const int ox2 = static_cast<int>(
          std::round((detection.x2 - letterboxed.pad_left) / letterboxed.ratio));
      const int oy2 = static_cast<int>(
          std::round((detection.y2 - letterboxed.pad_top) / letterboxed.ratio));
      std::cout << "  " << class_names[detection.cls] << " "
                << std::fixed << std::setprecision(3) << detection.score
                << " box=[" << ox1 << "," << oy1 << "," << ox2 << "," << oy2
                << "]\n";
    }

    if (options.draw) {
      draw_detections(source, detections, kept, proto_tensor, format, input_width,
                      input_height, letterboxed.ratio, letterboxed.pad_left,
                      letterboxed.pad_top, class_names);
      if (!cv::imwrite(options.output, source)) {
        fail("cannot write output image: " + options.output);
      }
      std::cout << "Wrote " << options.output << "\n";
    }

    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
