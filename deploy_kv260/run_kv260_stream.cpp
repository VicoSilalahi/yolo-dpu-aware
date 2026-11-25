// -----------------------------------------------------------------------------
// run_kv260_stream.cpp
//
// Streaming 2-class (crack, spalling) YOLOv8-seg DPU runner for Kria KV260.
// Designed for continuous video/camera/RealSense input and decoupled
// structured JSON output (stdout NDJSON or ZeroMQ PUB) for downstream
// deduplication and transmission subsystems.
//
// Build:
//   make stream
//   make stream USE_REALSENSE=1 USE_ZMQ=1
//
// Run:
//   ./run_kv260_stream <model.xmodel> --source <video.mp4|dir|cam_idx|realsense>
// -----------------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <xir/util/any.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>
#include <vart/runner.hpp>
#include <vart/experimental/runner_helper.hpp>

#include "byte_tracker.hpp"

#ifdef HAVE_REALSENSE
#include <librealsense2/rs.hpp>
#endif

#ifdef HAVE_ZMQ
#include <zmq.h>
#endif

namespace {

constexpr int kNumClasses = 2;
constexpr std::array<const char*, kNumClasses> kClassNames = {"crack", "spalling"};
constexpr int kDflBins = 16;
constexpr int kNumMaskCoeffs = 32;

[[noreturn]] void fail(const std::string& msg) {
  throw std::runtime_error(msg);
}

int fix_point_of(const xir::Tensor* tensor) {
  return tensor->has_attr("fix_point") ? tensor->get_attr<int>("fix_point") : 0;
}

float scale_of(const xir::Tensor* tensor) {
  return std::ldexp(1.0f, -fix_point_of(tensor));
}

inline float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

struct Detection {
  int class_id = 0;
  float confidence = 0.0f;
  float x1 = 0.0f;
  float y1 = 0.0f;
  float x2 = 0.0f;
  float y2 = 0.0f;
  float cx = 0.0f;
  float cy = 0.0f;
  int area_px = 0;
  std::uint64_t track_id = 0;
  bool is_new_unique = false;
  cv::Mat mask;  // binary uint8 mask in original frame coordinates
};

struct FrameTiming {
  double preprocess_ms = 0.0;
  double dpu_ms = 0.0;
  double postprocess_ms = 0.0;
};

struct FrameResult {
  std::uint64_t frame_id = 0;
  std::int64_t timestamp_ms = 0;
  std::uint64_t unique_total = 0;
  FrameTiming timing;
  std::vector<Detection> detections;
};

// =============================================================================
// FrameSource: unified input abstraction
// =============================================================================

class FrameSource {
 public:
  virtual ~FrameSource() = default;
  virtual bool next(cv::Mat& bgr, std::int64_t& timestamp_ms) = 0;
  virtual std::string description() const = 0;
};

class VideoCaptureSource final : public FrameSource {
 public:
  explicit VideoCaptureSource(const std::string& source) : source_(source) {
    // Check if source is a camera index (all digits)
    bool is_index = !source.empty() &&
                    std::all_of(source.begin(), source.end(),
                                [](unsigned char c) { return std::isdigit(c); });
    if (is_index) {
      if (!cap_.open(std::stoi(source))) {
        fail("Failed to open camera index: " + source);
      }
      mode_ = "camera";
      return;
    }

    // Try opening as video file or stream URL first
    if (cap_.open(source)) {
      mode_ = "video";
      return;
    }

    // Fall back to image directory / glob
    std::string pattern = source;
    if (!pattern.empty() && pattern.back() == '/') {
      pattern += "*.jpg";
    } else if (pattern.find('*') == std::string::npos &&
               pattern.find('.') == std::string::npos) {
      pattern += "/*.jpg";
    }
    cv::glob(pattern, image_paths_, false);
    if (image_paths_.empty()) {
      fail("Failed to open video, camera, or image glob: " + source);
    }
    std::sort(image_paths_.begin(), image_paths_.end());
    mode_ = "images";
  }

  bool next(cv::Mat& bgr, std::int64_t& timestamp_ms) override {
    const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
    timestamp_ms = now;

    if (mode_ == "images") {
      if (image_idx_ >= image_paths_.size()) {
        return false;
      }
      bgr = cv::imread(image_paths_[image_idx_++], cv::IMREAD_COLOR);
      return !bgr.empty();
    }
    return cap_.read(bgr) && !bgr.empty();
  }

  std::string description() const override {
    return mode_ + ":" + source_;
  }

 private:
  std::string source_;
  std::string mode_;
  cv::VideoCapture cap_;
  std::vector<std::string> image_paths_;
  std::size_t image_idx_ = 0;
};

#ifdef HAVE_REALSENSE
class RealSenseSource final : public FrameSource {
 public:
  RealSenseSource(int width, int height, int fps) {
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);
    pipe_.start(cfg);
  }

  bool next(cv::Mat& bgr, std::int64_t& timestamp_ms) override {
    rs2::frameset frames = pipe_.wait_for_frames();
    rs2::video_frame color = frames.get_color_frame();
    if (!color) return false;
    timestamp_ms = static_cast<std::int64_t>(color.get_timestamp());
    cv::Mat view(cv::Size(color.get_width(), color.get_height()), CV_8UC3,
                 const_cast<void*>(color.get_data()), cv::Mat::AUTO_STEP);
    view.copyTo(bgr);
    return true;
  }

  std::string description() const override { return "realsense:color"; }

 private:
  rs2::pipeline pipe_;
};
#endif

std::unique_ptr<FrameSource> make_source(const std::string& source) {
  if (source == "realsense") {
#ifdef HAVE_REALSENSE
    return std::make_unique<RealSenseSource>(640, 480, 30);
#else
    fail("RealSense support not compiled in. Rebuild with: make stream USE_REALSENSE=1");
#endif
  }
  return std::make_unique<VideoCaptureSource>(source);
}

// =============================================================================
// ResultSink: decoupled output abstraction (NDJSON / ZMQ / Video)
// =============================================================================

std::string to_json(const FrameResult& res) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(4);
  out << "{\"frame_id\":" << res.frame_id
      << ",\"timestamp_ms\":" << res.timestamp_ms
      << ",\"unique_total\":" << res.unique_total
      << ",\"timing\":{\"preprocess_ms\":" << res.timing.preprocess_ms
      << ",\"dpu_ms\":" << res.timing.dpu_ms
      << ",\"postprocess_ms\":" << res.timing.postprocess_ms << "}"
      << ",\"detections\":[";
  for (std::size_t i = 0; i < res.detections.size(); ++i) {
    const auto& d = res.detections[i];
    if (i > 0) out << ",";
    out << "{\"class_id\":" << d.class_id
        << ",\"class_name\":\"" << kClassNames[d.class_id] << "\""
        << ",\"confidence\":" << d.confidence
        << ",\"bbox_xyxy\":[" << d.x1 << "," << d.y1 << "," << d.x2 << "," << d.y2 << "]"
        << ",\"centroid_xy\":[" << d.cx << "," << d.cy << "]"
        << ",\"area_px\":" << d.area_px
        << ",\"track_id\":" << d.track_id
        << ",\"is_new_unique\":" << (d.is_new_unique ? "true" : "false") << "}";
  }
  out << "]}";
  return out.str();
}

class JsonStdoutSink {
 public:
  explicit JsonStdoutSink(bool enabled) : enabled_(enabled) {}
  void publish(const FrameResult& res) const {
    if (!enabled_) return;
    std::cout << to_json(res) << "\n" << std::flush;
  }

 private:
  bool enabled_;
};

#ifdef HAVE_ZMQ
class ZmqPubSink {
 public:
  explicit ZmqPubSink(const std::string& endpoint) {
    ctx_ = zmq_ctx_new();
    sock_ = zmq_socket(ctx_, ZMQ_PUB);
    if (zmq_bind(sock_, endpoint.c_str()) != 0) {
      fail("Failed to bind ZeroMQ PUB socket on " + endpoint);
    }
  }
  ~ZmqPubSink() {
    if (sock_) zmq_close(sock_);
    if (ctx_) zmq_ctx_destroy(ctx_);
  }
  void publish(const FrameResult& res) const {
    const std::string payload = to_json(res);
    zmq_send(sock_, payload.data(), payload.size(), ZMQ_DONTWAIT);
  }

 private:
  void* ctx_ = nullptr;
  void* sock_ = nullptr;
};
#endif

// =============================================================================
// DpuPipeline: pre-allocated 2-class YOLOv8-seg DPU runner
// =============================================================================

struct StrideOutputs {
  int h = 0;
  int w = 0;
  std::size_t box_idx = 0;
  std::size_t cls_idx = 0;
  std::size_t mask_idx = 0;
};

class DpuPipeline {
 public:
  explicit DpuPipeline(const std::string& xmodel_path) {
    graph_ = xir::Graph::deserialize(xmodel_path);
    auto root = graph_->get_root_subgraph();
    const xir::Subgraph* dpu_subgraph = nullptr;
    for (auto* child : root->children_topological_sort()) {
      if (child->has_attr("device") &&
          child->get_attr<std::string>("device") == "DPU") {
        dpu_subgraph = child;
        break;
      }
    }
    if (!dpu_subgraph) {
      fail("No DPU subgraph found in " + xmodel_path);
    }

    runner_ = vart::Runner::create_runner(dpu_subgraph, "run");
    auto in_tensors = runner_->get_input_tensors();
    auto out_tensors = runner_->get_output_tensors();
    if (in_tensors.size() != 1 || out_tensors.size() != 10) {
      fail("Expected 1 input and 10 output tensors for YOLOv8-seg");
    }

    in_tensor_ = in_tensors[0];
    const auto in_shape = in_tensor_->get_shape();  // [1, H, W, 3]
    in_h_ = in_shape[1];
    in_w_ = in_shape[2];
    in_scale_ = static_cast<float>(1 << fix_point_of(in_tensor_));
    in_buf_.resize(in_tensor_->get_element_num());
    letterbox_buf_.create(in_h_, in_w_, CV_8UC3);

    out_tensors_ = out_tensors;
    out_bufs_.resize(out_tensors_.size());
    out_scales_.resize(out_tensors_.size());
    for (std::size_t i = 0; i < out_tensors_.size(); ++i) {
      out_bufs_[i].resize(out_tensors_[i]->get_element_num());
      out_scales_[i] = scale_of(out_tensors_[i]);
    }

    // Map outputs by shape: proto=[1,160,160,32], strides=[80,40,20]
    std::map<int, StrideOutputs> by_h;
    for (std::size_t i = 0; i < out_tensors_.size(); ++i) {
      const auto s = out_tensors_[i]->get_shape();
      const int h = s[1];
      const int c = s[3];
      if (h == 160 && c == kNumMaskCoeffs) {
        proto_idx_ = i;
      } else if (c == 4 * kDflBins) {
        by_h[h].h = h;
        by_h[h].w = s[2];
        by_h[h].box_idx = i;
      } else if (c == kNumClasses) {
        by_h[h].h = h;
        by_h[h].w = s[2];
        by_h[h].cls_idx = i;
      } else if (c == kNumMaskCoeffs) {
        by_h[h].h = h;
        by_h[h].w = s[2];
        by_h[h].mask_idx = i;
      } else {
        fail("Unexpected output channel count " + std::to_string(c) +
             " (this runner only supports 2-class crack/spalling models)");
      }
    }
    if (by_h.size() != 3) {
      fail("Failed to map all 3 detection strides (80, 40, 20)");
    }
    for (auto it = by_h.rbegin(); it != by_h.rend(); ++it) {
      strides_.push_back(it->second);
    }
  }

  void infer(const cv::Mat& bgr, float conf_thresh, float iou_thresh,
             std::vector<Detection>& detections, FrameTiming& timing) {
    detections.clear();
    const auto t0 = std::chrono::steady_clock::now();

    // Letterbox into pre-allocated buffer
    const float r = std::min(static_cast<float>(in_w_) / bgr.cols,
                             static_cast<float>(in_h_) / bgr.rows);
    const int new_w = static_cast<int>(std::round(bgr.cols * r));
    const int new_h = static_cast<int>(std::round(bgr.rows * r));
    const int pad_x = (in_w_ - new_w) / 2;
    const int pad_y = (in_h_ - new_h) / 2;

    letterbox_buf_.setTo(cv::Scalar(114, 114, 114));
    cv::Mat roi(letterbox_buf_, cv::Rect(pad_x, pad_y, new_w, new_h));
    cv::resize(bgr, roi, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // BGR -> RGB + quantize to int8 NHWC
    const int pixels = in_h_ * in_w_;
    const std::uint8_t* src = letterbox_buf_.ptr<std::uint8_t>(0);
    std::int8_t* dst = in_buf_.data();
    const float scale = in_scale_ / 255.0f;
    for (int i = 0; i < pixels; ++i) {
      const int idx = i * 3;
      dst[idx + 0] = static_cast<std::int8_t>(
          std::clamp(static_cast<int>(std::lrintf(src[idx + 2] * scale)), -128, 127));
      dst[idx + 1] = static_cast<std::int8_t>(
          std::clamp(static_cast<int>(std::lrintf(src[idx + 1] * scale)), -128, 127));
      dst[idx + 2] = static_cast<std::int8_t>(
          std::clamp(static_cast<int>(std::lrintf(src[idx + 0] * scale)), -128, 127));
    }
    const auto t1 = std::chrono::steady_clock::now();

    // Execute DPU
    std::vector<std::unique_ptr<vart::TensorBuffer>> in_tbs;
    std::vector<std::unique_ptr<vart::TensorBuffer>> out_tbs;
    in_tbs.push_back(std::make_unique<vart::CpuFlatTensorBuffer>(
        in_buf_.data(), in_tensor_));
    for (std::size_t i = 0; i < out_tensors_.size(); ++i) {
      out_tbs.push_back(std::make_unique<vart::CpuFlatTensorBuffer>(
          out_bufs_[i].data(), out_tensors_[i]));
    }
    std::vector<vart::TensorBuffer*> in_ptrs = {in_tbs[0].get()};
    std::vector<vart::TensorBuffer*> out_ptrs;
    out_ptrs.reserve(out_tbs.size());
    for (auto& tb : out_tbs) out_ptrs.push_back(tb.get());

    auto job = runner_->execute_async(in_ptrs, out_ptrs);
    runner_->wait(job.first, -1);
    const auto t2 = std::chrono::steady_clock::now();

    // Decode 2-class detections
    struct Candidate {
      int class_id;
      float conf;
      cv::Rect2f box;
      std::array<float, kNumMaskCoeffs> coeffs;
    };
    std::vector<Candidate> candidates;

    for (const auto& st : strides_) {
      const float stride_x = static_cast<float>(in_w_) / st.w;
      const float stride_y = static_cast<float>(in_h_) / st.h;
      const std::int8_t* box_ptr = out_bufs_[st.box_idx].data();
      const std::int8_t* cls_ptr = out_bufs_[st.cls_idx].data();
      const std::int8_t* msk_ptr = out_bufs_[st.mask_idx].data();
      const float box_s = out_scales_[st.box_idx];
      const float cls_s = out_scales_[st.cls_idx];
      const float msk_s = out_scales_[st.mask_idx];

      for (int y = 0; y < st.h; ++y) {
        for (int x = 0; x < st.w; ++x) {
          const int cell = y * st.w + x;
          const float s0 = sigmoid(cls_ptr[cell * kNumClasses + 0] * cls_s);
          const float s1 = sigmoid(cls_ptr[cell * kNumClasses + 1] * cls_s);
          const int best_cls = (s0 >= s1) ? 0 : 1;
          const float best_conf = (s0 >= s1) ? s0 : s1;
          if (best_conf < conf_thresh) continue;

          // DFL decode 4 sides
          float dist[4] = {0, 0, 0, 0};
          const std::int8_t* b = box_ptr + cell * (4 * kDflBins);
          for (int side = 0; side < 4; ++side) {
            float max_v = -1e9f;
            float exps[kDflBins];
            for (int k = 0; k < kDflBins; ++k) {
              const float v = b[side * kDflBins + k] * box_s;
              if (v > max_v) max_v = v;
              exps[k] = v;
            }
            float sum = 0.0f;
            float acc = 0.0f;
            for (int k = 0; k < kDflBins; ++k) {
              const float e = std::exp(exps[k] - max_v);
              sum += e;
              acc += e * k;
            }
            dist[side] = acc / sum;
          }

          const float cx = x + 0.5f;
          const float cy = y + 0.5f;
          const float x1 = std::clamp(((cx - dist[0]) * stride_x - pad_x) / r,
                                      0.0f, static_cast<float>(bgr.cols));
          const float y1 = std::clamp(((cy - dist[1]) * stride_y - pad_y) / r,
                                      0.0f, static_cast<float>(bgr.rows));
          const float x2 = std::clamp(((cx + dist[2]) * stride_x - pad_x) / r,
                                      0.0f, static_cast<float>(bgr.cols));
          const float y2 = std::clamp(((cy + dist[3]) * stride_y - pad_y) / r,
                                      0.0f, static_cast<float>(bgr.rows));
          if (x2 <= x1 || y2 <= y1) continue;

          Candidate c;
          c.class_id = best_cls;
          c.conf = best_conf;
          c.box = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);
          const std::int8_t* m = msk_ptr + cell * kNumMaskCoeffs;
          for (int k = 0; k < kNumMaskCoeffs; ++k) {
            c.coeffs[k] = m[k] * msk_s;
          }
          candidates.push_back(c);
        }
      }
    }

    // Class-aware NMS
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    boxes.reserve(candidates.size());
    scores.reserve(candidates.size());
    for (const auto& c : candidates) {
      boxes.emplace_back(static_cast<int>(c.box.x), static_cast<int>(c.box.y),
                         static_cast<int>(c.box.width), static_cast<int>(c.box.height));
      scores.push_back(c.conf);
    }
    std::vector<int> keep;
    cv::dnn::NMSBoxes(boxes, scores, conf_thresh, iou_thresh, keep);

    const std::int8_t* proto_ptr = out_bufs_[proto_idx_].data();
    const float proto_s = out_scales_[proto_idx_];
    const auto proto_shape = out_tensors_[proto_idx_]->get_shape();  // [1, 160, 160, 32]
    const int proto_h = proto_shape[1];
    const int proto_w = proto_shape[2];
    const float sx_p = static_cast<float>(proto_w) / in_w_;
    const float sy_p = static_cast<float>(proto_h) / in_h_;

    for (int idx : keep) {
      const auto& c = candidates[idx];
      Detection d;
      d.class_id = c.class_id;
      d.confidence = c.conf;
      d.x1 = c.box.x;
      d.y1 = c.box.y;
      d.x2 = c.box.x + c.box.width;
      d.y2 = c.box.y + c.box.height;
      d.cx = 0.5f * (d.x1 + d.x2);
      d.cy = 0.5f * (d.y1 + d.y2);

      // Decode segmentation mask inside the detection bbox in proto space (160x160)
      const int px1 = std::clamp(static_cast<int>(std::floor((d.x1 * r + pad_x) * sx_p)), 0, proto_w - 1);
      const int py1 = std::clamp(static_cast<int>(std::floor((d.y1 * r + pad_y) * sy_p)), 0, proto_h - 1);
      const int px2 = std::clamp(static_cast<int>(std::ceil((d.x2 * r + pad_x) * sx_p)), px1 + 1, proto_w);
      const int py2 = std::clamp(static_cast<int>(std::ceil((d.y2 * r + pad_y) * sy_p)), py1 + 1, proto_h);

      cv::Mat proto_mask = cv::Mat::zeros(proto_h, proto_w, CV_32F);
      for (int py = py1; py < py2; ++py) {
        float* row = proto_mask.ptr<float>(py);
        for (int px = px1; px < px2; ++px) {
          const std::int8_t* p = proto_ptr + (py * proto_w + px) * kNumMaskCoeffs;
          float sum = 0.0f;
          for (int k = 0; k < kNumMaskCoeffs; ++k) {
            sum += c.coeffs[k] * (p[k] * proto_s);
          }
          row[px] = sigmoid(sum);
        }
      }

      // Un-letterbox proto_mask -> original frame size
      cv::Mat full_mask;
      const int crop_x = std::clamp(static_cast<int>(std::round(pad_x * sx_p)), 0, proto_w - 1);
      const int crop_y = std::clamp(static_cast<int>(std::round(pad_y * sy_p)), 0, proto_h - 1);
      const int crop_w = std::clamp(static_cast<int>(std::round(new_w * sx_p)), 1, proto_w - crop_x);
      const int crop_h = std::clamp(static_cast<int>(std::round(new_h * sy_p)), 1, proto_h - crop_y);
      cv::resize(proto_mask(cv::Rect(crop_x, crop_y, crop_w, crop_h)), full_mask, bgr.size(), 0, 0, cv::INTER_LINEAR);
      cv::threshold(full_mask, d.mask, 0.5, 255, cv::THRESH_BINARY);
      d.mask.convertTo(d.mask, CV_8U);
      d.area_px = cv::countNonZero(d.mask);
      detections.push_back(std::move(d));
    }

    const auto t3 = std::chrono::steady_clock::now();
    timing.preprocess_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    timing.dpu_ms =
        std::chrono::duration<double, std::milli>(t2 - t1).count();
    timing.postprocess_ms =
        std::chrono::duration<double, std::milli>(t3 - t2).count();
  }

 private:
  std::unique_ptr<xir::Graph> graph_;
  std::unique_ptr<vart::Runner> runner_;
  const xir::Tensor* in_tensor_ = nullptr;
  std::vector<const xir::Tensor*> out_tensors_;
  int in_h_ = 640;
  int in_w_ = 640;
  float in_scale_ = 1.0f;
  std::size_t proto_idx_ = 0;
  std::vector<StrideOutputs> strides_;
  std::vector<std::int8_t> in_buf_;
  std::vector<std::vector<std::int8_t>> out_bufs_;
  std::vector<float> out_scales_;
  cv::Mat letterbox_buf_;
};

void draw_annotations(cv::Mat& bgr, const std::vector<Detection>& detections) {
  cv::Mat overlay = bgr.clone();
  for (const auto& d : detections) {
    const cv::Scalar color =
        (d.class_id == 0) ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 0, 255);
    if (!d.mask.empty()) {
      overlay.setTo(color, d.mask);
    }
    cv::rectangle(bgr, cv::Point2f(d.x1, d.y1), cv::Point2f(d.x2, d.y2), color, 2);
    std::ostringstream label;
    if (d.track_id > 0) label << "#" << d.track_id << " ";
    label << kClassNames[d.class_id] << " " << std::fixed << std::setprecision(2)
          << d.confidence;
    cv::putText(bgr, label.str(), cv::Point2f(d.x1, std::max(15.0f, d.y1 - 5.0f)),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2);
  }
  cv::addWeighted(overlay, 0.4, bgr, 0.6, 0.0, bgr);
}

void print_usage(const char* prog) {
  std::cout << "Usage: " << prog << " <model.xmodel> --source <src> [options]\n"
            << "  --source <src>       Video file, image dir/glob, camera index, or 'realsense'\n"
            << "  --conf <float>       Confidence threshold (default: 0.25)\n"
            << "  --iou <float>        NMS IoU threshold (default: 0.45)\n"
            << "  --track              Enable ByteTrack-style unique defect tracking\n"
            << "  --report-interval <s> Print periodic summary every <s> seconds (e.g. 1.0)\n"
            << "  --json-out           Emit per-frame NDJSON to stdout (default: on)\n"
            << "  --no-json            Disable stdout NDJSON\n"
            << "  --zmq-pub <endpoint> Publish NDJSON on ZeroMQ PUB (requires USE_ZMQ=1)\n"
            << "  --save-video <path>  Write annotated video to <path>\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  std::string model_path;
  std::string source;
  std::string save_video;
  std::string zmq_endpoint;
  float conf = 0.25f;
  float iou = 0.45f;
  double report_interval_s = 0.0;
  bool json_out = true;
  bool enable_track = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else if (arg == "--source" && i + 1 < argc) {
      source = argv[++i];
    } else if (arg == "--conf" && i + 1 < argc) {
      conf = std::stof(argv[++i]);
    } else if (arg == "--iou" && i + 1 < argc) {
      iou = std::stof(argv[++i]);
    } else if (arg == "--track") {
      enable_track = true;
    } else if (arg == "--report-interval" && i + 1 < argc) {
      report_interval_s = std::stod(argv[++i]);
    } else if (arg == "--json-out") {
      json_out = true;
    } else if (arg == "--no-json") {
      json_out = false;
    } else if (arg == "--zmq-pub" && i + 1 < argc) {
      zmq_endpoint = argv[++i];
    } else if (arg == "--save-video" && i + 1 < argc) {
      save_video = argv[++i];
    } else if (model_path.empty() && arg[0] != '-') {
      model_path = arg;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return 1;
    }
  }

  if (model_path.empty() || source.empty()) {
    print_usage(argv[0]);
    return 1;
  }

  try {
    DpuPipeline pipeline(model_path);
    auto frame_source = make_source(source);
    JsonStdoutSink json_sink(json_out);

#ifdef HAVE_ZMQ
    std::unique_ptr<ZmqPubSink> zmq_sink;
    if (!zmq_endpoint.empty()) {
      zmq_sink = std::make_unique<ZmqPubSink>(zmq_endpoint);
    }
#else
    if (!zmq_endpoint.empty()) {
      fail("ZeroMQ support not compiled in. Rebuild with: make stream USE_ZMQ=1");
    }
#endif

    cv::VideoWriter writer;
    cv::Mat frame;
    std::int64_t ts_ms = 0;
    std::uint64_t frame_id = 0;
    ByteTracker tracker;
    auto report_t0 = std::chrono::steady_clock::now();
    std::uint64_t report_frames = 0;

    while (frame_source->next(frame, ts_ms)) {
      FrameResult res;
      res.frame_id = frame_id++;
      res.timestamp_ms = ts_ms;

      pipeline.infer(frame, conf, iou, res.detections, res.timing);
      if (enable_track) {
        auto tracked = tracker.update(res.detections);
        std::vector<Detection> prev = std::move(res.detections);
        res.detections.clear();
        for (const auto& t : tracked) {
          Detection d;
          d.class_id = t.class_id;
          d.confidence = t.confidence;
          d.x1 = t.box.x;
          d.y1 = t.box.y;
          d.x2 = t.box.x + t.box.width;
          d.y2 = t.box.y + t.box.height;
          d.cx = 0.5f * (d.x1 + d.x2);
          d.cy = 0.5f * (d.y1 + d.y2);
          d.track_id = t.track_id;
          d.is_new_unique = t.is_new_unique;
          // Preserve decoded segmentation mask & exact area from best matching detection
          for (const auto& p : prev) {
            if (p.class_id == d.class_id && std::abs(p.x1 - d.x1) < 1e-2f &&
                std::abs(p.y1 - d.y1) < 1e-2f) {
              d.mask = p.mask;
              d.area_px = p.area_px;
              break;
            }
          }
          if (d.area_px == 0) {
            d.area_px = static_cast<int>(t.box.width * t.box.height);
          }
          res.detections.push_back(std::move(d));
        }
        res.unique_total = tracker.unique_total();
      }
      json_sink.publish(res);
#ifdef HAVE_ZMQ
      if (zmq_sink) zmq_sink->publish(res);
#endif
      report_frames++;
      if (report_interval_s > 0.0) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed_s = std::chrono::duration<double>(now - report_t0).count();
        if (elapsed_s >= report_interval_s) {
          std::cout << std::fixed << std::setprecision(1)
                    << "[report] frames=" << report_frames
                    << " fps=" << (report_frames / elapsed_s)
                    << " dets=" << res.detections.size()
                    << " unique_total=" << res.unique_total << "\n"
                    << std::flush;
          report_t0 = now;
          report_frames = 0;
        }
      }

      if (!save_video.empty()) {
        if (!writer.isOpened()) {
          writer.open(save_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                      15.0, frame.size());
        }
        draw_annotations(frame, res.detections);
        writer.write(frame);
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
