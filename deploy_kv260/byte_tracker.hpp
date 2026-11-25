#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

// Lightweight, zero-dependency ByteTrack-style multi-object tracker.
// Performs two-stage same-class IoU association (high-confidence first, then
// low-confidence) with smoothed velocity prediction and unique-defect flagging.
class ByteTracker {
 public:
  struct Config {
    float high_conf = 0.35f;
    float low_conf = 0.15f;
    float iou_thresh = 0.25f;
    int min_hits = 3;
    int max_age = 15;
  };

  struct TrackedObject {
    std::uint64_t track_id = 0;
    int class_id = 0;
    float confidence = 0.0f;
    cv::Rect2f box;
    bool is_new_unique = false;
  };

  ByteTracker() : cfg_() {}
  explicit ByteTracker(const Config& cfg) : cfg_(cfg) {}

  template <typename DetT>
  std::vector<TrackedObject> update(const std::vector<DetT>& detections) {
    // 1. Predict next box position using smoothed velocity
    for (auto& t : tracks_) {
      t.box.x += t.vx;
      t.box.y += t.vy;
      t.time_since_update++;
    }

    // 2. Partition detections into high-conf and low-conf indices
    std::vector<std::size_t> high_idx, low_idx;
    for (std::size_t i = 0; i < detections.size(); ++i) {
      if (detections[i].confidence >= cfg_.high_conf) {
        high_idx.push_back(i);
      } else if (detections[i].confidence >= cfg_.low_conf) {
        low_idx.push_back(i);
      }
    }

    std::vector<bool> track_matched(tracks_.size(), false);
    std::vector<bool> det_matched(detections.size(), false);

    // 3. Stage 1: match active tracks with high-confidence detections
    associate(detections, high_idx, track_matched, det_matched);

    // 4. Stage 2: match remaining tracks with low-confidence detections
    associate(detections, low_idx, track_matched, det_matched);

    // 5. Spawn new tracks for unmatched high-confidence detections
    for (std::size_t idx : high_idx) {
      if (det_matched[idx]) continue;
      const auto& d = detections[idx];
      Track t;
      t.id = next_id_++;
      t.class_id = d.class_id;
      t.confidence = d.confidence;
      t.box = cv::Rect2f(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
      t.hits = 1;
      t.time_since_update = 0;
      t.confirmed = (cfg_.min_hits <= 1);
      tracks_.push_back(t);
    }

    // 6. Prune stale tracks
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [this](const Track& t) {
                         return t.time_since_update > cfg_.max_age;
                       }),
        tracks_.end());

    // 7. Emit currently matched & confirmed (or tentative) tracks
    std::vector<TrackedObject> out;
    for (auto& t : tracks_) {
      if (t.time_since_update != 0) continue;
      bool newly_unique = false;
      if (t.confirmed && !t.emitted_unique) {
        t.emitted_unique = true;
        newly_unique = true;
        unique_total_++;
      }
      TrackedObject obj;
      obj.track_id = t.id;
      obj.class_id = t.class_id;
      obj.confidence = t.confidence;
      obj.box = t.box;
      obj.is_new_unique = newly_unique;
      out.push_back(obj);
    }
    return out;
  }

  std::uint64_t unique_total() const { return unique_total_; }

 private:
  struct Track {
    std::uint64_t id = 0;
    int class_id = 0;
    float confidence = 0.0f;
    cv::Rect2f box;
    float vx = 0.0f;
    float vy = 0.0f;
    int hits = 0;
    int time_since_update = 0;
    bool confirmed = false;
    bool emitted_unique = false;
  };

  static float iou(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float inter = (a & b).area();
    if (inter <= 0.0f) return 0.0f;
    const float uni = a.area() + b.area() - inter;
    return (uni > 0.0f) ? (inter / uni) : 0.0f;
  }

  template <typename DetT>
  void associate(const std::vector<DetT>& detections,
                 const std::vector<std::size_t>& det_indices,
                 std::vector<bool>& track_matched,
                 std::vector<bool>& det_matched) {
    struct Pair {
      float score;
      std::size_t t_idx;
      std::size_t d_idx;
    };
    std::vector<Pair> pairs;
    for (std::size_t ti = 0; ti < tracks_.size(); ++ti) {
      if (track_matched[ti]) continue;
      for (std::size_t di : det_indices) {
        if (det_matched[di]) continue;
        if (tracks_[ti].class_id != detections[di].class_id) continue;
        const auto& d = detections[di];
        const cv::Rect2f dbox(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
        const float score = iou(tracks_[ti].box, dbox);
        if (score >= cfg_.iou_thresh) {
          pairs.push_back({score, ti, di});
        }
      }
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const Pair& a, const Pair& b) { return a.score > b.score; });

    for (const auto& p : pairs) {
      if (track_matched[p.t_idx] || det_matched[p.d_idx]) continue;
      track_matched[p.t_idx] = true;
      det_matched[p.d_idx] = true;

      auto& t = tracks_[p.t_idx];
      const auto& d = detections[p.d_idx];
      const cv::Rect2f new_box(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
      const float dx = new_box.x - t.box.x;
      const float dy = new_box.y - t.box.y;
      t.vx = 0.5f * t.vx + 0.5f * dx;
      t.vy = 0.5f * t.vy + 0.5f * dy;
      t.box = new_box;
      t.confidence = d.confidence;
      t.hits++;
      t.time_since_update = 0;
      if (t.hits >= cfg_.min_hits) {
        t.confirmed = true;
      }
    }
  }

  Config cfg_;
  std::vector<Track> tracks_;
  std::uint64_t next_id_ = 1;
  std::uint64_t unique_total_ = 0;
};
