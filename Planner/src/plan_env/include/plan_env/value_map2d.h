/**
 * @file value_map2d.h
 * @brief 2D semantic value map adapted for GridMap
 *
 * Confidence-weighted ITM (Image-Text Matching) score fusion on 2D occupancy grid.
 * Reference: "VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation"
 *
 * Adapted from ApexNav's ValueMap2D to work with GridMap's 2D projection layer.
 */

#ifndef _VALUE_MAP_2D_H_
#define _VALUE_MAP_2D_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <vector>
#include <nav_msgs/OccupancyGrid.h>
#include <plan_env/grid_map_new.h>

using Eigen::Vector2d;
using Eigen::Vector2i;
using std::vector;

class ValueMap2D {
public:
  ValueMap2D(GridMap::Ptr grid_map);
  ~ValueMap2D(){};

  /**
   * @brief Update value map with a new ITM observation
   * @param sensor_pos 2D position of the sensor
   * @param sensor_yaw Yaw angle of the sensor (rad)
   * @param free_grids Indices of free grids observed in this frame
   * @param itm_score ITM cosine similarity score [0, 1]
   */
  void updateValueMap(const Vector2d& sensor_pos, const double& sensor_yaw,
      const vector<Vector2i>& free_grids, const double& itm_score);

  /// Get semantic value at world position
  double getValue(const Vector2d& pos);
  /// Get semantic value at grid index
  double getValue(const Vector2i& idx);
  /// Get confidence at world position
  double getConfidence(const Vector2d& pos);
  /// Get confidence at grid index
  double getConfidence(const Vector2i& idx);

  /// Collect free grids visible from current sensor pose (within FOV, range, and line-of-sight)
  void getFreeGrids(vector<Vector2i>& free_grids,
                    const Vector2d& sensor_pos, const double& sensor_yaw);
  void getFreeGrids(vector<Vector2i>& free_grids);

  /// Check if value map buffers are initialized
  bool isInitialized() const { return initialized_; }

  /// Re-initialize buffers if 2D map size changed
  void ensureInitialized();

  // Accessors for visualization
  const vector<double>& getValueBuffer() const { return value_buffer_; }
  const vector<double>& getConfidenceBuffer() const { return confidence_buffer_; }
  int getWidth() const { return width_; }
  int getHeight() const { return height_; }
  double getOriginX() const { return origin_x_; }
  double getOriginY() const { return origin_y_; }
  double getResolution() const { return resolution_; }

private:
  double getFovConfidence(
      const Vector2d& sensor_pos, const double& sensor_yaw, const Vector2d& pt_pos);
  double normalizeAngle(double angle);

  // 2D grid helpers (using cached 2D map parameters from GridMap)
  int toAddress2D(const Vector2i& idx);
  void posToIndex2D(const Vector2d& pos, Vector2i& idx);
  void indexToPos2D(const Vector2i& idx, Vector2d& pos);
  bool isInMap2D(const Vector2i& idx);

  vector<double> value_buffer_;      ///< Per-grid semantic value
  vector<double> confidence_buffer_; ///< Per-grid confidence for weighted fusion

  GridMap::Ptr grid_map_;  ///< Reference to the GridMap

  // Cached 2D map parameters (updated from GridMap's 2D occupancy grid)
  double origin_x_, origin_y_, resolution_, resolution_inv_;
  int width_, height_, voxel_num_;
  bool initialized_ = false;

  // FOV parameter for confidence model
  double fov_angle_;  ///< Total FOV angle in radians (default 79°)
  double max_depth_;  ///< Maximum sensor depth/range in meters (default 5.0)
};

// ============== Inline implementations ==============

inline double ValueMap2D::normalizeAngle(double angle)
{
  while (angle > M_PI) angle -= 2.0 * M_PI;
  while (angle < -M_PI) angle += 2.0 * M_PI;
  return angle;
}

inline int ValueMap2D::toAddress2D(const Vector2i& idx)
{
  return idx(0) + idx(1) * width_;
}

inline void ValueMap2D::posToIndex2D(const Vector2d& pos, Vector2i& idx)
{
  idx(0) = int(floor((pos(0) - origin_x_) * resolution_inv_));
  idx(1) = int(floor((pos(1) - origin_y_) * resolution_inv_));
}

inline void ValueMap2D::indexToPos2D(const Vector2i& idx, Vector2d& pos)
{
  pos(0) = (idx(0) + 0.5) * resolution_ + origin_x_;
  pos(1) = (idx(1) + 0.5) * resolution_ + origin_y_;
}

inline bool ValueMap2D::isInMap2D(const Vector2i& idx)
{
  return idx(0) >= 0 && idx(0) < width_ && idx(1) >= 0 && idx(1) < height_;
}

inline double ValueMap2D::getConfidence(const Vector2d& pos)
{
  Vector2i idx;
  posToIndex2D(pos, idx);
  return getConfidence(idx);
}

inline double ValueMap2D::getConfidence(const Vector2i& idx)
{
  if (!isInMap2D(idx)) return 0.0;
  return confidence_buffer_[toAddress2D(idx)];
}

inline double ValueMap2D::getValue(const Vector2d& pos)
{
  Vector2i idx;
  posToIndex2D(pos, idx);
  return getValue(idx);
}

inline double ValueMap2D::getValue(const Vector2i& idx)
{
  if (!isInMap2D(idx)) return 0.0;
  return value_buffer_[toAddress2D(idx)];
}

#endif