/**
 * @file value_map2d.cpp
 * @brief Implementation of 2D semantic value map adapted for GridMap
 *
 * Confidence-weighted ITM score fusion on 2D occupancy grid.
 * Reference: "VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation"
 *
 * @author Adapted from ApexNav
 */

#include <plan_env/value_map2d.h>
#include <cmath>

// --- Static semantic transform parameters (defaults) ---
// Tuned for BLIP2 ITM scores which empirically max out around 0.55
double ValueMap2D::low_thresh    = 0.2;   // below → penalty
double ValueMap2D::high_thresh   = 0.3;  // above → bonus
double ValueMap2D::penalty_scale = 4.0;   // max penalty at ITM=0
double ValueMap2D::bonus_scale   = 3.0;   // max bonus at ITM=sat_score
double ValueMap2D::sat_score     = 0.6;   // effective saturation point

double ValueMap2D::transformITMScore(double itm_score)
{
  if (itm_score < low_thresh) {
    // Penalty zone: magnitude grows quadratically as score decreases
    // t ∈ [0, 1]  where 0 = at low_thresh, 1 = at ITM=0
    double t = (low_thresh - itm_score) / low_thresh;
    return -penalty_scale * t * t;
  }
  else if (itm_score > high_thresh) {
    // Bonus zone: magnitude grows quadratically as score increases
    // Normalized by (sat_score - high_thresh) so ITM=0.55 gets meaningful bonus
    // t ∈ [0, 1]  where 0 = at high_thresh, 1 = at ITM=sat_score
    // Clamp t ≤ 1 so scores beyond sat_score don't explode
    double t = (itm_score - high_thresh) / (sat_score - high_thresh);
    if (t > 1.0) t = 1.0;
    return bonus_scale * t * t;
  }
  else {
    // Ambiguous zone [low_thresh, high_thresh]: slight linear ramp
    double mid = (low_thresh + high_thresh) / 2.0;
    double half_range = (high_thresh - low_thresh) / 2.0;
    return 0.05 * (itm_score - mid) / half_range;
  }
}

ValueMap2D::ValueMap2D(GridMap::Ptr grid_map)
{
  this->grid_map_ = grid_map;

  // Default FOV angle (79° typical RGB camera), can be loaded from param
  // nh.param("value_map/fov_angle_deg", fov_angle_, 79.0);
  fov_angle_ = 79.0;
  fov_angle_ = fov_angle_ * M_PI / 180.0;

  // Default max depth/range for sensor visibility (meters)
  // nh.param("value_map/max_depth", max_depth_, 5.0);
  max_depth_ = 5.0;

  // Buffers will be initialized when 2D map is ready
  initialized_ = false;
  voxel_num_ = 0;
}

void ValueMap2D::ensureInitialized()
{
  if(grid_map_==nullptr){
    ROS_ERROR("No grid map");
    return;
  }
  if (!grid_map_->is2DMapReady()) {
    ROS_ERROR("2D map not ready");
    return;
  }

  const nav_msgs::OccupancyGrid& grid = grid_map_->get2DOccupancyGrid();
  int new_width = grid.info.width;
  int new_height = grid.info.height;

  // Check if size, origin, or resolution changed, or not yet initialized
  if (!initialized_ || new_width != width_ || new_height != height_ ||
      fabs(grid.info.origin.position.x - origin_x_) > 1e-4 ||
      fabs(grid.info.origin.position.y - origin_y_) > 1e-4 ||
      fabs(grid.info.resolution - resolution_) > 1e-6) {
    width_ = new_width;
    height_ = new_height;
    voxel_num_ = width_ * height_;
    origin_x_ = grid.info.origin.position.x;
    origin_y_ = grid.info.origin.position.y;
    resolution_ = grid.info.resolution;
    resolution_inv_ = 1.0 / resolution_;

    // Resize and reset buffers
    value_buffer_ = vector<double>(voxel_num_, 0.0);
    confidence_buffer_ = vector<double>(voxel_num_, 0.0);
    itm_buffer_ = vector<double>(voxel_num_,0.0);
    initialized_ = true;

    ROS_INFO("[ValueMap2D] Initialized: %dx%d, origin=(%.2f,%.2f), res=%.3f",
             width_, height_, origin_x_, origin_y_, resolution_);
  }
}

void ValueMap2D::updateValueMap(const Vector2d& sensor_pos, const double& sensor_yaw,
    const vector<Vector2i>& free_grids, const double& itm_score)
{
  if (!initialized_) return;

  for (const auto& grid : free_grids) {
    Vector2d pos;
    indexToPos2D(grid, pos);
    int adr = toAddress2D(grid);

    // Skip out-of-bounds
    if (adr < 0 || adr >= voxel_num_) continue;

    // Calculate FOV-based confidence for current observation
    double now_confidence = getFovConfidence(sensor_pos, sensor_yaw, pos);
    // Transform raw ITM → signed semantic contribution (penalty/bonus)
    double now_value = transformITMScore(itm_score);
    itm_buffer_[adr] = itm_score;

    // Retrieve existing confidence and value
    double last_confidence = confidence_buffer_[adr];
    double last_value = value_buffer_[adr];

    // Apply confidence-weighted fusion with quadratic confidence combination
    double total_confidence = now_confidence + last_confidence;
    if (total_confidence < 1e-6) {
      // Both confidences near zero — skip update to avoid division by zero / NaN
      continue;
    }
    confidence_buffer_[adr] =
        (now_confidence * now_confidence + last_confidence * last_confidence) /
        total_confidence;
    value_buffer_[adr] = (now_confidence * now_value + last_confidence * last_value) /
                         total_confidence;
  }
}

double ValueMap2D::getFovConfidence(
    const Vector2d& sensor_pos, const double& sensor_yaw, const Vector2d& pt_pos)
{
  // Calculate relative position vector from sensor to target point
  Vector2d rel_pos = pt_pos - sensor_pos;
  double angle_to_point = atan2(rel_pos(1), rel_pos(0));

  // Normalize angles to [-π, π] range
  double normalized_sensor_yaw = normalizeAngle(sensor_yaw);
  double normalized_angle_to_point = normalizeAngle(angle_to_point);
  double relative_angle = normalizeAngle(normalized_angle_to_point - normalized_sensor_yaw);

  // Apply cosine-squared FOV confidence model
  double value = std::cos(relative_angle / (fov_angle_ / 2) * (M_PI / 2));
  return value * value;  // Square for stronger center weighting
}
void ValueMap2D::getFreeGrids(vector<Vector2i>& free_grids)
{
}

void ValueMap2D::getFreeGrids(vector<Vector2i>& free_grids,
                               const Vector2d& sensor_pos, const double& sensor_yaw)
{
  free_grids.clear();
  if (!grid_map_->is2DMapReady()) return;

  const auto& occ_data = grid_map_->get2DOccupancyData();
  const nav_msgs::OccupancyGrid& grid = grid_map_->get2DOccupancyGrid();
  int w = grid.info.width;
  int h = grid.info.height;

  // Sensor position in grid indices
  Vector2i sensor_idx;
  posToIndex2D(sensor_pos, sensor_idx);

  // Compute bounding box in grid coordinates for max_depth_ range
  double range_cells = max_depth_ * resolution_inv_;
  int x_min = std::max(0, static_cast<int>(floor(sensor_idx(0) - range_cells)));
  int x_max = std::min(w - 1, static_cast<int>(ceil(sensor_idx(0) + range_cells)));
  int y_min = std::max(0, static_cast<int>(floor(sensor_idx(1) - range_cells)));
  int y_max = std::min(h - 1, static_cast<int>(ceil(sensor_idx(1) + range_cells)));

  // Half FOV angle for gating
  double half_fov = fov_angle_ / 2.0;
  double normalized_sensor_yaw = normalizeAngle(sensor_yaw);

  for (int y = y_min; y <= y_max; ++y) {
    for (int x = x_min; x <= x_max; ++x) {
      int idx = y * w + x;

      // Only consider free cells (0 = free in OccupancyGrid)
      if (occ_data[idx] != 0) continue;

      Vector2i grid_idx(x, y);
      Vector2d grid_pos;
      indexToPos2D(grid_idx, grid_pos);

      // 1. Distance check: within sensor max range
      Vector2d rel_pos = grid_pos - sensor_pos;
      double dist = rel_pos.norm();
      if (dist > max_depth_) continue;

      // 2. FOV angle check: within half FOV of sensor yaw
      double angle_to_point = atan2(rel_pos(1), rel_pos(0));
      double relative_angle = normalizeAngle(normalizeAngle(angle_to_point) - normalized_sensor_yaw);
      if (fabs(relative_angle) > half_fov) continue;

      // 3. Line-of-sight check: no occupied cell between sensor and this grid
      //    Bresenham ray-march from sensor_idx to grid_idx; if any intermediate
      //    cell is occupied, the target is occluded.
      bool occluded = false;
      int dx = abs(x - sensor_idx(0));
      int dy = abs(y - sensor_idx(1));
      int sx = (sensor_idx(0) < x) ? 1 : -1;
      int sy = (sensor_idx(1) < y) ? 1 : -1;
      int err = dx - dy;
      int cx = sensor_idx(0);
      int cy = sensor_idx(1);

      while (cx != x || cy != y) {
        int e2 = 2 * err;
        //int prev_cx = cx, prev_cy = cy;
        if (e2 > -dy) { err -= dy; cx += sx; }
        if (e2 < dx)  { err += dx; cy += sy; }
        // Skip the starting cell and the target cell itself
        if ((cx == sensor_idx(0) && cy == sensor_idx(1)) || (cx == x && cy == y))
          continue;
        // Check bounds
        if (cx < 0 || cx >= w || cy < 0 || cy >= h) break;
        // If an intermediate cell is occupied, the line of sight is blocked
        if (occ_data[cy * w + cx] > 0) {
          occluded = true;
          break;
        }
      }

      if (!occluded) {
        free_grids.push_back(grid_idx);
      }
    }
  }
}
