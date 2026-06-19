/**
 * @file object_map2d.cpp
 * @brief Implementation of semantic object mapping system — adapted for GridMap
 *
 * Provides semantic object detection, multi-view fusion, confidence scoring,
 * and 3D point cloud processing on top of GridMap's 2D occupancy layer.
 * Adapted from ApexNav's ObjectMap2D (originally built on SDFMap2D).
 *
 * @author Adapted from ApexNav (Zager-Zhang)
 */

#include <plan_env/object_map2d.h>

ObjectMap2D::ObjectMap2D(GridMap::Ptr grid_map, ros::NodeHandle& nh)
{
  // Store reference to the 3D GridMap (we work on its 2D projection layer)
  this->grid_map_ = grid_map;
  initialized_ = false;
  voxel_num_ = 0;

  // Initialize point cloud containers
  all_object_clouds_.reset(new pcl::PointCloud<pcl::PointXYZ>());
  over_depth_object_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>());

  // Load configuration parameters
  min_confidence_ = -1.0;  // Default to accept all detections
  nh.param("object/min_observation_num", min_observation_num_, 10);
  nh.param("object/fusion_type", fusion_type_, 1);
  nh.param("object/use_observation", use_observation_, true);
  nh.param("object/vis_cloud", is_vis_cloud_, true);

  // Setup ROS communication
  object_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/object/clouds", 10);

  // Resolution will be set from the 2D grid once initialized
  resolution_ = grid_map_->getRes();
  resolution_inv_ = 1.0 / resolution_;

  // Set point cloud processing parameters
  leaf_size_ = 0.1f;  // Voxel grid leaf size for downsampling
}

void ObjectMap2D::ensureInitialized()
{
  if (initialized_)
    return;

  if (!grid_map_->is2DMapReady())
    return;

  const nav_msgs::OccupancyGrid& grid = grid_map_->get2DOccupancyGrid();
  int new_width = grid.info.width;
  int new_height = grid.info.height;

  width_ = new_width;
  height_ = new_height;
  voxel_num_ = width_ * height_;
  origin_x_ = grid.info.origin.position.x;
  origin_y_ = grid.info.origin.position.y;
  resolution_ = grid.info.resolution;
  resolution_inv_ = 1.0 / resolution_;

  // Allocate object indexing buffers (sized to the full 2D grid)
  object_buffer_ = vector<char>(voxel_num_, 0);
  object_indexs_ = vector<int>(voxel_num_, -1);

  initialized_ = true;

  ROS_INFO("[ObjectMap2D] Initialized: %dx%d, origin=(%.2f,%.2f), res=%.3f",
           width_, height_, origin_x_, origin_y_, resolution_);
}

void ObjectMap2D::setConfidenceThreshold(double val)
{
  min_confidence_ = val;
  ROS_INFO("Set Confidence Threshold = %f", val);
}

/**
 * @brief Process observation clouds to adjust detection confidence
 *
 * Handles negative evidence from visual observations where objects were
 * expected but not detected. Computes spatial overlap between observation
 * regions and existing detections to reduce confidence scores.
 */
void ObjectMap2D::inputObservationObjectsCloud(
    const vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>> observation_clouds,
    const double& itm_score)
{
  // Only process observations in fusion mode with observation enabled
  if (fusion_type_ != 1 || !use_observation_)
    return;

  // Process each observation cloud against corresponding objects
  for (int i = 0; i < (int)observation_clouds.size(); i++) {
    auto observation_cloud = observation_clouds[i];
    auto object = objects_[i];

    if (observation_cloud->points.empty())
      continue;

    // Check overlap with each possible object classification
    for (int label = 0; label < 5; ++label) {
      if (object.confidence_scores_[label] < 1e-3)
        continue;  // Skip labels with negligible confidence

      // Setup spatial search for overlap computation
      pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
      kdtree.setInputCloud(observation_cloud);
      double distance_threshold = leaf_size_ * 1.1;  // Spatial overlap threshold
      int overlap_count = 0;

      // Count overlapping points between object and observation clouds
      for (const auto& point : object.clouds_[label]->points) {
        std::vector<int> point_idx_search;
        std::vector<float> point_squared_distance;
        if (kdtree.nearestKSearch(point, 1, point_idx_search, point_squared_distance) > 0) {
          // Points within threshold are considered overlapping
          if (point_squared_distance[0] <= distance_threshold * distance_threshold) {
            overlap_count++;
          }
        }
      }

      // Skip if no spatial overlap detected
      if (overlap_count == 0)
        continue;

      // Update confidence scores based on negative observation evidence
      auto& merged_object = objects_[i];
      merged_object.observation_cloud_sums_[label] += overlap_count;
      int total_last = merged_object.clouds_[label]->points.size();
      double confidence_last = merged_object.confidence_scores_[label];
      int observation_now = overlap_count;
      double confidence_now = 0.0;  // Negative evidence has zero confidence
      if (label == 0)
        confidence_now = itm_score;  // Use ITM score for primary label
      int total_now = merged_object.clouds_[label]->points.size();

      // Apply confidence fusion algorithm
      merged_object.confidence_scores_[label] = fusionConfidenceScore(total_last, confidence_last,
          observation_now, confidence_now, total_now, merged_object.observation_cloud_sums_[label]);
      printFusionInfo(merged_object, label, "[Observation]");
    }
  }
}

int ObjectMap2D::searchSingleObjectCluster(const DetectedObject& detected_object)
{
  ensureInitialized();
  if (!initialized_)
    return -1;

  auto object_cloud = detected_object.cloud;

  // Initialize clustering analysis variables
  int point_num = object_cloud->points.size();
  int obj_idx = -1;
  vector<Eigen::Vector2d> object_point2Ds;
  vector<char> flag_2d(voxel_num_, 0);  // Duplicate point prevention

  int out_of_bounds = 0, dup_count = 0, not_satisfy = 0, satisfy_count = 0;
  ROS_ERROR("[ObjectMap2D] searchSingleObjectCluster ENTER: cloud_points=%d label=%d score=%.3f initialized=%d",
      point_num, detected_object.label, detected_object.score, initialized_);

  // Process each point in the detected object cloud
  for (int i = 0; i < point_num; i++) {
    Eigen::Vector2i idx;
    Eigen::Vector2d pt_w;
    pt_w << object_cloud->points[i].x, object_cloud->points[i].y;
    posToIndex2D(pt_w, idx);
    int adr = toAddress2D(idx);

    // Skip out-of-bounds or duplicate grid cells
    if (adr < 0 || adr >= voxel_num_) {
      out_of_bounds++;
      continue;
    }
    if (flag_2d[adr] == 1) {
      dup_count++;
      continue;
    }

    flag_2d[adr] = 1;

    // Validate if point satisfies object characteristics
    // DISABLED: no longer require 2D occupancy grid to be occupied
    // if (isSatisfyObject(pt_w)) {
      object_buffer_[adr] = 1;  // Mark cell as containing object
      object_point2Ds.push_back(pt_w);
      satisfy_count++;
    // }
  }

  ROS_ERROR("[ObjectMap2D] Point stats: total_3d=%d unique_2d=%d oob=%d dup=%d",
      point_num, satisfy_count, out_of_bounds, dup_count);

  // Return early if no valid object points found
  if (object_point2Ds.empty()) {
    ROS_ERROR("[ObjectMap2D] FAIL: no valid 2D grid cells — all points out of bounds!");
    return -1;
  }

  // Search for existing object clusters in neighborhood
  for (auto pt_w : object_point2Ds) {
    Eigen::Vector2i idx;
    posToIndex2D(pt_w, idx);

    // Get neighboring grid cells within clustering radius
    auto nbrs = allGridsDistance(idx, 0.08);
    nbrs.push_back(idx);

    // Check neighbors for existing object associations
    for (auto nbr : nbrs) {
      if (!isInMap2D(nbr))
        continue;
      int nbr_adr = toAddress2D(nbr);
      if (object_indexs_[nbr_adr] != -1) {
        // Found existing object cluster - use first match
        obj_idx = object_indexs_[nbr_adr];
        break;
      }
    }
  }

  // Apply voxel grid filtering to reduce point cloud density
  pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
  voxel_filter.setInputCloud(object_cloud);
  voxel_filter.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
  voxel_filter.filter(*detected_object.cloud);

  // Either merge with existing cluster or create new one
  if (obj_idx != -1) {
    mergeCellsIntoObjectCluster(obj_idx, object_point2Ds, detected_object);
  }
  else {
    createNewObjectCluster(object_point2Ds, detected_object);
    obj_idx = object_indexs_[toAddress2D(object_point2Ds[0])];
  }

  // Update classification and visualization
  updateObjectBestLabel(obj_idx);
  if (is_vis_cloud_)
    publishObjectClouds();

  // Validate successful clustering
  if (obj_idx == -1) {
    ROS_ERROR("[bug] Why not find the object cluster!?");
    return obj_idx;
  }

  return obj_idx;
}

void ObjectMap2D::updateObjectBestLabel(int obj_idx)
{
  double max_func_score = 0.1;  // Minimum threshold for valid classification
  int best_label = -1;

  // Evaluate each possible classification label
  for (int label = 0; label < (int)objects_[obj_idx].clouds_.size(); label++) {
    auto obs_sum = objects_[obj_idx].observation_cloud_sums_[label];
    auto score = objects_[obj_idx].confidence_scores_[label];
    int func_score = obs_sum * score;  // Combined reliability metric

    ROS_ERROR("[ObjectMap2D] updateBestLabel obj=%d label=%d obs_sum=%d score=%.3f func_score=%d",
        obj_idx, label, obs_sum, score, func_score);

    if (func_score > max_func_score) {
      max_func_score = func_score;
      best_label = label;
    }
  }
  objects_[obj_idx].best_label_ = best_label;
  ROS_ERROR("[ObjectMap2D] updateBestLabel obj=%d → best_label=%d (max_func_score=%d, threshold=0.1)",
      obj_idx, best_label, (int)max_func_score);
}

void ObjectMap2D::createNewObjectCluster(
    const std::vector<Eigen::Vector2d>& cells, const DetectedObject& detected_object)
{
  int label = detected_object.label;

  // Initialize new object cluster with unique ID
  ObjectCluster obj;
  obj.id_ = (int)objects_.size();
  obj.max_seen_count_ = 0;
  obj.good_cells_.clear();
  obj.seen_counts_.clear();
  obj.best_label_ = -1;

  // Process spatial cells and establish grid associations
  std::vector<Eigen::Vector2d> real_new_cells;
  for (auto cell : cells) {
    int adr = toAddress2D(cell);
    if (adr < 0 || adr >= voxel_num_)
      continue;
    object_indexs_[adr] = obj.id_;  // Associate grid cell with object
    obj.visited_[adr] = 1;

    // Track high-confidence observations for label 0
    if (label == 0) {
      obj.seen_counts_[adr] = 1;
      obj.max_seen_count_ = 1;
      obj.good_cells_.push_back(cell);
    }
  }

  // Compute spatial properties of the object cluster
  obj.cells_ = cells;
  obj.average_.setZero();
  obj.box_max2d_ = obj.cells_.front();
  obj.box_min2d_ = obj.cells_.front();

  for (auto cell : obj.cells_) {
    obj.average_ += cell;
    for (int i = 0; i < 2; ++i) {
      obj.box_min2d_[i] = min(obj.box_min2d_[i], cell[i]);
      obj.box_max2d_[i] = max(obj.box_max2d_[i], cell[i]);
    }
  }
  obj.average_ /= double(obj.cells_.size());

  // Initialize point cloud storage for the detected label
  obj.clouds_[label].reset(new pcl::PointCloud<pcl::PointXYZ>());
  *obj.clouds_[label] = *detected_object.cloud;

  // Compute 3D bounding box from point cloud
  obj.box_max3d_ = Vector3d(obj.clouds_[label]->points[0].x, obj.clouds_[label]->points[0].y,
      obj.clouds_[label]->points[0].z);
  obj.box_min3d_ = Vector3d(obj.clouds_[label]->points[0].x, obj.clouds_[label]->points[0].y,
      obj.clouds_[label]->points[0].z);

  for (auto pt : obj.clouds_[label]->points) {
    Vector3d vec_pt = Vector3d(pt.x, pt.y, pt.z);
    for (int i = 0; i < 3; ++i) {
      obj.box_min3d_[i] = min(obj.box_min3d_[i], vec_pt[i]);
      obj.box_max3d_[i] = max(obj.box_max3d_[i], vec_pt[i]);
    }
  }

  // Initialize confidence tracking for this object
  obj.confidence_scores_[label] = detected_object.score;
  obj.observation_cloud_sums_[label] = detected_object.cloud->points.size();
  obj.observation_nums_[label] = 1;

  // Add to global object registry
  objects_.push_back(obj);
  printFusionInfo(obj, label, "[New Object Cluster]");
}

void ObjectMap2D::mergeCellsIntoObjectCluster(const int& merged_object_id,
    const std::vector<Eigen::Vector2d>& new_cells, const DetectedObject& detected_object)
{
  int label = detected_object.label;
  const auto last_objects = objects_;

  ObjectCluster& merged_object = objects_[merged_object_id];
  std::vector<Eigen::Vector2d> real_new_cells;

  // Process new spatial cells for integration
  for (auto new_cell : new_cells) {
    int adr = toAddress2D(new_cell);
    if (adr < 0 || adr >= voxel_num_)
      continue;
    object_indexs_[adr] = merged_object_id;  // Associate with this object cluster

    // Add only genuinely new cells to avoid duplicates
    if (!merged_object.visited_.count(adr)) {
      real_new_cells.push_back(new_cell);
      merged_object.visited_[adr] = 1;
    }

    // Track observation frequency for high-confidence detections (label 0)
    if (label == 0) {
      if (merged_object.seen_counts_.count(adr))
        merged_object.seen_counts_[adr] += 1;
      else
        merged_object.seen_counts_[adr] = 1;

      // Update maximum observation count for this cluster
      if (merged_object.seen_counts_[adr] > merged_object.max_seen_count_)
        merged_object.max_seen_count_ = merged_object.seen_counts_[adr];
    }
  }

  // Extend cluster's spatial coverage
  merged_object.cells_.insert(
      merged_object.cells_.end(), real_new_cells.begin(), real_new_cells.end());

  // Update high-confidence cell tracking for label 0
  if (label == 0) {
    merged_object.good_cells_.clear();
    for (auto cell : merged_object.cells_) {
      int adr = toAddress2D(cell);
      if (merged_object.seen_counts_.count(adr)) {
        // Cells with sufficient observations are considered "good"
        if (merged_object.seen_counts_[adr] >= min(4, merged_object.max_seen_count_))
          merged_object.good_cells_.push_back(cell);
      }
    }
    ROS_ERROR("merged_object good cells size = %ld", merged_object.good_cells_.size());
  }

  // Recompute spatial properties
  merged_object.average_.setZero();
  merged_object.box_max2d_ = merged_object.cells_.front();
  merged_object.box_min2d_ = merged_object.cells_.front();
  for (auto cell : merged_object.cells_) {
    merged_object.average_ += cell;
    for (int i = 0; i < 2; ++i) {
      merged_object.box_min2d_[i] = min(merged_object.box_min2d_[i], cell[i]);
      merged_object.box_max2d_[i] = max(merged_object.box_max2d_[i], cell[i]);
    }
  }
  merged_object.average_ /= double(merged_object.cells_.size());

  // Handle point cloud fusion based on observation history
  if (!merged_object.observation_nums_[label]) {
    // First observation of this label - initialize directly
    merged_object.clouds_[label].reset(new pcl::PointCloud<pcl::PointXYZ>());
    *merged_object.clouds_[label] = *(detected_object.cloud);
    merged_object.confidence_scores_[label] = detected_object.score;
    merged_object.observation_cloud_sums_[label] = detected_object.cloud->points.size();
    merged_object.observation_nums_[label] = 1;
    printFusionInfo(merged_object, label, "[New Label Merged]");
  }
  else {
    // Merge with existing observations using point cloud fusion
    pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    *merged_cloud = *(merged_object.clouds_[label]);  // Copy existing cloud
    *merged_cloud += *(detected_object.cloud);        // Add new observations

    // Apply voxel grid downsampling to manage point cloud size
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(merged_cloud);
    voxel_filter.setLeafSize(leaf_size_, leaf_size_, leaf_size_);
    voxel_filter.filter(*merged_cloud);
    merged_object.clouds_[label] = merged_cloud;

    // Update 3D bounding box from merged point cloud
    merged_object.box_max3d_ = Vector3d(merged_object.clouds_[label]->points[0].x,
        merged_object.clouds_[label]->points[0].y, merged_object.clouds_[label]->points[0].z);
    merged_object.box_min3d_ = Vector3d(merged_object.clouds_[label]->points[0].x,
        merged_object.clouds_[label]->points[0].y, merged_object.clouds_[label]->points[0].z);

    for (auto pt : merged_object.clouds_[label]->points) {
      Vector3d vec_pt = Vector3d(pt.x, pt.y, pt.z);
      for (int i = 0; i < 3; ++i) {
        merged_object.box_min3d_[i] = min(merged_object.box_min3d_[i], vec_pt[i]);
        merged_object.box_max3d_[i] = max(merged_object.box_max3d_[i], vec_pt[i]);
      }
    }

    // Update observation tracking
    merged_object.observation_nums_[label]++;
    merged_object.observation_cloud_sums_[label] += detected_object.cloud->points.size();

    // Prepare confidence fusion parameters
    int last_total = last_objects[merged_object_id].clouds_[label]->points.size();
    double last_total_confidence = last_objects[merged_object_id].confidence_scores_[label];
    int now_observation = detected_object.cloud->points.size();
    double now_confidence = detected_object.score;
    int now_total = merged_object.clouds_[label]->points.size();

    // Apply confidence fusion strategy based on fusion type
    if (fusion_type_ == 0)
      merged_object.confidence_scores_[label] = now_confidence;  // Replace with current
    else if (fusion_type_ == 1)
      merged_object.confidence_scores_[label] =
          fusionConfidenceScore(last_total, last_total_confidence, now_observation, now_confidence,
              now_total, merged_object.observation_cloud_sums_[label]);  // Weighted fusion
    else if (fusion_type_ == 2)
      merged_object.confidence_scores_[label] =
          max(merged_object.confidence_scores_[label], now_confidence);  // Maximum confidence
    printFusionInfo(merged_object, label, "[Fusion]");
  }
}

/**
 * @brief Fusion algorithm for combining confidence scores from multiple observations
 *
 * Implements a weighted confidence fusion strategy that combines historical
 * confidence with new observations, considering both the quantity of evidence
 * and the quality of individual detections.
 */
double ObjectMap2D::fusionConfidenceScore(
    int total_num_last, double c_last, int n_num_now, double c_now, int total_now, int sum)
{
  double n_now = (double)n_num_now;
  double w_last, w_now, final_score;
  // Calculate weighted fusion based on observation counts
  w_last = (sum - n_now) / sum;                   // Weight for historical evidence
  w_now = n_now / sum;                            // Weight for current observation
  final_score = w_last * c_last + w_now * c_now;  // Weighted combination
  return final_score;
}

void ObjectMap2D::getObjects(
    vector<vector<Eigen::Vector2d>>& clusters, vector<Vector2d>& averages, vector<int>& labels)
{
  clusters.clear();
  averages.clear();
  labels.clear();
  for (auto object : objects_) {
    clusters.push_back(object.cells_);
    averages.push_back(object.average_);
    labels.push_back(object.best_label_);
  }
}

void ObjectMap2D::getObjectBoxes(vector<pair<Eigen::Vector2d, Eigen::Vector2d>>& boxes)
{
  boxes.clear();
  for (auto object : objects_) {
    Vector2d center = (object.box_max2d_ + object.box_min2d_) * 0.5;
    Vector2d scale = object.box_max2d_ - object.box_min2d_;
    boxes.push_back(make_pair(center, scale));
  }
}

void ObjectMap2D::getObjectBoxes(vector<pair<Eigen::Vector3d, Eigen::Vector3d>>& boxes)
{
  boxes.clear();
  for (auto object : objects_) {
    Vector3d center = (object.box_max3d_ + object.box_min3d_) * 0.5;
    Vector3d scale = object.box_max3d_ - object.box_min3d_;
    boxes.push_back(make_pair(center, scale));
  }
}

void ObjectMap2D::getObjectBoxes(vector<Vector3d>& bmin, vector<Vector3d>& bmax)
{
  bmin.clear();
  bmax.clear();
  for (auto object : objects_) {
    bmin.push_back(object.box_min3d_);
    bmax.push_back(object.box_max3d_);
  }
}

bool ObjectMap2D::getBestObjectTarget(Eigen::Vector3d& target, const Eigen::Vector3d& drone_pos)
{
  // Find the highest-confidence valid object and pick the nearest AABB face center
  double best_score = -1.0;
  int best_idx = -1;
  Eigen::Vector3d best_face = Eigen::Vector3d::Zero();

  for (size_t i = 0; i < objects_.size(); i++) {
    const auto& obj = objects_[i];
    int label = obj.best_label_;
    if (label < 0 || label >= (int)obj.clouds_.size()) continue;
    if (obj.confidence_scores_[label] < min_confidence_) continue;

    // Compute 6 AABB face centers
    Eigen::Vector3d center = (obj.box_min3d_ + obj.box_max3d_) * 0.5;
    std::vector<Eigen::Vector3d> faces = {
      Eigen::Vector3d(obj.box_min3d_.x(), center.y(), center.z()),
      Eigen::Vector3d(obj.box_max3d_.x(), center.y(), center.z()),
      Eigen::Vector3d(center.x(), obj.box_min3d_.y(), center.z()),
      Eigen::Vector3d(center.x(), obj.box_max3d_.y(), center.z()),
      Eigen::Vector3d(center.x(), center.y(), obj.box_min3d_.z()),
      Eigen::Vector3d(center.x(), center.y(), obj.box_max3d_.z()),
    };

    // Find nearest face to drone
    double min_dist = std::numeric_limits<double>::max();
    Eigen::Vector3d nearest_face = center;
    for (const auto& fc : faces) {
      double d = (fc - drone_pos).norm();
      if (d < min_dist) {
        min_dist = d;
        nearest_face = fc;
      }
    }

    // Score: confidence weighted by inverse distance
    double score = obj.confidence_scores_[label] / (min_dist + 0.5);
    if (score > best_score) {
      best_score = score;
      best_idx = i;
      best_face = nearest_face;
    }
  }

  if (best_idx < 0) return false;
  target = best_face;
  return true;
}

void ObjectMap2D::getAllConfidenceObjectClouds(
    pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>& object_clouds)
{
  object_clouds.reset(new pcl::PointCloud<pcl::PointXYZ>());

  // Extract high-confidence object cells
  for (auto object : objects_) {
    if (object.confidence_scores_[0] >= min_confidence_) {
      for (auto cell : object.good_cells_) {
        pcl::PointXYZ point;
        point.x = cell[0];
        point.y = cell[1];
        point.z = 0;
        object_clouds->push_back(point);
      }
    }
  }
}

void ObjectMap2D::getTopConfidenceObjectCloud(
    vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& top_object_clouds,
    bool limited_confidence, bool extreme)
{
  top_object_clouds.clear();
  vector<ObjectCluster> top_objects;

  // Confidence filtering strategy
  if (!limited_confidence) {
    // Include all objects without confidence filtering
    for (auto object : objects_) top_objects.push_back(object);

    // Sort by confidence score in descending order
    std::sort(
        top_objects.begin(), top_objects.end(), [](const ObjectCluster& a, const ObjectCluster& b) {
          return a.confidence_scores_[0] > b.confidence_scores_[0];
        });

    // Extract point clouds for top-ranked objects
    for (auto top_obj : top_objects) {
      if (top_obj.confidence_scores_[0] <= 0.01)
        break;  // Skip extremely low confidence objects

      pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> top_object_cloud;
      top_object_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());

      for (auto cell : top_obj.good_cells_) {
        pcl::PointXYZ point;
        point.x = cell(0);
        point.y = cell(1);
        point.z = 0;
        top_object_cloud->push_back(point);
      }
      top_object_clouds.push_back(top_object_cloud);
    }

    // Fallback for extreme mode when no high-confidence objects exist
    if (extreme && top_object_clouds.empty()) {
      pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> others_object_cloud;
      others_object_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());

      // Include all object cells regardless of confidence
      for (auto object : objects_) {
        for (auto cell : object.cells_) {
          pcl::PointXYZ point;
          point.x = cell(0);
          point.y = cell(1);
          point.z = 0;
          others_object_cloud->push_back(point);
        }
      }
      top_object_clouds.push_back(others_object_cloud);
    }
  }
  else {
    // Apply confidence filtering with functional scoring
    for (auto object : objects_) {
      int max_func_score = 0, best_label = -1;

      // Find best label using functional score (observation count * confidence)
      for (int label = 0; label < (int)object.clouds_.size(); label++) {
        auto obs_sum = object.observation_cloud_sums_[label];
        auto score = object.confidence_scores_[label];
        int func_score = obs_sum * score;
        if (func_score > max_func_score) {
          max_func_score = func_score;
          best_label = label;
        }
      }

      // Include only high-confidence objects with primary label (0)
      if (best_label == 0 && isConfidenceObject(object))
        top_objects.push_back(object);
    }

    // Sort filtered objects by confidence
    std::sort(
        top_objects.begin(), top_objects.end(), [](const ObjectCluster& a, const ObjectCluster& b) {
          return a.confidence_scores_[0] > b.confidence_scores_[0];
        });

    // Extract point clouds from filtered objects
    for (auto top_obj : top_objects) {
      pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>> top_object_cloud;
      top_object_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>());
      for (auto cell : top_obj.good_cells_) {
        pcl::PointXYZ point;
        point.x = cell(0);
        point.y = cell(1);
        point.z = 0;
        top_object_cloud->push_back(point);
      }
      top_object_clouds.push_back(top_object_cloud);
    }
  }
}
