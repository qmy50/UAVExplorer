/**
 * @file object_map_manager.cpp
 * @brief Integration of YOLO per-instance masks with ObjectMap2D
 *
 * Full pipeline:
 *   ROS subs (SingleMasksWithConfidence + depth + pose + camera_info)
 *     → for each instance: extract 3D points from masked depth pixels
 *     → VoxelGrid downsampling + EuclideanClusterExtraction
 *     → DetectedObject{cloud, label, confidence}
 *     → ObjectMap2D::searchSingleObjectCluster()
 *     → Publish cluster visualization markers
 *
 * Adapted from:
 *   - onboard_detector/cluster.cpp (mask→3D extraction, clustering, ROS I/O)
 *   - ApexNav's map_ros.cpp (detection → ObjectMap2D pipeline)
 */

#include <plan_env/object_map_manager.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>

ObjectMapManager::ObjectMapManager(GridMap::Ptr grid_map, ros::NodeHandle& nh)
  : grid_map_(grid_map)
  , nh_(nh)
  , mask_received_(false)
  , depth_received_(false)
  , pose_received_(false)
  , intrinsics_set_(false)
{
  // Default Realsense intrinsics (overwritten by camera_info or params)
  fx_ = 608.08740234375;
  fy_ = 608.08740234375;
  cx_ = 317.48284912109375;
  cy_ = 234.11557006835938;

  latest_camera_pos_ = Eigen::Vector3d::Zero();
  latest_camera_R_ = Eigen::Matrix3d::Identity();

  // Body to camera transform (from detector_param.yaml / cluster.cpp)
  body_to_camera_ << 0.0,  0.0,  1.0,  0.09,
                    -1.0,  0.0,  0.0,  0.0,
                     0.0, -1.0,  0.0,  0.095,
                     0.0,  0.0,  0.0,  1.0;
                     

  // Create the underlying ObjectMap2D
  object_map_.reset(new ObjectMap2D(grid_map_, nh_));
}

void ObjectMapManager::init()
{
  ros::NodeHandle pnh("~");

  // --- Clustering parameters ---
  pnh.param("depth_min_value", depth_min_value_, 0.5);
  pnh.param("depth_max_value", depth_max_value_, 5.0);
  pnh.param("depth_scale_factor", depth_scale_factor_, 1000.0);
  pnh.param("skip_pixel", skip_pixel_, 2);
  pnh.param("cluster_tolerance", cluster_tolerance_, 0.3);
  pnh.param("min_cluster_size", min_cluster_size_, 100);
  pnh.param("voxel_leaf_size", voxel_leaf_size_, 0.05);

  // --- Camera intrinsics from params (fallback if no camera_info topic) ---
  double fx_param, fy_param, cx_param, cy_param;
  if (pnh.getParam("fx", fx_param) && pnh.getParam("fy", fy_param) &&
      pnh.getParam("cx", cx_param) && pnh.getParam("cy", cy_param)) {
    fx_ = fx_param; fy_ = fy_param; cx_ = cx_param; cy_ = cy_param;
    intrinsics_set_ = true;
    ROS_INFO("[ObjectMapManager] Camera intrinsics from params: fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
             fx_, fy_, cx_, cy_);
  }

  // --- Body-to-camera transform from params ---
  vector<double> b2c_vec;
  if (pnh.getParam("body_to_camera", b2c_vec) && b2c_vec.size() == 16) {
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        body_to_camera_(i, j) = b2c_vec[i * 4 + j];
    ROS_INFO("[ObjectMapManager] Loaded body_to_camera transform from params");
  }

  // --- Topic names ---
  pnh.param<std::string>("single_mask_topic", single_mask_topic_,
                         "/yolo_detector/single_mask");
  pnh.param<std::string>("mask_topic", mask_topic_,
                         "/yolo_detector/mask_image");  // legacy
  pnh.param<std::string>("depth_topic", depth_topic_,
                         "/iris_0/realsense/depth_camera/depth/image_raw");
  pnh.param<std::string>("camera_info_topic", camera_info_topic_,
                         "/iris_0/realsense/depth_camera/color/camera_info");
  pnh.param<std::string>("pose_topic", pose_topic_,
                         "/iris_0/mavros/odometry/in");

  // --- Subscribers ---
  single_mask_sub_ = nh_.subscribe(single_mask_topic_, 10,
                                   &ObjectMapManager::singleMaskCallback, this);
  // mask_sub_ = nh_.subscribe(mask_topic_, 1,
  //                           &ObjectMapManager::maskCallback, this);  // legacy
  depth_sub_ = nh_.subscribe(depth_topic_, 1,
                             &ObjectMapManager::depthCallback, this);
  camera_info_sub_ = nh_.subscribe(camera_info_topic_, 1,
                                   &ObjectMapManager::cameraInfoCallback, this);
  pose_sub_ = nh_.subscribe(pose_topic_, 1,
                            &ObjectMapManager::poseCallback, this);

  // LLM-supplied confidence threshold
  pnh.param<std::string>("conf_thresh_topic", conf_thresh_topic_,
                         "/detector/confidence_threshold");
  conf_thresh_sub_ = nh_.subscribe(conf_thresh_topic_, 1,
                                   &ObjectMapManager::confidenceThresholdCallback, this);

  // --- Publishers ---
  cluster_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
      "cluster_centers_marker", 10);
  cluster_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
      "cluster_cloud", 10);
  cluster_pos_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/cluster_target", 10);

  ROS_INFO("[ObjectMapManager] Initialized:");
  ROS_INFO("  single_mask_topic:  %s", single_mask_topic_.c_str());
  ROS_INFO("  mask_topic (legacy):%s", mask_topic_.c_str());
  ROS_INFO("  depth_topic:        %s", depth_topic_.c_str());
  ROS_INFO("  camera_info_topic:  %s", camera_info_topic_.c_str());
  ROS_INFO("  pose_topic:         %s", pose_topic_.c_str());
  ROS_INFO("  conf_thresh_topic:  %s", conf_thresh_topic_.c_str());
  ROS_INFO("  depth_range: [%.1f, %.1f] m  cluster_tol=%.2fm  min_size=%d  leaf=%.2fm",
           depth_min_value_, depth_max_value_, cluster_tolerance_,
           min_cluster_size_, voxel_leaf_size_);
}

// ==================== ROS Callbacks ====================

void ObjectMapManager::singleMaskCallback(
    const plan_env::SingleMasksWithConfidenceConstPtr& msg)
{
  // Convert mono8 mask Image to cv::Mat
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg->mask, "mono8");
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("[ObjectMapManager] singleMask cv_bridge exception: %s", e.what());
    return;
  }
  // ROS_ERROR("[Object map] We got the yolo message !!!");
  PendingDetection det;
  det.mask = cv_ptr->image.clone();
  det.label_index = msg->label_index;
  det.confidence = msg->confidence;
  det.label_name = msg->label_name;
  // ROS_ERROR("[ObjectMapManager] We got the %s",msg->label_name.c_str());

  pending_detections_.push_back(det);
}

void ObjectMapManager::maskCallback(const sensor_msgs::ImageConstPtr& msg)
{
  // Legacy: BGR8 visual mask
  try {
    latest_mask_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
    mask_received_ = true;
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("[ObjectMapManager] mask cv_bridge exception: %s", e.what());
  }
}

void ObjectMapManager::depthCallback(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("[ObjectMapManager] depth cv_bridge exception: %s", e.what());
    return;
  }

  if (msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
    cv_ptr->image.convertTo(latest_depth_, CV_16UC1, depth_scale_factor_);
  } else if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
    latest_depth_ = cv_ptr->image;
  } else {
    ROS_ERROR_THROTTLE(5.0, "[ObjectMapManager] Unsupported depth encoding: %s",
                       msg->encoding.c_str());
    return;
  }
  depth_received_ = true;

  processIfReady();
}

void ObjectMapManager::cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg)
{
  if (!intrinsics_set_) {
    fx_ = msg->K[0];
    fy_ = msg->K[4];
    cx_ = msg->K[2];
    cy_ = msg->K[5];
    intrinsics_set_ = true;
    ROS_INFO("[ObjectMapManager] Camera intrinsics from camera_info: "
             "fx=%.2f fy=%.2f cx=%.2f cy=%.2f", fx_, fy_, cx_, cy_);
  }
}

void ObjectMapManager::poseCallback(const nav_msgs::OdometryConstPtr& msg)
{
  Eigen::Vector3d body_pos(msg->pose.pose.position.x,
                            msg->pose.pose.position.y,
                            msg->pose.pose.position.z);
  Eigen::Quaterniond body_quat(msg->pose.pose.orientation.w,
                                msg->pose.pose.orientation.x,
                                msg->pose.pose.orientation.y,
                                msg->pose.pose.orientation.z);
  Eigen::Matrix3d R_mb = body_quat.toRotationMatrix();

  Eigen::Matrix3d R_bc = body_to_camera_.block<3, 3>(0, 0);
  Eigen::Vector3d t_bc = body_to_camera_.block<3, 1>(0, 3);

  latest_camera_R_ = R_mb * R_bc;
  latest_camera_pos_ = R_mb * t_bc + body_pos;
  pose_received_ = true;

  ROS_ERROR_THROTTLE(2.0, "[ObjectMapManager] poseCallback: frame=%s body_pos=(%.2f,%.2f,%.2f) yaw=%.1f cam_pos=(%.2f,%.2f,%.2f)",
      msg->header.frame_id.c_str(),
      body_pos.x(), body_pos.y(), body_pos.z(),
      atan2(R_mb(1,0), R_mb(0,0)) * 180.0 / M_PI,
      latest_camera_pos_.x(), latest_camera_pos_.y(), latest_camera_pos_.z());
}

void ObjectMapManager::confidenceThresholdCallback(const std_msgs::Float64ConstPtr& msg)
{
  double val = msg->data;
  object_map_->setConfidenceThreshold(val);
  ROS_WARN_THROTTLE(3.0, "[ObjectMapManager] LLM confidence threshold updated: %.3f", val);
}

void ObjectMapManager::processIfReady()
{
  if (!pose_received_ || !depth_received_ || !intrinsics_set_) {
    ROS_ERROR_THROTTLE(1.0, "[ObjectMapManager] WAIT: pose=%d depth=%d intrinsics=%d",
        pose_received_, depth_received_, intrinsics_set_);
    return;
  }

  object_map_->ensureInitialized();

  // ---- Priority 1: per-instance masks with real labels/confidence ----
  if (!pending_detections_.empty()) {
    // Steal the list so we can process without holding data during callback
    // vector<PendingDetection> batch;
    // batch.swap(pending_detections_);

    vector<PendingDetection> batch = std::move(pending_detections_);
    ROS_ERROR("[ObjectMapManager] Processing %lu pending detections, camera_pos=(%.2f,%.2f,%.2f)",
        batch.size(), latest_camera_pos_.x(), latest_camera_pos_.y(), latest_camera_pos_.z());

    // Collect all cluster centers and clouds for this frame
    vector<Eigen::Vector3d> all_centers;
    pcl::PointCloud<Point3D>::Ptr all_cluster_cloud(new pcl::PointCloud<Point3D>);

    for (auto& det : batch) {
      if (det.mask.empty()) continue;

      int n = processMaskAndDepth(det.mask, latest_depth_,
          latest_camera_pos_, latest_camera_R_,
          det.label_index, det.confidence);

      ROS_ERROR("[ObjectMapManager] processMaskAndDepth returns n=%d for label=%d conf=%.2f",
          n, det.label_index, det.confidence);
    }

    return;
  }

  // ---- Priority 2: legacy BGR8 visual mask (backward compatible) ----
  if (mask_received_) {
    processMaskAndDepth(latest_mask_, latest_depth_,
        latest_camera_pos_, latest_camera_R_, 0, 1.0);
    return;
  }
}

// ==================== Main Pipeline ====================

int ObjectMapManager::processMaskAndDepth(const cv::Mat& mask, const cv::Mat& depth_16u,
    const Eigen::Vector3d& camera_pos, const Eigen::Matrix3d& camera_R,
    int label_index, double confidence)
{
  if (mask.empty() || depth_16u.empty()) {
    ROS_ERROR("[ObjectMapManager] processMaskAndDepth: mask_empty=%d depth_empty=%d",
        mask.empty(), depth_16u.empty());
    return 0;
  }

  ros::WallTime t_start = ros::WallTime::now();

  // Step 1: Extract 3D points from masked depth pixels
  pcl::PointCloud<Point3D>::Ptr raw_cloud(new pcl::PointCloud<Point3D>);
  extractMaskPoints(mask, depth_16u, camera_R, camera_pos, raw_cloud);

  ROS_ERROR("[ObjectMapManager] extractMaskPoints: raw_cloud size = %lu", raw_cloud->size());

  if (raw_cloud->empty()) {
    ROS_ERROR("[ObjectMapManager] FAIL: raw_cloud is empty — check mask/depth alignment and depth range [%.1f, %.1f]",
        depth_min_value_, depth_max_value_);
    publishEmptyMarkers();
    return 0;
  }

  // Step 2: Cluster
  vector<Eigen::Vector3d> cluster_centers;
  auto cluster_clouds = clusterPointCloud(raw_cloud, cluster_centers);

  ROS_ERROR("[ObjectMapManager] clusterPointCloud: %lu clusters found, min_cluster_size=%d tolerance=%.2f",
      cluster_clouds.size(), min_cluster_size_, cluster_tolerance_);

  // Step 3: Collect all clustered points for visualization
  pcl::PointCloud<Point3D>::Ptr all_cluster_cloud(new pcl::PointCloud<Point3D>);

  // Step 4: Feed each cluster to ObjectMap2D with real label + confidence
  int fed_count = 0;
  for (size_t i = 0; i < cluster_clouds.size(); i++) {
    auto& cluster_cloud = cluster_clouds[i];
    if (cluster_cloud->points.empty())
      continue;

    *all_cluster_cloud += *cluster_cloud;

    DetectedObject obj;
    obj.cloud = cluster_cloud;
    obj.score = confidence;      // From YOLO detection
    obj.label = label_index;     // From YOLO classification

    ROS_ERROR("[ObjectMapManager] Feeding cluster %lu: points=%lu score=%.3f label=%d",
        i, cluster_cloud->points.size(), confidence, label_index);

    int cluster_id = object_map_->searchSingleObjectCluster(obj);
    ROS_ERROR("[ObjectMapManager] searchSingleObjectCluster returned cluster_id=%d", cluster_id);
    if (cluster_id >= 0)
      fed_count++;
  }

  // Step 5: Publish visualization
  publishClusterMarkers(cluster_centers);
  publishClusterCloud(all_cluster_cloud);

  double elapsed = (ros::WallTime::now() - t_start).toSec() * 1000.0;
  ROS_DEBUG("[ObjectMapManager] %lu clusters, %d fed, label=%d conf=%.2f, %.1f ms",
            cluster_clouds.size(), fed_count, label_index, confidence, elapsed);

  return fed_count;
}

void ObjectMapManager::processDetectedObjects(const vector<DetectedObject>& detected_objects)
{
  for (const auto& obj : detected_objects) {
    if (obj.cloud->points.empty())
      continue;
    object_map_->searchSingleObjectCluster(obj);
  }
}

// ==================== Query Methods ====================

void ObjectMapManager::getTopConfidenceObjectClouds(
    vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& object_clouds,
    bool limited_confidence, bool extreme)
{
  object_map_->getTopConfidenceObjectCloud(object_clouds, limited_confidence, extreme);
}

void ObjectMapManager::getObjects(
    vector<vector<Vector2d>>& clusters, vector<Vector2d>& averages, vector<int>& labels)
{
  object_map_->getObjects(clusters, averages, labels);
}

void ObjectMapManager::getObjectBoxes(vector<pair<Vector2d, Vector2d>>& boxes)
{
  object_map_->getObjectBoxes(boxes);
}

void ObjectMapManager::getObjectBoxes(vector<pair<Vector3d, Vector3d>>& boxes)
{
  object_map_->getObjectBoxes(boxes);
}

void ObjectMapManager::getObjectBoxes(vector<Vector3d>& bmin, vector<Vector3d>& bmax)
{
  object_map_->getObjectBoxes(bmin, bmax);
}

void ObjectMapManager::setConfidenceThreshold(double val)
{
  object_map_->setConfidenceThreshold(val);
}

// ==================== Private: Mask → 3D Extraction ====================

void ObjectMapManager::extractMaskPoints(const cv::Mat& mask, const cv::Mat& depth_16u,
    const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
    pcl::PointCloud<Point3D>::Ptr cloud)
{
  int H = mask.rows;
  int W = mask.cols;

  if (H == 0 || W == 0) return;

  // Validate depth dimensions match mask
  cv::Mat depth;
  if (depth_16u.rows != H || depth_16u.cols != W) {
    cv::resize(depth_16u, depth, cv::Size(W, H), 0, 0, cv::INTER_NEAREST);
  } else {
    depth = depth_16u;
  }

  double inv_fx = 1.0 / fx_;
  double inv_fy = 1.0 / fy_;
  double inv_scale = 1.0 / depth_scale_factor_;

  cloud->reserve((H / skip_pixel_) * (W / skip_pixel_) / 4);

  // Detect mask type: mono8 (single channel) vs BGR8 (3 channels)
  bool is_mono8 = (mask.channels() == 1);

  int mask_pixel_cnt = 0, depth_valid_cnt = 0, depth_oob_cnt = 0;
  double depth_min_seen = 999, depth_max_seen = 0;
  int sample_u = -1, sample_v = -1;
  double sample_depth = 0;

  for (int v = 0; v < H; v += skip_pixel_) {
    const uint16_t* depth_row = depth.ptr<uint16_t>(v);

    if (is_mono8) {
      // --- mono8 binary mask (from SingleMasksWithConfidence) ---
      const uint8_t* mask_row = mask.ptr<uint8_t>(v);
      for (int u = 0; u < W; u += skip_pixel_) {
        if (mask_row[u] == 0) continue;  // background
        mask_pixel_cnt++;

        double depth_m = static_cast<double>(depth_row[u]) * inv_scale;
        if (depth_m < depth_min_value_ || depth_m > depth_max_value_) {
          depth_oob_cnt++;
          continue;
        }
        depth_valid_cnt++;
        if (depth_m < depth_min_seen) depth_min_seen = depth_m;
        if (depth_m > depth_max_seen) depth_max_seen = depth_m;
        if (sample_u < 0) { sample_u = u; sample_v = v; sample_depth = depth_m; }

        double x_cam = (static_cast<double>(u) - cx_) * depth_m * inv_fx;
        double y_cam = (static_cast<double>(v) - cy_) * depth_m * inv_fy;

        Point3D pt;
        pt.x = R(0, 0) * x_cam + R(0, 1) * y_cam + R(0, 2) * depth_m + t(0);
        pt.y = R(1, 0) * x_cam + R(1, 1) * y_cam + R(1, 2) * depth_m + t(1);
        pt.z = R(2, 0) * x_cam + R(2, 1) * y_cam + R(2, 2) * depth_m + t(2);
        cloud->push_back(pt);
      }
    } else {
      // --- BGR8 visual mask (legacy) ---
      const cv::Vec3b* mask_row = mask.ptr<cv::Vec3b>(v);
      for (int u = 0; u < W; u += skip_pixel_) {
        const cv::Vec3b& pixel = mask_row[u];
        if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0) continue;
        mask_pixel_cnt++;

        double depth_m = static_cast<double>(depth_row[u]) * inv_scale;
        if (depth_m < depth_min_value_ || depth_m > depth_max_value_) {
          depth_oob_cnt++;
          continue;
        }
        depth_valid_cnt++;
        if (depth_m < depth_min_seen) depth_min_seen = depth_m;
        if (depth_m > depth_max_seen) depth_max_seen = depth_m;
        if (sample_u < 0) { sample_u = u; sample_v = v; sample_depth = depth_m; }

        double x_cam = (static_cast<double>(u) - cx_) * depth_m * inv_fx;
        double y_cam = (static_cast<double>(v) - cy_) * depth_m * inv_fy;

        Point3D pt;
        pt.x = R(0, 0) * x_cam + R(0, 1) * y_cam + R(0, 2) * depth_m + t(0);
        pt.y = R(1, 0) * x_cam + R(1, 1) * y_cam + R(1, 2) * depth_m + t(1);
        pt.z = R(2, 0) * x_cam + R(2, 1) * y_cam + R(2, 2) * depth_m + t(2);
        cloud->push_back(pt);
      }
    }
  }

  // ---- Debug: projection summary ----
  if (cloud->empty()) {
    ROS_ERROR("[ObjectMapManager] extractMaskPoints: FAILED — mask_px=%d depth_valid=%d depth_oob=%d (range=[%.1f,%.1f]m) "
              "mask=%dx%d depth=%dx%d skip=%d fx=%.1f cx=%.1f",
        mask_pixel_cnt, depth_valid_cnt, depth_oob_cnt, depth_min_value_, depth_max_value_,
        W, H, depth_16u.cols, depth_16u.rows, skip_pixel_, fx_, cx_);
  } else {
    // Sample: first valid pixel → 3D
    double sx = (sample_u - cx_) * sample_depth / fx_;
    double sy = (sample_v - cy_) * sample_depth / fy_;
    Point3D sp;
    sp.x = R(0,0)*sx + R(0,1)*sy + R(0,2)*sample_depth + t(0);
    sp.y = R(1,0)*sx + R(1,1)*sy + R(1,2)*sample_depth + t(1);
    sp.z = R(2,0)*sx + R(2,1)*sy + R(2,2)*sample_depth + t(2);
    ROS_ERROR("[ObjectMapManager] extractMaskPoints: cloud=%lu pts mask_px=%d depth_valid=%d depth_oob=%d "
              "depth_range=[%.2f,%.2f]m mask=%dx%d skip=%d "
              "SAMPLE: pixel(%d,%d) depth=%.2fm → cam(%.2f,%.2f,%.2f) → world(%.2f,%.2f,%.2f)",
        cloud->size(), mask_pixel_cnt, depth_valid_cnt, depth_oob_cnt,
        depth_min_seen, depth_max_seen, W, H, skip_pixel_,
        sample_u, sample_v, sample_depth, sx, sy, sample_depth,
        sp.x, sp.y, sp.z);
  }
}

// ==================== Private: Euclidean Clustering ====================

vector<pcl::PointCloud<Point3D>::Ptr> ObjectMapManager::clusterPointCloud(
    const pcl::PointCloud<Point3D>::Ptr& cloud, vector<Eigen::Vector3d>& centers)
{
  vector<pcl::PointCloud<Point3D>::Ptr> result;
  centers.clear();

  pcl::PointCloud<Point3D>::Ptr filtered(new pcl::PointCloud<Point3D>);
  pcl::VoxelGrid<Point3D> voxel;
  voxel.setInputCloud(cloud);
  voxel.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
  voxel.filter(*filtered);

  if (filtered->size() < static_cast<size_t>(min_cluster_size_))
    return result;

  pcl::search::KdTree<Point3D>::Ptr kdtree(new pcl::search::KdTree<Point3D>);
  kdtree->setInputCloud(filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<Point3D> ec;
  ec.setClusterTolerance(cluster_tolerance_);
  ec.setMinClusterSize(min_cluster_size_);
  ec.setMaxClusterSize(filtered->size());
  ec.setSearchMethod(kdtree);
  ec.setInputCloud(filtered);
  ec.extract(cluster_indices);

  for (const auto& indices : cluster_indices) {
    pcl::PointCloud<Point3D>::Ptr cluster_cloud(new pcl::PointCloud<Point3D>);

    // Eigen::Vector3d aabb_min(std::numeric_limits<double>::max(),
    //                           std::numeric_limits<double>::max(),
    //                           std::numeric_limits<double>::max());
    // Eigen::Vector3d aabb_max(-std::numeric_limits<double>::max(),
    //                           -std::numeric_limits<double>::max(),
    //                           -std::numeric_limits<double>::max());

    for (int idx : indices.indices) {
      const auto& pt = filtered->points[idx];
      cluster_cloud->push_back(pt);
      // aabb_min = aabb_min.cwiseMin(Eigen::Vector3d(pt.x, pt.y, pt.z));
      // aabb_max = aabb_max.cwiseMax(Eigen::Vector3d(pt.x, pt.y, pt.z));
    }

    // Eigen::Vector3d aabb_center = (aabb_min + aabb_max) * 0.5;

    // std::vector<Eigen::Vector3d> face_centers = {
    //     Eigen::Vector3d(aabb_min.x(), aabb_center.y(), aabb_center.z()),
    //     Eigen::Vector3d(aabb_max.x(), aabb_center.y(), aabb_center.z()),
    //     Eigen::Vector3d(aabb_center.x(), aabb_min.y(), aabb_center.z()),
    //     Eigen::Vector3d(aabb_center.x(), aabb_max.y(), aabb_center.z()),
    //     Eigen::Vector3d(aabb_center.x(), aabb_center.y(), aabb_min.z()),
    //     Eigen::Vector3d(aabb_center.x(), aabb_center.y(), aabb_max.z())
    // };
    // double min_dist = std::numeric_limits<double>::max();
    // Eigen::Vector3d nearest_face = aabb_center;
    // for (const auto& fc : face_centers) {
    //   double d = (fc - latest_camera_pos_).norm();
    //   if (d < min_dist) {
    //     min_dist = d;
    //     nearest_face = fc;
    //   }
    // }
    // centers.push_back(nearest_face);
    result.push_back(cluster_cloud);
  }

  return result;
}

// ==================== Private: Visualization ====================

void ObjectMapManager::publishClusterMarkers(const vector<Eigen::Vector3d>& centers)
{
  visualization_msgs::MarkerArray marker_array;

  if (centers.empty()) {
    publishEmptyMarkers();
    return;
  }

  for (size_t i = 0; i < centers.size(); ++i) {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.ns = "cluster_centers";
    marker.id = static_cast<int>(i);
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = centers[i].x();
    marker.pose.position.y = centers[i].y();
    marker.pose.position.z = centers[i].z();
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.2;
    marker.scale.y = 0.2;
    marker.scale.z = 0.2;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.lifetime = ros::Duration(0.5);
    marker_array.markers.push_back(marker);

    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.frame_id = "map";
    pose_stamped.header.stamp = ros::Time::now();
    pose_stamped.pose.position.x = centers[i].x();
    pose_stamped.pose.position.y = centers[i].y();
    pose_stamped.pose.position.z = centers[i].z();
    pose_stamped.pose.orientation.w = 1.0;
    cluster_pos_pub_.publish(pose_stamped);
  }

  cluster_marker_pub_.publish(marker_array);
}

void ObjectMapManager::publishClusterCloud(const pcl::PointCloud<Point3D>::Ptr& cloud)
{
  if (cloud->empty()) return;
  sensor_msgs::PointCloud2 msg;
  pcl::toROSMsg(*cloud, msg);
  msg.header.frame_id = "map";
  msg.header.stamp = ros::Time::now();
  cluster_cloud_pub_.publish(msg);
}

void ObjectMapManager::publishEmptyMarkers()
{
  visualization_msgs::MarkerArray marker_array;
  visualization_msgs::Marker clear;
  clear.action = visualization_msgs::Marker::DELETEALL;
  marker_array.markers.push_back(clear);
  cluster_marker_pub_.publish(marker_array);
}
