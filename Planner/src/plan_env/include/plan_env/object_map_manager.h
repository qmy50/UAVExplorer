/**
 * @file object_map_manager.h
 * @brief Integration layer — clustering pipeline (from cluster.cpp) + ObjectMap2D
 *
 * ObjectMapManager bridges the gap between raw sensor data (mask + depth)
 * and the ObjectMap2D multi-frame object mapping system. It:
 *   1. Subscribes to per-instance YOLO masks (SingleMasksWithConfidence),
 *      depth, camera_info, and pose topics
 *   2. Extracts 3D points from per-instance segmented depth regions
 *   3. Applies voxel downsampling + Euclidean cluster extraction
 *   4. Wraps each cluster as a DetectedObject (with real confidence + label)
 *   5. Feeds them into ObjectMap2D::searchSingleObjectCluster()
 *   6. Publishes cluster markers/clouds for visualization
 *   7. Exposes query methods for the planner
 *
 * Adapted from onboard_detector/cluster.cpp (mask→3D extraction + clustering)
 * and ApexNav's MapROS (detection → ObjectMap2D pipeline).
 */

#ifndef _OBJECT_MAP_MANAGER_H_
#define _OBJECT_MAP_MANAGER_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <string>

// ROS messages
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float64.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <cv_bridge/cv_bridge.h>

// Custom messages
#include <plan_env/SingleMasksWithConfidence.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

// Internal
#include <plan_env/object_map2d.h>
#include <plan_env/grid_map_indoor.h>

using Eigen::Vector2d;
using Eigen::Vector2i;
using Eigen::Vector3d;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

typedef pcl::PointXYZ Point3D;

class ObjectMapManager {
public:
  ObjectMapManager(GridMap::Ptr grid_map, ros::NodeHandle& nh);
  ~ObjectMapManager() = default;

  /// Load parameters, setup subscribers & publishers
  void init();

  /**
   * @brief Full pipeline: mask + depth → cluster → ObjectMap2D
   * @param mask      Binary mono8 mask (0=bg, 255=object)
   * @param depth_16u Depth image in 16UC1 (mm), must match mask dimensions
   * @param camera_pos Camera position in world/map frame
   * @param camera_R   Camera orientation matrix (camera→world)
   * @param label_index Numeric class label
   * @param confidence  Detection confidence [0,1]
   * @return Number of object clusters fed to ObjectMap2D this frame
   */
  int processMaskAndDepth(const cv::Mat& mask, const cv::Mat& depth_16u,
      const Eigen::Vector3d& camera_pos, const Eigen::Matrix3d& camera_R,
      int label_index = 0, double confidence = 1.0);

  /**
   * @brief Direct feed: pass already-detected objects to ObjectMap2D
   */
  void processDetectedObjects(const vector<DetectedObject>& detected_objects);

  // ---- Query methods (delegated to ObjectMap2D) ----

  void getTopConfidenceObjectClouds(
      vector<pcl::shared_ptr<pcl::PointCloud<pcl::PointXYZ>>>& object_clouds,
      bool limited_confidence = true, bool extreme = false);

  void getObjects(
      vector<vector<Vector2d>>& clusters, vector<Vector2d>& averages, vector<int>& labels);

  void getObjectBoxes(vector<pair<Vector2d, Vector2d>>& boxes);
  void getObjectBoxes(vector<pair<Vector3d, Vector3d>>& boxes);
  void getObjectBoxes(vector<Vector3d>& bmin, vector<Vector3d>& bmax);

  bool getBestObjectTarget(Eigen::Vector3d& target, const Eigen::Vector3d& drone_pos)
  {
    return object_map_->getBestObjectTarget(target, drone_pos);
  }

  void setConfidenceThreshold(double val);

  /// Access underlying ObjectMap2D (e.g., for over_depth_object_cloud_)
  ObjectMap2D* getObjectMap() { return object_map_.get(); }

  /// Camera intrinsics (set once, or loaded from camera_info topic)
  void setCameraIntrinsics(double fx, double fy, double cx, double cy)
  {
    fx_ = fx; fy_ = fy; cx_ = cx; cy_ = cy;
    intrinsics_set_ = true;
  }
  void publishEmptyMarkers();
  void publishEmptyCloud();

private:
  // ==================== ROS Callbacks ====================

  /// New per-instance mask message (recommended)
  void singleMaskCallback(const plan_env::SingleMasksWithConfidenceConstPtr& msg);

  /// Legacy: single BGR8 visual mask (backward compatible)
  void maskCallback(const sensor_msgs::ImageConstPtr& msg);

  void depthCallback(const sensor_msgs::ImageConstPtr& msg);
  void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg);
  void poseCallback(const nav_msgs::OdometryConstPtr& msg);

  /// LLM-supplied confidence threshold for ObjectMap2D gate
  void confidenceThresholdCallback(const std_msgs::Float64ConstPtr& msg);

  /// Check if all sensor data is ready, then run the pipeline
  void processIfReady();

  // ==================== Pipeline Internals ====================

  /// Extract 3D points from binary mask (mono8) + depth
  void extractMaskPoints(const cv::Mat& mask, const cv::Mat& depth_16u,
      const Eigen::Matrix3d& R, const Eigen::Vector3d& t,
      pcl::PointCloud<Point3D>::Ptr cloud);

  /// Cluster a point cloud using EuclideanClusterExtraction
  vector<pcl::PointCloud<Point3D>::Ptr> clusterPointCloud(
      const pcl::PointCloud<Point3D>::Ptr& cloud, vector<Eigen::Vector3d>& centers);

  // ==================== Visualization ====================

  void publishClusterMarkers(const vector<Eigen::Vector3d>& centers);
  void publishClusterCloud(const pcl::PointCloud<Point3D>::Ptr& cloud);


  // ==================== Data Members ====================

  // Core components
  GridMap::Ptr grid_map_;
  unique_ptr<ObjectMap2D> object_map_;
  ros::NodeHandle nh_;

  // ---- Per-instance pending detections ----
  struct PendingDetection {
    cv::Mat mask;             // mono8 binary mask
    int label_index;
    float confidence;
    std::string label_name;
  };
  vector<PendingDetection> pending_detections_;

  // ---- ROS Subscribers ----
  ros::Subscriber single_mask_sub_;  ///< New: per-instance mask + label + conf
  ros::Subscriber mask_sub_;         ///< Legacy: BGR8 visual mask
  ros::Subscriber depth_sub_;
  ros::Subscriber camera_info_sub_;
  ros::Subscriber pose_sub_;
  ros::Subscriber conf_thresh_sub_;  ///< LLM confidence threshold from YOLO bridge

  // ---- ROS Publishers (visualization) ----
  ros::Publisher cluster_marker_pub_;
  ros::Publisher cluster_cloud_pub_;
  ros::Publisher cluster_pos_pub_;

  // ---- Latest sensor data ----
  cv::Mat latest_mask_;            ///< Legacy BGR8 mask (backward compat)
  cv::Mat latest_depth_;
  Eigen::Vector3d latest_camera_pos_;
  Eigen::Matrix3d latest_camera_R_;
  bool mask_received_;
  bool depth_received_;
  bool pose_received_;

  // ---- Camera configuration ----
  double fx_, fy_, cx_, cy_;
  bool intrinsics_set_;

  /// Body-to-camera transform matrix (4x4)
  Eigen::Matrix4d body_to_camera_;

  // ---- Topic names ----
  std::string single_mask_topic_;  ///< New topic
  std::string mask_topic_;         ///< Legacy topic
  std::string depth_topic_;
  std::string camera_info_topic_;
  std::string pose_topic_;  ///< Odometry topic for camera pose
  std::string conf_thresh_topic_;  ///< Topic for LLM confidence threshold

  // ---- Clustering parameters ----
  double depth_min_value_;
  double depth_max_value_;
  double depth_scale_factor_;
  int skip_pixel_;
  double cluster_tolerance_;
  int min_cluster_size_;
  double voxel_leaf_size_;
};

#endif  // _OBJECT_MAP_MANAGER_H_
