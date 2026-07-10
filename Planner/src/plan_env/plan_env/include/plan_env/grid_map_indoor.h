#ifndef _GRID_MAP_INDOOR_H
#define _GRID_MAP_INDOOR_H

#include <memory>
#include <mutex>
#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/OccupancyGrid.h>
#include <queue>
#include <ros/ros.h>
#include <tuple>
#include <visualization_msgs/Marker.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>

#include <plan_env/raycast.h>

#define logit(x) (log((x) / (1 - (x))))

using namespace std;

// voxel hashing
template <typename T>
struct matrix_hash : std::unary_function<T, size_t> {
  std::size_t operator()(T const& matrix) const {
    size_t seed = 0;
    for (size_t i = 0; i < matrix.size(); ++i) {
      auto elem = *(matrix.data() + i);
      seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

struct MappingParameters {
  Eigen::Vector3d map_origin_, map_size_;
  Eigen::Vector3d map_min_boundary_, map_max_boundary_;
  Eigen::Vector3i map_voxel_num_;
  Eigen::Vector3d local_update_range_;
  double resolution_, resolution_inv_;
  double obstacles_inflation_;
  string frame_id_;
  int pose_type_;

  double cx_, cy_, fx_, fy_;
  double depth_filter_maxdist_, depth_filter_mindist_, depth_filter_tolerance_;
  int depth_filter_margin_;
  bool use_depth_filter_;
  double k_depth_scaling_factor_;
  int skip_pixel_;

  double p_hit_, p_miss_, p_min_, p_max_, p_occ_;
  double prob_hit_log_, prob_miss_log_, clamp_min_log_, clamp_max_log_, min_occupancy_log_;
  double min_ray_length_, max_ray_length_, max_2d_ray_length_;

  // 高度带过滤：只处理此 Z 范围 [height_range_min_, height_range_max_] 内的 voxel
  double height_range_min_, height_range_max_;
  int height_range_z_min_idx_, height_range_z_max_idx_;

  int local_map_margin_;
  double visualization_truncate_height_, virtual_ceil_height_, ground_height_;
  bool show_occ_time_;
  double unknown_flag_;
};

// 前置声明
struct MappingData;

class GridMap {
public:
  GridMap();
  ~GridMap();

  enum { POSE_STAMPED = 1, ODOMETRY = 2, INVALID_IDX = -10000 };

  // occupancy map management
  void resetBuffer();
  void resetBuffer(Eigen::Vector3d min, Eigen::Vector3d max);

  void posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& id);
  void indexToPos(const Eigen::Vector3i& id, Eigen::Vector3d& pos);
  int toAddress(const Eigen::Vector3i& id);
  int toAddress(int& x, int& y, int& z);
  bool isInMap(const Eigen::Vector3d& pos);
  bool isInMap(const Eigen::Vector3i& idx);

  void setOccupancy(Eigen::Vector3d pos, double occ = 1);
  void setOccupied(Eigen::Vector3d pos);
  int getOccupancy(Eigen::Vector3d pos);
  int getOccupancy(Eigen::Vector3i id);
  int getInflateOccupancy(Eigen::Vector3d pos);

  void boundIndex(Eigen::Vector3i& id);
  bool isUnknown(const Eigen::Vector3i& id);
  bool isUnknown(const Eigen::Vector3d& pos);
  bool isKnownFree(const Eigen::Vector3i& id);
  bool isKnownFree(const Eigen::Vector3d& pos);
  bool isKnownOccupied(const Eigen::Vector3i& id);
  bool isKnownOccupied(const Eigen::Vector3d& pose);

  void initMap(ros::NodeHandle& nh);
  void publishMap();
  void publishMapInflate(bool all_info = false);
  void publishUnknown();

  bool hasDepthObservation();
  bool odomValid();
  void getRegion(Eigen::Vector3d& ori, Eigen::Vector3d& size);
  double getResolution();
  Eigen::Vector3d getOrigin();
  int getVoxelNum();

  typedef std::shared_ptr<GridMap> Ptr;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  MappingParameters mp_;
  std::unique_ptr<MappingData> md_;

  // 回调函数
  void depthPoseCallback(const sensor_msgs::ImageConstPtr& img,
                         const geometry_msgs::PoseStampedConstPtr& pose);
  void depthOdomCallback(const sensor_msgs::ImageConstPtr& img, const nav_msgs::OdometryConstPtr& odom);
  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& img);
  void odomCallback(const nav_msgs::OdometryConstPtr& odom);

  void updateOccupancyCallback(const ros::TimerEvent& /*event*/);
  void visCallback(const ros::TimerEvent& /*event*/);

  void projectDepthImage();
  void raycastProcess();
  void clearAndInflateLocalMap();

  void inflatePoint(const Eigen::Vector3i& pt, int step, vector<Eigen::Vector3i>& pts);
  int setCacheOccupancy(Eigen::Vector3d pos, int occ);
  Eigen::Vector3d closetPointInMap(const Eigen::Vector3d& pt, const Eigen::Vector3d& camera_pt);

  // 辅助函数：判断 voxel 的 Z 坐标是否在 height_range 内
  inline bool inHeightRange(int z_idx) const {
    return z_idx >= mp_.height_range_z_min_idx_ && z_idx <= mp_.height_range_z_max_idx_;
  }

  // 同步策略
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry>
      SyncPolicyImageOdom;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped>
      SyncPolicyImagePose;
  typedef shared_ptr<message_filters::Synchronizer<SyncPolicyImagePose>> SynchronizerImagePose;
  typedef shared_ptr<message_filters::Synchronizer<SyncPolicyImageOdom>> SynchronizerImageOdom;

  ros::NodeHandle node_;
  shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> depth_sub_;
  shared_ptr<message_filters::Subscriber<geometry_msgs::PoseStamped>> pose_sub_;
  shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub_;
  SynchronizerImagePose sync_image_pose_;
  SynchronizerImageOdom sync_image_odom_;

  ros::Subscriber indep_cloud_sub_, indep_odom_sub_;
  ros::Publisher map_pub_, map_inf_pub_;
  ros::Publisher unknown_pub_;
  ros::Publisher map2DPub_;
  ros::Timer occ_timer_, vis_timer_;

  // exploration相关
  float groundHeight_;
  void publish2DOccupancyGrid();
  nav_msgs::OccupancyGrid cached_2d_map_;          // 缓存的2D OccupancyGrid
  std::vector<int8_t> occupancy_2d_persistent_;     // 持久化2D占据状态，只增不删（建图效果）
  bool has_2d_map_initialized_ = false;             // 是否已完成首次初始化
  bool use_8neighbor_frontier_ = false;             // false=4邻域frontier(默认), true=8邻域
  double current_z_;

  // 缓存的2D地图参数，避免每次查询时访问ROS消息字段
  double map_2d_origin_x_ = 0.0;
  double map_2d_origin_y_ = 0.0;
  double map_2d_res_inv_ = 1.0;
  int map_2d_width_ = 0;
  int map_2d_height_ = 0;
  std::vector<Eigen::Vector2i> free_2d_temp_;
  mutable std::mutex map_2d_mutex_;  // 保护 free_2d_temp_ 等二维地图数据的线程锁

public:
  bool isInflatedOccupiedLine(const Eigen::Vector3d& pos1, const Eigen::Vector3d& pos2);
  bool isInflatedFreeLine(const Eigen::Vector3d& pos1, const Eigen::Vector3d& pos2);
  inline float getRes(){return mp_.resolution_;};

  inline void getCurrMapRange(Eigen::Vector3d& currRangeMin, Eigen::Vector3d& currRangeMax){
    currRangeMax << currMapRangeMax_(0),currMapRangeMax_(1),0.5f;
    currRangeMin << currMapRangeMin_(0),currMapRangeMin_(1),0.5f;
	}

  Eigen::Vector2d currMapRangeMin_,currMapRangeMax_;

  bool is2DOccupied(double x, double y);
  bool is2DFree(double x, double y);
  bool is2DUnknown(double x, double y);
  bool is2DInflatedOccupiedLine2D(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2);

  // 强制设置2D占据（脱困失败后标记障碍，使DEP规划绕开）
  void setForceOcc2D(double x, double y);

  const nav_msgs::OccupancyGrid& get2DOccupancyGrid() const { return cached_2d_map_; }
  std::vector<Eigen::Vector2i> get2DFreeGrid() const { std::lock_guard<std::mutex> lock(map_2d_mutex_); return free_2d_temp_; }
  const std::vector<int8_t>& get2DOccupancyData() const { return occupancy_2d_persistent_; }
  bool is2DMapReady() const { return has_2d_map_initialized_; }

  // ━━━ 前沿检测 (BFS + PCA 递归分裂) ━━━
  void setUse8NeighborFrontier(bool use_8n) { use_8neighbor_frontier_ = use_8n; }
  // 输出格式: vector<(world_center, size_meters)>, 对齐 DEP frontierPointPairs_
  void detectFrontierClusters(std::vector<std::pair<Eigen::Vector3d, double>>& frontier_pairs,
                              double z_height,
                              double cluster_size_xy = 2.0,
                              int min_cluster_cells = 5,
                              double min_frontier_size = 0.5,
                              double min_center_dist = 2.0);

private:
  // 检查 (gx, gy) 是否为前沿 cell: unknown(-1) 且 4邻域有 free(0)
  bool isFrontierCell(int gx, int gy) const;

  // 8邻域 BFS 区域生长，收集连通的前沿 cell
  void bfsGrowFrontier(int seed_gx, int seed_gy,
                       const std::vector<int8_t>& occupancy,
                       std::vector<bool>& visited,
                       std::vector<std::pair<int, int>>& cluster_cells) const;

  // PCA 递归分裂：簇跨度 > cluster_size_xy 时沿第一主成分方向二分
  void splitClusterPCA(const std::vector<std::pair<int, int>>& cluster_cells,
                       double cluster_size_xy,
                       std::vector<std::vector<std::pair<int, int>>>& result) const;

  // 计算簇的世界坐标中心
  Eigen::Vector3d computeClusterCenter(const std::vector<std::pair<int, int>>& cluster_cells,
                                       double z_height) const;

  // 计算簇的世界坐标跨度 (max of x_span, y_span)
  double computeClusterSize(const std::vector<std::pair<int, int>>& cluster_cells) const;

  // grid 坐标 → 世界坐标 (cell center)
  inline void gridToWorld(int gx, int gy, double& wx, double& wy) const {
    wx = map_2d_origin_x_ + (gx + 0.5) * (1.0 / map_2d_res_inv_);
    wy = map_2d_origin_y_ + (gy + 0.5) * (1.0 / map_2d_res_inv_);
  }

};

#endif