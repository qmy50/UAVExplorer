#include "plan_env/grid_map_indoor.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <queue>
#include <algorithm> 

struct MappingData 
{
  // main map data, occupancy of each voxel and Euclidean distance
  MappingData() = default;
  
  std::vector<double> occupancy_buffer_;
  std::vector<char> occupancy_buffer_inflate_;

  // camera position and pose data

  Eigen::Vector3d camera_pos_, last_camera_pos_;
  Eigen::Quaterniond camera_q_, last_camera_q_;

  // depth image data

  cv::Mat depth_image_, last_depth_image_;
  int image_cnt_;

  Eigen::Matrix4d cam2body_;

  // flags of map state

  bool occ_need_update_, local_updated_;
  bool has_first_depth_;
  bool has_odom_, has_cloud_;

  // depth image projected point cloud

  vector<Eigen::Vector3d> proj_points_;
  int proj_points_cnt;

  // flag buffers for speeding up raycasting

  vector<short> count_hit_, count_hit_and_miss_;
  vector<char> flag_traverse_, flag_rayend_;
  char raycast_num_;
  queue<Eigen::Vector3i> cache_voxel_;

  // range of updating grid

  Eigen::Vector3i local_bound_min_, local_bound_max_;

  // computation time

  double fuse_time_, max_fuse_time_;
  int update_num_;
  int raycast_cnt_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

GridMap::GridMap() : md_(std::make_unique<MappingData>()) {
    
}

GridMap::~GridMap() = default;   // unique_ptr 自动释放


int GridMap::toAddress(const Eigen::Vector3i& id) {
  return id(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2) + id(1) * mp_.map_voxel_num_(2) + id(2);
}

int GridMap::toAddress(int& x, int& y, int& z) {
  return x * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2) + y * mp_.map_voxel_num_(2) + z;
}

void GridMap::boundIndex(Eigen::Vector3i& id) {
  Eigen::Vector3i id1;
  id1(0) = max(min(id(0), mp_.map_voxel_num_(0) - 1), 0);
  id1(1) = max(min(id(1), mp_.map_voxel_num_(1) - 1), 0);
  id1(2) = max(min(id(2), mp_.map_voxel_num_(2) - 1), 0);
  id = id1;
}

bool GridMap::isUnknown(const Eigen::Vector3i& id) {
  Eigen::Vector3i id1 = id;
  boundIndex(id1);
  return md_->occupancy_buffer_[toAddress(id1)] < mp_.clamp_min_log_ - 1e-3;
}

bool GridMap::isUnknown(const Eigen::Vector3d& pos) {
  Eigen::Vector3i idc;
  posToIndex(pos, idc);
  return isUnknown(idc);
}

bool GridMap::isKnownFree(const Eigen::Vector3i& id) {
  Eigen::Vector3i id1 = id;
  boundIndex(id1);
  int adr = toAddress(id1);

  return md_->occupancy_buffer_[adr] >= mp_.clamp_min_log_ && md_->occupancy_buffer_inflate_[adr] == 0;
}

bool GridMap::isKnownFree(const Eigen::Vector3d& pos) {
  Eigen::Vector3i id1;
  posToIndex(pos,id1);
  boundIndex(id1);
  int adr = toAddress(id1);

  return md_->occupancy_buffer_[adr] >= mp_.clamp_min_log_ && md_->occupancy_buffer_inflate_[adr] == 0;
}

bool GridMap::isKnownOccupied(const Eigen::Vector3i& id) {
  Eigen::Vector3i id1 = id;
  boundIndex(id1);
  int adr = toAddress(id1);

  return md_->occupancy_buffer_inflate_[adr] == 1;
}

bool GridMap::isKnownOccupied(const Eigen::Vector3d& pos) {
  Eigen::Vector3i id1;
  posToIndex(pos,id1);
  boundIndex(id1);
  int adr = toAddress(id1);

  return md_->occupancy_buffer_inflate_[adr] == 1;
}

void GridMap::setOccupied(Eigen::Vector3d pos) {
  if (!isInMap(pos)) return;

  Eigen::Vector3i id;
  posToIndex(pos, id);

  md_->occupancy_buffer_inflate_[id(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2) +
                                id(1) * mp_.map_voxel_num_(2) + id(2)] = 1;
}

void GridMap::setOccupancy(Eigen::Vector3d pos, double occ) {
  if (occ != 1 && occ != 0) {
    cout << "occ value error!" << endl;
    return;
  }

  if (!isInMap(pos)) return;

  Eigen::Vector3i id;
  posToIndex(pos, id);

  md_->occupancy_buffer_[toAddress(id)] = occ;
}

int GridMap::getOccupancy(Eigen::Vector3d pos) {
  if (!isInMap(pos)) return -1;

  Eigen::Vector3i id;
  posToIndex(pos, id);

  return md_->occupancy_buffer_[toAddress(id)] > mp_.min_occupancy_log_ ? 1 : 0;
}

int GridMap::getInflateOccupancy(Eigen::Vector3d pos) {
    if (!isInMap(pos)) return -1;

    Eigen::Vector3i id;
    posToIndex(pos, id);
    int addr = toAddress(id);

    if (md_->occupancy_buffer_inflate_[addr] > 0) return 1;
    return 0;
}

int GridMap::getOccupancy(Eigen::Vector3i id) {
  if (id(0) < 0 || id(0) >= mp_.map_voxel_num_(0) || id(1) < 0 || id(1) >= mp_.map_voxel_num_(1) ||
      id(2) < 0 || id(2) >= mp_.map_voxel_num_(2))
    return -1;

  return md_->occupancy_buffer_[toAddress(id)] > mp_.min_occupancy_log_ ? 1 : 0;
}

bool GridMap::isInMap(const Eigen::Vector3d& pos) {
  if (pos(0) < mp_.map_min_boundary_(0) + 1e-4 || pos(1) < mp_.map_min_boundary_(1) + 1e-4 ||
      pos(2) < mp_.map_min_boundary_(2) + 1e-4) {
    return false;
  }
  if (pos(0) > mp_.map_max_boundary_(0) - 1e-4 || pos(1) > mp_.map_max_boundary_(1) - 1e-4 ||
      pos(2) > mp_.map_max_boundary_(2) - 1e-4) {
    return false;
  }
  return true;
}

bool GridMap::isInMap(const Eigen::Vector3i& idx) {
  if (idx(0) < 0 || idx(1) < 0 || idx(2) < 0) {
    return false;
  }
  if (idx(0) > mp_.map_voxel_num_(0) - 1 || idx(1) > mp_.map_voxel_num_(1) - 1 ||
      idx(2) > mp_.map_voxel_num_(2) - 1) {
    return false;
  }
  return true;
}

void GridMap::posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& id) {
  for (int i = 0; i < 3; ++i) id(i) = floor((pos(i) - mp_.map_origin_(i)) * mp_.resolution_inv_);
}

void GridMap::indexToPos(const Eigen::Vector3i& id, Eigen::Vector3d& pos) {
  for (int i = 0; i < 3; ++i) pos(i) = (id(i) + 0.5) * mp_.resolution_ + mp_.map_origin_(i);
}

void GridMap::inflatePoint(const Eigen::Vector3i& pt, int step, vector<Eigen::Vector3i>& pts) {
  int num = 0;

  /* ---------- all inflate ---------- */
  for (int x = -step; x <= step; ++x)
    for (int y = -step; y <= step; ++y)
      for (int z = -step; z <= step; ++z) {
        pts[num++] = Eigen::Vector3i(pt(0) + x, pt(1) + y, pt(2) + z);
      }
}

double GridMap::getResolution() { return mp_.resolution_; }


void GridMap::initMap(ros::NodeHandle &nh)
{
  node_ = nh;

  /* get parameter */
  double x_size, y_size, z_size;
  node_.param("grid_map/resolution", mp_.resolution_, -1.0);
  node_.param("grid_map/map_size_x", x_size, -1.0);
  node_.param("grid_map/map_size_y", y_size, -1.0);
  node_.param("grid_map/map_size_z", z_size, -1.0);
  node_.param("grid_map/local_update_range_x", mp_.local_update_range_(0), -1.0);
  node_.param("grid_map/local_update_range_y", mp_.local_update_range_(1), -1.0);
  node_.param("grid_map/local_update_range_z", mp_.local_update_range_(2), -1.0);
  node_.param("grid_map/obstacles_inflation", mp_.obstacles_inflation_, -1.0);

  node_.param("grid_map/fx", mp_.fx_, -1.0);
  node_.param("grid_map/fy", mp_.fy_, -1.0);
  node_.param("grid_map/cx", mp_.cx_, -1.0);
  node_.param("grid_map/cy", mp_.cy_, -1.0);

  node_.param("grid_map/use_depth_filter", mp_.use_depth_filter_, true);
  node_.param("grid_map/depth_filter_tolerance", mp_.depth_filter_tolerance_, -1.0);
  node_.param("grid_map/depth_filter_maxdist", mp_.depth_filter_maxdist_, -1.0);
  node_.param("grid_map/depth_filter_mindist", mp_.depth_filter_mindist_, -1.0);
  node_.param("grid_map/depth_filter_margin", mp_.depth_filter_margin_, -1);
  node_.param("grid_map/k_depth_scaling_factor", mp_.k_depth_scaling_factor_, -1.0);
  node_.param("grid_map/skip_pixel", mp_.skip_pixel_, -1);

  node_.param("grid_map/p_hit", mp_.p_hit_, 0.70);
  node_.param("grid_map/p_miss", mp_.p_miss_, 0.35);
  node_.param("grid_map/p_min", mp_.p_min_, 0.12);
  node_.param("grid_map/p_max", mp_.p_max_, 0.97);
  node_.param("grid_map/p_occ", mp_.p_occ_, 0.80);
  node_.param("grid_map/min_ray_length", mp_.min_ray_length_, -0.1);
  node_.param("grid_map/max_ray_length", mp_.max_ray_length_, -0.1);
  node_.param("grid_map/max_2d_ray_length", mp_.max_2d_ray_length_, mp_.max_ray_length_);

  // 高度带过滤参数
  node_.param("grid_map/height_range_min", mp_.height_range_min_, 0.0);
  node_.param("grid_map/height_range_max", mp_.height_range_max_, 3.0);

  node_.param("grid_map/visualization_truncate_height", mp_.visualization_truncate_height_, 999.0);
  node_.param("grid_map/virtual_ceil_height", mp_.virtual_ceil_height_, -0.1);

  node_.param("grid_map/show_occ_time", mp_.show_occ_time_, false);
  node_.param("grid_map/pose_type", mp_.pose_type_, 1);

  node_.param("grid_map/frame_id", mp_.frame_id_, string("world"));
  node_.param("grid_map/local_map_margin", mp_.local_map_margin_, 1);
  node_.param("grid_map/ground_height", mp_.ground_height_, 1.0);

  // map_origin: default centers map at world (0,0); override to shift the map
  double origin_x, origin_y;
  node_.param("grid_map/map_origin_x", origin_x, -x_size / 2.0);
  node_.param("grid_map/map_origin_y", origin_y, -y_size / 2.0);

  mp_.resolution_inv_ = 1 / mp_.resolution_;
  mp_.map_origin_ = Eigen::Vector3d(origin_x, origin_y, mp_.ground_height_);
  mp_.map_size_ = Eigen::Vector3d(x_size, y_size, z_size);

  mp_.prob_hit_log_ = logit(mp_.p_hit_);
  mp_.prob_miss_log_ = logit(mp_.p_miss_);
  mp_.clamp_min_log_ = logit(mp_.p_min_);
  mp_.clamp_max_log_ = logit(mp_.p_max_);
  mp_.min_occupancy_log_ = logit(mp_.p_occ_);
  mp_.unknown_flag_ = 0.01;

  cout << "hit: " << mp_.prob_hit_log_ << endl;
  cout << "miss: " << mp_.prob_miss_log_ << endl;
  cout << "min log: " << mp_.clamp_min_log_ << endl;
  cout << "max: " << mp_.clamp_max_log_ << endl;
  cout << "thresh log: " << mp_.min_occupancy_log_ << endl;

  for (int i = 0; i < 3; ++i)
    mp_.map_voxel_num_(i) = ceil(mp_.map_size_(i) / mp_.resolution_);

  // 计算高度带对应的 Z 坐标范围索引（必须在 map_voxel_num_ 之后）
  mp_.height_range_z_min_idx_ = floor((mp_.height_range_min_ - mp_.map_origin_(2)) * mp_.resolution_inv_);
  mp_.height_range_z_max_idx_ = floor((mp_.height_range_max_ - mp_.map_origin_(2)) * mp_.resolution_inv_);
  // clamp 到地图 Z 范围
  mp_.height_range_z_min_idx_ = max(0, mp_.height_range_z_min_idx_);
  mp_.height_range_z_max_idx_ = min(mp_.map_voxel_num_(2) - 1, mp_.height_range_z_max_idx_);
  cout << "height_range: [" << mp_.height_range_min_ << ", " << mp_.height_range_max_ << "]"
       << " z_idx: [" << mp_.height_range_z_min_idx_ << ", " << mp_.height_range_z_max_idx_ << "]" << endl;

  mp_.map_min_boundary_ = mp_.map_origin_;
  mp_.map_max_boundary_ = mp_.map_origin_ + mp_.map_size_;

  // initialize data buffers

  int buffer_size = mp_.map_voxel_num_(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2);

  md_->occupancy_buffer_ = vector<double>(buffer_size, mp_.clamp_min_log_ - mp_.unknown_flag_);
  md_->occupancy_buffer_inflate_ = vector<char>(buffer_size, 0);

  md_->count_hit_and_miss_ = vector<short>(buffer_size, 0);
  md_->count_hit_ = vector<short>(buffer_size, 0);
  md_->flag_rayend_ = vector<char>(buffer_size, -1);
  md_->flag_traverse_ = vector<char>(buffer_size, -1);

  md_->raycast_num_ = 0;
  md_->raycast_cnt_ = 0;

  md_->proj_points_.resize(640 * 480 / mp_.skip_pixel_ / mp_.skip_pixel_ * 2);
  md_->proj_points_cnt = 0;
  md_->cam2body_ << 0.0, 0.0, 1.0, 0.0,
      -1.0, 0.0, 0.0, 0.0,
      0.0, -1.0, 0.0, -0.02,
      0.0, 0.0, 0.0, 1.0;

  /* init callback */

  depth_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(node_, "grid_map/depth", 50));

  if (mp_.pose_type_ == POSE_STAMPED)
  {
    pose_sub_.reset(
        new message_filters::Subscriber<geometry_msgs::PoseStamped>(node_, "grid_map/pose", 25));

    sync_image_pose_.reset(new message_filters::Synchronizer<SyncPolicyImagePose>(
        SyncPolicyImagePose(100), *depth_sub_, *pose_sub_));
    sync_image_pose_->registerCallback(boost::bind(&GridMap::depthPoseCallback, this, _1, _2));
  }
  else if (mp_.pose_type_ == ODOMETRY)
  {
    odom_sub_.reset(new message_filters::Subscriber<nav_msgs::Odometry>(node_, "grid_map/odom", 100));

    sync_image_odom_.reset(new message_filters::Synchronizer<SyncPolicyImageOdom>(
        SyncPolicyImageOdom(100), *depth_sub_, *odom_sub_));
    sync_image_odom_->registerCallback(boost::bind(&GridMap::depthOdomCallback, this, _1, _2));
  }

  // use odometry and point cloud
  indep_cloud_sub_ =
      node_.subscribe<sensor_msgs::PointCloud2>("grid_map/cloud", 10, &GridMap::cloudCallback, this);
  indep_odom_sub_ =
      node_.subscribe<nav_msgs::Odometry>("grid_map/odom", 10, &GridMap::odomCallback, this);

  occ_timer_ = node_.createTimer(ros::Duration(0.05), &GridMap::updateOccupancyCallback, this);
  vis_timer_ = node_.createTimer(ros::Duration(0.1), &GridMap::visCallback, this);

  map_pub_ = node_.advertise<sensor_msgs::PointCloud2>("grid_map/occupancy", 10);
  map_inf_pub_ = node_.advertise<sensor_msgs::PointCloud2>("grid_map/occupancy_inflate", 10);

  unknown_pub_ = node_.advertise<sensor_msgs::PointCloud2>("grid_map/unknown", 10);

  md_->occ_need_update_ = false;
  md_->local_updated_ = false;
  md_->has_first_depth_ = false;
  md_->has_odom_ = false;
  md_->has_cloud_ = false;
  md_->image_cnt_ = 0;

  md_->fuse_time_ = 0.0;
  md_->update_num_ = 0;
  md_->max_fuse_time_ = 0.0;

  // exploration相关
  map2DPub_ = node_.advertise<nav_msgs::OccupancyGrid>("grid_map/occupancy_2d", 10);
  currMapRangeMax_ << -100.0f,-1000.0f;
  currMapRangeMin_ << 100.0f,100.0f;
  groundHeight_ = mp_.ground_height_;

}

// ---------------------------------- exploration相关实现 ----------------------------------

void GridMap::publish2DOccupancyGrid(){
  //ROS_ERROR("Going to send 2d map !!");
		Eigen::Vector3d minRange, maxRange;
		minRange = mp_.map_min_boundary_;
		maxRange = mp_.map_max_boundary_;
		minRange(2) = mp_.ground_height_;

		Eigen::Vector3i minRangeIdx, maxRangeIdx;
		posToIndex(minRange, minRangeIdx);
		posToIndex(maxRange, maxRangeIdx);
		boundIndex(minRangeIdx);
		boundIndex(maxRangeIdx);

		int width  = mp_.map_voxel_num_(0);
		int height = mp_.map_voxel_num_(1);
		// double z = 0.45f;
    double z = current_z_;
		int zIdx   =  floor((z - mp_.map_origin_(2)) * mp_.resolution_inv_);

		if (!has_2d_map_initialized_) {
				// -1=未知(灰), 0=自由(白), 100=占据(黑)
				occupancy_2d_persistent_.assign(width * height, -1);

				cached_2d_map_.data.resize(width * height);
				for (int i = 0; i < width * height; ++i)
						cached_2d_map_.data[i] = occupancy_2d_persistent_[i];

				cached_2d_map_.header.frame_id = "map";
				cached_2d_map_.info.resolution = mp_.resolution_;
				cached_2d_map_.info.width = width;
				cached_2d_map_.info.height = height;
				cached_2d_map_.info.origin.position.x = minRange(0);
				cached_2d_map_.info.origin.position.y = minRange(1);
				cached_2d_map_.info.origin.orientation.w = 1.0;

				// 缓存2D地图参数，避免每次查询时访问ROS消息字段
				map_2d_origin_x_ = minRange(0);
				map_2d_origin_y_ = minRange(1);
				map_2d_res_inv_ = 1.0 / mp_.resolution_;
				map_2d_width_ = width;
				map_2d_height_ = height;
        if(md_->has_first_depth_){
            ROS_WARN("Have first depth");
        }else{
            ROS_WARN("Don noe have first depth");
        }

				// 等待首次深度数据投影完成后再标记2D地图就绪，
				// 否则此时 occupancy_2d_persistent_ 全为 -1（unknown）
				if (md_->has_first_depth_) {
					has_2d_map_initialized_ = true;
					ROS_INFO("[GridMap] 2D occupancy grid initialized with real depth data");
				}
		}

		// 从3D缓冲区投影到2D地图（首次初始化+数据就绪时立即执行，后续周期持续更新）
		if (has_2d_map_initialized_) {
        // ROS_ERROR("2D map init !");
				int inf_step = ceil(mp_.obstacles_inflation_ / mp_.resolution_);
				int margin   = mp_.local_map_margin_ + inf_step;

				Eigen::Vector3i update_min = md_->local_bound_min_ - Eigen::Vector3i(margin, margin, 0);
				Eigen::Vector3i update_max = md_->local_bound_max_ + Eigen::Vector3i(margin, margin, 0);

				update_min(0) = max(update_min(0), minRangeIdx(0));
				update_min(1) = max(update_min(1), minRangeIdx(1));
				update_max(0) = min(update_max(0), width  - 1);
				update_max(1) = min(update_max(1), height - 1);

				// 直接使用 occupancy_buffer_ 在 z 切片上的数据：
				// - occupancy_buffer_ > min_occupancy_log_ → 占据(100/黑)
				// - 3D 已知且非占据 → 自由(0/白)
				// - 未知 → 保持原值(-1/灰)

				// Habitat 模式: 0.5/0.6 两层 OR 逻辑
				//   - 任意层占据 → 2D 占据
				//   - 任意层已知且非占据 → 2D free (不再要求所有层都 free)
				bool habitat_mode = false;
				ros::param::param("/use_habitat_mode", habitat_mode, false);
				std::vector<double> z_layers;
        std::vector<double> z_obstacle_layers;
				// if (habitat_mode){
					z_layers = {0.5};
          z_obstacle_layers = {0.2, 0.5};
        // }
				// else{
				// 	z_layers = {0.5};
        // }

				for (int x = update_min(0); x <= update_max(0); ++x) {
					for (int y = update_min(1); y <= update_max(1); ++y) {
						int map2DIdx = x + y * width;
						bool any_occ = false, any_known_free = false;

						for (double z_layer : z_layers) {
							int zi = floor((z_layer - mp_.map_origin_(2)) * mp_.resolution_inv_);
							Eigen::Vector3i pt(x, y, zi);
							int addr = toAddress(pt);
							// Habitat 模式用 inflate buffer（膨胀障碍更保守），否则用原始 buffer
							if (habitat_mode) {
								// if (md_->occupancy_buffer_inflate_[addr] > 0)
								// 	any_occ = true;
								if (!any_known_free) {
									// 已知（>= clamp_min_log）且非占据 → free
									if (md_->occupancy_buffer_[addr] >= mp_.clamp_min_log_ - 1e-3
									    && md_->occupancy_buffer_inflate_[addr] == 0)
										any_known_free = true;
								}
							}else {
								if (md_->occupancy_buffer_[addr] > mp_.min_occupancy_log_)
									any_occ = true;
								if (!any_known_free && !isUnknown(pt)
								    && md_->occupancy_buffer_[addr] <= mp_.min_occupancy_log_)
									any_known_free = true;
							}
						}

            for (double z_layer : z_obstacle_layers) {
              int zi = floor((z_layer - mp_.map_origin_(2)) * mp_.resolution_inv_);
              Eigen::Vector3i pt(x, y, zi);
              int addr = toAddress(pt);
              if(habitat_mode){
                if (md_->occupancy_buffer_inflate_[addr] > 0)
									any_occ = true;
              }
            }

						if (any_occ || occupancy_2d_persistent_[map2DIdx] == 100) {
							occupancy_2d_persistent_[map2DIdx] = 100;
						} else if (any_known_free) {
							occupancy_2d_persistent_[map2DIdx] = 0;
						}
					}
				}

				// 更新地图范围缓存（用主 z 层）
				for (int x = update_min(0); x <= update_max(0); ++x) {
						for (int y = update_min(1); y <= update_max(1); ++y) {
								Eigen::Vector3i pointIdx(x, y, zIdx);
								Eigen::Vector3d cachePos;
								indexToPos(pointIdx, cachePos);
								if(cachePos(0) > currMapRangeMax_(0)) currMapRangeMax_(0) = cachePos(0);
								if(cachePos(0) < currMapRangeMin_(0)) currMapRangeMin_(0) = cachePos(0);
								if(cachePos(1) > currMapRangeMax_(1)) currMapRangeMax_(1) = cachePos(1);
								if(cachePos(1) < currMapRangeMin_(1)) currMapRangeMin_(1) = cachePos(1);
						}
				}
		}

		for (int i = 0; i < width * height; ++i)
				cached_2d_map_.data[i] = occupancy_2d_persistent_[i];

		cached_2d_map_.header.stamp = ros::Time::now();
		this->map2DPub_.publish(cached_2d_map_);
    // ROS_ERROR("Send 2d map !!");
}


bool GridMap::isInflatedOccupiedLine(const Eigen::Vector3d& pos1, const Eigen::Vector3d& pos2){		
		if (isKnownOccupied(pos1) || isKnownOccupied(pos2)){
			return true;
		}

		Eigen::Vector3d diff = pos2 - pos1;
		double dist = diff.norm();
		Eigen::Vector3d diffUnit = diff/dist;
		int stepNum = int(dist/mp_.resolution_);
		Eigen::Vector3d pCheck;
		Eigen::Vector3d unitIncrement = diffUnit * mp_.resolution_;
		bool isOccupied = false;
		for (int i=1; i<stepNum; ++i){
			pCheck = pos1 + i * unitIncrement;
			isOccupied = isKnownOccupied(pCheck);
			if (isOccupied){
				return true;
			}
		}
		return false;
}

bool GridMap::isInflatedFreeLine(const Eigen::Vector3d& pos1, const Eigen::Vector3d& pos2){

		if (isKnownFree(pos1) || isKnownFree(pos2)){
			return false;
		}

		Eigen::Vector3d diff = pos2 - pos1;
		double dist = diff.norm();
		Eigen::Vector3d diffUnit = diff/dist;
		int stepNum = int(dist/mp_.resolution_);
		Eigen::Vector3d pCheck;
		Eigen::Vector3d unitIncrement = diffUnit * mp_.resolution_;
		bool isFree = true;
		for (int i=1; i<stepNum; ++i){
			pCheck = pos1 + i * unitIncrement;
			isFree = isKnownFree(pCheck);
			if (not isFree){
				return false;
			}
		}
		return true;
}

bool GridMap::is2DOccupied(double x, double y) {
	if (!has_2d_map_initialized_) return false;
	
	int gx = floor((x - map_2d_origin_x_) * map_2d_res_inv_);
	int gy = floor((y - map_2d_origin_y_) * map_2d_res_inv_);
	
	if (gx < 0 || gx >= map_2d_width_ || gy < 0 || gy >= map_2d_height_)
		return true;  // 地图外视为占据
	
	return occupancy_2d_persistent_[gy * map_2d_width_ + gx] == 100;
}

bool GridMap::is2DFree(double x, double y) {
	if (!has_2d_map_initialized_) return false;
	
	int gx = floor((x - map_2d_origin_x_) * map_2d_res_inv_);
	int gy = floor((y - map_2d_origin_y_) * map_2d_res_inv_);
	
	if (gx < 0 || gx >= map_2d_width_ || gy < 0 || gy >= map_2d_height_)
		return false;
	
	return occupancy_2d_persistent_[gy * map_2d_width_ + gx] == 0;
}

bool GridMap::is2DUnknown(double x, double y) {
	if (!has_2d_map_initialized_) return true;
	
	int gx = floor((x - map_2d_origin_x_) * map_2d_res_inv_);
	int gy = floor((y - map_2d_origin_y_) * map_2d_res_inv_);
	
	if (gx < 0 || gx >= map_2d_width_ || gy < 0 || gy >= map_2d_height_)
		return true;
	
	return occupancy_2d_persistent_[gy * map_2d_width_ + gx] == -1;
}

bool GridMap::is2DInflatedOccupiedLine2D(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) {
	if (!has_2d_map_initialized_) return false;

	int x0 = floor((p1.x() - map_2d_origin_x_) * map_2d_res_inv_);
	int y0 = floor((p1.y() - map_2d_origin_y_) * map_2d_res_inv_);
	int x1 = floor((p2.x() - map_2d_origin_x_) * map_2d_res_inv_);
	int y1 = floor((p2.y() - map_2d_origin_y_) * map_2d_res_inv_);

	auto isOccAt = [&](int gx, int gy) -> bool {
		if (gx < 0 || gx >= map_2d_width_ || gy < 0 || gy >= map_2d_height_)
			return true;
		return occupancy_2d_persistent_[gy * map_2d_width_ + gx] == 100;
	};

	if (isOccAt(x0, y0) || isOccAt(x1, y1))
		return true;

	// Bresenham整数直线算法
	int dx = std::abs(x1 - x0);
	int dy = std::abs(y1 - y0);
	int sx = x0 < x1 ? 1 : -1;
	int sy = y0 < y1 ? 1 : -1;
	int err = dx - dy;

	while (x0 != x1 || y0 != y1) {
		int e2 = 2 * err;
		if (e2 > -dy) {
			err -= dy;
			x0 += sx;
		}
		if (e2 < dx) {
			err += dx;
			y0 += sy;
		}
		if (isOccAt(x0, y0))
			return true;
	}
	return false;
}

// 强制设置2D占据（脱困失败时标记障碍，使DEP规划绕开该位置）
void GridMap::setForceOcc2D(double x, double y) {
  if (!has_2d_map_initialized_) return;

  int ix = static_cast<int>((x - map_2d_origin_x_) * map_2d_res_inv_);
  int iy = static_cast<int>((y - map_2d_origin_y_) * map_2d_res_inv_);
  if (ix < 0 || ix >= map_2d_width_ || iy < 0 || iy >= map_2d_height_) return;

  int idx = iy * map_2d_width_ + ix;
  {
    std::lock_guard<std::mutex> lock(map_2d_mutex_);
    occupancy_2d_persistent_[idx] = 100;  // 占据
    cached_2d_map_.data[idx] = 100;
  }
}

// ---------------------------------------------------------------------------------------

void GridMap::resetBuffer()
{
  Eigen::Vector3d min_pos = mp_.map_min_boundary_;
  Eigen::Vector3d max_pos = mp_.map_max_boundary_;

  resetBuffer(min_pos, max_pos);

  md_->local_bound_min_ = Eigen::Vector3i::Zero();
  md_->local_bound_max_ = mp_.map_voxel_num_ - Eigen::Vector3i::Ones();

  // 重置2D地图缓存，下次 publish 时会全量重建
  ROS_WARN("Reset Buffer!!!!!!!");
  has_2d_map_initialized_ = false;
}

void GridMap::resetBuffer(Eigen::Vector3d min_pos, Eigen::Vector3d max_pos)
{

  Eigen::Vector3i min_id, max_id;
  posToIndex(min_pos, min_id);
  posToIndex(max_pos, max_id);

  boundIndex(min_id);
  boundIndex(max_id);

  /* reset occ and dist buffer */
  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
      for (int z = min_id(2); z <= max_id(2); ++z)
      {
        md_->occupancy_buffer_inflate_[toAddress(x, y, z)] = 0;
        md_->occupancy_buffer_[toAddress(x, y, z)] = mp_.clamp_min_log_ - mp_.unknown_flag_;
      }
}

int GridMap::setCacheOccupancy(Eigen::Vector3d pos, int occ)
{
  if (occ != 1 && occ != 0)
    return INVALID_IDX;

  Eigen::Vector3i id;
  posToIndex(pos, id);
  int idx_ctns = toAddress(id);

  // 高度带过滤：跳过 Z 不在 range 内的 voxel
  if (!inHeightRange(id(2)))
    return INVALID_IDX;

  md_->count_hit_and_miss_[idx_ctns] += 1;

  if (md_->count_hit_and_miss_[idx_ctns] == 1)
  {
    md_->cache_voxel_.push(id);
  }

  if (occ == 1)
    md_->count_hit_[idx_ctns] += 1;

  return idx_ctns;
}

void GridMap::projectDepthImage()
{
  md_->proj_points_cnt = 0;

  uint16_t *row_ptr;
  int cols = md_->depth_image_.cols;
  int rows = md_->depth_image_.rows;

  double depth;

  Eigen::Matrix3d camera_r = md_->camera_q_.toRotationMatrix();

  if (!mp_.use_depth_filter_)
  {
    for (int v = 0; v < rows; v++)
    {
      row_ptr = md_->depth_image_.ptr<uint16_t>(v);

      for (int u = 0; u < cols; u++)
      {

        Eigen::Vector3d proj_pt;
        depth = (*row_ptr++) / mp_.k_depth_scaling_factor_;
        if (depth == 0.0)  // 无有效深度：跳过，不生成 free 射线（避免地图外区域误标 free）
            continue;
        proj_pt(0) = (u - mp_.cx_) * depth / mp_.fx_;
        proj_pt(1) = (v - mp_.cy_) * depth / mp_.fy_;
        proj_pt(2) = depth;

        proj_pt = camera_r * proj_pt + md_->camera_pos_;

        if (u == 320 && v == 240)
          std::cout << "depth: " << depth << std::endl;

        md_->proj_points_[md_->proj_points_cnt++] = proj_pt;
      }
    }
  }
  /* use depth filter */
  else
  {

    if (!md_->has_first_depth_)
      md_->has_first_depth_ = true;
    else
    {
      Eigen::Vector3d pt_cur, pt_world, pt_reproj;

      Eigen::Matrix3d last_camera_r_inv;
      last_camera_r_inv = md_->last_camera_q_.inverse();
      const double inv_factor = 1.0 / mp_.k_depth_scaling_factor_;

      for (int v = mp_.depth_filter_margin_; v < rows - mp_.depth_filter_margin_; v += mp_.skip_pixel_)
      {
        row_ptr = md_->depth_image_.ptr<uint16_t>(v) + mp_.depth_filter_margin_;

        for (int u = mp_.depth_filter_margin_; u < cols - mp_.depth_filter_margin_;
             u += mp_.skip_pixel_)
        {

          depth = (*row_ptr) * inv_factor;
          // bool is_free_ray = false;
          if (depth == 0.0)                    // 无有效深度：跳过，不生成 free 射线（避免地图外区域误标 free）
          {
              row_ptr += mp_.skip_pixel_;
              continue;
          }
          else if (depth < mp_.depth_filter_mindist_)
          {
              continue;
          }
          else if (depth > mp_.depth_filter_maxdist_)
          {
              depth = mp_.max_ray_length_ + 0.1;
              // is_free_ray = true;
          }
          row_ptr = row_ptr + mp_.skip_pixel_;
          
          // project to world frame
          pt_cur(0) = (u - mp_.cx_) * depth / mp_.fx_;
          pt_cur(1) = (v - mp_.cy_) * depth / mp_.fy_;
          pt_cur(2) = depth;

          pt_world = camera_r * pt_cur + md_->camera_pos_;

          md_->proj_points_[md_->proj_points_cnt++] = pt_world;

          // check consistency with last image, disabled...
          if (false)
          {
            pt_reproj = last_camera_r_inv * (pt_world - md_->last_camera_pos_);
            double uu = pt_reproj.x() * mp_.fx_ / pt_reproj.z() + mp_.cx_;
            double vv = pt_reproj.y() * mp_.fy_ / pt_reproj.z() + mp_.cy_;

            if (uu >= 0 && uu < cols && vv >= 0 && vv < rows)
            {
              if (fabs(md_->last_depth_image_.at<uint16_t>((int)vv, (int)uu) * inv_factor -
                       pt_reproj.z()) < mp_.depth_filter_tolerance_)
              {
                md_->proj_points_[md_->proj_points_cnt++] = pt_world;
              }
            }
            else
            {
              md_->proj_points_[md_->proj_points_cnt++] = pt_world;
            }
          }
        }
      }

      // [禁用] 处理 margin 区域的 depth=0 像素，为视野边缘生成 free 射线
      // 这些边缘 free 射线会打到墙外/室外，是"地图外区域被误标 free"的主要来源
#if 0
      for (int v = 0; v < rows; v += mp_.skip_pixel_)
      {
        for (int u = 0; u < cols; u += mp_.skip_pixel_)
        {
          // 跳过内部已处理的区域
          if (v >= mp_.depth_filter_margin_ && v < rows - mp_.depth_filter_margin_ &&
              u >= mp_.depth_filter_margin_ && u < cols - mp_.depth_filter_margin_)
            continue;

          double margin_depth = md_->depth_image_.at<uint16_t>(v, u) * inv_factor;
          if (margin_depth == 0.0)
          {
            margin_depth = mp_.max_ray_length_ + 0.1;
            Eigen::Vector3d margin_pt;
            margin_pt(0) = (u - mp_.cx_) * margin_depth / mp_.fx_;
            margin_pt(1) = (v - mp_.cy_) * margin_depth / mp_.fy_;
            margin_pt(2) = margin_depth;
            margin_pt = camera_r * margin_pt + md_->camera_pos_;
            md_->proj_points_[md_->proj_points_cnt++] = margin_pt;
          }
        }
      }
#endif
    }
  }

  /* maintain camera pose for consistency check */

  md_->last_camera_pos_ = md_->camera_pos_;
  md_->last_camera_q_ = md_->camera_q_;
  md_->last_depth_image_ = md_->depth_image_;
}

void GridMap::raycastProcess()
{
  if (md_->proj_points_cnt == 0)
    return;

  ros::Time t1, t2;

  md_->raycast_num_ += 1;

  int vox_idx;
  double length;
  std::vector<Eigen::Vector2i> free_2d_local;

  // 2D切片使用高度带中间值
  const double z_mid = (mp_.height_range_min_ + mp_.height_range_max_) * 0.5;
  const int z_slice_idx = static_cast<int>(floor((z_mid - mp_.map_origin_(2)) * mp_.resolution_inv_));

  // bounding box of updated region
  double min_x = mp_.map_max_boundary_(0);
  double min_y = mp_.map_max_boundary_(1);
  double min_z = mp_.map_max_boundary_(2);

  double max_x = mp_.map_min_boundary_(0);
  double max_y = mp_.map_min_boundary_(1);
  double max_z = mp_.map_min_boundary_(2);

  RayCaster raycaster;
  Eigen::Vector3d half = Eigen::Vector3d(0.5, 0.5, 0.5);
  Eigen::Vector3d ray_pt, pt_w;
  for (int i = 0; i < md_->proj_points_cnt; ++i)
  {
    pt_w = md_->proj_points_[i];

    // set flag for projected point

    if (!isInMap(pt_w))
    {
      pt_w = closetPointInMap(pt_w, md_->camera_pos_);

      length = (pt_w - md_->camera_pos_).norm();
      if (length > mp_.max_ray_length_)
      {
        pt_w = (pt_w - md_->camera_pos_) / length * mp_.max_ray_length_ + md_->camera_pos_;
      }
      vox_idx = setCacheOccupancy(pt_w, 0);
    }
    else
    {
      length = (pt_w - md_->camera_pos_).norm();

      if (length > mp_.max_ray_length_)
      {
        pt_w = (pt_w - md_->camera_pos_) / length * mp_.max_ray_length_ + md_->camera_pos_;
        vox_idx = setCacheOccupancy(pt_w, 0);
      }
      else
      {
        vox_idx = setCacheOccupancy(pt_w, 1);
      }
    }

    max_x = max(max_x, pt_w(0));
    max_y = max(max_y, pt_w(1));
    max_z = max(max_z, pt_w(2));

    min_x = min(min_x, pt_w(0));
    min_y = min(min_y, pt_w(1));
    min_z = min(min_z, pt_w(2));

    // raycasting between camera center and point

    if (vox_idx != INVALID_IDX)
    {
      if (md_->flag_rayend_[vox_idx] == md_->raycast_num_)
      {
        continue;
      }
      else
      {
        md_->flag_rayend_[vox_idx] = md_->raycast_num_;
      }
    }

    raycaster.setInput(pt_w / mp_.resolution_, md_->camera_pos_ / mp_.resolution_);

    while (raycaster.step(ray_pt))
    {
      Eigen::Vector3d tmp = (ray_pt + half) * mp_.resolution_;
      length = (tmp - md_->camera_pos_).norm();

      vox_idx = setCacheOccupancy(tmp, 0);

      if (vox_idx != INVALID_IDX)
      {
        if (md_->flag_traverse_[vox_idx] == md_->raycast_num_)
        {
          break;
        }
        else
        {
          md_->flag_traverse_[vox_idx] = md_->raycast_num_;
        }
      }
    }
  }

  min_x = min(min_x, md_->camera_pos_(0));
  min_y = min(min_y, md_->camera_pos_(1));
  min_z = min(min_z, md_->camera_pos_(2));

  max_x = max(max_x, md_->camera_pos_(0));
  max_y = max(max_y, md_->camera_pos_(1));
  max_z = max(max_z, md_->camera_pos_(2));
  max_z = max(max_z, mp_.ground_height_);

  posToIndex(Eigen::Vector3d(max_x, max_y, max_z), md_->local_bound_max_);
  posToIndex(Eigen::Vector3d(min_x, min_y, min_z), md_->local_bound_min_);
  boundIndex(md_->local_bound_min_);
  boundIndex(md_->local_bound_max_);

  md_->local_updated_ = true;

  // update occupancy cached in queue
  Eigen::Vector3d local_range_min = md_->camera_pos_ - mp_.local_update_range_;
  Eigen::Vector3d local_range_max = md_->camera_pos_ + mp_.local_update_range_;

  Eigen::Vector3i min_id, max_id;
  posToIndex(local_range_min, min_id);
  posToIndex(local_range_max, max_id);
  boundIndex(min_id);
  boundIndex(max_id);

  while (!md_->cache_voxel_.empty())
  {

    Eigen::Vector3i idx = md_->cache_voxel_.front();
    int idx_ctns = toAddress(idx);
    md_->cache_voxel_.pop();

    double log_odds_update =
        md_->count_hit_[idx_ctns] >= md_->count_hit_and_miss_[idx_ctns] - md_->count_hit_[idx_ctns] ? mp_.prob_hit_log_ : mp_.prob_miss_log_;

    md_->count_hit_[idx_ctns] = md_->count_hit_and_miss_[idx_ctns] = 0;

    if (log_odds_update >= 0 && md_->occupancy_buffer_[idx_ctns] >= mp_.clamp_max_log_)
    {
      continue;
    }
    else if (log_odds_update <= 0 && md_->occupancy_buffer_[idx_ctns] <= mp_.clamp_min_log_)
    {
      md_->occupancy_buffer_[idx_ctns] = mp_.clamp_min_log_;
      if (idx(2) == z_slice_idx)
      {
        free_2d_local.push_back({idx(0), idx(1)});
      }
      continue;
    }

    bool in_local = idx(0) >= min_id(0) && idx(0) <= max_id(0) && idx(1) >= min_id(1) &&
                    idx(1) <= max_id(1) && idx(2) >= min_id(2) && idx(2) <= max_id(2);
    if (!in_local)
    {
      continue;
    }

    
    md_->occupancy_buffer_[idx_ctns] =
        std::min(std::max(md_->occupancy_buffer_[idx_ctns] + log_odds_update, mp_.clamp_min_log_),
                 mp_.clamp_max_log_);

    if (idx(2) == z_slice_idx && md_->occupancy_buffer_[idx_ctns] >= mp_.clamp_min_log_
        && md_->occupancy_buffer_[idx_ctns] < mp_.min_occupancy_log_)
    {
      free_2d_local.push_back({idx(0), idx(1)});
    }
  }

  // 将当前帧FOV可见的free grid原子性地换入 free_2d_temp_，供 ValueMap2D 读取
  {
    std::lock_guard<std::mutex> lock(map_2d_mutex_);
    free_2d_temp_.swap(free_2d_local);
  }
}

Eigen::Vector3d GridMap::closetPointInMap(const Eigen::Vector3d &pt, const Eigen::Vector3d &camera_pt)
{
  Eigen::Vector3d diff = pt - camera_pt;
  Eigen::Vector3d max_tc = mp_.map_max_boundary_ - camera_pt;
  Eigen::Vector3d min_tc = mp_.map_min_boundary_ - camera_pt;

  double min_t = 1000000;

  for (int i = 0; i < 3; ++i)
  {
    if (fabs(diff[i]) > 0)
    {

      double t1 = max_tc[i] / diff[i];
      if (t1 > 0 && t1 < min_t)
        min_t = t1;

      double t2 = min_tc[i] / diff[i];
      if (t2 > 0 && t2 < min_t)
        min_t = t2;
    }
  }

  return camera_pt + (min_t - 1e-3) * diff;
}

void GridMap::clearAndInflateLocalMap()
{
  /*clear outside local*/
  const int vec_margin = 5;

  Eigen::Vector3i min_cut = md_->local_bound_min_ -
                            Eigen::Vector3i(mp_.local_map_margin_, mp_.local_map_margin_, mp_.local_map_margin_);
  Eigen::Vector3i max_cut = md_->local_bound_max_ +
                            Eigen::Vector3i(mp_.local_map_margin_, mp_.local_map_margin_, mp_.local_map_margin_);
  boundIndex(min_cut);
  boundIndex(max_cut);

  // Z 方向 clamp 到 height_range
  min_cut(2) = max(min_cut(2), mp_.height_range_z_min_idx_);
  max_cut(2) = min(max_cut(2), mp_.height_range_z_max_idx_);

  Eigen::Vector3i min_cut_m = min_cut - Eigen::Vector3i(vec_margin, vec_margin, 0);
  Eigen::Vector3i max_cut_m = max_cut + Eigen::Vector3i(vec_margin, vec_margin, 0);
  min_cut_m(2) = mp_.height_range_z_min_idx_;
  max_cut_m(2) = mp_.height_range_z_max_idx_;
  boundIndex(min_cut_m);
  boundIndex(max_cut_m);

  // clear data outside the local range (only XY, Z stays in height_range)
  for (int x = min_cut_m(0); x <= max_cut_m(0); ++x)
    for (int y = min_cut_m(1); y <= max_cut_m(1); ++y)
    {
      for (int z = min_cut_m(2); z < min_cut(2); ++z)
      {
        int idx = toAddress(x, y, z);
        md_->occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
      }

      for (int z = max_cut(2) + 1; z <= max_cut_m(2); ++z)
      {
        int idx = toAddress(x, y, z);
        md_->occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
      }
    }

  for (int z = min_cut_m(2); z <= max_cut_m(2); ++z)
    for (int x = min_cut_m(0); x <= max_cut_m(0); ++x)
    {

      for (int y = min_cut_m(1); y < min_cut(1); ++y)
      {
        int idx = toAddress(x, y, z);
        md_->occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
      }

      for (int y = max_cut(1) + 1; y <= max_cut_m(1); ++y)
      {
        int idx = toAddress(x, y, z);
        md_->occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
      }
    }

  for (int y = min_cut_m(1); y <= max_cut_m(1); ++y)
    for (int z = min_cut_m(2); z <= max_cut_m(2); ++z)
    {

      for (int x = min_cut_m(0); x < min_cut(0); ++x)
      {
        int idx = toAddress(x, y, z);
        md_->occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
      }

      for (int x = max_cut(0) + 1; x <= max_cut_m(0); ++x)
      {
        int idx = toAddress(x, y, z);
        md_->occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
      }
    }

  // inflate occupied voxels to compensate robot size

  int inf_step = ceil(mp_.obstacles_inflation_ / mp_.resolution_);
  vector<Eigen::Vector3i> inf_pts(pow(2 * inf_step + 1, 3));
  Eigen::Vector3i inf_pt;

  // clear outdated data — only clear inflate when occupancy is at the clamp floor
  // (repeatedly hit by free-space rays, very confident it's free)
  for (int x = md_->local_bound_min_(0); x <= md_->local_bound_max_(0); ++x)
    for (int y = md_->local_bound_min_(1); y <= md_->local_bound_max_(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z)
      {
        int idx = toAddress(x, y, z);
        // clamp_min_log_ = logit(p_min) ≈ floor of occupancy: only clear inflate
        // when the voxel is maximally confident to be free (hard threshold)
        if (md_->occupancy_buffer_[idx] <= mp_.clamp_min_log_)
        {
          md_->occupancy_buffer_inflate_[idx] = 0;
        }
      }

  // inflate obstacles
  for (int x = md_->local_bound_min_(0); x <= md_->local_bound_max_(0); ++x)
    for (int y = md_->local_bound_min_(1); y <= md_->local_bound_max_(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z)
      {

        if (md_->occupancy_buffer_[toAddress(x, y, z)] > mp_.min_occupancy_log_)
        {
          inflatePoint(Eigen::Vector3i(x, y, z), inf_step, inf_pts);

          for (int k = 0; k < (int)inf_pts.size(); ++k)
          {
            inf_pt = inf_pts[k];
            int idx_inf = toAddress(inf_pt);
            if (idx_inf < 0 ||
                idx_inf >= mp_.map_voxel_num_(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2))
            {
              continue;
            }
            md_->occupancy_buffer_inflate_[idx_inf] = 1;
          }
        }
      }
}

void GridMap::visCallback(const ros::TimerEvent & /*event*/)
{
  if(md_->raycast_cnt_ > 10)
  {
    publishMap();
    publishMapInflate(true);
    publish2DOccupancyGrid();
  }
}

void GridMap::updateOccupancyCallback(const ros::TimerEvent & /*event*/)
{
  if (!md_->occ_need_update_)
    return;

  projectDepthImage();
  raycastProcess();
  md_->raycast_cnt_ += 1;

  if (md_->local_updated_){
    clearAndInflateLocalMap();
  }

  md_->occ_need_update_ = false;
  md_->local_updated_ = false;
}

void GridMap::depthPoseCallback(const sensor_msgs::ImageConstPtr &img,
                                const geometry_msgs::PoseStampedConstPtr &pose)
{
  /* get depth image */
  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(img, img->encoding);

  if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
  {
    (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, mp_.k_depth_scaling_factor_);
  }
  cv_ptr->image.copyTo(md_->depth_image_);

  /* get pose */
  md_->camera_pos_(0) = pose->pose.position.x;
  md_->camera_pos_(1) = pose->pose.position.y;
  md_->camera_pos_(2) = pose->pose.position.z;
  md_->camera_q_ = Eigen::Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x,
                                     pose->pose.orientation.y, pose->pose.orientation.z);
  if (isInMap(md_->camera_pos_))
  {
    md_->has_odom_ = true;
    md_->update_num_ += 1;
    md_->occ_need_update_ = true;
  }
  else
  {
    md_->occ_need_update_ = false;
  }
}
void GridMap::odomCallback(const nav_msgs::OdometryConstPtr &odom)
{
  Eigen::Quaterniond body_q = Eigen::Quaterniond(odom->pose.pose.orientation.w,
                                                 odom->pose.pose.orientation.x,
                                                 odom->pose.pose.orientation.y,
                                                 odom->pose.pose.orientation.z);
  Eigen::Matrix3d body_r_m = body_q.toRotationMatrix();
  Eigen::Matrix4d body2world;
  body2world.block<3, 3>(0, 0) = body_r_m;
  body2world(0, 3) = odom->pose.pose.position.x;
  body2world(1, 3) = odom->pose.pose.position.y;
  body2world(2, 3) = odom->pose.pose.position.z;
  body2world(3, 3) = 1.0;
  current_z_ = odom->pose.pose.position.z;

  Eigen::Matrix4d cam_T = body2world * md_->cam2body_;
  md_->camera_pos_(0) = cam_T(0, 3);
  md_->camera_pos_(1) = cam_T(1, 3);
  md_->camera_pos_(2) = cam_T(2, 3);
  md_->camera_q_ = Eigen::Quaterniond(cam_T.block<3, 3>(0, 0));

  md_->has_odom_ = true;
}

void GridMap::cloudCallback(const sensor_msgs::PointCloud2ConstPtr &img)
{
  pcl::PointCloud<pcl::PointXYZ> latest_cloud;
  pcl::fromROSMsg(*img, latest_cloud);

  md_->has_cloud_ = true;

  if (!md_->has_odom_)
  {
    std::cout << "no odom!" << std::endl;
    return;
  }

  if (latest_cloud.points.size() == 0)
    return;

  if (isnan(md_->camera_pos_(0)) || isnan(md_->camera_pos_(1)) || isnan(md_->camera_pos_(2)))
    return;

  pcl::PointXYZ pt;
  Eigen::Vector3d p3d, p3d_inf;

  int inf_step = ceil(mp_.obstacles_inflation_ / mp_.resolution_);
  int inf_step_z = 1;

  double max_x, max_y, max_z, min_x, min_y, min_z;

  min_x = mp_.map_max_boundary_(0);
  min_y = mp_.map_max_boundary_(1);
  min_z = mp_.map_max_boundary_(2);

  max_x = mp_.map_min_boundary_(0);
  max_y = mp_.map_min_boundary_(1);
  max_z = mp_.map_min_boundary_(2);

  // 所有障碍点（包括远距离），用于射线追踪
  std::vector<Eigen::Vector3d> all_obstacle_points;

  for (size_t i = 0; i < latest_cloud.points.size(); ++i)
  {
    pt = latest_cloud.points[i];
    p3d(0) = pt.x, p3d(1) = pt.y, p3d(2) = pt.z;

    // 过滤 NaN 点
    if (std::isnan(p3d(0)) || std::isnan(p3d(1)) || std::isnan(p3d(2)))
      continue;

    Eigen::Vector3d devi = p3d - md_->camera_pos_;
    double dist = devi.norm();

    // 超过 max_ray_length 的点跳过（太远不可靠）
    if (dist > mp_.max_ray_length_)
      continue;

    // 所有范围内的障碍点都记录，用于射线追踪
    all_obstacle_points.push_back(p3d);

    // 在 occupancy_buffer_ 中标记占据
    Eigen::Vector3i occ_idx;
    posToIndex(p3d, occ_idx);
    if (isInMap(occ_idx) && inHeightRange(occ_idx(2)))
    {
      int idx = toAddress(occ_idx);
      md_->occupancy_buffer_[idx] = std::min(md_->occupancy_buffer_[idx] + mp_.prob_hit_log_, mp_.clamp_max_log_);
    }

    // 仅对 local_update_range 内的点做膨胀
    if (fabs(devi(0)) < mp_.local_update_range_(0) && fabs(devi(1)) < mp_.local_update_range_(1) &&
        fabs(devi(2)) < mp_.local_update_range_(2))
    {
      Eigen::Vector3i inf_pt;

      /* inflate the point */
      for (int x = -inf_step; x <= inf_step; ++x)
        for (int y = -inf_step; y <= inf_step; ++y)
          for (int z = -inf_step_z; z <= inf_step_z; ++z)
          {

            p3d_inf(0) = pt.x + x * mp_.resolution_;
            p3d_inf(1) = pt.y + y * mp_.resolution_;
            p3d_inf(2) = pt.z + z * mp_.resolution_;

            max_x = max(max_x, p3d_inf(0));
            max_y = max(max_y, p3d_inf(1));
            max_z = max(max_z, p3d_inf(2));

            min_x = min(min_x, p3d_inf(0));
            min_y = min(min_y, p3d_inf(1));
            min_z = min(min_z, p3d_inf(2));

            posToIndex(p3d_inf, inf_pt);

            if (!isInMap(inf_pt) || !inHeightRange(inf_pt(2)))
              continue;

            int idx_inf = toAddress(inf_pt);

            md_->occupancy_buffer_inflate_[idx_inf] = 1;
          }
    }
    else
    {
      // 远距离障碍点：不膨胀，但必须标记 occupancy_buffer_inflate_
      Eigen::Vector3i inf_pt;
      posToIndex(p3d, inf_pt);
      if (isInMap(inf_pt) && inHeightRange(inf_pt(2)))
      {
        md_->occupancy_buffer_inflate_[toAddress(inf_pt)] = 1;
      }

      max_x = max(max_x, p3d(0));
      max_y = max(max_y, p3d(1));
      max_z = max(max_z, p3d(2));

      min_x = min(min_x, p3d(0));
      min_y = min(min_y, p3d(1));
      min_z = min(min_z, p3d(2));
    }
  }

  // 3D射线追踪：从相机到每个障碍点，沿途标记自由空间
  {
    RayCaster raycaster;
    Eigen::Vector3d half = Eigen::Vector3d(0.5, 0.5, 0.5);
    Eigen::Vector3d ray_pt;

    for (size_t i = 0; i < all_obstacle_points.size(); ++i)
    {
      Eigen::Vector3d end_pt = all_obstacle_points[i];

      // 限制射线最大长度
      double length = (end_pt - md_->camera_pos_).norm();
      if (length > mp_.max_ray_length_)
      {
        end_pt = (end_pt - md_->camera_pos_) / length * mp_.max_ray_length_ + md_->camera_pos_;
      }

      raycaster.setInput(end_pt / mp_.resolution_, md_->camera_pos_ / mp_.resolution_);

      while (raycaster.step(ray_pt))
      {
        Eigen::Vector3d tmp = (ray_pt + half) * mp_.resolution_;
        Eigen::Vector3i tmp_idx;
        posToIndex(tmp, tmp_idx);
        if (!isInMap(tmp_idx) || !inHeightRange(tmp_idx(2)))
          continue;

        int idx = toAddress(tmp_idx);
        // 遇到占据体素就停止
        if (md_->occupancy_buffer_inflate_[idx] > 0)
          break;

        // 对自由空间使用概率递减
        if (md_->occupancy_buffer_[idx] < mp_.clamp_min_log_) {
          md_->occupancy_buffer_[idx] = mp_.clamp_min_log_;
        } else {
          md_->occupancy_buffer_[idx] = std::max(
              md_->occupancy_buffer_[idx] + mp_.prob_miss_log_, mp_.clamp_min_log_);
        }
      }
    }
  }

  // FOV自由射线：投射到 max_ray_length，遇到占据体素就停止
  {
    Eigen::Matrix3d camera_r = md_->camera_q_.toRotationMatrix();
    RayCaster raycaster;
    Eigen::Vector3d half = Eigen::Vector3d(0.5, 0.5, 0.5);
    Eigen::Vector3d ray_pt;

    const double fx = mp_.fx_, fy = mp_.fy_, cx = mp_.cx_, cy = mp_.cy_;
    const int img_w = 640, img_h = 480;
    const double ray_len = mp_.max_ray_length_;
    const int fov_skip = 10;

    for (int v = 0; v < img_h; v += fov_skip)
    {
      for (int u = 0; u < img_w; u += fov_skip)
      {
        Eigen::Vector3d cam_dir;
        cam_dir(0) = (u - cx) / fx;
        cam_dir(1) = (v - cy) / fy;
        cam_dir(2) = 1.0;
        cam_dir.normalize();

        Eigen::Vector3d world_dir = camera_r * cam_dir;
        Eigen::Vector3d end_pt = md_->camera_pos_ + world_dir * ray_len;

        raycaster.setInput(end_pt / mp_.resolution_, md_->camera_pos_ / mp_.resolution_);

        while (raycaster.step(ray_pt))
        {
          Eigen::Vector3d tmp = (ray_pt + half) * mp_.resolution_;
          Eigen::Vector3i tmp_idx;
          posToIndex(tmp, tmp_idx);
          if (!isInMap(tmp_idx) || !inHeightRange(tmp_idx(2)))
            continue;

          int idx = toAddress(tmp_idx);
          // 遇到占据体素就停止
          if (md_->occupancy_buffer_inflate_[idx] > 0)
            break;

          if (md_->occupancy_buffer_[idx] < mp_.clamp_min_log_)
          {
            md_->occupancy_buffer_[idx] = mp_.clamp_min_log_;
          }
        }
      }
    }
  }

  min_x = min(min_x, md_->camera_pos_(0));
  min_y = min(min_y, md_->camera_pos_(1));
  min_z = min(min_z, md_->camera_pos_(2));

  max_x = max(max_x, md_->camera_pos_(0));
  max_y = max(max_y, md_->camera_pos_(1));
  max_z = max(max_z, md_->camera_pos_(2));

  max_z = max(max_z, mp_.ground_height_);

  posToIndex(Eigen::Vector3d(max_x, max_y, max_z), md_->local_bound_max_);
  posToIndex(Eigen::Vector3d(min_x, min_y, min_z), md_->local_bound_min_);

  boundIndex(md_->local_bound_min_);
  boundIndex(md_->local_bound_max_);
}

void GridMap::publishMap()
{

  if (map_pub_.getNumSubscribers() <= 0)
    return;

  ROS_INFO_THROTTLE(1.0,"Get the occ ");
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  Eigen::Vector3i min_cut = md_->local_bound_min_;
  Eigen::Vector3i max_cut = md_->local_bound_max_;

  int lmm = mp_.local_map_margin_ ;
  min_cut -= Eigen::Vector3i(lmm, lmm, lmm);
  max_cut += Eigen::Vector3i(lmm, lmm, lmm);

  boundIndex(min_cut);
  boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z)
      {
        if (md_->occupancy_buffer_[toAddress(x, y, z)] < mp_.min_occupancy_log_ + 1e-3){
          continue;
        }

        Eigen::Vector3d pos;
        indexToPos(Eigen::Vector3i(x, y, z), pos);
        if (pos(2) > mp_.visualization_truncate_height_)
          continue;
        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = pos(2);
        cloud.push_back(pt);
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;
  cloud.header.stamp = ros::Time::now().toNSec() / 1000;  // PCL stamp uses microseconds
  sensor_msgs::PointCloud2 cloud_msg;

  pcl::toROSMsg(cloud, cloud_msg);
  map_pub_.publish(cloud_msg);
}

void GridMap::publishMapInflate(bool all_info)
{
  if (map_inf_pub_.getNumSubscribers() <= 0)
    return;
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  Eigen::Vector3i min_cut = Eigen::Vector3i::Zero();
  Eigen::Vector3i max_cut = mp_.map_voxel_num_ - Eigen::Vector3i::Ones();

  if (all_info)
  {
    int lmm = mp_.local_map_margin_;
    min_cut -= Eigen::Vector3i(lmm, lmm, lmm);
    max_cut += Eigen::Vector3i(lmm, lmm, lmm);
  }
  boundIndex(min_cut);
  boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z)
      {
        if (md_->occupancy_buffer_inflate_[toAddress(x, y, z)] == 0)
          continue;

        Eigen::Vector3d pos;
        indexToPos(Eigen::Vector3i(x, y, z), pos);
        if (pos(2) > mp_.visualization_truncate_height_)
          continue;

        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = pos(2);
        cloud.push_back(pt);
      }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;
  cloud.header.stamp = ros::Time::now().toNSec() / 1000;
  sensor_msgs::PointCloud2 cloud_msg;

  pcl::toROSMsg(cloud, cloud_msg);
  map_inf_pub_.publish(cloud_msg);
}

void GridMap::publishUnknown()
{
  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  Eigen::Vector3i min_cut = md_->local_bound_min_;
  Eigen::Vector3i max_cut = md_->local_bound_max_;

  boundIndex(max_cut);
  boundIndex(min_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z)
      {

        if (md_->occupancy_buffer_[toAddress(x, y, z)] < mp_.clamp_min_log_ - 1e-3)
        {
          Eigen::Vector3d pos;
          indexToPos(Eigen::Vector3i(x, y, z), pos);
          if (pos(2) > mp_.visualization_truncate_height_)
            continue;

          pt.x = pos(0);
          pt.y = pos(1);
          pt.z = pos(2);
          cloud.push_back(pt);
        }
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;
  cloud.header.stamp = ros::Time::now().toNSec() / 1000;

  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  unknown_pub_.publish(cloud_msg);
}

bool GridMap::odomValid() { return md_->has_odom_; }

bool GridMap::hasDepthObservation() { return md_->has_first_depth_; }

Eigen::Vector3d GridMap::getOrigin() { return mp_.map_origin_; }

void GridMap::getRegion(Eigen::Vector3d &ori, Eigen::Vector3d &size)
{
  ori = mp_.map_origin_, size = mp_.map_size_;
}

void GridMap::depthOdomCallback(const sensor_msgs::ImageConstPtr &img,
                                const nav_msgs::OdometryConstPtr &odom)
{
  /* get pose */
  Eigen::Quaterniond body_q = Eigen::Quaterniond(odom->pose.pose.orientation.w,
                                                 odom->pose.pose.orientation.x,
                                                 odom->pose.pose.orientation.y,
                                                 odom->pose.pose.orientation.z);    
  Eigen::Matrix3d body_r_m = body_q.toRotationMatrix();   
  Eigen::Matrix4d body2world;
  body2world.block<3, 3>(0, 0) = body_r_m;
  body2world(0, 3) = odom->pose.pose.position.x;
  body2world(1, 3) = odom->pose.pose.position.y;
  body2world(2, 3) = odom->pose.pose.position.z;
  body2world(3, 3) = 1.0;
  
  Eigen::Matrix4d cam_T = body2world * md_->cam2body_;
  md_->camera_pos_(0) = cam_T(0, 3);
  md_->camera_pos_(1) = cam_T(1, 3);
  md_->camera_pos_(2) = cam_T(2, 3);
  md_->camera_q_ = Eigen::Quaterniond(cam_T.block<3, 3>(0, 0));

  /* get depth image */
  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(img, img->encoding);
  if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
  {
    (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, mp_.k_depth_scaling_factor_);
  }
  cv_ptr->image.copyTo(md_->depth_image_);

  md_->occ_need_update_ = true;
}

// ══════════════════════════════════════════════════════════════
// 前沿检测 (BFS + PCA 递归分裂)
// ══════════════════════════════════════════════════════════════

bool GridMap::isFrontierCell(int gx, int gy) const {
  if (gx < 0 || gx >= map_2d_width_ || gy < 0 || gy >= map_2d_height_)
    return false;
  int idx = gy * map_2d_width_ + gx;
  if (occupancy_2d_persistent_[idx] != -1)  // 不是 unknown
    return false;

  // 4/8 邻域检查是否有 free cell（由 use_8neighbor_frontier_ 控制）
  const int dx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
  const int dy[8] = {0, 0, -1, 1, -1, 1, -1, 1};
  const int num_dirs = use_8neighbor_frontier_ ? 8 : 4;
  for (int k = 0; k < num_dirs; ++k) {
    int nx = gx + dx[k], ny = gy + dy[k];
    if (nx >= 0 && nx < map_2d_width_ && ny >= 0 && ny < map_2d_height_) {
      if (occupancy_2d_persistent_[ny * map_2d_width_ + nx] == 0)
        return true;  // unknown 邻接 free → 前沿
    }
  }
  return false;
}

void GridMap::bfsGrowFrontier(int seed_gx, int seed_gy,
                              const std::vector<int8_t>& occupancy,
                              std::vector<bool>& visited,
                              std::vector<std::pair<int, int>>& cluster_cells) const {
  const int w = map_2d_width_, h = map_2d_height_;
  std::queue<std::pair<int, int>> q;
  q.push({seed_gx, seed_gy});
  visited[seed_gy * w + seed_gx] = true;
  cluster_cells.push_back({seed_gx, seed_gy});

  // 8 邻域偏移
  const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
  const int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

  while (!q.empty()) {
    auto [cx, cy] = q.front(); q.pop();

    for (int k = 0; k < 8; ++k) {
      int nx = cx + dx[k], ny = cy + dy[k];
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      int nidx = ny * w + nx;
      if (visited[nidx]) continue;
      if (occupancy[nidx] != -1) continue;  // 不是 unknown

      // 检查是否是前沿 cell (有 free 邻域, 4或8邻域由标志位控制)
      bool has_free = false;
      const int fdx[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
      const int fdy[8] = {0, 0, -1, 1, -1, 1, -1, 1};
      const int num_dirs = use_8neighbor_frontier_ ? 8 : 4;
      for (int d = 0; d < num_dirs; ++d) {
        int nnx = nx + fdx[d], nny = ny + fdy[d];
        if (nnx >= 0 && nnx < w && nny >= 0 && nny < h &&
            occupancy[nny * w + nnx] == 0) {
          has_free = true; break;
        }
      }
      if (has_free) {
        visited[nidx] = true;
        cluster_cells.push_back({nx, ny});
        q.push({nx, ny});
      }
    }
  }
}

Eigen::Vector3d GridMap::computeClusterCenter(
    const std::vector<std::pair<int, int>>& cluster_cells, double z_height) const {
  double sx = 0, sy = 0;
  for (auto [gx, gy] : cluster_cells) {
    double wx, wy;
    gridToWorld(gx, gy, wx, wy);
    sx += wx; sy += wy;
  }
  double n = cluster_cells.size();
  return Eigen::Vector3d(sx / n, sy / n, z_height);
}

double GridMap::computeClusterSize(
    const std::vector<std::pair<int, int>>& cluster_cells) const {
  double min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
  for (auto [gx, gy] : cluster_cells) {
    double wx, wy;
    gridToWorld(gx, gy, wx, wy);
    if (wx < min_x) min_x = wx;
    if (wx > max_x) max_x = wx;
    if (wy < min_y) min_y = wy;
    if (wy > max_y) max_y = wy;
  }
  return std::max(max_x - min_x, max_y - min_y);
}

void GridMap::splitClusterPCA(
    const std::vector<std::pair<int, int>>& cluster_cells,
    double cluster_size_xy,
    std::vector<std::vector<std::pair<int, int>>>& result) const {

  double sz = computeClusterSize(cluster_cells);
  // 簇够小或点数太少 → 不分裂
  if (sz <= cluster_size_xy || cluster_cells.size() <= 10) {
    result.push_back(cluster_cells);
    return;
  }

  // 转世界坐标矩阵 (N x 2)
  int N = cluster_cells.size();
  Eigen::MatrixXd pos(N, 2);
  for (int i = 0; i < N; ++i) {
    double wx, wy;
    gridToWorld(cluster_cells[i].first, cluster_cells[i].second, wx, wy);
    pos(i, 0) = wx; pos(i, 1) = wy;
  }

  // 协方差
  Eigen::Vector2d mean = pos.colwise().mean();
  Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
  for (int i = 0; i < N; ++i) {
    Eigen::Vector2d diff = pos.row(i).transpose() - mean;
    cov += diff * diff.transpose();
  }
  cov /= N;

  // 特征分解 → 第一主成分
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
  Eigen::Vector2d pc1 = es.eigenvectors().col(1);  // 最大特征值对应向量

  // 沿 PC1 投影 → 中位数二分
  std::vector<double> proj(N);
  for (int i = 0; i < N; ++i) {
    proj[i] = (pos.row(i).transpose() - mean).dot(pc1);
  }
  std::vector<double> proj_sorted = proj;
  std::nth_element(proj_sorted.begin(),
                   proj_sorted.begin() + N / 2, proj_sorted.end());
  double median = proj_sorted[N / 2];

  std::vector<std::pair<int, int>> sub_a, sub_b;
  for (int i = 0; i < N; ++i) {
    if (proj[i] < median)
      sub_a.push_back(cluster_cells[i]);
    else
      sub_b.push_back(cluster_cells[i]);
  }

  // 避免退化 (一边为空)
  if (sub_a.empty() || sub_b.empty()) {
    result.push_back(cluster_cells);
    return;
  }

  splitClusterPCA(sub_a, cluster_size_xy, result);
  splitClusterPCA(sub_b, cluster_size_xy, result);
}

void GridMap::detectFrontierClusters(
    std::vector<std::pair<Eigen::Vector3d, double>>& frontier_pairs,
    double z_height, double cluster_size_xy,
    int min_cluster_cells, double min_frontier_size,
    double min_center_dist) {

  frontier_pairs.clear();
  if (!has_2d_map_initialized_) return;

  // Snapshot occupancy 数据，避免和 publish 线程竞争
  std::vector<int8_t> occ_snapshot;
  {
    std::lock_guard<std::mutex> lock(map_2d_mutex_);
    occ_snapshot = occupancy_2d_persistent_;
  }

  // 扫描范围：已探索区域 (currMapRange) → grid 索引
  int xmin = 0, xmax = map_2d_width_ - 1;
  int ymin = 0, ymax = map_2d_height_ - 1;
  {
    double wx_min, wy_min, wx_max, wy_max;
    // getCurrMapRange 返回 (world_min, world_max) in XY
    wx_min = currMapRangeMin_(0); wy_min = currMapRangeMin_(1);
    wx_max = currMapRangeMax_(0); wy_max = currMapRangeMax_(1);
    if (wx_max > wx_min + 0.01 && wy_max > wy_min + 0.01) {
      xmin = std::max(0, (int)std::floor((wx_min - map_2d_origin_x_) * map_2d_res_inv_));
      ymin = std::max(0, (int)std::floor((wy_min - map_2d_origin_y_) * map_2d_res_inv_));
      xmax = std::min(map_2d_width_ - 1,
                      (int)std::ceil((wx_max - map_2d_origin_x_) * map_2d_res_inv_));
      ymax = std::min(map_2d_height_ - 1,
                      (int)std::ceil((wy_max - map_2d_origin_y_) * map_2d_res_inv_));
    }
  }

  // 外部 visited 数组
  std::vector<bool> visited(map_2d_width_ * map_2d_height_, false);

  // 逐 cell 扫描 → BFS
  for (int gy = ymin; gy <= ymax; ++gy) {
    for (int gx = xmin; gx <= xmax; ++gx) {
      int idx = gy * map_2d_width_ + gx;
      if (visited[idx]) continue;
      if (occ_snapshot[idx] != -1) continue;

      // 用作 isFrontierCell 检查：用 snapshot 数据 (4或8邻域由标志位控制)
      bool is_ftr = false;
      const int d8[8][2] = {{-1,0},{1,0},{0,-1},{0,1},{-1,-1},{-1,1},{1,-1},{1,1}};
      const int num_dirs = use_8neighbor_frontier_ ? 8 : 4;
      for (int k = 0; k < num_dirs; ++k) {
        int nx = gx + d8[k][0], ny = gy + d8[k][1];
        if (nx >= 0 && nx < map_2d_width_ && ny >= 0 && ny < map_2d_height_) {
          if (occ_snapshot[ny * map_2d_width_ + nx] == 0) {
            is_ftr = true; break;
          }
        }
      }
      if (!is_ftr) continue;

      // BFS 收集连通前沿
      std::vector<std::pair<int, int>> cluster_cells;
      bfsGrowFrontier(gx, gy, occ_snapshot, visited, cluster_cells);

      // PCA 分裂
      std::vector<std::vector<std::pair<int, int>>> sub_clusters;
      splitClusterPCA(cluster_cells, cluster_size_xy, sub_clusters);

      // 输出
      for (auto& sub : sub_clusters) {
        if ((int)sub.size() < min_cluster_cells) continue;
        double sz = computeClusterSize(sub);
        if (sz < min_frontier_size) continue;
        Eigen::Vector3d center = computeClusterCenter(sub, z_height);
        // Skip if too close to any already-accepted frontier center
        bool too_close = false;
        for (auto& fp : frontier_pairs) {
          if ((center - fp.first).norm() < min_center_dist) {
            too_close = true;
            break;
          }
        }
        if (too_close) continue;
        frontier_pairs.push_back({center, sz});
      }
    }
  }
}

// GridMap