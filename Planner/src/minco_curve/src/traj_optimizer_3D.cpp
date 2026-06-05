#include "traj_optmizer_3D.h"


PathPlannerSim3D::PathPlannerSim3D(ros::NodeHandle &nh,const int& id)
        : public_nh(nh), has_map_(false), planned_(false), drone_id_(id) {
        std::string cloud_topic;
        //public_nh.param<std::string>("cloud_topic", cloud_topic, "/random_complex/global_map");
        cloud_topic = "/drone_0_fake_explorer_node/grid_map/occupancy_inflate";
        ROS_WARN("The cloud topic is %s",cloud_topic.c_str());
        ROS_WARN("The cloud topic is %s",cloud_topic.c_str());
        ROS_WARN("The cloud topic is %s",cloud_topic.c_str());
        public_nh.param("use_real_model",use_real_model_,false);
        map_sub_ = public_nh.subscribe(cloud_topic, 1, &PathPlannerSim3D::MapCallback, this);

        path_pub_ = public_nh.advertise<nav_msgs::Path>("search_path", 1);
        traj_pub_ = public_nh.advertise<visualization_msgs::Marker>("opt_traj", 1);
        grid_map_vis_pub_ = public_nh.advertise<sensor_msgs::PointCloud2>("grid_map_vis", 1);
        waypoint_pub_ = public_nh.advertise<visualization_msgs::Marker>("minco_waypoints", 10);
        minisnap_pub_ = public_nh.advertise<visualization_msgs::Marker>("minisnap_waypoints", 10);
        wp_traj_vis_pub_ = public_nh.advertise<visualization_msgs::Marker>("minijerk_waypoints", 10);
        // poly_traj_pub_ = public_nh.advertise<traj_utils::PolyTraj>("planning/trajectory", 10);


        public_nh.param<std::string>("astar_topic", astar_topic_, "/a_star_planned_path");
        ROS_INFO("set a* topic");

        // check_timer_ = public_nh.createTimer(ros::Duration(0.2), &PathPlannerSim3D::checkAndPlan, this);

        ROS_INFO("3D Path Planner initialized. Waiting for point cloud and A* path...");
        have_a_star_path_ = false;
        have_minco_waypoints_ = false;
        have_current_traj_ = false;
        have_minisnap_waypoints_ = false;
        
    }

    void PathPlannerSim3D::setParam(ros::NodeHandle &nh){
        nh.param("manager/max_vel", max_vel_, 2.0);
        nh.param("manager/max_acc", max_acc_, 1.5);
    }

    void PathPlannerSim3D::setEnvironment(const GridMap::Ptr &map){
        grid_map_ = map;

        a_star_.reset(new AStar(drone_id_));
        a_star_->initGridMap(grid_map_, Eigen::Vector3i(100, 100, 100));
        has_map_ = true;

    }

    bool PathPlannerSim3D::checkTrajCollision(){
        if(! have_minco_waypoints_ ){
            return true;
        }
        double offset = 0.15; 
        Eigen::Vector3d dir(1.0, 1.0, 0.0); 
        for(const auto& waypoint_position : current_minco_waypoints_){
            if(a_star_->checkOccupancy(waypoint_position,drone_id_)){
                have_a_star_path_ = false;
                have_current_traj_ = false;
                have_minco_waypoints_ = false;
                return true;
            }
        }
        return false;
    }

    PathPlannerSim3D::VectorXd PathPlannerSim3D::timeAllocation(const std::vector<Eigen::Vector3d>& Path,
                        double max_vel, double max_acc,
                        double start_vel, double end_vel) {
        int n_seg = Path.size() - 1;
        VectorXd time(n_seg);
        
        // 1. 计算各段长度及累积弧长
        std::vector<double> seg_len(n_seg);
        std::vector<double> cum_len(n_seg + 1, 0.0);
        double total_dist = 0.0;
        for (int i = 0; i < n_seg; ++i) {
            double dist = (Path[i+1] - Path[i]).norm();
            seg_len[i] = dist;
            total_dist += dist;
            cum_len[i+1] = total_dist;
        }
        
        // 如果总长度接近零，所有段时间为零
        if (total_dist < 1e-6) {
            time.setZero();
            return time;
        }
        
        // 参数有效性检查
        if (max_acc <= 1e-6 || max_vel <= 1e-6) {
            // 采用匀速模型（按长度比例分配时间）
            double total_time = total_dist / max_vel;
            for (int i = 0; i < n_seg; ++i) {
                time(i) = (seg_len[i] / total_dist) * total_time;
            }
            return time;
        }
        
        // 2. 梯形/三角形速度曲线规划
        double v_max_actual = max_vel;
        // 计算加速、减速所需距离（使用 std::max 防止负值）
        double d_acc = std::max(0.0, (v_max_actual * v_max_actual - start_vel * start_vel) / (2.0 * max_acc));
        double d_dec = std::max(0.0, (v_max_actual * v_max_actual - end_vel * end_vel) / (2.0 * max_acc));
        double d_const = total_dist - d_acc - d_dec;
        
        if (d_const < 0.0) {
            // 无法达到 max_vel，降低峰值速度
            // 解方程：total_dist = (v_peak^2 - start_vel^2)/(2a) + (v_peak^2 - end_vel^2)/(2a)
            double v_peak_sq = max_acc * total_dist + (start_vel * start_vel + end_vel * end_vel) / 2.0;
            if (v_peak_sq < 0.0) v_peak_sq = 0.0;
            v_max_actual = std::sqrt(v_peak_sq);
            // 重新计算距离
            d_acc = std::max(0.0, (v_max_actual * v_max_actual - start_vel * start_vel) / (2.0 * max_acc));
            d_dec = std::max(0.0, (v_max_actual * v_max_actual - end_vel * end_vel) / (2.0 * max_acc));
            d_const = 0.0;
            // 数值修正：确保 d_acc + d_dec 不超过总距离
            if (d_acc + d_dec > total_dist) {
                double scale = total_dist / (d_acc + d_dec);
                d_acc *= scale;
                d_dec *= scale;
            }
        }
        
        // 计算各阶段时间
        double t_acc = (v_max_actual - start_vel) / max_acc;
        double t_dec = (v_max_actual - end_vel) / max_acc;
        double t_const = d_const / v_max_actual;
        if (t_const < 0.0) t_const = 0.0;
        
        // 3. 弧长 → 时间映射函数（带数值保护）
        auto time_at_s = [&](double s) -> double {
            if (s <= 0.0) return 0.0;
            if (s >= total_dist) return t_acc + t_const + t_dec;
            
            // 加速段
            if (s <= d_acc) {
                // 解二次方程：0.5 * max_acc * t^2 + start_vel * t - s = 0
                double a = 0.5 * max_acc;
                double b = start_vel;
                double c = -s;
                double discriminant = b * b - 4.0 * a * c;
                if (discriminant < 0.0) discriminant = 0.0;
                double t = (-b + std::sqrt(discriminant)) / (2.0 * a);
                return std::max(0.0, t);
            }
            // 匀速段
            else if (s <= d_acc + d_const) {
                return t_acc + (s - d_acc) / v_max_actual;
            }
            // 减速段
            else {
                double s_dec = s - (d_acc + d_const);
                // 减速方程：s_dec = v_max_actual * t - 0.5 * max_acc * t^2
                // => 0.5 * max_acc * t^2 - v_max_actual * t + s_dec = 0
                double a = 0.5 * max_acc;
                double b = -v_max_actual;
                double c = s_dec;
                double discriminant = b * b - 4.0 * a * c;
                if (discriminant < 0.0) discriminant = 0.0;
                double t = (-b - std::sqrt(discriminant)) / (2.0 * a);
                return t_acc + t_const + std::max(0.0, t);
            }
        };
        
        // 4. 计算每段时间
        for (int i = 0; i < n_seg; ++i) {
            double s_start = cum_len[i];
            double s_end   = cum_len[i+1];
            double t_start = time_at_s(s_start);
            double t_end   = time_at_s(s_end);
            time(i) = std::max(0.0, t_end - t_start);
            if (seg_len[i] < 1e-6) time(i) = 0.0;
        }
        
        // 调试信息
        double total_time = time.sum();
        // ROS_WARN("timeAllocation: start_vel=%.2f, end_vel=%.2f, v_max_actual=%.2f, total_dist=%.2f, total_time=%.2f",
                // start_vel, end_vel, v_max_actual, total_dist, total_time);
        if (std::isnan(total_time)) {
            ROS_ERROR("timeAllocation produced NaN total_time! Using fallback.");
            // 保底匀速分配
            double total_time_fallback = total_dist / max_vel;
            for (int i = 0; i < n_seg; ++i) {
                time(i) = (seg_len[i] / total_dist) * total_time_fallback;
            }
        }
    
    return time;
}

    void PathPlannerSim3D::Planning(const std::vector<Eigen::Vector3d>& waypoints,const Eigen::Vector3d& start_vel,const Eigen::Vector3d& start_acc,
                                    const Eigen::Vector3d& end_vel,const Eigen::Vector3d& end_acc) {
        current_minco_waypoints_.clear();
        current_a_star_waypoints_.clear();
        current_minisnap_waypoints_.clear();
        current_traj_.clear();
        have_current_traj_=false;
        have_minisnap_waypoints_ = false;
        current_minco_waypoints_ = waypoints;
        have_minco_waypoints_=true;
        have_a_star_path_ = true;

        pieceNum_ = current_minco_waypoints_ .size() - 1;
        Eigen::Vector3d start_dir = (current_minco_waypoints_ [1] - current_minco_waypoints_ [0]).normalized();
        Eigen::Vector3d end_dir   = (current_minco_waypoints_ .back() - current_minco_waypoints_ [current_minco_waypoints_ .size()-2]).normalized();

        double start_speed = start_vel.norm();
        double end_speed = end_vel.norm();
        if(start_speed > max_vel_)start_speed = max_vel_;
        // if(end_speed > max_vel_)end_speed = max_vel_;
        Eigen::Vector3d start_vel_calc = start_dir * start_speed;
        Eigen::Vector3d end_vel_calc   = end_dir   * end_speed;

        Eigen::Matrix<double, 3, 3> headState, tailState;
        headState.row(0) = current_minco_waypoints_ .front().transpose();
        headState.row(1) = start_vel_calc.transpose();
        // headState.row(2) = Eigen::Vector3d::Zero().transpose();
        headState.row(2) = start_acc.transpose();
        tailState.row(0) = current_minco_waypoints_ .back().transpose();
        tailState.row(1) = end_vel_calc.transpose();
        // tailState.row(2) = Eigen::Vector3d::Zero().transpose();
        tailState.row(2) = end_acc.transpose();

        opt_.setConditions(headState, tailState, pieceNum_);
        Eigen::Matrix<double, 3, -1> inPos(3, pieceNum_ - 1);
        for (int i = 1; i < pieceNum_; ++i)
            inPos.col(i - 1) = current_minco_waypoints_ [i];
        ts_ = Eigen::VectorXd::Constant(pieceNum_, 1);

        Eigen::VectorXd polytime_init_ = timeAllocation(current_minco_waypoints_ , max_vel_,max_acc_,start_speed,end_speed); 
        opt_.setParameters(inPos.transpose(), polytime_init_);

        poly_traj::Trajectory traj;
        opt_.getTrajectory(traj);
        current_traj_ = traj;
        have_current_traj_ = true;

        std::vector<Eigen::Vector3d> traj_pts;
        for (double t = 0.0; t <= traj.getTotalDuration(); t += 0.05)
            traj_pts.push_back(traj.getPos(t));
        VisuaTraj(traj_pts,traj_pub_);                            
    
    }

    void PathPlannerSim3D::Planning(const Eigen::Vector3d& start_position,const Eigen::Vector3d& start_vel,const Eigen::Vector3d& start_acc,
                                    const Eigen::Vector3d& end_position, const Eigen::Vector3d& end_vel,const Eigen::Vector3d& end_acc) {

        ros::Time start = ros::Time::now();
        current_minco_waypoints_.clear();
        current_a_star_waypoints_.clear();
        current_minisnap_waypoints_.clear();
        current_traj_.clear();
        have_a_star_path_ = false;
        have_minco_waypoints_=false;
        have_current_traj_=false;
        have_minisnap_waypoints_ = false;

        double step_size = grid_map_->getResolution();
        // if(use_real_model_){
        //     step_size = grid_map_->getResolution() + 0.2;
        //     //ROS_WARN("USED");
        // }else{
        //     step_size = grid_map_->getResolution() + 0.2;
        //     //ROS_WARN("DID NOT USED");
        // }
        int flag = a_star_->AstarSearch(step_size, start_position, end_position);

        if(flag == ASTAR_RET::SUCCESS){
           current_minco_waypoints_ = a_star_->getPath();
           have_a_star_path_ = true;
           have_minco_waypoints_=true;
        }else{
            ROS_WARN("A star Failed, quit");
            return;
        }

        // double cost = stc_gen_3d_->getPlanPath(path);
        if (current_minco_waypoints_.size() < 2) {
            ROS_WARN("Invalid A* path (size < 2), skipping optimization.");
            return;
        }


        pieceNum_ = current_minco_waypoints_ .size() - 1;
        Eigen::Vector3d start_dir = (current_minco_waypoints_ [1] - current_minco_waypoints_ [0]).normalized();
        Eigen::Vector3d end_dir   = (current_minco_waypoints_ .back() - current_minco_waypoints_ [current_minco_waypoints_ .size()-2]).normalized();

        double start_speed = start_vel.norm();
        double end_speed = end_vel.norm();
        if(start_speed > max_vel_)start_speed = max_vel_;
        // if(end_speed > max_vel_)end_speed = max_vel_;
        Eigen::Vector3d start_vel_calc = start_dir * start_speed;
        Eigen::Vector3d end_vel_calc   = end_dir   * end_speed;

        Eigen::Matrix<double, 3, 3> headState, tailState;
        headState.row(0) = current_minco_waypoints_ .front().transpose();
        headState.row(1) = start_vel_calc.transpose();
        // headState.row(2) = Eigen::Vector3d::Zero().transpose();
        headState.row(2) = start_acc.transpose();
        tailState.row(0) = current_minco_waypoints_ .back().transpose();
        tailState.row(1) = end_vel_calc.transpose();
        // tailState.row(2) = Eigen::Vector3d::Zero().transpose();
        tailState.row(2) = end_acc.transpose();

        opt_.setConditions(headState, tailState, pieceNum_);
        Eigen::Matrix<double, 3, -1> inPos(3, pieceNum_ - 1);
        for (int i = 1; i < pieceNum_; ++i)
            inPos.col(i - 1) = current_minco_waypoints_ [i];
        ts_ = Eigen::VectorXd::Constant(pieceNum_, 1);

        Eigen::VectorXd polytime_init_ = timeAllocation(current_minco_waypoints_ , max_vel_,max_acc_,start_speed,end_speed); 
        opt_.setParameters(inPos.transpose(), polytime_init_);

        poly_traj::Trajectory traj;
        opt_.getTrajectory(traj);
        current_traj_ = traj;
        have_current_traj_ = true;

        std::vector<Eigen::Vector3d> traj_pts;
        for (double t = 0.0; t <= traj.getTotalDuration(); t += 0.05)
            traj_pts.push_back(traj.getPos(t));
        VisuaTraj(traj_pts,traj_pub_);              
    }
    
    // decompose velocity into x/y components
    // theta: velocity direction（maybe yaw）
void PathPlannerSim3D::DecompVel(const double theta, const double vel, double &vx, double &vy,double &vz){
    const double epsi = 1e-8;
    double vt = std::abs(vel) < epsi?epsi:std::abs(vel);
    vx = vt*std::cos(theta);
    vy = vt*std::sin(theta);
    vz = 0.0;
}


double pointToSegmentSqrDist(const Eigen::Vector3d& pt, const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) {
    Eigen::Vector3d ab = p2 - p1;
    Eigen::Vector3d ac = pt - p1;
    double t = ac.dot(ab) / ab.squaredNorm();
    if (t <= 0.0) return ac.squaredNorm();
    if (t >= 1.0) return (pt - p2).squaredNorm();
    return (ac - t * ab).squaredNorm();
}


    // VisuaTraj
// void PathPlannerSim3D::VisuaTraj(const std::vector<Eigen::Vector3d> &path,ros::Publisher path_pub){
//         nav_msgs::Path vis_path;
//         vis_path.header.frame_id = "map";
//         vis_path.header.stamp = ros::Time::now();
//         vis_path.poses.resize(path.size());
//         for(int i = 0;i<path.size();i++){
//             vis_path.poses[i].header.frame_id = "map";
//             vis_path.poses[i].header.stamp = ros::Time::now();
//             vis_path.poses[i].pose.position.x = path[i].x();
//             vis_path.poses[i].pose.position.y = path[i].y();
//             vis_path.poses[i].pose.position.z = path[i].z();
//         }
//         path_pub.publish(vis_path);
// }
    void PathPlannerSim3D::VisuaTraj(const std::vector<Eigen::Vector3d> &path, ros::Publisher marker_pub) {
        visualization_msgs::Marker line_strip;

        line_strip.header.frame_id = "map";
        line_strip.header.stamp = ros::Time::now();
        line_strip.ns = "minco_opt";      
        line_strip.id = 0;                    
        line_strip.type = visualization_msgs::Marker::LINE_STRIP;
        line_strip.action = visualization_msgs::Marker::ADD;
        
        line_strip.scale.x = 0.1;             
        line_strip.color.a = 1.0;            
        line_strip.color.r = 0.0;            
        line_strip.color.g = 0.0;             
        line_strip.color.b = 1.0;            
        for (const auto& point : path) {
            geometry_msgs::Point p;
            p.x = point.x();
            p.y = point.y();
            p.z = point.z();
            line_strip.points.push_back(p);
        }
        marker_pub.publish(line_strip);
    }

void PathPlannerSim3D::VisuaWaypoints(const std::vector<Eigen::Vector3d> &traj, ros::Publisher marker_pub){
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.ns = "minisnap_waypoints";
    marker.id = 0;              

    // 1. 先删除相同 id 的旧 Marker
    marker.action = visualization_msgs::Marker::DELETE;
    marker_pub.publish(marker);

    // 2. 重新配置为添加模式
    marker.action = visualization_msgs::Marker::ADD;
    marker.type = visualization_msgs::Marker::SPHERE_LIST;
    marker.scale.x = marker.scale.y = marker.scale.z = 0.15;
    marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0; marker.color.a = 1.0;
    marker.pose.orientation.w = 1.0;
    
    marker.points.clear();
    for (const auto &pt : traj) {
        geometry_msgs::Point p;
        p.x = pt.x(); p.y = pt.y(); p.z = pt.z();
        marker.points.push_back(p);
    }
    marker_pub.publish(marker);
}

void PathPlannerSim3D::MapCallback(const sensor_msgs::PointCloud2::ConstPtr& pointcloud_map){
    if(!has_map_ || have_current_traj_)return;
    ROS_INFO_THROTTLE(1.0,"Have Local Map");
    pcl::PointCloud<pcl::PointXYZ> cloud;
    local_obstacles_.clear();
    // pcl::PointCloud<pcl::PointXYZ> cloud_vis;
    // sensor_msgs::PointCloud2 map_vis;

    pcl::fromROSMsg(*pointcloud_map,cloud);
    pcl::PointXYZ pt;
    for (const auto& pt : cloud.points) {
        Eigen::Vector3d temp(pt.x, pt.y, pt.z);
        local_obstacles_.push_back(temp);
        // cloud_vis.points.push_back(pt);
    }
    // cloud_vis.width    = cloud_vis.points.size();
    // cloud_vis.height   = 1;
    // cloud_vis.is_dense = true;
    // pcl::toROSMsg(cloud_vis, map_vis);
    // map_vis.header.frame_id = "map";
    // grid_map_vis_pub_.publish(map_vis);
    // has_map_ = true;
    // ROS_INFO("SimpleMoveBase::MapCallback");
}



void PathPlannerSim3D::polyTraj2ROSMsg(traj_utils::PolyTraj &poly_msg)
  {
    VectorXd durs = minco_traj_ -> getDurations();
    ROS_WARN("Total time is %f",durs.sum());
    int piece_num = minco_traj_ -> getPieceNum();

    poly_msg.drone_id = 0;
    poly_msg.traj_id = 0;
    poly_msg.start_time = ros::Time::now();
    poly_msg.order = 5; // todo, only support order = 5 now.
    poly_msg.duration.resize(piece_num);
    poly_msg.coef_x.resize(6 * piece_num);
    poly_msg.coef_y.resize(6 * piece_num);
    poly_msg.coef_z.resize(6 * piece_num);
    for (int i = 0; i < piece_num; ++i)
    {
      poly_msg.duration[i] = durs(i);

      poly_traj::CoefficientMat cMat = minco_traj_ -> getPiece(i).getCoeffMat();
      int i6 = i * 6;
      for (int j = 0; j < 6; j++)
      {
        poly_msg.coef_x[i6 + j] = cMat(0, j);
        poly_msg.coef_y[i6 + j] = cMat(1, j);
        poly_msg.coef_z[i6 + j] = cMat(2, j);
      }
    }
  }