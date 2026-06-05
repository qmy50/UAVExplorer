#include "fake_explorer_fsm.h"


namespace fake_planner
{

// bool  have_plan_traj_ = false;
// bool have_plan_traj_1 = false;
void FakeExploreFSM::init(ros::NodeHandle &nh)
{
    node_ = nh;
    node_.param("fsm/replan_thresh", replan_thresh_, 0.5);
    node_.param("fsm/planning_horizen", planning_horizen_, 8.0);
    node_.param("fsm/emergency_stop_time", emergency_stop_time_, 3.0);
    node_.param("fsm/target_vel_x", target_vel_(0), 0.0);
    node_.param("fsm/target_vel_y", target_vel_(1), 0.0);
    node_.param("fsm/target_vel_z", target_vel_(2), 0.0);
    node_.param("fsm/target_acc_x", target_acc_(0), 0.0);
    node_.param("fsm/target_acc_y", target_acc_(1), 0.0);
    node_.param("fsm/target_acc_z", target_acc_(2), 0.0);

    nh.param("fsm/waypoint_num", waypoint_num_, -1);
      for (int i = 0; i < waypoint_num_; i++)
      {
          nh.param("fsm/waypoint" + to_string(i) + "_x", waypoints_[i][0], -1.0);
          nh.param("fsm/waypoint" + to_string(i) + "_y", waypoints_[i][1], -1.0);
          nh.param("fsm/waypoint" + to_string(i) + "_z", waypoints_[i][2], -1.0);
      }

    planner_manager_.reset(new FakePlanManager);
    planner_manager_->initPlanModules(node_);

    exec_state_ = INIT;
    have_odom_ = false;
    have_traj_ = false;
    touch_goal_ = false;
    current_wp_idx_ = 0;
    has_cluster_target_ = false;
    executing_cluster_target_ = false;
    cluster_path_ready_ = false;
    task_complete_ = false;

    exec_timer_ = node_.createTimer(ros::Duration(0.02), &FakeExploreFSM::execFSMCallback, this);
    dep_timer_ = node_.createTimer(ros::Duration(0.02),&FakeExploreFSM::execDepCallback, this);
    // safety_timer_ = node_.createTimer(ros::Duration(0.05), &FakeExploreFSM::checkCollisionCallback, this);

    odom_sub_ = node_.subscribe("odom_world", 10, &FakeExploreFSM::odometryCallback, this);

    clicked_point_sub_ = node_.subscribe("/cluster_target", 1, &FakeExploreFSM::clusterTargetCallback, this);

    // trigger_sub_ = node_.subscribe("trigger", 1, &FakeExploreFSM::triggerCallback, this);
    poly_traj_pub_ = node_.advertise<traj_utils::PolyTraj>("planning/trajectory", 10);
    waypoint_pub_ = node_.advertise<visualization_msgs::Marker>("minco_waypoints", 10);
    cluster_target_marker_pub_ = node_.advertise<visualization_msgs::Marker>("cluster_target_marker", 10);
      
    ROS_INFO("FSM initialized, waiting for odom and target.");
    nh.param("fsm/predict_dt", predict_dt_, 0.01);      
    nh.param("fsm/use_kalman_filter", use_kalman_filter_, true);
    
    // parameters for rising
    consecutive_replan_cnt_ = 0;
    rising_traj_generated_ = false;
    
    // explore
    expPlanner_.reset(new globalPlanner::DEP (node_));
		expPlanner_->setMap(planner_manager_->grid_map_);
		expPlanner_->loadVelocity(0.5, 0.5);
    last_dep_plan_time_ = ros::Time(0);
    node_.param("fsm/dep_plan_interval", dep_plan_interval_, 2.0);  // DEP replan interval (seconds)
    node_.param("fsm/interstep_dist", interstep_dist_, 0.1);       // interpolation distance between waypoints
    dep_has_new_path_ = false;

    // early replan based on path progress
    node_.param("fsm/path_progress_thresh", path_progress_thresh_, 0.5);  // progress threshold (0.0~1.0) to allow early replan, default 50%
    initial_dist_to_goal_ = 0.0;
    early_replan_requested_ = false;

    // rotation state
    is_rotating_ = false;
    rotation_end_time_ = ros::Time(0);

    client_ = nh.serviceClient<minco_curve::RotateDrone>("/rotate_drone");
    client_.waitForExistence(ros::Duration(5.0));

}

void FakeExploreFSM::execDepCallback(const ros::TimerEvent &e){
    if(task_complete_){
      ROS_INFO_THROTTLE(1.0,"✅ TASK COMPLETE !!!");
      return;
    }
     // === DEP exploration: call makePlan at throttled interval ===
    ros::Time now = ros::Time::now();
    if(!have_odom_)return;

    // When executing a cluster target, DEP exploration is completely disabled
    if (executing_cluster_target_) {
        return;
    }

    if (exec_state_ == RISING) { expPlanner_->updateRoadmapOnly(); return; }

    // if (have_odom_ && (now - last_dep_plan_time_).toSec() >= dep_plan_interval_) {
        // Skip replanning if we already have a path
        if(dep_has_new_path_){
            ROS_WARN_THROTTLE(1.0,"On our way to current traj,plan latter");
            return;
        }
        bool replanSuccess = expPlanner_->makePlan();
        last_dep_plan_time_ = now;
        if (replanSuccess) {
            // Extract best_path from DEP and convert to waypoints
            nav_msgs::Path bestPathMsg = expPlanner_->getBestPath();
            if (!bestPathMsg.poses.empty()) {
                waypoint_list_.clear();
                current_wp_idx_ = 0;
                for (auto& pose : bestPathMsg.poses) {
                    Eigen::Vector3d wp(pose.pose.position.x,
                                       pose.pose.position.y,
                                       pose.pose.position.z);
                    waypoint_list_.push_back(wp);
                }
                dep_has_new_path_ = true;
                trigger_ = true;
                have_traj_ = false;
                touch_goal_ = false;
                // Record initial distance to goal for progress tracking
                initial_dist_to_goal_ = (odom_pos_ - waypoint_list_.back()).norm();
                early_replan_requested_ = false;
                ROS_WARN("[ExploreFSM] Got new best_path with %zu waypoints, first: (%.2f, %.2f, %.2f), last: (%.2f, %.2f, %.2f)",
                         waypoint_list_.size(),
                         waypoint_list_.front().x(), waypoint_list_.front().y(), waypoint_list_.front().z(),
                         waypoint_list_.back().x(), waypoint_list_.back().y(), waypoint_list_.back().z());
                // If currently executing, interrupt and replan with new path
                if (exec_state_ == EXEC_TRAJ) {
                    changeFSMExecState(WAIT_TRAJ, "New DEP path, replan");
                }
            }
        } else {
            ROS_WARN_THROTTLE(1.0,"[ExploreFSM] DEP makePlan failed, rotating to explore");
            srv_.request.angle_rad = M_PI/12;
            srv_.request.speed_rad_sec = 0.5f;
            srv_.request.pos_x = odom_pos_.x();
            srv_.request.pos_y = odom_pos_.y();
            srv_.request.pos_z = odom_pos_.z();
            srv_.request.angle_start = current_yaw_;
            if(client_.call(srv_)){
              if(srv_.response.success){
                ROS_INFO_THROTTLE(1.0,"Rotation service: %s", srv_.response.message.c_str());
                is_rotating_ = true;
              }else{
                ROS_WARN_THROTTLE(1.0,"Rotation service rejected: %s", srv_.response.message.c_str());
              }
            }else{
              ROS_ERROR_THROTTLE(1.0,"Rotation service call failed (service unavailable)");
            }
        }
}

void FakeExploreFSM::execFSMCallback(const ros::TimerEvent &e)
{
    if(task_complete_){
      ROS_INFO("✅ TASK COMPLETE !!!");
      return;
    }
    if (exec_state_ == EMERGENCY_STOP)
    {
      return;
    }

    switch (exec_state_)
    {
      case INIT:{
      if(! have_odom_){
        ROS_WARN_THROTTLE(1.0,"No ODOM !!");
        goto force_return;
      }
      // For exploration: auto-trigger when DEP gives a path
      if(! trigger_ && !dep_has_new_path_) goto force_return;
      if(! trigger_) goto force_return;
        changeFSMExecState(WAIT_TRAJ, "INIT -> WAIT_TRAJ");
        break;
      }

      case WAIT_TRAJ:{
        if(! have_odom_) goto force_return;
        // For exploration: keep triggering if we have waypoints from DEP
        if(! trigger_ && waypoint_list_.empty()) goto force_return;
        if(!waypoint_list_.empty() && current_wp_idx_ < (int)waypoint_list_.size()){
            // Interpolate between DEP waypoints to get dense waypoints, then plan smooth trajectory
            // First, collect remaining waypoints from current_wp_idx_ to the end
            std::vector<Eigen::Vector3d> remaining_wps(
                waypoint_list_.begin() + current_wp_idx_, waypoint_list_.end());

            // Interpolate to fill gaps between sparse DEP waypoints
           
              std::vector<Eigen::Vector3d> interp_wps = interpolateWaypoints(remaining_wps, interstep_dist_);
              VisuaWaypoints(interp_wps,waypoint_pub_);

              if (interp_wps.size() >= 2) {
                  bool ok = planToTarget(interp_wps);
                  if (ok) {
                      changeFSMExecState(EXEC_TRAJ, "WAIT_TRAJ -> EXEC_TRAJ");
                  } else {
                      // Planning failed, wait for next DEP plan
                      trigger_ = false;
                      have_traj_ = false;
                      ROS_WARN("[ExploreFSM] planToTarget with interpolated waypoints failed, waiting for next DEP plan");
                  }
              } else {
                  // Not enough waypoints after interpolation, wait for next DEP plan
                  trigger_ = false;
                  have_traj_ = false;
                  ROS_INFO("[ExploreFSM] Not enough interpolated waypoints, waiting for next DEP plan");
              }
              
          }else{
            // No waypoints, wait for DEP to provide new path
            goto force_return;
          }
        break;
      }
      
      case EXEC_TRAJ:{
          if(!have_traj_) {
              ROS_WARN("EXEC_TRAJ without valid trajectory, switching to WAIT_TRAJ");
              changeFSMExecState(WAIT_TRAJ, "Missing traj in EXEC_TRAJ");
              break;
          }
          // Check if we've reached the final goal of the current waypoint list
          bool touch_the_goal =  (odom_pos_ - waypoint_list_.back()).norm() < replan_thresh_; 
          if(touch_the_goal){
              ROS_INFO("[ExploreFSM] Reached final goal of current best_path");
              current_wp_idx_ = waypoint_list_.size();  // mark all waypoints as completed
              have_traj_ = false;
              // All waypoints in current best_path reached
              touch_goal_ = true;
              trigger_ = false;
              have_traj_ = false;
              dep_has_new_path_ = false;
              // If this was a cluster target, clear the cluster mode and resume exploration
              if (executing_cluster_target_) {
                  executing_cluster_target_ = false;
                  has_cluster_target_ = false;
                  cluster_path_ready_ = false;
                  task_complete_ = true;
                  ROS_INFO("[ExploreFSM] Cluster target reached, finish task");
              }
              // Request immediate DEP replan for next exploration target
              last_dep_plan_time_ = ros::Time(0);
              changeFSMExecState(INIT, "EXEC_TRAJ -> INIT (wait for next DEP path)");
          }else{
              // If executing a cluster target, skip DEP replan — just keep going
              if (executing_cluster_target_) {
                  // Still check collision safety; if need replan, re-plan via cluster PRM path
                  static ros::Time last_cluster_replan_time = ros::Time::now();
                  if (planner_manager_->needRePlan() && (ros::Time::now() - last_cluster_replan_time).toSec()>0.02) {
                      ROS_WARN("[ExploreFSM] Collision risk on cluster path, re-planning via PRM...");
                      last_cluster_replan_time = ros::Time::now();
                      have_traj_ = false;
                      cluster_path_ready_ = false;
                      changeFSMExecState(GOTOCLUSTER, "Replan cluster path due to collision");
                      break;
                  }
                  break;  // remain in EXEC_TRAJ, don't go into DEP replan
              }
              static ros::Time last_replan_time = ros::Time::now();
              if (planner_manager_->needRePlan() && (ros::Time::now() - last_replan_time).toSec()>0.02) {   
                ROS_WARN_THROTTLE(1.0,"Need Replan");
                if((ros::Time::now() - last_replan_time).toSec()<0.04){
                    consecutive_replan_cnt_++;}
                else if((ros::Time::now() - last_replan_time).toSec() > 1.2){
                    consecutive_replan_cnt_ = 0;
                }
                last_replan_time = ros::Time::now();
                ROS_WARN("Current consecutive_replan_cnt = %d",consecutive_replan_cnt_);
                if (consecutive_replan_cnt_ >= 30) {
                    changeFSMExecState(RISING, "Too many replans, rising");
                }else{
                  have_traj_ = false;
                  dep_has_new_path_ = false;       // allow DEP to generate new plan
                  changeFSMExecState(WAIT_TRAJ, "Need replan due to collision");
                }
            }
          }
        break;
      }

      case RISING: {
        if (!rising_traj_generated_){

            std::vector<Eigen::Vector3d> wps;
            wps.push_back(odom_pos_);
            rise_target_ << odom_pos_(0),odom_pos_(1),odom_pos_(2) + 0.5f;
            wps.push_back(rise_target_);
            bool success = planner_manager_->planGlobalTraj(wps, odom_vel_, odom_acc_,Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
        if (success) {
            traj_utils::PolyTraj poly_msg;
            planner_manager_->polyTraj2ROSMsg(poly_msg);
            publishTraj(poly_msg);
            have_traj_ = true;
            rising_traj_generated_ = true;
            // dep_has_new_path_ = true; // 如果在上升，禁止dep工作防止和旋转相互冲突
        } else {
            ROS_ERROR("Failed to generate rising trajectory, emergency stop?");
            changeFSMExecState(EMERGENCY_STOP, "Rising failed");
        }
      }
        else{
        if ((odom_pos_ - rise_target_).norm() < replan_thresh_) {
            consecutive_replan_cnt_ = 0;   
            have_traj_ = false;
            rising_traj_generated_ = false;
            dep_has_new_path_ = false;
            waypoint_list_.clear();
            changeFSMExecState(WAIT_TRAJ, "Rising completed, resume original target");
        }
      }
        break;
      }

      case GOTOCLUSTER: {
        if (!have_odom_) goto force_return;
        if (!cluster_path_ready_) {
            ROS_INFO("[ExploreFSM] GOTOCLUSTER: planning direct trajectory to cluster target...");
            bool ok = gotoClusterPlan();
            if (ok) {
                cluster_path_ready_ = true;
                changeFSMExecState(EXEC_TRAJ, "GOTOCLUSTER -> EXEC_TRAJ");
            } else {
                ROS_WARN("[ExploreFSM] GOTOCLUSTER: planToTarget failed, retrying...");
                // Stay in GOTOCLUSTER, will retry next cycle
            }
        }
        break;
      }
      default:
        break;
      }
    force_return:;
    exec_timer_.start();
}

std::vector<Eigen::Vector3d> FakeExploreFSM::interpolateWaypoints(const std::vector<Eigen::Vector3d>& waypoints, double step) {
    std::vector<Eigen::Vector3d> result;
    if (waypoints.size() < 2 || step <= 1e-6) {
        return waypoints;  // nothing to interpolate
    }

    result.push_back(waypoints.front());
    for (size_t i = 1; i < waypoints.size(); ++i) {
        const Eigen::Vector3d& p0 = waypoints[i - 1];
        const Eigen::Vector3d& p1 = waypoints[i];
        double seg_len = (p1 - p0).norm();
        if (seg_len < 1e-6) continue;  // skip duplicate points

        int n_interp = std::max(1, static_cast<int>(std::floor(seg_len / step)));
        for (int k = 1; k <= n_interp; ++k) {
            double t = static_cast<double>(k) / static_cast<double>(n_interp + 1);
            result.push_back(p0 + t * (p1 - p0));
        }
        result.push_back(p1);
    }
    return result;
}

bool FakeExploreFSM::planToTarget(const std::vector<Eigen::Vector3d> &target_waypoints) {
    if (target_waypoints.size() < 2) {
        ROS_WARN("[ExploreFSM] planToTarget(vector): need at least 2 waypoints, got %zu", target_waypoints.size());
        return false;
    }

    Eigen::Vector3d start_pt = odom_pos_ + odom_vel_ * predict_dt_;
    ROS_WARN("STARTPOINT x = %f,y = %f,z = %f", start_pt[0], start_pt[1], start_pt[2]);
    Eigen::Vector3d start_vel = odom_vel_ * 0.5;
    Eigen::Vector3d start_acc = odom_acc_;

    // Replace first waypoint with predicted start position so the trajectory starts from current state
    std::vector<Eigen::Vector3d> wps = target_waypoints;
    wps.front() = start_pt;

    bool success = planner_manager_->planGlobalTraj(wps, start_vel, start_acc, target_vel_, target_acc_);

    if (success) {
        traj_utils::PolyTraj poly_msg;
        traj_utils::MINCOTraj MINCO_msg;
        planner_manager_->polyTraj2ROSMsg(poly_msg, MINCO_msg);
        publishTraj(poly_msg);
        have_traj_ = true;
        // broadcast_ploytraj_pub_.publish(MINCO_msg);
        ROS_INFO("[ExploreFSM] planToTarget(vector): smooth trajectory generated with %zu waypoints", wps.size());
    } else {
        ROS_WARN("[ExploreFSM] planToTarget(vector): planGlobalTraj failed");
    }
    return success;
}

bool FakeExploreFSM::planToTarget(const Eigen::Vector3d &target_pt) {
    // Build a 2-point waypoint list: predicted start → target, then reuse vector version
    Eigen::Vector3d start_pt = odom_pos_ + odom_vel_ * predict_dt_;
    Eigen::Vector3d end_vel, end_acc;
    Eigen::Vector3d start_vel = odom_vel_ * 0.5;
    Eigen::Vector3d start_acc = odom_acc_;
    
    bool success = false;
    if (current_wp_idx_ == waypoint_list_.size() - 1) {
        success = planner_manager_->planGlobalTraj(start_pt, start_vel, start_acc,
                                                   target_pt, target_vel_, target_acc_);
    } else {
        end_vel.setZero();
        end_acc.setZero();
        success = planner_manager_->planGlobalTraj(start_pt, start_vel, start_acc,
                                                   target_pt, end_vel, end_acc);
    }
    
    if (success) {
        traj_utils::PolyTraj poly_msg;
        traj_utils::MINCOTraj MINCO_msg;
        //traj_utils::Waypoints Waypoints_msg;
        planner_manager_->polyTraj2ROSMsg(poly_msg,MINCO_msg);
        publishTraj(poly_msg);
        have_traj_ = true;
        // broadcast_ploytraj_pub_.publish(MINCO_msg);
    }
    return success;
}

void FakeExploreFSM::checkCollisionCallback(const ros::TimerEvent &e)
{
  if (exec_state_ == EMERGENCY_STOP) return;

}

void FakeExploreFSM::emergencyStop()
{
  if (exec_state_ != EMERGENCY_STOP)
  {
    changeFSMExecState(EMERGENCY_STOP, "Emergency stop");

  }
}

bool FakeExploreFSM::gotoClusterPlan()
{
  if (waypoint_list_.empty()) {
    ROS_ERROR("[ExploreFSM] gotoClusterPlan: waypoint_list_ is empty");
    return false;
  }

  // Plan a direct smooth trajectory from current position to the cluster target point
  // This uses planToTarget(single-point) which calls planGlobalTraj for MINCO optimization
  Eigen::Vector3d target = waypoint_list_[0];
  bool success = planToTarget(target);

  if (success) {
    ROS_INFO("[ExploreFSM] gotoClusterPlan: direct trajectory planned to cluster target (%.2f, %.2f, %.2f)",
             target.x(), target.y(), target.z());
    return true;
  } else {
    ROS_WARN("[ExploreFSM] gotoClusterPlan: planToTarget(single) failed");
    return false;
  }
}

void FakeExploreFSM::clusterTargetCallback(const geometry_msgs::PoseStampedConstPtr &msg)
{
  // Ignore new cluster targets if already executing one
  if (executing_cluster_target_) {
    ROS_WARN("[ExploreFSM] Already executing a cluster target, ignoring new one");
    return;
  }

  cluster_target_pt_ = Eigen::Vector3d(msg->pose.position.x,
                                        msg->pose.position.y,
                                        msg->pose.position.z);
  ROS_INFO("[ExploreFSM] Cluster target received at (%.2f, %.2f, %.2f), switching to GOTOCLUSTER",
           cluster_target_pt_.x(), cluster_target_pt_.y(), cluster_target_pt_.z());

  // Publish cluster target as a red sphere Marker in RViz
  {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.ns = "cluster_target";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = cluster_target_pt_.x();
    marker.pose.position.y = cluster_target_pt_.y();
    marker.pose.position.z = cluster_target_pt_.z();
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.3;
    marker.scale.y = 0.3;
    marker.scale.z = 0.3;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker.lifetime = ros::Duration(0);  // persistent until new target received
    cluster_target_marker_pub_.publish(marker);
  }

  // Set single waypoint and enter cluster mode (DEP exploration fully disabled)
  waypoint_list_.clear();
  waypoint_list_.push_back(cluster_target_pt_);
  current_wp_idx_ = 0;
  has_cluster_target_ = true;
  executing_cluster_target_ = true;
  cluster_path_ready_ = false;
  have_traj_ = false;
  dep_has_new_path_ = true;   // block DEP from overwriting waypoint_list_
  // trigger_ = false;            // don't use DEP exploration trigger

  changeFSMExecState(GOTOCLUSTER, "Cluster target received");
}

void FakeExploreFSM::odometryCallback(const nav_msgs::OdometryConstPtr &msg)
{
  Eigen::Vector3d new_pos;
  have_odom_ = true;
  new_pos(0) = msg->pose.pose.position.x;
  new_pos(1) = msg->pose.pose.position.y;
  new_pos(2) = msg->pose.pose.position.z;

  last_odom_pos_ = new_pos;
  has_last_odom_ = true;

  odom_pos_ = new_pos;
  odom_vel_(0) = msg->twist.twist.linear.x;
  odom_vel_(1) = msg->twist.twist.linear.y;
  odom_vel_(2) = msg->twist.twist.linear.z;

  tf::Quaternion q(msg->pose.pose.orientation.x,msg->pose.pose.orientation.y,
                   msg->pose.pose.orientation.z,msg->pose.pose.orientation.w);
  current_yaw_ = tf::getYaw(q);
  current_angular_z_ = msg->twist.twist.angular.z;

  odom_acc_.setZero();

}

void FakeExploreFSM::publishTraj(const traj_utils::PolyTraj &traj_msg)
{
  poly_traj_pub_.publish(traj_msg);
}

void FakeExploreFSM::printFSMExecState()
{
  const char *state_names[] = {"INIT", "WAIT_TRAJ", "EXEC_TRAJ", "RISING", "GOTOCLUSTER", "EMERGENCY_STOP"};
  ROS_INFO("FSM State: %s", state_names[exec_state_]);
}

void FakeExploreFSM::changeFSMExecState(FSM_EXEC_STATE new_state, const std::string &pos_call)
{
  if (exec_state_ == new_state) return;
//   ROS_INFO("[%s] -> %s", pos_call.c_str(), exec_state_.c_str());
  exec_state_ = new_state;
  printFSMExecState();
}

void FakeExploreFSM::VisuaWaypoints(const std::vector<Eigen::Vector3d> &traj, ros::Publisher marker_pub){
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.ns = "minisnap_waypoints";
    marker.id = 0;              

    marker.action = visualization_msgs::Marker::DELETE;
    marker_pub.publish(marker);
    marker.action = visualization_msgs::Marker::ADD;
    marker.type = visualization_msgs::Marker::SPHERE_LIST;
    marker.scale.x = marker.scale.y = marker.scale.z = 0.15;
    marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0; marker.color.a = 1.0;
    marker.pose.orientation.w = 1.0;
    
    marker.points.clear();
    for(int i=0 ;i < traj.size(); i++){
      geometry_msgs::Point p;
      p.x = traj[i].x(),p.y = traj[i].y(),p.z = traj[i].z();
      marker.points.push_back(p);
    }
    marker_pub.publish(marker);
}

} // namespace fake_planner