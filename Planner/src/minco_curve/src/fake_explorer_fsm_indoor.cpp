#include "fake_explorer_fsm.h"
#include <tf/tf.h>

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

    // safety_timer_ = node_.createTimer(ros::Duration(0.05), &FakeExploreFSM::checkCollisionCallback, this);

    odom_sub_ = node_.subscribe("odom_world", 10, &FakeExploreFSM::odometryCallback, this);

    // clicked_point_sub_ = node_.subscribe("/cluster_target", 1, &FakeExploreFSM::clusterTargetCallback, this);

    // trigger_sub_ = node_.subscribe("trigger", 1, &FakeExploreFSM::triggerCallback, this);
    poly_traj_pub_ = node_.advertise<traj_utils::PolyTraj>("planning/trajectory", 10);
    waypoint_pub_ = node_.advertise<visualization_msgs::Marker>("minco_waypoints", 10);
    cluster_target_marker_pub_ = node_.advertise<visualization_msgs::Marker>("cluster_target_marker", 10);
    object_cloud_viz_pub_ = node_.advertise<visualization_msgs::Marker>("object_cloud_viz", 10);
      
    ROS_INFO("FSM initialized, waiting for odom and target.");
    nh.param("fsm/predict_dt", predict_dt_, 0.01);      
    nh.param("fsm/use_kalman_filter", use_kalman_filter_, true);
    consecutive_replan_cnt_ = 0;
    
    // explore
    expPlanner_.reset(new globalPlanner::DEP (node_));
		expPlanner_->setMap(planner_manager_->grid_map_);
		expPlanner_->loadVelocity(0.5, 0.5);
    last_dep_plan_time_ = ros::Time(0);
    node_.param("fsm/dep_plan_interval", dep_plan_interval_, 2.0);  // DEP replan interval (seconds)
    node_.param("fsm/interstep_dist", interstep_dist_, 0.1);       // interpolation distance between waypoints
    dep_has_new_path_ = false;

    // object mapping — replaces /cluster_target with multi-frame fused objects
    object_manager_.reset(new ObjectMapManager(planner_manager_->grid_map_, nh));
    object_manager_->init();

    // early replan based on path progress
    node_.param("fsm/path_progress_thresh", path_progress_thresh_, 0.5);  // progress threshold (0.0~1.0) to allow early replan, default 50%
    // node_.param("/change_layer",change_layer_,false);
    initial_dist_to_goal_ = 0.0;
    early_replan_requested_ = false;

    // rotation state
    just_rotating_ = false;
    rotation_end_time_ = ros::Time(0);

    // ━━━ 初始环顾旋转 ━━━
    node_.param("fsm/init_rotate", init_rotate_total_steps_, 4);       // 总步数，默认 12 步
    node_.param("fsm/init_rotate_angle_deg", init_rotate_angle_per_step_, 90.0);  // 每步角度(度)
    init_rotate_angle_per_step_ = init_rotate_angle_per_step_ * M_PI / 180.0;     // 转 rad
    init_rotate_step_ = 0;
    init_rotate_start_time_ = ros::Time(0);
    // ━━━━━━━━━━━━━━━━━━━━━

    // backtracking state
    is_backtracking_ = false;
    dep_fail_cnt_ = 0;

    // wait for 2D map + ValueMap before starting DEP exploration
    maps_ready_ = false;
    maps_ready_since_ = ros::Time(0);

    // Rotation action client — traj_server notifies us when rotation actually completes
    rotate_action_client_.reset(new RotateActionClient("/rotate_drone_action", true));
    rotate_goal_sent_ = false;
    ROS_INFO("[ExploreFSM] Waiting for rotation action server...");
    if (!rotate_action_client_->waitForServer(ros::Duration(5.0))) {
      ROS_ERROR("[ExploreFSM] Rotation action server not available!");
    } else {
      ROS_INFO("[ExploreFSM] Rotation action server connected.");
    }

    change_layer_sub_ = node_.subscribe("/change_layer", 10, &FakeExploreFSM::changeLayerCallback, this);
    drop_goal_sub_ = node_.subscribe("/drop_goal", 10, &FakeExploreFSM::dropGoalCallback, this);

    change_layer_test_ = false;
    drop_goal_test_ = false;

    ros::Duration(0.5).sleep(); // wait for maps's preperation

    exec_timer_ = node_.createTimer(ros::Duration(0.02), &FakeExploreFSM::execFSMCallback, this);
    dep_timer_ = node_.createTimer(ros::Duration(0.02),&FakeExploreFSM::execDepCallback, this);

}

void FakeExploreFSM::changeLayerCallback(const std_msgs::Bool& msg){
    change_layer_test_ = !change_layer_test_;
    if(change_layer_test_){
        expPlanner_->switchHeightLayer();
        ROS_WARN("Change layer test !!");
    }
}

void FakeExploreFSM::dropGoalCallback(const std_msgs::Bool& msg){
    drop_goal_test_ = !drop_goal_test_;
    if(drop_goal_test_){
        expPlanner_->dropCurrentGoalNode();
        ROS_WARN("Drop goal test !!");
    }
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
    // Skip DEP replanning while rotating (ROTATING state handles the wait)
    if(exec_state_ == ROTATING) return;
    // Skip DEP replanning during init look-around (value map builds up via valuemapCB)
    if(exec_state_ == INIT_ROTATE) return;

    // Wait for 2D grid map and ValueMap to be fully initialized, then allow a short
    // warmup period for depth data to accumulate before starting DEP exploration.
    // Without this warmup, DEP samples with only ~1 depth frame of data and produces
    // waypoints that the trajectory optimizer cannot handle → flickering replan loop.
    if (!maps_ready_) {
      bool grid_ready = planner_manager_->grid_map_->is2DMapReady();
      bool value_ready = expPlanner_->isValueMapReady();
      if (grid_ready && value_ready) {
        if (maps_ready_since_.isZero()) {
          maps_ready_since_ = ros::Time::now();
        }
        double waited = (ros::Time::now() - maps_ready_since_).toSec();
        if (waited > 1.0) {  // 500ms warmup ≈ 10 depth frames at 20Hz
          maps_ready_ = true;
          ROS_INFO("[ExploreFSM] Maps ready (2D grid + ValueMap), starting DEP exploration");
        }
        // else: still warming up, return without making a plan
        return;
      } else {
        maps_ready_since_ = ros::Time(0);  // reset if maps become unready
        ROS_INFO_THROTTLE(2.0, "[ExploreFSM] Waiting for maps (grid=%s, value=%s)...",
                          grid_ready ? "OK" : "WAIT",
                          value_ready ? "OK" : "WAIT");
        return;
      }
    }

    // if (have_odom_ && (now - last_dep_plan_time_).toSec() >= dep_plan_interval_) {
        // Skip replanning if we already have a path
        if(dep_has_new_path_ ){
            // ROS_WARN_THROTTLE(1.0,"On our way to current traj,plan latter");
            return;
        }

        // === Priority: high-confidence semantic objects (multi-frame fused) ===
        // Only check when not already executing a cluster/object target
        if (!executing_cluster_target_ && object_manager_) {
          Eigen::Vector3d object_target;
          if (object_manager_->getBestObjectTarget(object_target, odom_pos_)) {
            ROS_WARN("[ExploreFSM] ObjectMap2D: AABB target (%.2f, %.2f, %.2f) → GOTOCLUSTER",
                     object_target.x(), object_target.y(), object_target.z());

            // Visualize target marker
            {
              visualization_msgs::Marker target_marker;
              target_marker.header.frame_id = "map";
              target_marker.header.stamp = ros::Time::now();
              target_marker.ns = "cluster_target";
              target_marker.id = 0;
              target_marker.type = visualization_msgs::Marker::SPHERE;
              target_marker.action = visualization_msgs::Marker::ADD;
              target_marker.pose.position.x = object_target.x();
              target_marker.pose.position.y = object_target.y();
              target_marker.pose.position.z = object_target.z();
              target_marker.pose.orientation.w = 1.0;
              target_marker.scale.x = target_marker.scale.y = target_marker.scale.z = 0.3;
              target_marker.color.r = 1.0;
              target_marker.color.g = 0.0;
              target_marker.color.b = 0.0;
              target_marker.color.a = 1.0;
              target_marker.lifetime = ros::Duration(1.0);
              cluster_target_marker_pub_.publish(target_marker);
            }

            waypoint_list_.clear();
            waypoint_list_.push_back(object_target);
            current_wp_idx_ = 0;
            dep_has_new_path_ = true;
            trigger_ = true;
            have_traj_ = false;
            touch_goal_ = false;
            initial_dist_to_goal_ = (odom_pos_ - object_target).norm();
            early_replan_requested_ = false;

            executing_cluster_target_ = true;
            cluster_path_ready_ = false;
            changeFSMExecState(GOTOCLUSTER, "ObjectMap2D → GOTOCLUSTER");
            return;
          }
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
                is_backtracking_ = false;
                dep_fail_cnt_ = 0;
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
        } else if (!is_backtracking_ && !waypoint_list_.empty()) {
            dep_fail_cnt_++;
            ROS_WARN("[ExploreFSM] DEP makePlan FAILED (%d/10)", dep_fail_cnt_);
            // Only trigger backtracking when drone has reached the end of current waypoint_list
            // Otherwise, let the drone finish its current path first — DEP failure may be temporary
            bool at_end_of_waypoints = (odom_pos_ - waypoint_list_.back()).norm() < replan_thresh_;
            if (dep_fail_cnt_ >= 10 && at_end_of_waypoints) {
                // DEP failed to find a forward path, and we are at the end of current path
                // → backtrack along the last successful path to escape this dead-end area
                ROS_WARN("[ExploreFSM] DEP makePlan FAILED 10 times at end of waypoint_list, backtracking along previous path (%zu waypoints)",
                         waypoint_list_.size());
                std::reverse(waypoint_list_.begin(), waypoint_list_.end());
                current_wp_idx_ = 0;
                dep_has_new_path_ = true;
                trigger_ = true;
                have_traj_ = false;
                touch_goal_ = false;
                is_backtracking_ = true;
                dep_fail_cnt_ = 0;
                ROS_WARN("[ExploreFSM] Backtracking: reversed waypoint_list_, first: (%.2f, %.2f, %.2f), last: (%.2f, %.2f, %.2f)",
                         waypoint_list_.front().x(), waypoint_list_.front().y(), waypoint_list_.front().z(),
                         waypoint_list_.back().x(), waypoint_list_.back().y(), waypoint_list_.back().z());
            } else if (dep_fail_cnt_ >= 10 && !at_end_of_waypoints) {
                // DEP keeps failing but drone hasn't reached the end yet
                // Reset counter so it can retry after reaching the end
                ROS_WARN("[ExploreFSM] DEP failed 10 times but not at end of waypoints yet (dist=%.2f > thresh=%.2f), resetting counter",
                         (odom_pos_ - waypoint_list_.back()).norm(), replan_thresh_);
                dep_fail_cnt_ = 0;
            }
        }
}

void FakeExploreFSM::execFSMCallback(const ros::TimerEvent &e)
{
    if(task_complete_){
      ROS_INFO_THROTTLE(1.0,"✅ TASK COMPLETE !!!");
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
        goto force_return;
      }
      if(! maps_ready_) goto force_return;
      // ━━━ 首次启动: 先环顾一周收集语义 value map ━━━
      // 重新进入 INIT 时（后续探索周期）跳过环顾，直接规划
      if (init_rotate_step_ < init_rotate_total_steps_) {
        ROS_INFO("[ExploreFSM] Starting initial look-around rotation (%d steps × %.1f°)",
                 init_rotate_total_steps_, init_rotate_angle_per_step_ * 180.0 / M_PI);
        changeFSMExecState(INIT_ROTATE, "INIT -> INIT_ROTATE");
      } else {
        // 后续周期: 正常 DEP 规划
        if(! trigger_ && !dep_has_new_path_) goto force_return;
        if(! trigger_) goto force_return;
        changeFSMExecState(WAIT_TRAJ, "INIT -> WAIT_TRAJ");
      }
      break;
      }

      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      // 初始环顾旋转: 使用 action 通信，traj_server 旋转完成后主动通知
      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      case INIT_ROTATE:{
        if (init_rotate_step_ >= init_rotate_total_steps_) {
          last_dep_plan_time_ = ros::Time(0);
          dep_has_new_path_ = false;
          just_rotating_ = false;
          rotate_goal_sent_ = false;
          ROS_INFO("[ExploreFSM] Init rotation complete (%d steps), starting exploration",
                   init_rotate_total_steps_);
          changeFSMExecState(WAIT_TRAJ, "INIT_ROTATE -> WAIT_TRAJ");
          break;
        }

        // Goal 已发出 → 等待 traj_server 通知完成
        if (rotate_goal_sent_) {
          auto state = rotate_action_client_->getState();
          if (state == actionlib::SimpleClientGoalState::SUCCEEDED) {
            auto result = rotate_action_client_->getResult();
            rotate_goal_sent_ = false;
            init_rotate_step_++;
            ROS_INFO("[ExploreFSM] Init rotation step %d/%d done (action result: %s, final_yaw=%.1f°)",
                     init_rotate_step_, init_rotate_total_steps_,
                     result->message.c_str(), result->final_yaw * 180.0 / M_PI);
          } else if (state == actionlib::SimpleClientGoalState::ABORTED) {
            ROS_WARN("[ExploreFSM] Rotation step %d ABORTED, retrying",
                     init_rotate_step_ + 1);
            rotate_goal_sent_ = false;  // 下一周期重发
          } else {
            // 还在执行中: 最大超时保护
            constexpr double kMaxTimeout = 8.0;
            double elapsed = (ros::Time::now() - init_rotate_start_time_).toSec();
            if (elapsed > kMaxTimeout) {
              ROS_ERROR("[ExploreFSM] Rotation step %d timeout (%.1fs), cancelling and skipping",
                        init_rotate_step_ + 1, elapsed);
              rotate_action_client_->cancelGoal();
              rotate_goal_sent_ = false;
              init_rotate_step_++;  // 跳过这一步步
            }
          }
          break;
        }

        // 发送下一步旋转 goal
        {
          ROS_INFO("[ExploreFSM] Init rotation step %d/%d: rotate %.1f°",
                   init_rotate_step_ + 1, init_rotate_total_steps_,
                   init_rotate_angle_per_step_ * 180.0 / M_PI);

          minco_curve::RotateDroneGoal goal;
          goal.angle_rad = init_rotate_angle_per_step_;
          goal.speed_rad_sec = 0.8;
          goal.pos_x = odom_pos_.x();
          goal.pos_y = odom_pos_.y();
          goal.pos_z = odom_pos_.z();
          goal.angle_start = current_yaw_;

          rotate_action_client_->sendGoal(goal);
          rotate_goal_sent_ = true;
          init_rotate_start_time_ = ros::Time::now();
          ROS_INFO("[ExploreFSM] Rotation action goal sent, step %d/%d",
                   init_rotate_step_ + 1, init_rotate_total_steps_);
        }
        break;
      }

      case WAIT_TRAJ:{
        if(! have_odom_) goto force_return;
        // For exploration: keep triggering if we have waypoints from DEP
        if(! trigger_ && waypoint_list_.empty()) goto force_return;
        if(!waypoint_list_.empty()){

            std::vector<Eigen::Vector3d> remaining_wps(
                waypoint_list_.begin(), waypoint_list_.end());

            // Interpolate to fill gaps between sparse DEP waypoints

              std::vector<Eigen::Vector3d> interp_wps = interpolateWaypoints(remaining_wps, interstep_dist_);
              VisuaWaypoints(interp_wps,waypoint_pub_);

              if (interp_wps.size() >= 2) {
                  // bool ok = planToTarget(waypoint_list_.back());
                  bool ok = planToTarget(interp_wps);
                  if (ok) {
                      changeFSMExecState(EXEC_TRAJ, "WAIT_TRAJ -> EXEC_TRAJ");
                  } else {
                      // Planning failed, wait for next DEP plan
                      trigger_ = false;
                      have_traj_ = false;
                      dep_has_new_path_ = false; 
                      ROS_WARN("[ExploreFSM] planToTarget with interpolated waypoints failed, waiting for next DEP plan");
                  }
              } else {
                  // Not enough waypoints after interpolation, wait for next DEP plan
                  trigger_ = false;
                  have_traj_ = false;
                  dep_has_new_path_ = false;  // FIX: allow DEP to replan instead of deadlocking
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
              // If this was a cluster/object target, clear the mode and resume exploration
              if (executing_cluster_target_) {
                  executing_cluster_target_ = false;
                  has_cluster_target_ = false;
                  cluster_path_ready_ = false;
                  task_complete_ = true;
                  ROS_INFO("[ExploreFSM] Object target reached, resuming exploration");
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
              /* 脱困策略：尝试旋转360度进行重规划 -> 舍弃当前目标点重新选点 -> 切换高度层选点*/
              if (planner_manager_->needRePlan()) {
                ros::Time now = ros::Time::now();
                double dt = (now - last_replan_time_).toSec();
                ROS_INFO("Current dt : %f",dt);
                if(dt < 0.06){
                    consecutive_replan_cnt_++;}
                else if(just_rotating_){
                    // rotation just finished, force-increment counter so it can progress
                    // from 5→6→7→8→9, completing 4×90° rotations
                    consecutive_replan_cnt_++;
                }
                else if(dt > 0.8){
                    consecutive_replan_cnt_ = 0;
                }
                last_replan_time_ = now;
                ROS_WARN("[ExploreFSM] Current replans count (%d)", consecutive_replan_cnt_);
                //if(true){
                if (consecutive_replan_cnt_ >= 30){
                    ROS_WARN("[ExploreFSM] Too many replans (%d), Use my A star!", consecutive_replan_cnt_);
                    consecutive_replan_cnt_ = 0;
                    bool ok = planToTarget(waypoint_list_.back());
                    if(ok){
                        ROS_INFO("MY A star succeed");
                    }else{
                        dep_has_new_path_ = false;
                        changeFSMExecState(WAIT_TRAJ, "Too many replans, resetting");
                    }
                }

                if (consecutive_replan_cnt_ >= 20) {
                    ROS_WARN("[ExploreFSM] Too many replans (%d), switching height layer!", consecutive_replan_cnt_);
                    expPlanner_->switchHeightLayer();
                    // consecutive_replan_cnt_ = 0;
                    dep_has_new_path_ = false;
                    changeFSMExecState(WAIT_TRAJ, "Too many replans, resetting");
                }
                else if (consecutive_replan_cnt_ == 10) {
                    ROS_WARN("[ExploreFSM] Too many replans (%d), dropping current goal node!", consecutive_replan_cnt_);
                    expPlanner_->dropCurrentGoalNode();
                    // consecutive_replan_cnt_ = 0;
                    have_traj_ = false;
                    dep_has_new_path_ = false;
                    changeFSMExecState(INIT, "Drop goal, re-explore");
                }
                else if (consecutive_replan_cnt_ >= 5 && consecutive_replan_cnt_ <= 8) {
                    ROS_WARN("[ExploreFSM] Replan count %d, rotating to explore...", consecutive_replan_cnt_);
                    {
                        minco_curve::RotateDroneGoal goal;
                        goal.angle_rad = M_PI/2;
                        goal.speed_rad_sec = 1.0;
                        goal.pos_x = odom_pos_.x();
                        goal.pos_y = odom_pos_.y();
                        goal.pos_z = odom_pos_.z();
                        goal.angle_start = current_yaw_;
                        rotate_action_client_->sendGoal(goal);
                        rotate_goal_sent_ = true;
                        start_rotation_time_ = ros::Time::now();
                        have_traj_ = false;
                        dep_has_new_path_ = false;
                        changeFSMExecState(ROTATING, "Replan count 5, rotating");
                    }
                }else if(consecutive_replan_cnt_ == 9){
                    just_rotating_ = false;
                }
                 else {
                  have_traj_ = false;
                  dep_has_new_path_ = false;
                  changeFSMExecState(WAIT_TRAJ, "Need replan due to collision");
                }
            }
          }
        break;
      }

      case ROTATING:{
        // Wait for traj_server rotation to complete (M_PI/12 / 0.5 ≈ 0.52s, add margin)
        constexpr double kRotationTimeout = 1.68f;
        double elapsed = (ros::Time::now() - start_rotation_time_).toSec();
        if (elapsed >= kRotationTimeout) {
          just_rotating_ = true;
          dep_has_new_path_ = false;
          last_dep_plan_time_ = ros::Time(0);  // trigger immediate DEP replan
          // Bump consecutive_replan_cnt_ past the rotate zone (5-8) so higher thresholds can activate
          // if (consecutive_replan_cnt_ >= 5 && consecutive_replan_cnt_ <= 8) {
          //   consecutive_replan_cnt_ = 9;
          // }
          last_replan_time_ = ros::Time::now();  // reset timer so counter doesn't immediately reset after rotation
          ROS_INFO("[ExploreFSM] Rotation complete after %.2fs, back to WAIT_TRAJ", elapsed);
          changeFSMExecState(WAIT_TRAJ, "ROTATING -> WAIT_TRAJ");
        }
        // While rotating, traj_server cmdCallback publishes rotation commands,
        // so PX4 offboard keeps alive — no need to publish anything here.
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
    // ROS_WARN("STARTPOINT x = %f,y = %f,z = %f", start_pt[0], start_pt[1], start_pt[2]);
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
        // ROS_INFO("[ExploreFSM] planToTarget(vector): smooth trajectory generated with %zu waypoints", wps.size());
    } else {
        // ROS_WARN("[ExploreFSM] planToTarget(vector): planGlobalTraj failed");
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
    end_vel.setZero();
    end_acc.setZero();
    success = planner_manager_->planGlobalTraj(start_pt, start_vel, start_acc,
                                                target_pt, end_vel, end_acc);
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

void FakeExploreFSM::setWaypointsFromObjectCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& object_cloud)
{
  if (!object_cloud || object_cloud->points.empty()) return;

  // Find the nearest point in the object point cloud to the drone
  double min_dist = std::numeric_limits<double>::max();
  Eigen::Vector3d nearest_pt(0, 0, odom_pos_.z());  // keep current z
  for (const auto& pt : object_cloud->points) {
    double d = std::hypot(pt.x - odom_pos_.x(), pt.y - odom_pos_.y());
    if (d < min_dist) {
      min_dist = d;
      nearest_pt = Eigen::Vector3d(pt.x, pt.y, odom_pos_.z());
    }
  }

  ROS_INFO("[ExploreFSM] Object target: nearest point (%.2f, %.2f) at dist=%.2fm",
           nearest_pt.x(), nearest_pt.y(), min_dist);

  // ---- Visualize object cloud candidate points (green SPHERE_LIST) ----
  {
    visualization_msgs::Marker cloud_marker;
    cloud_marker.header.frame_id = "map";
    cloud_marker.header.stamp = ros::Time::now();
    cloud_marker.ns = "object_cloud_viz";
    cloud_marker.id = 0;
    cloud_marker.type = visualization_msgs::Marker::SPHERE_LIST;
    cloud_marker.action = visualization_msgs::Marker::ADD;
    cloud_marker.scale.x = cloud_marker.scale.y = cloud_marker.scale.z = 0.1;
    cloud_marker.color.r = 0.0;
    cloud_marker.color.g = 1.0;
    cloud_marker.color.b = 0.0;
    cloud_marker.color.a = 0.6;
    cloud_marker.pose.orientation.w = 1.0;
    cloud_marker.lifetime = ros::Duration(0.5);

    for (const auto& pt : object_cloud->points) {
      geometry_msgs::Point p;
      p.x = pt.x;
      p.y = pt.y;
      p.z = odom_pos_.z();
      cloud_marker.points.push_back(p);
    }
    object_cloud_viz_pub_.publish(cloud_marker);
  }

  // ---- Visualize selected target point (red SPHERE) ----
  {
    visualization_msgs::Marker target_marker;
    target_marker.header.frame_id = "map";
    target_marker.header.stamp = ros::Time::now();
    target_marker.ns = "cluster_target";
    target_marker.id = 0;
    target_marker.type = visualization_msgs::Marker::SPHERE;
    target_marker.action = visualization_msgs::Marker::ADD;
    target_marker.pose.position.x = nearest_pt.x();
    target_marker.pose.position.y = nearest_pt.y();
    target_marker.pose.position.z = odom_pos_.z();
    target_marker.pose.orientation.w = 1.0;
    target_marker.scale.x = target_marker.scale.y = target_marker.scale.z = 0.3;
    target_marker.color.r = 1.0;
    target_marker.color.g = 0.0;
    target_marker.color.b = 0.0;
    target_marker.color.a = 1.0;
    target_marker.lifetime = ros::Duration(0.5);
    cluster_target_marker_pub_.publish(target_marker);
  }

  waypoint_list_.clear();
  waypoint_list_.push_back(nearest_pt);
  current_wp_idx_ = 0;
  dep_has_new_path_ = true;   // block DEP from overwriting
  trigger_ = true;
  have_traj_ = false;
  touch_goal_ = false;
  initial_dist_to_goal_ = (odom_pos_ - nearest_pt).norm();
  early_replan_requested_ = false;

}

// void FakeExploreFSM::clusterTargetCallback(const geometry_msgs::PoseStampedConstPtr &msg)
// {
//   // Ignore new cluster targets if already executing one
//   if (executing_cluster_target_) {
//     ROS_WARN("[ExploreFSM] Already executing a cluster target, ignoring new one");
//     return;
//   }

//   cluster_target_pt_ = Eigen::Vector3d(msg->pose.position.x,
//                                         msg->pose.position.y,
//                                         msg->pose.position.z);
//   ROS_INFO("[ExploreFSM] Cluster target received at (%.2f, %.2f, %.2f), switching to GOTOCLUSTER",
//            cluster_target_pt_.x(), cluster_target_pt_.y(), cluster_target_pt_.z());

//   // Publish cluster target as a red sphere Marker in RViz
//   {
//     visualization_msgs::Marker marker;
//     marker.header.frame_id = "map";
//     marker.header.stamp = ros::Time::now();
//     marker.ns = "cluster_target";
//     marker.id = 0;
//     marker.type = visualization_msgs::Marker::SPHERE;
//     marker.action = visualization_msgs::Marker::ADD;
//     marker.pose.position.x = cluster_target_pt_.x();
//     marker.pose.position.y = cluster_target_pt_.y();
//     marker.pose.position.z = cluster_target_pt_.z();
//     marker.pose.orientation.w = 1.0;
//     marker.scale.x = 0.3;
//     marker.scale.y = 0.3;
//     marker.scale.z = 0.3;
//     marker.color.a = 1.0;
//     marker.color.r = 1.0;
//     marker.color.g = 0.0;
//     marker.color.b = 0.0;
//     marker.lifetime = ros::Duration(0);  // persistent until new target received
//     cluster_target_marker_pub_.publish(marker);
//   }

//   // Set single waypoint and enter cluster mode (DEP exploration fully disabled)
//   waypoint_list_.clear();
//   waypoint_list_.push_back(cluster_target_pt_);
//   current_wp_idx_ = 0;
//   has_cluster_target_ = true;
//   executing_cluster_target_ = true;
//   cluster_path_ready_ = false;
//   have_traj_ = false;
//   dep_has_new_path_ = true;   // block DEP from overwriting waypoint_list_
//   // trigger_ = false;            // don't use DEP exploration trigger

//   changeFSMExecState(GOTOCLUSTER, "Cluster target received");
// }

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
  const char *state_names[] = {"INIT", "INIT_ROTATE", "WAIT_TRAJ", "EXEC_TRAJ", "ROTATING", "GOTOCLUSTER", "EMERGENCY_STOP"};
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