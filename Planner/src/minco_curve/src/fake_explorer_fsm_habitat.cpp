#include "fake_explorer_fsm_habitat.h"
#include <tf/tf.h>

namespace fake_planner
{

void FakeExploreFSM::init(ros::NodeHandle &nh)
{
    node_ = nh;
    node_.param("fsm/replan_thresh", replan_thresh_, 0.3);
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

    odom_sub_ = node_.subscribe("odom_world", 10, &FakeExploreFSM::odometryCallback, this);
    poly_traj_pub_ = node_.advertise<traj_utils::PolyTraj>("planning/trajectory", 10);
    waypoint_pub_ = node_.advertise<visualization_msgs::Marker>("minco_waypoints", 10);
    cluster_target_marker_pub_ = node_.advertise<visualization_msgs::Marker>("cluster_target_marker", 10);
    object_cloud_viz_pub_ = node_.advertise<visualization_msgs::Marker>("object_cloud_viz", 10);

    ROS_INFO("FSM initialized, waiting for odom and target.");
    nh.param("fsm/predict_dt", predict_dt_, 0.01);
    nh.param("fsm/use_kalman_filter", use_kalman_filter_, true);

    // explore
    expPlanner_.reset(new globalPlanner::DEP (node_));
    expPlanner_->setMap(planner_manager_->grid_map_);
    expPlanner_->loadVelocity(0.5, 0.5);
    last_dep_plan_time_ = ros::Time(0);
    node_.param("fsm/dep_plan_interval", dep_plan_interval_, 2.0);
    node_.param("fsm/interstep_dist", interstep_dist_, 0.1);
    dep_has_new_path_ = false;

    // object mapping
    object_manager_.reset(new ObjectMapManager(planner_manager_->grid_map_, nh));
    object_manager_->init();

    // early replan
    node_.param("fsm/path_progress_thresh", path_progress_thresh_, 0.7);
    initial_dist_to_goal_ = 0.0;
    early_replan_requested_ = false;

    // stuck detection
    node_.param("fsm/stuck_moved_thresh", stuck_moved_thresh_, 0.02);
    node_.param("fsm/turn_angle", turn_angle_, 10.0);  // 对齐 Habitat turn_angle, 默认 30°
    init_rotate_half_steps_ = static_cast<int>(360.0 / turn_angle_);  // 30°→12步, 10°→36步
    ROS_INFO("[ExploreFSM] turn_angle=%.0f°  init_rotate_half=%d  total_init=%d",
             turn_angle_, init_rotate_half_steps_, 2 + 2 * init_rotate_half_steps_);

    // backtracking
    is_backtracking_ = false;
    dep_fail_cnt_ = 0;
    goto_cluster_retry_count_ = 0;

    // 2D map + ValueMap readiness
    maps_ready_ = false;
    maps_ready_since_ = ros::Time(0);

    change_layer_sub_ = node_.subscribe("/change_layer", 10, &FakeExploreFSM::changeLayerCallback, this);
    drop_goal_sub_ = node_.subscribe("/drop_goal", 10, &FakeExploreFSM::dropGoalCallback, this);
    toggle_early_replan_sub_ = node_.subscribe("/toggle_early_replan", 10, &FakeExploreFSM::toggleEarlyReplanCallback, this);

    // ── Habitat 离散模式 ──
    node_.param("use_habitat_mode", use_habitat_mode_, false);
    node_.param("use_habitat_mode", use_object_nav_, false);
    ros::param::set("/use_habitat_mode", use_habitat_mode_);  // 全局可见，供 GridMap 读取
    if (use_habitat_mode_) {
      // 离散动作发布（直接对 Habitat）
      discrete_action_pub_ = node_.advertise<std_msgs::Int32>("/habitat/plan_action", 10);
      // 接收 Habitat 动作完成信号
      habitat_state_sub_ = node_.subscribe("/habitat/state", 10,
          &FakeExploreFSM::habitatStateCallback, this);
      // 保留 waypoints 通道（exploration 阶段直接发 waypoints 给 Habitat bridge）
      // habitat_wp_pub_ = node_.advertise<nav_msgs::Path>("/habitat/waypoints", 10);
      habitat_arrive_sub_ = node_.subscribe("/habitat/arrived", 10,
          &FakeExploreFSM::habitatArriveCB, this);
      // 发布 FSM 最终状态给 Python habitat_evaluation（对齐 /ros/expl_state）
      ros_expl_state_pub_ = node_.advertise<std_msgs::Int32>("/ros/expl_state", 10);
      ROS_WARN("[ExploreFSM] HABITAT MODE enabled — discrete actions via /habitat/plan_action");
    }

    change_layer_test_ = false;
    drop_goal_test_ = false;

    // FSM 数据 & 参数
    fd_.reset(new FSMData);
    fp_.reset(new FSMParam);

    // ros::Duration(0.5).sleep();

    exec_timer_ = node_.createTimer(ros::Duration(0.1), &FakeExploreFSM::execFSMCallback, this);
    dep_timer_ = node_.createTimer(ros::Duration(0.1), &FakeExploreFSM::execDepCallback, this);

    consecutive_replan_cnt_ = 0;
    // value_map_.reset(new ValueMap2D(map_));
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

void FakeExploreFSM::toggleEarlyReplanCallback(const std_msgs::Bool& msg){
  if(msg.data == true){
    early_replan_enabled_ = true;
    ROS_WARN("Early replan ENABLED");
  }else{
    early_replan_enabled_ = false;
    ROS_WARN("Early replan DISABLED");
  }
}

void FakeExploreFSM::habitatArriveCB(const std_msgs::Float64ConstPtr& msg){
    habitat_arrived_ = true;
    ROS_INFO("[ExploreFSM] Habitat arrived signal received");
}

// ━━━ Habitat 状态回调 ━━━
void FakeExploreFSM::habitatStateCallback(const std_msgs::Int32::ConstPtr& msg) {
  if (msg->data == HABITAT_STATE::ACTION_FINISH) {
    fd_->action_done_ = true;
    // 根据 FSM 上下文路由: 脱困/初始旋转/探索 各回各的状态
    if (fd_->escape_stucking_flag_)
      changeFSMExecState(ESCAPE_STUCK, "ACTION_FINISH -> ESCAPE_STUCK");
    else if (fd_->init_action_count_ < 2 + 2 * init_rotate_half_steps_)
      changeFSMExecState(INIT_ROTATE, "ACTION_FINISH -> INIT_ROTATE");
    else
      changeFSMExecState(PLAN_ACTION, "ACTION_FINISH -> PLAN_ACTION");
  }
  if (msg->data == HABITAT_STATE::EPISODE_FINISH) {
    ROS_WARN("[ExploreFSM] Habitat episode finished — resetting FSM + maps");
    // 重置关键状态以迎接新 episode
    task_complete_ = false;
    stop_sent_ = false;
    fd_->final_result_ = -1;
    executing_cluster_target_ = false;
    has_cluster_target_ = false;
    cluster_path_ready_ = false;
    goto_cluster_retry_count_ = 0;
    waypoint_list_.clear();
    current_wp_idx_ = 0;
    dep_has_new_path_ = false;
    trigger_ = true;   // 新 episode 自动重启探索
    fd_->action_done_ = true;     // 清除旧 episode 遗留的未完成动作
    fd_->stucking_action_count_ = 0;
    fd_->init_action_count_ = 0;  // 新场景重新初始旋转
    fd_->past_stuck_rotate_remaining_ = 0;
    exec_state_ = INIT; // 回到 INIT 重新开始
    maps_ready_= false;
    object_manager_->publishEmptyMarkers();  // 清除 ObjectMap 可视化 (Marker + /object/clouds)
    object_manager_->publishEmptyCloud();    // 确保 PointCloud2 也被清空

    // planner_manager_->grid_map_->resetBuffer();
    planner_manager_.reset(new FakePlanManager);
    planner_manager_->initPlanModules(node_);
    expPlanner_.reset(new globalPlanner::DEP(node_));
    expPlanner_->setMap(planner_manager_->grid_map_);   // 重新绑定 GridMap + 创建新 ValueMap
    expPlanner_->loadVelocity(0.5, 0.5);
    object_manager_.reset(new ObjectMapManager(planner_manager_->grid_map_, node_));
    object_manager_->init();

    auto makeDeleteAll = [](const std::string& ns) {
      visualization_msgs::Marker m;
      m.action = visualization_msgs::Marker::DELETEALL;
      m.ns = ns;
      return m;
    };
    waypoint_pub_.publish(makeDeleteAll("minisnap_waypoints"));
    waypoint_pub_.publish(makeDeleteAll("minco_opt"));
    cluster_target_marker_pub_.publish(makeDeleteAll("cluster_target"));
    //object_cloud_viz_pub_.publish(makeDeleteAll("object_cloud"));

    ROS_WARN("[ExploreFSM] Maps + viz cleared — ready for new episode");
    //ros::Duration(1.0).sleep();
  }
}

void FakeExploreFSM::execDepCallback(const ros::TimerEvent &e){
    if(task_complete_){
      if (!stop_sent_) {
        stop_sent_ = true;
        std_msgs::Int32 stop_msg;
        stop_msg.data = ACT_STOP;
        discrete_action_pub_.publish(stop_msg);
        std_msgs::Int32 expl_state_msg;
        expl_state_msg.data = fd_->final_result_;
        ros_expl_state_pub_.publish(expl_state_msg);
        ROS_WARN("[ExploreFSM] ✅ TASK COMPLETE → STOP sent, expl_state=%d", fd_->final_result_);
      }
      ROS_INFO_THROTTLE(1.0,"✅ TASK COMPLETE !!!");
      return;
    }
    if(!have_odom_) return;
    if (executing_cluster_target_) return;

    if (!maps_ready_) {
      bool grid_ready = planner_manager_->grid_map_->is2DMapReady();
      bool value_ready = expPlanner_->isValueMapReady();
      if (grid_ready && value_ready) {
        if (maps_ready_since_.isZero())
          maps_ready_since_ = ros::Time::now();
        double waited = (ros::Time::now() - maps_ready_since_).toSec();
        if (waited > 1.0) {
          maps_ready_ = true;
          ROS_INFO("[ExploreFSM] Maps ready (2D grid + ValueMap), starting DEP exploration");
        }
        return;
      } else {
        maps_ready_since_ = ros::Time(0);
        ROS_INFO_THROTTLE(2.0, "[ExploreFSM] Waiting for maps (grid=%s, value=%s)...",
                          grid_ready ? "OK" : "WAIT",
                          value_ready ? "OK" : "WAIT");
        return;
      }
    }

    // 初始旋转未完成 → 禁止 DEP 规划
    int total_init_steps_ = 2 + 2 * init_rotate_half_steps_;
    if (fd_->init_action_count_ < total_init_steps_ || exec_state_ == INIT_ROTATE) return;

    // Priority: semantic objects — check BEFORE dep_has_new_path_ guard
    // so ObjectMap2D can interrupt an ongoing DEP path at any time,
    // not just during the brief re-planning window between paths.
    if (!executing_cluster_target_ && object_manager_) {
      Eigen::Vector3d object_target;
      if (object_manager_->getBestObjectTarget(object_target, odom_pos_)) {
        // object_target.z() = odom_pos_.z();
         object_target.z() = 0.4;
        ROS_WARN("[ExploreFSM] ObjectMap2D: target (%.2f, %.2f, %.2f) → GOTOCLUSTER",
                 object_target.x(), object_target.y(), object_target.z());

        // Visualize target
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
          target_marker.color.r = 1.0; target_marker.color.g = 0.0;
          target_marker.color.b = 0.0; target_marker.color.a = 1.0;
          //target_marker.lifetime = ros::Duration(1.0);
          cluster_target_marker_pub_.publish(target_marker);
        }
        if(!use_object_nav_){
          fd_->final_result_ = FINAL_RESULT::REACH_OBJECT;
          task_complete_ = true;
          return;  // 直接完成，不需要走 GOTOCLUSTER 流程
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
        goto_cluster_retry_count_ = 0;
        changeFSMExecState(GOTOCLUSTER, "ObjectMap2D → GOTOCLUSTER");
        return;
      }
    }

    if(dep_has_new_path_) return;

    ros::Time now = ros::Time::now();
    bool replanSuccess = expPlanner_->makePlan();
    last_dep_plan_time_ = now;
    if (replanSuccess) {
        // Large initial yaw change → rotating will reveal new areas
        // → schedule replan after first action completes (only once per cycle)
        init_yaw_ = expPlanner_->getInitialYawChange();
        double yaw_thresh = expPlanner_->getYawReplanThreshold();
        if (init_yaw_ > yaw_thresh) {
            ROS_ERROR("[ExploreFSM] Large initial yaw %.1f° > thresh %.1f°, will replan after initial rotation",
                     init_yaw_ * 180.0 / M_PI, yaw_thresh * 180.0 / M_PI);
            pending_yaw_replan_ = 1;  // state 1: defer to next PLAN_ACTION cycle after first action
            yaw_cnt_ = 0;
        }
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
            VisuaWaypoints(waypoint_list_, waypoint_pub_);
            dep_has_new_path_ = true;
            trigger_ = true;
            have_traj_ = false;
            touch_goal_ = false;
            is_backtracking_ = false;
            dep_fail_cnt_ = 0;
            initial_dist_to_goal_ = (odom_pos_ - waypoint_list_.back()).norm();
            early_replan_requested_ = false;
            ROS_WARN("[ExploreFSM] New DEP path: %zu waypoints, last: (%.2f, %.2f, %.2f)",
                     waypoint_list_.size(),
                     waypoint_list_.back().x(), waypoint_list_.back().y(), waypoint_list_.back().z());
            // if (exec_state_ == EXEC_TRAJ)
            //     changeFSMExecState(WAIT_TRAJ, "New DEP path, replan");
        }
    } else if (!is_backtracking_ && !waypoint_list_.empty()) {
        dep_fail_cnt_++;
        ROS_WARN("[ExploreFSM] DEP makePlan FAILED (%d/10)", dep_fail_cnt_);
        bool at_end = (odom_pos_ - waypoint_list_.back()).norm() < replan_thresh_;
        if (dep_fail_cnt_ >= 10 && at_end) {
            ROS_WARN("[ExploreFSM] Backtracking along reversed path (%zu waypoints)",
                     waypoint_list_.size());
            std::reverse(waypoint_list_.begin(), waypoint_list_.end());
            current_wp_idx_ = 0;
            dep_has_new_path_ = true;
            trigger_ = true;
            have_traj_ = false;
            touch_goal_ = false;
            is_backtracking_ = true;
            dep_fail_cnt_ = 0;
        } else if (dep_fail_cnt_ >= 10 && !at_end) {
            dep_fail_cnt_ = 0;
        }
    }
}

void FakeExploreFSM::execFSMCallback(const ros::TimerEvent &e)
{
    if (exec_state_ == EMERGENCY_STOP) return;

    if(task_complete_){
      if (!stop_sent_) {
        stop_sent_ = true;
        std_msgs::Int32 stop_msg;
        stop_msg.data = ACT_STOP;
        discrete_action_pub_.publish(stop_msg);
        std_msgs::Int32 expl_state_msg;
        expl_state_msg.data = fd_->final_result_;
        ros_expl_state_pub_.publish(expl_state_msg);
        ROS_WARN("[ExploreFSM] ✅ TASK COMPLETE → STOP sent, expl_state=%d", fd_->final_result_);
      }
      ROS_INFO_THROTTLE(1.0,"✅ TASK COMPLETE !!!");
      return;
    }

    switch (exec_state_)
    {
      case INIT: {
        if (!have_odom_) goto force_return;
        if (!maps_ready_) goto force_return;

        // 首次启动 → 初始环绕旋转 (步数 = 2 + 2*half)
        int total_init = 2 + 2 * init_rotate_half_steps_;
        if (fd_->init_action_count_ < total_init) {
          ROS_INFO("[ExploreFSM] Starting initial look-around rotation (%d steps)", total_init);
          fd_->action_done_ = true;  // 允许发第一个动作
          changeFSMExecState(INIT_ROTATE, "INIT -> INIT_ROTATE");
        } else {
          // 后续周期 → 正常规划
          if (!trigger_ && !dep_has_new_path_) goto force_return;
          changeFSMExecState(WAIT_TRAJ, "INIT -> WAIT_TRAJ");
        }
        break;
      }

      case INIT_ROTATE: {
        int total_init = 2 + 2 * init_rotate_half_steps_;
        if (fd_->init_action_count_ >= total_init) {
          ROS_INFO("[ExploreFSM] Init rotation complete (%d steps), starting exploration", total_init);
          changeFSMExecState(WAIT_TRAJ, "INIT_ROTATE -> WAIT_TRAJ");
          break;
        }

        // 等上一个动作完成
        // if (!fd_->action_done_) break;
        fd_->action_done_ = false;

        // 序列: TURN_DOWN → TURN_LEFT×half → TURN_UP → TURN_LEFT×half
        if (fd_->init_action_count_ < 1)
          fd_->newest_action_ = ACT_TURN_LEFT;
        else if (fd_->init_action_count_ < 1 + init_rotate_half_steps_)
          fd_->newest_action_ = ACT_TURN_LEFT;
        else if (fd_->init_action_count_ < 1 + init_rotate_half_steps_ + 1)
          fd_->newest_action_ = ACT_TURN_LEFT;
        else
          fd_->newest_action_ = ACT_TURN_LEFT;

        ROS_WARN("Init Mode Process -----> (%d/%d)", fd_->init_action_count_, total_init);
        fd_->init_action_count_++;

        // 不直接发动作，转到 PUB_ACTION 统一发布
        changeFSMExecState(PUB_ACTION, "INIT_ROTATE -> PUB_ACTION");
        break;
      }

      case WAIT_TRAJ: {
        if (!have_odom_) goto force_return;
        if (waypoint_list_.empty()) goto force_return;

        if (use_habitat_mode_) {
          current_wp_idx_ = 0;
          initial_dist_to_goal_ = (odom_pos_ - waypoint_list_.back()).norm();  // 记录初始距离
          early_replan_requested_ = false;
          have_traj_ = true;
          dep_has_new_path_ = false;
          fd_->action_done_ = true;
          // ROS_INFO("[ExploreFSM] Habitat: %zu waypoints, init_dist=%.2fm",
          //          waypoint_list_.size(), initial_dist_to_goal_);
          changeFSMExecState(PLAN_ACTION, "WAIT_TRAJ -> PLAN_ACTION");
          break;
        }
        break;
      }

      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      // PLAN_ACTION: 从 waypoint_list_ 选局部目标 → 离散动作
      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      case PLAN_ACTION: {
        if (!have_odom_) goto force_return;
        if (waypoint_list_.empty()) {
          changeFSMExecState(INIT, "PLAN_ACTION: no waypoints");
          break;
        }

        // 等上一个动作完成
        if (!fd_->action_done_) break;
        fd_->action_done_ = false;

        if (fd_->past_stuck_rotate_remaining_ > 0) {
          dep_has_new_path_ = false;
          fd_->newest_action_ = ACT_TURN_LEFT;
          fd_->past_stuck_rotate_remaining_--;
          ROS_INFO("[ExploreFSM] Past-stuck rotate: %d steps remaining",
                   fd_->past_stuck_rotate_remaining_);
          changeFSMExecState(PUB_ACTION, "PLAN_ACTION: past-stuck rotate -> PUB_ACTION");
          break;
        }

        // if (executing_cluster_target_ && !waypoint_list_.empty()) {
        //     VisuaWaypoints(waypoint_list_, waypoint_pub_);
        // }


        if(pending_yaw_replan_ == 1 && (fd_->newest_action_ == ACT_TURN_LEFT || fd_ ->newest_action_ == ACT_TURN_RIGHT)){
              yaw_cnt_++;
        }

        // Large initial rotation → replan to capture newly revealed frontiers
        // State machine: 0=inactive, 1=defer first action, 2=keep checking until aligned
        if (yaw_cnt_ * turn_angle_ >= 0.8 * init_yaw_ * 180 / M_PI) {
                ROS_ERROR("[ExploreFSM] Yaw aligned to target, replanning to capture new frontiers");
                yaw_cnt_ = 0;
                pending_yaw_replan_ = 0;
                //yaw_replan_done_ = true;
                have_traj_ = false;
                dep_has_new_path_ = false;
                last_dep_plan_time_ = ros::Time(0);
                changeFSMExecState(INIT, "Yaw-triggered replan after rotation");
                break;
          
        } 

        Eigen::Vector2d cur2d(odom_pos_.x(), odom_pos_.y());
        Eigen::Vector2d goal2d(waypoint_list_.back().x(), waypoint_list_.back().y());

        // ━━━ 物理困住检测（连续 N 次 MOVE_FORWARD 移动不足才触发）━━━
        if (!fd_->escape_stucking_flag_ &&
            fd_->newest_action_ == ACT_MOVE_FORWARD) {
          double moved = (cur2d - fd_->action_start_odom_pos_).norm();
          if (moved < stuck_moved_thresh_) {
            fd_->consecutive_stuck_count_++;
            ROS_WARN("[ExploreFSM] Slow step %d/3 (moved=%.3fm < %.3fm)",
                     fd_->consecutive_stuck_count_, moved, stuck_moved_thresh_);
            if (fd_->consecutive_stuck_count_ >= 3) {
              bool past_stuck = false;
              for (auto& sp : fd_->stucking_points_) {
                Eigen::Vector2d sp2d(sp.x(), sp.y());
                if ((sp2d - cur2d).norm() < stuck_moved_thresh_ &&
                    std::fabs(sp.z() - current_yaw_) < M_PI / 6.0) {
                  past_stuck = true;
                  ROS_ERROR("[ExploreFSM] Still stuck at same place, skip");
                  break;
                }
              }
              if (!past_stuck) {
                ROS_ERROR("[ExploreFSM] STUCK! (%d slow steps, moved=%.3fm). Entering ESCAPE_STUCK...",
                         fd_->consecutive_stuck_count_, moved);
                fd_->consecutive_stuck_count_ = 0;
                fd_->escape_stucking_flag_ = true;
                fd_->escape_stucking_count_ = 0;
                fd_->escape_stucking_pos_ = cur2d;
                fd_->escape_stucking_yaw_ = current_yaw_;
                fd_->stucking_action_count_++;
                fd_->action_done_ = true;
                changeFSMExecState(ESCAPE_STUCK, "PLAN_ACTION: stuck -> ESCAPE_STUCK");
                break;
              } else {
                // 老卡点: 先旋转一圈再继续，避免在同一障碍前反复卡住
                expPlanner_->dropCurrentGoalNode();
                fd_->consecutive_stuck_count_ = 0;
                int full_rotation_steps = static_cast<int>(360.0 / turn_angle_);
                fd_->past_stuck_rotate_remaining_ = full_rotation_steps;
                ROS_WARN("[ExploreFSM] Past stuck at same place, rotating %d steps before continue",
                         full_rotation_steps);
              }
            }
          } else {
            fd_->consecutive_stuck_count_ = 0;  // 移动正常，重置连续计数器
          }
        } else if (fd_->newest_action_ != ACT_MOVE_FORWARD) {
          fd_->consecutive_stuck_count_ = 0;    // 非前进动作，重置
        }
        if (fd_->stucking_action_count_ >= 25) {
          ROS_ERROR("[ExploreFSM] Stuck too long (25+), stopping episode.");
          fd_->final_result_ = FINAL_RESULT::STUCKING;
          task_complete_ = true;
          break;
        }

        // 提前重规划：走过 path_progress_thresh_ 比例后不等走完就重新 DEP
        if (early_replan_enabled_ && !early_replan_requested_ && initial_dist_to_goal_ > 0.2) {
          double traveled = initial_dist_to_goal_ - (cur2d - goal2d).norm();
          if (traveled > initial_dist_to_goal_ * path_progress_thresh_) {
            // ROS_INFO("[ExploreFSM] PLAN_ACTION: %.0f%% path traveled (%.2f/%.2fm), early replan...",
            //          path_progress_thresh_ * 100, traveled, initial_dist_to_goal_);
            early_replan_requested_ = true;
            have_traj_ = false;
            dep_has_new_path_ = false;
            last_dep_plan_time_ = ros::Time(0);
            changeFSMExecState(INIT, "PLAN_ACTION: early replan -> INIT");
            break;
          }
        }
        // double itm_score = expPlanner_->GetITM(goal2d);
        // ROS_ERROR("The itm score is %f",itm_score);
        // 到达最终目标 → 重新 DEP 规划
        if ((cur2d - goal2d).norm() < replan_thresh_) {
          pending_yaw_replan_ = 0;       // clear yaw-triggered state machine
          yaw_replan_done_ = false;     // allow yaw replan in next exploration cycle
          // ROS_INFO("[ExploreFSM] PLAN_ACTION: reached waypoint end (dist=%.2f), replanning...",
          //          (cur2d - goal2d).norm());
          have_traj_ = false;
          dep_has_new_path_ = false;
          last_dep_plan_time_ = ros::Time(0);
          if (executing_cluster_target_) {
            executing_cluster_target_ = false;
            has_cluster_target_ = false;
            goto_cluster_retry_count_ = 0;
            fd_->final_result_ = FINAL_RESULT::REACH_OBJECT;
            task_complete_ = true;
            ROS_WARN("Task COmplete, reach object");
          }
          changeFSMExecState(INIT, "PLAN_ACTION -> INIT");
          break;
        }

        // 碰撞重规划检测：当前路径不安全 → 放弃，重新 DEP
        std::vector<Eigen::Vector3d> dense_waypoints = interpolateWaypoints(waypoint_list_,0.1);
        if (!executing_cluster_target_ && planner_manager_->needRePlanHabitat(dense_waypoints)) {
            ros::Time now = ros::Time::now();
            double dt = (now - last_dep_plan_time_).toSec();
            ROS_INFO("Current dt : %f",dt);
            if(dt < 0.2){
                consecutive_replan_cnt_++;}
            else{
                consecutive_replan_cnt_ = 0;
            }
            ROS_WARN("[ExploreFSM] PLAN_ACTION: path unsafe, replanning...");
            have_traj_ = false;
            dep_has_new_path_ = false;
            last_dep_plan_time_ = ros::Time(0);
            if(consecutive_replan_cnt_ >= 5){
              consecutive_replan_cnt_ = 0;
              expPlanner_->dropCurrentGoalNode();
            }
            changeFSMExecState(INIT, "PLAN_ACTION: unsafe -> INIT");
            break;
          }

        // 选择当前 waypoint 目标
        Eigen::Vector2d wp_target;
        if (!executing_cluster_target_) {
          // ObjectMap 目标导航 → 顺序跟随，不跳点
          wp_target = Eigen::Vector2d(waypoint_list_[current_wp_idx_].x(),
                                      waypoint_list_[current_wp_idx_].y());
          if ((cur2d - wp_target).norm() < replan_thresh_) {
            current_wp_idx_++;
            if (current_wp_idx_ >= (int)waypoint_list_.size()) {
              current_wp_idx_ = 0;
              have_traj_ = false;
              dep_has_new_path_ = false;
              last_dep_plan_time_ = ros::Time(0);
              executing_cluster_target_ = false;
              has_cluster_target_ = false;
              cluster_path_ready_ = false;
              goto_cluster_retry_count_ = 0;
              fd_->final_result_ = FINAL_RESULT::REACH_OBJECT;
              task_complete_ = true;
              ROS_INFO("[ExploreFSM] PLAN_ACTION: cluster target reached!");
              changeFSMExecState(INIT, "PLAN_ACTION -> INIT");
              break;
            }
            wp_target = Eigen::Vector2d(waypoint_list_[current_wp_idx_].x(),
                                        waypoint_list_[current_wp_idx_].y());
          }
        } else {
          // DEP 探索路径 → 沿路前瞻选取，自动跳过已越过的点
          wp_target = selectLocalTarget(cur2d, waypoint_list_, 0.5);
        }

        double target_yaw = std::atan2(wp_target.y() - cur2d.y(),
                                       wp_target.x() - cur2d.x());
        fd_->newest_action_ = decideNextAction(current_yaw_, target_yaw);

        // ROS_INFO("[ExploreFSM] PLAN_ACTION: wp[%d/%zu]=(%.2f,%.2f) yaw=%.0f° → action=%d",
        //          current_wp_idx_, waypoint_list_.size(),
        //          wp_target.x(), wp_target.y(),
        //          target_yaw * 180.0 / M_PI, fd_->newest_action_);

        changeFSMExecState(PUB_ACTION, "PLAN_ACTION -> PUB_ACTION");
        break;
      }

      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      // ESCAPE_STUCK: 物理困住逃逸状态机
      // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      case ESCAPE_STUCK: {
        if (!fd_->escape_stucking_flag_) {
          changeFSMExecState(WAIT_TRAJ, "ESCAPE_STUCK -> WAIT_TRAJ");
          break;
        }

        // 等上一个动作完成
        if (!fd_->action_done_) break;
        fd_->action_done_ = false;

        // 检测是否已脱困
        Eigen::Vector2d cur2d(odom_pos_.x(), odom_pos_.y());
        if (fd_->newest_action_ == ACT_MOVE_FORWARD &&
            (cur2d - fd_->action_start_odom_pos_).norm() >= 0.05) {
          ROS_INFO("[ExploreFSM] Escaped! (moved %.3fm)",
                   (cur2d - fd_->action_start_odom_pos_).norm());
          fd_->escape_stucking_flag_ = false;
          fd_->escape_stucking_count_ = 0;
          changeFSMExecState(WAIT_TRAJ, "Escape succeeded -> WAIT_TRAJ");
          break;
        }

        // ApexNav 10 步逃逸序列: R→F→R→F→L→L→L→F→L→F
        if (fd_->escape_stucking_count_ == 0)
          fd_->newest_action_ = ACT_TURN_RIGHT;
        else if (fd_->escape_stucking_count_ == 1)
          fd_->newest_action_ = ACT_MOVE_FORWARD;
        else if (fd_->escape_stucking_count_ == 2)
          fd_->newest_action_ = ACT_TURN_RIGHT;
        else if (fd_->escape_stucking_count_ == 3)
          fd_->newest_action_ = ACT_MOVE_FORWARD;
        else if (fd_->escape_stucking_count_ == 4)
          fd_->newest_action_ = ACT_TURN_LEFT;
        else if (fd_->escape_stucking_count_ == 5)
          fd_->newest_action_ = ACT_TURN_LEFT;
        else if (fd_->escape_stucking_count_ == 6)
          fd_->newest_action_ = ACT_TURN_LEFT;
        else if (fd_->escape_stucking_count_ == 7)
          fd_->newest_action_ = ACT_MOVE_FORWARD;
        else if (fd_->escape_stucking_count_ == 8)
          fd_->newest_action_ = ACT_TURN_LEFT;
        else if (fd_->escape_stucking_count_ == 9)
          fd_->newest_action_ = ACT_MOVE_FORWARD;
        else {
          // 10步全失败 → 标记占据
          ROS_ERROR("[ExploreFSM] Cannot escape after 10 attempts!");
          fd_->escape_stucking_flag_ = false;

          Eigen::Vector2d esp = fd_->escape_stucking_pos_;
          double eyaw = fd_->escape_stucking_yaw_;
          const double kFwd = 0.15;
          planner_manager_->grid_map_->setOccupancy(
              Eigen::Vector3d(esp.x(), esp.y(), odom_pos_.z()), 1);
          planner_manager_->grid_map_->setForceOcc2D(esp.x(), esp.y());
          for (int k = 1; k <= 2; k++) {
            double fx = esp.x() + k * kFwd * cos(eyaw);
            double fy = esp.y() + k * kFwd * sin(eyaw);
            planner_manager_->grid_map_->setOccupancy(
                Eigen::Vector3d(fx, fy, odom_pos_.z()), 1);
            planner_manager_->grid_map_->setForceOcc2D(fx, fy);
          }
          ROS_WARN("[ExploreFSM] Marked 3 positions occupied (3D+2D)");

          fd_->stucking_points_.push_back(
              Eigen::Vector3d(esp.x(), esp.y(), eyaw));
          fd_->stucking_action_count_ = 0;
          changeFSMExecState(INIT, "Escape failed -> INIT");
          break;
        }

        ROS_WARN("[ExploreFSM] Escape step %d/10: action=%d",
                 fd_->escape_stucking_count_, fd_->newest_action_);
        publishDiscreteAction(fd_->newest_action_);
        fd_->escape_stucking_count_++;
        break;
      }

      case GOTOCLUSTER: {
        if (!have_odom_) goto force_return;
        if (!cluster_path_ready_) {
          bool ok = gotoClusterPlan();
          if(ok){
            ROS_ERROR("AStar ok !!!!!!!!!");
          }else{
            ROS_ERROR("AStar failed !!!!!!!!!");
          }
          if (!ok) {
             ROS_ERROR("AStar failed !!!!!!!!!");
            goto_cluster_retry_count_++;
            if (goto_cluster_retry_count_ > 10) {
              ROS_ERROR("[ExploreFSM] GOTOCLUSTER: waypoint_list empty after %d retries, giving up",
                       goto_cluster_retry_count_);
              executing_cluster_target_ = false;
              has_cluster_target_ = false;
              cluster_path_ready_ = false;
              dep_has_new_path_ = false;
              goto_cluster_retry_count_ = 0;
              changeFSMExecState(INIT, "GOTOCLUSTER: failed -> INIT (resume exploration)");
              break;
            }
            ROS_WARN("[ExploreFSM] GOTOCLUSTER: waypoint_list empty, retry %d/10...",
                     goto_cluster_retry_count_);
            break;
          }

          early_replan_requested_ = false;
          fd_->action_done_ = true;
          cluster_path_ready_ = true;
          goto_cluster_retry_count_ = 0;
          ROS_INFO("The size of waypoint_list_ is %ld",waypoint_list_.size());
          VisuaWaypoints(waypoint_list_, waypoint_pub_);
         
          changeFSMExecState(PLAN_ACTION, "GOTOCLUSTER -> PLAN_ACTION");
        }
        break;
      }

      case PUB_ACTION: {
        // ROS_WARN("We are going to pub ation !");
        publishDiscreteAction(fd_->newest_action_);
        changeFSMExecState(WAIT_ACTION_FINISH, "PUB_ACTION -> WAIT_ACTION_FINISH");
        break;
      }

      case WAIT_ACTION_FINISH: {
        // 等待 habitatStateCallback 收 ACTION_FINISH → 切回 INIT_ROTATE
        break;
      }

      default:
        break;
    }
    force_return:;
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

void FakeExploreFSM::publishDiscreteAction(int action) {
  fd_->newest_action_ = action;
  fd_->action_done_ = false;
  fd_->action_start_odom_pos_ = Eigen::Vector2d(odom_pos_.x(), odom_pos_.y());
  fd_->action_start_yaw_ = current_yaw_;
  std_msgs::Int32 msg;
  msg.data = action;
  discrete_action_pub_.publish(msg);
}

// ━━━ 离散动作辅助函数━━━

void FakeExploreFSM::wrapAngle(double& angle) {
  while (angle < -M_PI) angle += 2 * M_PI;
  while (angle > M_PI)  angle -= 2 * M_PI;
}

int FakeExploreFSM::decideNextAction(double current_yaw, double target_yaw) {
  wrapAngle(target_yaw);
  wrapAngle(current_yaw);
  double yaw_diff = target_yaw - current_yaw;
  wrapAngle(yaw_diff);

  // 阈值 ≈ 0.9 × turn_angle, 防抖 ≈ 1.8 × turn_angle（保证一次转向收敛）
  double abs_diff = std::fabs(yaw_diff);
  double turn_rad = turn_angle_ * M_PI / 180.0;
  if (abs_diff > turn_rad * 0.8) {
    // 如果上次是前进且偏差还不大 (<1.8×turn_angle)，保持前进防止震荡
    if (fd_->newest_action_ == ACT_MOVE_FORWARD && abs_diff < 0.9 * turn_rad)
      return ACT_MOVE_FORWARD;
    return (yaw_diff > 0) ? ACT_TURN_LEFT : ACT_TURN_RIGHT;
  } else {
    return ACT_MOVE_FORWARD;
  }
}

// int FakeExploreFSM::decideNextAction(double current_yaw, double target_yaw) {
//   wrapAngle(target_yaw);
//   wrapAngle(current_yaw);
//   double yaw_diff = target_yaw - current_yaw;
//   wrapAngle(yaw_diff);

//   // 阈值 = turn_angle / 1.9, 对齐 ApexNav; 保证一次转向收敛 (turn_angle < 2*threshold)
//   double threshold = turn_angle_ * M_PI / 180.0 / 1.9;
//   if (std::fabs(yaw_diff) > threshold) {
//     return (yaw_diff > 0) ? ACT_TURN_LEFT : ACT_TURN_RIGHT;
//   }
//   return ACT_MOVE_FORWARD;
// }


Eigen::Vector2d FakeExploreFSM::selectLocalTarget(
    const Eigen::Vector2d& current_pos,
    const std::vector<Eigen::Vector3d>& path,
    double local_distance) {
  Eigen::Vector2d target = Eigen::Vector2d(path.back().x(), path.back().y());

  // 找离当前位置最近的路径点作为搜索起点
  int start_id = 0;
  double min_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < (int)path.size() - 1; i++) {
    Eigen::Vector2d pos(path[i].x(), path[i].y());
    double d = (pos - current_pos).norm();
    if (d < min_dist) { min_dist = d; start_id = i + 1; }
  }

  // 沿路径累积距离，找到 local_distance 处的点
  double len = (Eigen::Vector2d(path[start_id].x(), path[start_id].y()) - current_pos).norm();
  for (int i = start_id + 1; i < (int)path.size(); i++) {
    len += (Eigen::Vector2d(path[i].x(), path[i].y()) -
            Eigen::Vector2d(path[i - 1].x(), path[i - 1].y())).norm();
    if (len > local_distance &&
        (current_pos - Eigen::Vector2d(path[i - 1].x(), path[i - 1].y())).norm() > 0.30) {
      target = Eigen::Vector2d(path[i - 1].x(), path[i - 1].y());
      break;
    }
  }
  return target;
}

bool FakeExploreFSM::planToTarget(const Eigen::Vector3d &target_pt) {
    Eigen::Vector3d start_pt = odom_pos_;
    start_pt.z() = 0.4;
    Eigen::Vector3d end_vel, end_acc;
    Eigen::Vector3d start_vel = odom_vel_ * 0.5;
    Eigen::Vector3d start_acc = odom_acc_;

    end_vel.setZero();
    end_acc.setZero();
    bool success = planner_manager_->planGlobalTraj(start_pt, start_vel, start_acc,
                                                    target_pt, end_vel, end_acc);
                                                  
    if (success) {
        // traj_utils::PolyTraj poly_msg;
        // traj_utils::MINCOTraj MINCO_msg;
        // planner_manager_->polyTraj2ROSMsg(poly_msg, MINCO_msg);
        // publishTraj(poly_msg);
        std::vector<Eigen::Vector3d> traj_pts = planner_manager_->getCurrentWaypoints();
        ROS_INFO("[ExploreFSM] planToTarget: traj_pts size=%zu", traj_pts.size());
        waypoint_list_ = traj_pts;
        ROS_INFO("[ExploreFSM] The size of waypoint_list_ is %zu", waypoint_list_.size());
        have_traj_ = true;
    }
    return success;
}

bool FakeExploreFSM::gotoClusterPlan()
{
  if (waypoint_list_.empty()) {
    ROS_ERROR("[ExploreFSM] gotoClusterPlan: waypoint_list_ is empty");
    return false;
  }
  Eigen::Vector3d target = waypoint_list_.back();
  bool success = planToTarget(target);
  if (success) {
    ROS_INFO("[ExploreFSM] gotoClusterPlan: trajectory to (%.2f, %.2f, %.2f)",
             target.x(), target.y(), target.z());
    return true;
  } else {
    ROS_WARN("[ExploreFSM] gotoClusterPlan: planToTarget failed");
    return false;
  }
}

void FakeExploreFSM::setWaypointsFromObjectCloud(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& object_cloud)
{
  if (!object_cloud || object_cloud->points.empty()) return;

  double min_dist = std::numeric_limits<double>::max();
  Eigen::Vector3d nearest_pt(0, 0, odom_pos_.z());
  for (const auto& pt : object_cloud->points) {
    double d = std::hypot(pt.x - odom_pos_.x(), pt.y - odom_pos_.y());
    if (d < min_dist) {
      min_dist = d;
      nearest_pt = Eigen::Vector3d(pt.x, pt.y, odom_pos_.z());
    }
  }

  ROS_INFO("[ExploreFSM] Object target: (%.2f, %.2f) dist=%.2fm",
           nearest_pt.x(), nearest_pt.y(), min_dist);

  // Visualize target
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
    target_marker.color.r = 1.0; target_marker.color.g = 0.0;
    target_marker.color.b = 0.0; target_marker.color.a = 1.0;
    target_marker.lifetime = ros::Duration(0.5);
    cluster_target_marker_pub_.publish(target_marker);
  }

  waypoint_list_.clear();
  waypoint_list_.push_back(nearest_pt);
  current_wp_idx_ = 0;
  dep_has_new_path_ = true;
  trigger_ = true;
  have_traj_ = false;
  touch_goal_ = false;
  initial_dist_to_goal_ = (odom_pos_ - nearest_pt).norm();
  early_replan_requested_ = false;
}

void FakeExploreFSM::odometryCallback(const nav_msgs::OdometryConstPtr &msg)
{
  Eigen::Vector3d new_pos;
  have_odom_ = true;
  new_pos(0) = msg->pose.pose.position.x;
  new_pos(1) = msg->pose.pose.position.y;
  new_pos(2) = msg->pose.pose.position.z;

  odom_pos_ = new_pos;
  odom_vel_(0) = msg->twist.twist.linear.x;
  odom_vel_(1) = msg->twist.twist.linear.y;
  odom_vel_(2) = msg->twist.twist.linear.z;

  tf::Quaternion q(msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
                   msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
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
  const char *state_names[] = {
    "INIT", "INIT_ROTATE", "WAIT_TRAJ",
    "PLAN_ACTION", "GOTOCLUSTER", "ESCAPE_STUCK",
    "PUB_ACTION", "WAIT_ACTION_FINISH", "EMERGENCY_STOP"
  };
  // ROS_INFO("FSM State: %s", state_names[exec_state_]);
}

void FakeExploreFSM::changeFSMExecState(FSM_EXEC_STATE new_state, const std::string &pos_call)
{
  if (exec_state_ == new_state) return;
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
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.scale.x = marker.scale.y = marker.scale.z = 0.15;
    marker.color.r = 0.0; marker.color.g = 0.0; marker.color.b = 1.0; marker.color.a = 1.0;
    marker.pose.orientation.w = 1.0;

    marker.points.clear();
    for(int i=0 ;i < traj.size(); i++){
      geometry_msgs::Point p;
      p.x = traj[i].x(), p.y = traj[i].y(), p.z = traj[i].z();
      marker.points.push_back(p);
    }
    marker_pub.publish(marker);
}

} // namespace fake_planner
