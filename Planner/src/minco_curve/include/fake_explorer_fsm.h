# ifndef _FAKE_EXPLORER_FSM_
# define _FAKE_EXPLORER_FSM_


#include <utils/poly_traj_utils.hpp>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Bool.h>
#include <quadrotor_msgs/GoalSet.h>
#include <traj_utils/DataDisp.h>
#include <plan_manager.h>
#include <global_planner/dep.h>
#include <traj_utils/PolyTraj.h>
// #include <plan_env/grid_map_new.h>
#include <plan_env/grid_map_indoor.h>
#include <plan_env/object_map_manager.h>
#include <actionlib/client/simple_action_client.h>
#include "minco_curve/RotateDroneAction.h"

using std::vector;

namespace fake_planner
{

class FakeExploreFSM
{
public:
  FakeExploreFSM() {}
  ~FakeExploreFSM() {}

  void init(ros::NodeHandle &nh);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    enum FSM_EXEC_STATE
    {
      INIT,
      INIT_ROTATE,    // 初始环顾旋转，收集 value map 语义信息
      WAIT_TRAJ,
      EXEC_TRAJ,
      ROTATING,
      GOTOCLUSTER,
      // RISING,
      EMERGENCY_STOP
    };

  FakePlanManager::Ptr planner_manager_;
  double predict_dt_;
  bool use_kalman_filter_;
 
  double replan_thresh_;          
  double planning_horizen_;      
  double emergency_stop_time_;   
  bool have_odom_, have_traj_,touch_goal_,trigger_;
  bool executing_cluster_target_;   // true: cluster target mode, disable DEP exploration replanning
  bool cluster_path_ready_;        // true: PRM path to cluster target found, proceed to WAIT_TRAJ
  int waypoint_num_;
  int target_type_;
  double waypoints_[50][3];
  double max_vel_,max_acc_;
  FSM_EXEC_STATE exec_state_;



  Eigen::Vector3d odom_pos_, odom_vel_, odom_acc_;  
  Eigen::Vector3d target_pt_, target_vel_,target_acc_;
  Eigen::Vector3d cluster_target_pt_;
  bool has_cluster_target_;

  // Odometry jump detection
  Eigen::Vector3d last_odom_pos_;
  bool has_last_odom_;
  double odom_jump_thresh_;
  bool odom_jumped_;           // flag: odom just jumped, wait before replanning
  ros::Time odom_jump_time_;   // when the jump happened
  double odom_jump_cooldown_;  // seconds to wait after a jump before replanning
  
  std::vector<Eigen::Vector3d> waypoint_list_;          
  int current_wp_idx_;   
  int consecutive_replan_cnt_;                           
  
  ros::NodeHandle node_;
  ros::Timer exec_timer_, dep_timer_;
  ros::Subscriber odom_sub_, waypoint_sub_, trigger_sub_, mandatory_stop_sub_,plan_sub_;
  ros::Subscriber clicked_point_sub_;
  ros::Publisher poly_traj_pub_,waypoint_pub_,cluster_target_marker_pub_,object_cloud_viz_pub_;

  void execFSMCallback(const ros::TimerEvent &e);
  void execDepCallback(const ros::TimerEvent &e);
  void clusterTargetCallback(const geometry_msgs::PoseStampedConstPtr &msg);
  void changeFSMExecState(FSM_EXEC_STATE new_state, const std::string &pos_call);
  void printFSMExecState();

  bool planToTarget(const std::vector<Eigen::Vector3d> &target_waypoints);
  bool planToTarget(const Eigen::Vector3d &target_pt);
  std::vector<Eigen::Vector3d> interpolateWaypoints(const std::vector<Eigen::Vector3d>& waypoints, double step);
  // bool planToGivenCallbackWps(const std::vector<Eigen::Vector3d>& wps);

  void checkCollisionCallback(const ros::TimerEvent &e);
  void emergencyStop();

  void odometryCallback(const nav_msgs::OdometryConstPtr &msg);
  //void triggerCallback(const geometry_msgs::PoseStampedPtr &msg);
  void triggerCallback(const nav_msgs::PathPtr &msg);
  void triggerCallback(const quadrotor_msgs::GoalSetConstPtr &msg);
//   void mandatoryStopCallback(const std_msgs::Empty &msg);

  void publishTraj(const traj_utils::PolyTraj &traj_msg);
  // dwa
  double current_yaw_,current_angular_z_;
  // swarm
  // void RecvBroadcastMINCOTrajCallback(const traj_utils::MINCOTrajConstPtr &msg);
  void VisuaWaypoints(const std::vector<Eigen::Vector3d> &traj, ros::Publisher marker_pub);
  // bool have_recv_pre_agent_;
  // ros::Publisher checkpoints_pub_ ,broadcast_ploytraj_pub_;
  // ros::Subscriber broadcast_ploytraj_sub_;
  // double des_clearence_;

  // exploration
  std::shared_ptr<globalPlanner::DEP>expPlanner_;
  ros::Time last_dep_plan_time_;
  double dep_plan_interval_;
  bool dep_has_new_path_;
  double interstep_dist_;  // interpolation distance between waypoints

  // object mapping — multi-frame semantic object fusion
  std::shared_ptr<ObjectMapManager> object_manager_;
  void setWaypointsFromObjectCloud(
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& object_cloud);

  // early replan based on path progress
  double path_progress_thresh_;   // progress threshold (0.0~1.0) to trigger early replan, default 0.5
  double initial_dist_to_goal_;   // initial distance to goal when path was set
  bool early_replan_requested_;   // flag to avoid repeated early replan requests

  // Rotation action client (replaces service — traj_server notifies when done)
  typedef actionlib::SimpleActionClient<minco_curve::RotateDroneAction> RotateActionClient;
  boost::shared_ptr<RotateActionClient> rotate_action_client_;
  bool rotate_goal_sent_;  // true: goal sent, waiting for result

  // rotation state: prevent replanning during drone rotation
  bool just_rotating_;
  ros::Time rotation_end_time_;

  // goal-reached rotation scan: rotate ±30° to enrich PRM before next DEP plan
  int rotation_scan_step_;       // 0=rotate-left, 1=rotate-right, 2=done
  bool rotation_scan_requested_; // whether rotation service has been called for current step
  ros::Time rotation_scan_start_time_;

  // cluster goto state: PRM-based collision-free path to clicked cluster target
  bool gotoClusterPlan();
  bool task_complete_;
  bool change_layer_test_,drop_goal_test_;
  void changeLayerCallback(const std_msgs::Bool& msg);
  void dropGoalCallback(const std_msgs::Bool& msg);
  ros::Subscriber change_layer_sub_,drop_goal_sub_;
  ros::Time start_rotation_time_;

  // map readiness: wait for 2D grid map + ValueMap before DEP exploration
  bool maps_ready_;
  ros::Time maps_ready_since_;   // timestamp when both maps first became structurally ready

  // replan tracking: tracks time of last replan for consecutive_replan_cnt_ logic
  ros::Time last_replan_time_;

  // backtracking: when DEP fails, reverse previous waypoint_list_ to escape
  bool is_backtracking_;
  int dep_fail_cnt_;

  int init_rotate_step_;            // 当前步 (0 ~ total_steps-1)
  int init_rotate_total_steps_;     // 总旋转步数 (默认 12×30°=360°)
  double init_rotate_angle_per_step_; // 每步旋转角度 (rad, 默认 M_PI/6)
  ros::Time init_rotate_start_time_;  // 当前步开始时间                                              

};

}// namespace fake_planner

# endif