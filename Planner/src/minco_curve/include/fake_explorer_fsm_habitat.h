# ifndef _FAKE_EXPLORER_FSM_HABITAT_
# define _FAKE_EXPLORER_FSM_HABITAT_


#include <utils/poly_traj_utils.hpp>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Float64.h>
#include <traj_utils/DataDisp.h>
#include <plan_manager.h>
#include <global_planner/dep.h>
#include <traj_utils/PolyTraj.h>
#include <plan_env/grid_map_indoor.h>
#include <plan_env/object_map_manager.h>
#include <plan_env/value_map2d.h>
#include <plan_env/value_map2d.h>

using std::vector;

namespace fake_planner
{

// ━━━ 离散动作枚举（对齐 Habitat /habitat/plan_action 动作号）━━━
enum FSM_ACTION {
  ACT_STOP         = 0,
  ACT_MOVE_FORWARD = 1,
  ACT_TURN_LEFT    = 2,
  ACT_TURN_RIGHT   = 3,
  // ACT_TURN_DOWN    = 4,
  // ACT_TURN_UP      = 5
};

// Habitat 状态常量
enum HABITAT_STATE {
  READY = 0, ACTION_EXEC = 1, ACTION_FINISH = 2, EPISODE_FINISH = 3
};

// FSM 最终结果枚举（对齐 Python params.py FINAL_RESULT）
enum FINAL_RESULT {
  EXPLORE       = 0,
  SEARCH_OBJECT = 1,
  STUCKING      = 2,
  NO_FRONTIER   = 3,
  REACH_OBJECT  = 4
};

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
    INIT_ROTATE,
    WAIT_TRAJ,
    EXEC_TRAJ,
    PLAN_ACTION,
    GOTOCLUSTER,
    ESCAPE_STUCK,
    PUB_ACTION,
    WAIT_ACTION_FINISH,
    EMERGENCY_STOP
  };

  // ━━━ FSM 运行时数据（参照 ApexNav FSMData）━━━
  struct FSMData {
    FSMData()
    {
      trigger_ = false; have_odom_ = false; have_finished_ = false;
      init_action_count_ = 0; newest_action_ = -1; final_result_ = -1;
      escape_stucking_flag_ = false; escape_stucking_count_ = 0;
      stucking_action_count_ = 0; action_done_ = true;
    }
    bool trigger_, have_odom_, have_finished_;
    int  init_action_count_, newest_action_, final_result_;
    bool escape_stucking_flag_;
    int  escape_stucking_count_;
    Eigen::Vector2d escape_stucking_pos_;
    double escape_stucking_yaw_ = 0.0;
    std::vector<Eigen::Vector3d> stucking_points_;
    int  stucking_action_count_ = 0;
    bool action_done_;
    Eigen::Vector2d action_start_odom_pos_;
    double action_start_yaw_ = 0.0;
    int consecutive_stuck_count_ = 0;
    int past_stuck_rotate_remaining_ = 0;  // 老卡点旋转剩余步数 (0=不旋转)
  };

  struct FSMParam {
    FSMParam() { vis_scale_ = 0.1; }
    double vis_scale_;
  };

  // ── 核心模块 ──
  FakePlanManager::Ptr planner_manager_;
  std::shared_ptr<globalPlanner::DEP> expPlanner_;
  std::shared_ptr<ObjectMapManager> object_manager_;
  std::shared_ptr<FSMData> fd_;
  std::shared_ptr<FSMParam> fp_;

  // ── 状态 & 标志 ──
  FSM_EXEC_STATE exec_state_;
  double replan_thresh_, planning_horizen_, emergency_stop_time_;
  bool have_odom_, have_traj_, touch_goal_, trigger_;
  bool executing_cluster_target_, cluster_path_ready_, has_cluster_target_;
  bool task_complete_;

  // ── odom ──
  Eigen::Vector3d odom_pos_, odom_vel_, odom_acc_;
  Eigen::Vector3d target_pt_, target_vel_, target_acc_;
  Eigen::Vector3d cluster_target_pt_;
  double current_yaw_, current_angular_z_;

  // ── waypoints ──
  std::vector<Eigen::Vector3d> waypoint_list_;
  int current_wp_idx_, waypoint_num_;
  double waypoints_[50][3];

  // ── DEP exploration ──
  ros::Time last_dep_plan_time_;
  double dep_plan_interval_;
  bool dep_has_new_path_;
  double interstep_dist_;
  bool maps_ready_;
  ros::Time maps_ready_since_;
  bool is_backtracking_;
  int  dep_fail_cnt_;
  int  goto_cluster_retry_count_;
  double path_progress_thresh_, initial_dist_to_goal_;
  bool early_replan_requested_;
  bool early_replan_enabled_ = true;  // /toggle_early_replan 开关
  int  pending_yaw_replan_ = 0;        // 0=off, 1=wait first action, 2=trigger next cycle
  bool yaw_replan_done_ = false;       // prevent infinite yaw-replan loop
  double predict_dt_;
  bool use_kalman_filter_;

  // ── Habitat 离散模式 ──
  bool use_habitat_mode_ = false;
  bool habitat_arrived_ = false;
  bool use_object_nav_ = false;
  double turn_angle_ = 30.0;               // Habitat turn_angle (度), 决定 decideNextAction 阈值 = turn_angle/1.9
  int    init_rotate_half_steps_ = 12;      // 初始旋转半圈步数 = 360/turn_angle, 30°→12, 10°→36
  ros::Publisher  discrete_action_pub_;    // → /habitat/plan_action
  ros::Subscriber habitat_state_sub_;      // ← /habitat/state
  ros::Publisher  habitat_wp_pub_;         // → /habitat/waypoints (可视化)
  ros::Subscriber habitat_arrive_sub_;     // ← /habitat/arrived
  ros::Publisher  ros_expl_state_pub_;     // → /ros/expl_state (通知 Python FSM 最终状态)
  bool stop_sent_ = false;                 // 防止重复发送 STOP

  // ── ROS 接口 ──
  ros::NodeHandle node_;
  ros::Timer exec_timer_, dep_timer_;
  ros::Subscriber odom_sub_, change_layer_sub_, drop_goal_sub_, toggle_early_replan_sub_;
  ros::Publisher  poly_traj_pub_, waypoint_pub_;
  ros::Publisher  cluster_target_marker_pub_, object_cloud_viz_pub_;

  // ── 回调 ──
  void execFSMCallback(const ros::TimerEvent &e);
  void execDepCallback(const ros::TimerEvent &e);
  void odometryCallback(const nav_msgs::OdometryConstPtr &msg);
  void changeLayerCallback(const std_msgs::Bool& msg);
  void dropGoalCallback(const std_msgs::Bool& msg);
  void toggleEarlyReplanCallback(const std_msgs::Bool& msg);
  void habitatArriveCB(const std_msgs::Float64ConstPtr& msg);
  void habitatStateCallback(const std_msgs::Int32::ConstPtr& msg);

  // ── 规划 ──
  bool planToTarget(const Eigen::Vector3d &target_pt);
  bool gotoClusterPlan();
  void publishTraj(const traj_utils::PolyTraj &traj_msg);

  // ── 离散动作 ──
  void publishDiscreteAction(int action);
  int  decideNextAction(double current_yaw, double target_yaw);
  Eigen::Vector2d selectLocalTarget(const Eigen::Vector2d& current_pos,
                                    const std::vector<Eigen::Vector3d>& path,
                                    double local_distance);
  void wrapAngle(double& angle);

  // ── 辅助 ──
  void changeFSMExecState(FSM_EXEC_STATE new_state, const std::string &pos_call);
  void printFSMExecState();
  void VisuaWaypoints(const std::vector<Eigen::Vector3d> &traj, ros::Publisher marker_pub);
  void setWaypointsFromObjectCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& object_cloud);
  std::vector<Eigen::Vector3d> interpolateWaypoints(const std::vector<Eigen::Vector3d>& waypoints, double step);

  bool change_layer_test_, drop_goal_test_;
  int consecutive_replan_cnt_;
  double  stuck_moved_thresh_ = 0.0f;

  int yaw_cnt_ = 0;
  double init_yaw_ = 0.0f;
};

}// namespace fake_planner

# endif
