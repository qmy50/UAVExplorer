# ifndef _FAKE_EXPLORER_FSM_
# define _FAKE_EXPLORER_FSM_


#include <utils/poly_traj_utils.hpp>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <quadrotor_msgs/GoalSet.h>
#include <traj_utils/DataDisp.h>
#include <plan_manager.h>
#include <global_planner/dep.h>
#include <traj_utils/PolyTraj.h>
#include <plan_env/grid_map_new.h>
#include "minco_curve/RotateDrone.h"

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
      WAIT_TRAJ,   
      EXEC_TRAJ,    
      RISING,
      GOTOCLUSTER,
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

  int consecutive_replan_cnt_;
  bool is_rising_;
  bool rising_traj_generated_; 
  Eigen::Vector3d rise_target_;

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
  
  ros::NodeHandle node_;
  ros::Timer exec_timer_, dep_timer_;
  ros::Subscriber odom_sub_, waypoint_sub_, trigger_sub_, mandatory_stop_sub_,plan_sub_;
  ros::Subscriber clicked_point_sub_;
  ros::Publisher poly_traj_pub_,waypoint_pub_,cluster_target_marker_pub_;

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

  // early replan based on path progress
  double path_progress_thresh_;   // progress threshold (0.0~1.0) to trigger early replan, default 0.5
  double initial_dist_to_goal_;   // initial distance to goal when path was set
  bool early_replan_requested_;   // flag to avoid repeated early replan requests

  ros::ServiceClient client_;
  minco_curve::RotateDrone srv_;

  // rotation state: prevent replanning during drone rotation
  bool is_rotating_;
  ros::Time rotation_end_time_;

  // cluster goto state: PRM-based collision-free path to clicked cluster target
  bool gotoClusterPlan();
  bool task_complete_;

};

}// namespace fake_planner

# endif