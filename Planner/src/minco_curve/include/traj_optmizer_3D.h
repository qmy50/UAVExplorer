#ifndef TRAJ_OPTMIZER_3D_HPP
#define TRAJ_OPTMIZER_3D_HPP


#include <algorithm>
#include <memory>
#include <Eigen/Eigen>

#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <utils/poly_traj_utils.hpp>
#include <traj_utils/PolyTraj.h> 
#include <traj_utils/MINCOTraj.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <tf2/utils.h>

#include "utils/minco.hpp"
#include <path_searching/dyn_a_star_new.h>
#include <plan_env/grid_map_new.h>


class PathPlannerSim3D{
    public:
        template<typename T>
        using VecE = std::vector<T, Eigen::aligned_allocator<T>>;
        template <typename T>
        using vec_E = std::vector<T, Eigen::aligned_allocator<T>>;
        template <int N>
        using Vecf = Eigen::Matrix<double, N, 1>;
        typedef Vecf<3> Vec3f;
        typedef vec_E<Vec3f> vec_Vec3f;
        typedef Eigen::VectorXd VectorXd;

        int pieceNum_;
        int optDim_;
        int iter;

    private:
        ros::NodeHandle &public_nh;

        bool has_map_;                     // 点云是否已收到
        bool planned_;                     // 是否已执行过规划（只执行一次）
        ros::Subscriber map_sub_;
        ros::Publisher path_pub_, traj_pub_, grid_map_vis_pub_,wp_traj_vis_pub_;
        ros::Publisher poly_traj_pub_;
        ros::Publisher waypoint_pub_;
        ros::Publisher minisnap_pub_;

        VecE<Eigen::MatrixX4d> hpolys;
        minco::MINCO_S3NU<3> opt_;
        Eigen::VectorXd ts_;
        ros::Timer check_timer_;

        // 路径规划相关
        std::string astar_topic_;
        vec_Vec3f local_obstacles_;
        double last_planned_cost_ ;
        double max_vel_;
        double max_acc_; 
        Eigen::Vector3d start_point_,target_point_;
        std::shared_ptr<poly_traj::Trajectory> minco_traj_;
        GridMap::Ptr grid_map_;
        AStar::Ptr a_star_;
        std::vector<Eigen::Vector3d> current_a_star_waypoints_;
        std::vector<Eigen::Vector3d> current_minco_waypoints_;
        poly_traj::Trajectory current_traj_;

    public:
        std::vector<Eigen::Vector3d> current_minisnap_waypoints_;
        PathPlannerSim3D(ros::NodeHandle &nh,const int& id);
        void setParam(ros::NodeHandle &nh);
        void setEnvironment(const GridMap::Ptr &map);
        // void checkAndPlan(const ros::TimerEvent&);  
        
        void Planning(const std::vector<Eigen::Vector3d>& waypoints,const Eigen::Vector3d& start_vel,const Eigen::Vector3d& start_acc,
                                const Eigen::Vector3d& end_vel,const Eigen::Vector3d& end_acc);
        
        void Planning(const Eigen::Vector3d& start_position,const Eigen::Vector3d& start_vel,const Eigen::Vector3d& start_acc,
                                const Eigen::Vector3d& end_position, const Eigen::Vector3d& end_vel,const Eigen::Vector3d& end_acc);
        void DecompVel(const double theta, const double vel, double &vx, double &vy, double &vz);
        void VisuaTraj(const std::vector<Eigen::Vector3d> &path, ros::Publisher marker_pub);
        void VisuaWaypoints(const std::vector<Eigen::Vector3d> &traj,ros::Publisher path_pub);    
        void MapCallback(const sensor_msgs::PointCloud2::ConstPtr& pointcloud_map);
        void polyTraj2ROSMsg(traj_utils::PolyTraj &poly_msg);
        bool checkTrajCollision();
        VectorXd timeAllocation(const std::vector<Eigen::Vector3d>& Path,
                        double max_vel, double max_acc,
                        double start_vel, double end_vel);
        bool have_a_star_path_;
        bool have_minco_waypoints_,have_minisnap_waypoints_;
        bool have_current_traj_;
        bool use_real_model_;
        int drone_id_;

        inline std::vector<Eigen::Vector3d> getCurrentAstarPath(){
        return current_a_star_waypoints_;
        }

        inline std::vector<Eigen::Vector3d> getCurrentMincoWayPoints(){
            return current_minco_waypoints_;
        }

        inline poly_traj::Trajectory getCurrentTraj(){
            return current_traj_;
        }

        inline bool getCurrentMinisnapPoints(){

            return have_minisnap_waypoints_;
        }
    
    public:
        typedef std::unique_ptr<PathPlannerSim3D> Ptr;
        
};


#endif