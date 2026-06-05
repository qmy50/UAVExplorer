/*
*	File: dep.h
*	---------------
*   dynamic exploration planner header file
*/

#ifndef DEP_H
#define DEP_H

// #include <map_manager/dynamicMap.h>
#include <plan_env/grid_map_new.h>
#include <plan_env/value_map2d.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <global_planner/PRMKDTree.h>
#include <global_planner/PRMAstar.h>
#include <global_planner/utils.h>


namespace globalPlanner{
	class DEP{
	private:
		std::string ns_;
		std::string hint_;

		ros::NodeHandle nh_;
		ros::Publisher roadmapPub_;
		ros::Publisher candidatePathPub_;
		ros::Publisher bestPathPub_;
		ros::Publisher frontierVisPub_;
		ros::Publisher valueMapPub_;      ///< Publisher for value map heatmap (OccupancyGrid)
		ros::Subscriber odomSub_;
		ros::Timer visTimer_;
		ros::Timer valuemapTimer_;

		nav_msgs::Odometry odom_;
		GridMap::Ptr map_; 
		std::shared_ptr<ValueMap2D> value_map_;
		std::shared_ptr<PRM::KDTree> roadmap_;

		// semantic value map parameters
		double semanticWeight_ = 1.0;  ///< Weight for semantic value in path scoring
		bool useValueMap_ = true;      ///< Whether to use semantic value map




		// parameters
		double vel_ = 1.0;
		double angularVel_ = 1.0;
		std::string odomTopic_;
		Eigen::Vector3d globalRegionMin_, globalRegionMax_;
		Eigen::Vector3d localRegionMin_, localRegionMax_;
		int localSampleThresh_;
		int globalSampleThresh_;
		int frontierSampleThresh_;
		double distThresh_;
		double safeDistXY_;
		double safeDistZ_;
		bool safeDistCheckUnknown_;
		double horizontalFOV_;
		double verticalFOV_;
		double dmin_;
		double dmax_;
		int nnNum_;
		int nnNumFrontier_;
		double maxConnectDist_;
		std::vector<double> yaws_;
		double minVoxelThresh_;
		int minCandidateNum_;
		int maxCandidateNum_;
		double updateDist_;
		double yawPenaltyWeight_;

		// data
		bool odomReceived_ = false;
		Eigen::Vector3d position_;
		double currYaw_;
		std::deque<Eigen::Vector3d> histTraj_; // historic trajectory for information gain update 
		// std::vector<std::shared_ptr<PRM::Node>> prmNodeVec_; // all nodes		
		std::unordered_set<std::shared_ptr<PRM::Node>> prmNodeVec_; // all nodes
		std::vector<std::shared_ptr<PRM::Node>> goalCandidates_;
		std::vector<std::vector<std::shared_ptr<PRM::Node>>> candidatePaths_;
		std::vector<std::shared_ptr<PRM::Node>> bestPath_;
		std::vector<std::pair<Eigen::Vector3d, double>> frontierPointPairs_;  //前沿区域（Frontier Region）的中心位置及其区域大小


	public:
		DEP(const ros::NodeHandle& nh);

				// 新增2D地图相关
		bool use2DMap_ = true;  // 是否使用2D地图做碰撞检测

		// // 2D碰撞检查辅助方法
		// bool is2DOccupied(double x, double y);        // 查询2D栅格是否占据
		// bool is2DFree(double x, double y);            // 查询2D栅格是否自由  
		// bool is2DUnknown(double x, double y);         // 查询2D栅格是否未知
		// bool is2DInflatedOccupiedLine2D(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2); // 2D线段碰撞检测


		void setMap(const GridMap::Ptr& map);
		void setValueMap(const std::shared_ptr<ValueMap2D>& value_map);
		void loadVelocity(double vel, double angularVel);
		void initParam();
		void initModules();
		void registerPub();
		void registerCallback();

		bool makePlan();
		nav_msgs::Path getBestPath();
		Eigen::Vector3d getNearestPRMNode(const Eigen::Vector3d& target_point);
		void detectFrontierRegion(std::vector<std::pair<Eigen::Vector3d, double>>& frontierPointPairs);
		void buildRoadMap();
		void pruneNodes();
		void updateInformationGain();
		void getBestViewCandidates(std::vector<std::shared_ptr<PRM::Node>>& goalCandidates);
		bool findCandidatePath(const std::vector<std::shared_ptr<PRM::Node>>& goalCandidates,  std::vector<std::vector<std::shared_ptr<PRM::Node>>>& candidatePaths);
		void findBestPath(const std::vector<std::vector<std::shared_ptr<PRM::Node>>>& candidatePaths, std::vector<std::shared_ptr<PRM::Node>>& bestPath);
		bool findPathToPoint(const Eigen::Vector3d& target, std::vector<Eigen::Vector3d>& path_points);
		void updateRoadmapOnly();
		

		// callback functions
		void odomCB(const nav_msgs::OdometryConstPtr& odom);
		void visCB(const ros::TimerEvent&);
		void valuemapCB(const ros::TimerEvent&);


		// help function
		bool isPosValid(const Eigen::Vector3d& p);
		bool isPosValid(const Eigen::Vector3d& p, double safeDistXY, double safeDistZ);
		std::shared_ptr<PRM::Node> randomConfigBBox(const Eigen::Vector3d& minRegion, const Eigen::Vector3d& maxRegion);
		bool sensorRangeCondition(const shared_ptr<PRM::Node>& n1, const shared_ptr<PRM::Node>& n2);
		bool sensorFOVCondition(const Eigen::Vector3d& sample, const Eigen::Vector3d& pos);
		int calculateUnknown(const shared_ptr<PRM::Node>& n, std::unordered_map<double, int>& yawNumVoxels);
		double calculatePathLength(const std::vector<shared_ptr<PRM::Node>>& path);
		void shortcutPath(const std::vector<std::shared_ptr<PRM::Node>>& path, std::vector<std::shared_ptr<PRM::Node>>& pathSc);
		int weightedSample(const std::vector<double>& weights);
		std::shared_ptr<PRM::Node> sampleFrontierPoint(const std::vector<double>& sampleWeights);
		std::shared_ptr<PRM::Node> extendNode(const std::shared_ptr<PRM::Node>& n, const std::shared_ptr<PRM::Node>& target);

		// visualization functions
		void publishRoadmap();
		void publishCandidatePaths();
		void publishBestPath();
		void publishFrontier();
		void publishValueMap();           ///< Publish value map as OccupancyGrid + OpenCV heatmap
	};
}


#endif


