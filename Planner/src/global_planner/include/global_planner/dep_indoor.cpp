/*
*	File: dep.cpp
*	---------------
*   dynamic exploration planner implementation
*/

#include <global_planner/dep.h>
#include <algorithm>
#include <random>
#include <visualization_msgs/MarkerArray.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


namespace globalPlanner{
	DEP::DEP(const ros::NodeHandle& nh) : nh_(nh){
		this->ns_ = "/DEP";
		this->hint_ = "[DEP]";
		this->initParam();
		ROS_WARN("We are going to init the module");
		this->initModules();
		ROS_WARN("We have init the module");
		this->registerPub();
		this->registerCallback();
	}

	void DEP::setMap(const GridMap::Ptr &map){
		map_ = map;
		// 在 map_ 有效后再创建 ValueMap2D，避免传入空指针
		value_map_.reset(new ValueMap2D(map_));
		value_map_->ClearBuffer();
		map_->setUse8NeighborFrontier(this->use_8neighbor_frontier_);
		ROS_INFO("[DEP]: ValueMap2D created with valid GridMap");
	}

	void DEP::setValueMap(const std::shared_ptr<ValueMap2D>& value_map){
		value_map_ = value_map;
		ROS_INFO("[DEP]: ValueMap2D set successfully");
	}

	void DEP::loadVelocity(double vel, double angularVel){
		this->vel_ = vel;
		this->angularVel_ = angularVel;
	}

	void DEP::initParam(){
		// odom topic name
		if (not this->nh_.getParam(this->ns_ + "/odom_topic", this->odomTopic_)){
			// this->odomTopic_ = "/drone_0_visual_slam/odom";
			// this->odomTopic_ = "/iris_0/mavros/odometry/in";
			cout << this->hint_ << ": No odom topic name. Use default: /CERLAB/quadcopter/odom" << endl;
		}
		else{
			cout << this->hint_ << ": Odom topic: " << this->odomTopic_ << endl;
		}

		// local sample region min
		std::vector<double> localRegionMinTemp;	
		if (not this->nh_.getParam(this->ns_ + "/local_region_min", localRegionMinTemp)){
			this->localRegionMin_(0) = -5.0;
			this->localRegionMin_(1) = -5.0;
			this->localRegionMin_(2) = -2.0;
			cout << this->hint_ << ": No local region min param. Use default: [-5 -5 -2]" <<endl;
		}
		else{
			this->localRegionMin_(0) = localRegionMinTemp[0];
			this->localRegionMin_(1) = localRegionMinTemp[1];
			this->localRegionMin_(2) = localRegionMinTemp[2];
			cout << this->hint_ << ": Local Region Min: " << this->localRegionMin_[0] <<" " <<this->localRegionMin_[1]<<" "<< this->localRegionMin_[2]<< endl;
		}

		// local sample region max
		std::vector<double> localRegionMaxTemp;	
		if (not this->nh_.getParam(this->ns_ + "/local_region_max", localRegionMaxTemp)){
			this->localRegionMax_(0) = 5.0;
			this->localRegionMax_(1) = 5.0;
			this->localRegionMax_(2) = 2.0;
			cout << this->hint_ << ": No local region max param. Use default: [5 5 2]" <<endl;
		}
		else{
			this->localRegionMax_(0) = localRegionMaxTemp[0];
			this->localRegionMax_(1) = localRegionMaxTemp[1];
			this->localRegionMax_(2) = localRegionMaxTemp[2];
			cout << this->hint_ << ": Local Region Max: " << this->localRegionMax_[0] <<" " <<this->localRegionMax_[1]<<" "<< this->localRegionMax_[2]<< endl;
		}

		// global sample region min
		std::vector<double> globalRegionMinTemp;	
		if (not this->nh_.getParam(this->ns_ + "/global_region_min", globalRegionMinTemp)){
			this->globalRegionMin_(0) = -20.0;
			this->globalRegionMin_(1) = -20.0;
			this->globalRegionMin_(2) = 0.0;
			cout << this->hint_ << ": No global region min param. Use default: [-20 -20 0]" <<endl;
		}
		else{
			this->globalRegionMin_(0) = globalRegionMinTemp[0];
			this->globalRegionMin_(1) = globalRegionMinTemp[1];
			this->globalRegionMin_(2) = globalRegionMinTemp[2];
			cout << this->hint_ << ": Global Region Min: " << this->globalRegionMin_[0] <<" "<< this->globalRegionMin_[1]<<" "<< this->globalRegionMin_[2]<< endl;
		}

		// global sample region max
		std::vector<double> globalRegionMaxTemp;	
		if (not this->nh_.getParam(this->ns_ + "/global_region_max", globalRegionMaxTemp)){
			this->globalRegionMax_(0) = 20.0;
			this->globalRegionMax_(1) = 20.0;
			this->globalRegionMax_(2) = 3.0;
			cout << this->hint_ << ": No global region max param. Use default: [20 20 3]" <<endl;
		}
		else{
			this->globalRegionMax_(0) = globalRegionMaxTemp[0];
			this->globalRegionMax_(1) = globalRegionMaxTemp[1];
			this->globalRegionMax_(2) = globalRegionMaxTemp[2];
			cout << this->hint_ << ": Global Region Max: " << this->globalRegionMax_[0] <<" "<< this->globalRegionMax_[1]<<" "<< this->globalRegionMax_[2]<< endl;
		}

		// Local Sample Threshold Value
		if (not this->nh_.getParam(this->ns_ + "/local_sample_thresh", this->localSampleThresh_)){
			this->localSampleThresh_ = 50;
			cout << this->hint_ << ": No local sample thresh param. Use default: 50" << endl;
		}
		else{
			cout << this->hint_ << ": Local sample Thresh: " << this->localSampleThresh_ << endl;
		}
		
		// Global Sample Threshold Value
		if (not this->nh_.getParam(this->ns_ + "/global_sample_thresh", this->globalSampleThresh_)){
			this->globalSampleThresh_ = 50;
			cout << this->hint_ << ": No global sample thresh param. Use default: 50" << endl;
		}
		else{
			cout << this->hint_ << ": Global sample Thresh: " << this->globalSampleThresh_ << endl;
		}

		// Frontier Sample Threshold Value
		if (not this->nh_.getParam(this->ns_ + "/frontier_sample_thresh", this->frontierSampleThresh_)){
			this->frontierSampleThresh_ = 50;
			cout << this->hint_ << ": No frontier sample thresh param. Use default: 50" << endl;
		}
		else{
			cout << this->hint_ << ": Frontier sample Thresh: " << this->frontierSampleThresh_ << endl;
		}

		// BFS+PCA frontier cluster size threshold (world meters)
		if (not this->nh_.getParam(this->ns_ + "/cluster_size_xy", this->cluster_size_xy_)){
			this->cluster_size_xy_ = 1.0;
			cout << this->hint_ << ": No cluster_size_xy param. Use default: 2.0" << endl;
		}
		else{
			cout << this->hint_ << ": Cluster size XY: " << this->cluster_size_xy_ << endl;
		}

		// Minimum frontier cluster cells
		if (not this->nh_.getParam(this->ns_ + "/min_cluster_cells", this->min_cluster_cells_)){
			this->min_cluster_cells_ = 5;
			cout << this->hint_ << ": No min_cluster_cells param. Use default: 5" << endl;
		}
		else{
			cout << this->hint_ << ": Min cluster cells: " << this->min_cluster_cells_ << endl;
		}

		// Minimum frontier size (meters) to report
		if (not this->nh_.getParam(this->ns_ + "/min_frontier_size", this->min_frontier_size_)){
			this->min_frontier_size_ = 0.3;
			cout << this->hint_ << ": No min_frontier_size param. Use default: 0.3" << endl;
		}
		else{
			cout << this->hint_ << ": Min frontier size: " << this->min_frontier_size_ << endl;
		}

		// Minimum distance between frontier centers (meters)
		if (not this->nh_.getParam(this->ns_ + "/min_frontier_center_dist", this->min_frontier_center_dist_)){
			this->min_frontier_center_dist_ = 0.4;
			cout << this->hint_ << ": No min_frontier_center_dist param. Use default: 0.4" << endl;
		}
		else{
			cout << this->hint_ << ": Min frontier center distance: " << this->min_frontier_center_dist_ << endl;
		}

		// 8邻域 frontier 检测开关
		if (not this->nh_.getParam(this->ns_ + "/use_8neighbor_frontier", this->use_8neighbor_frontier_)){
			this->use_8neighbor_frontier_ = false;
			cout << this->hint_ << ": No use_8neighbor_frontier param. Use default: false (4-neighbor)" << endl;
		}
		else{
			cout << this->hint_ << ": Use 8-neighbor frontier: " << (this->use_8neighbor_frontier_ ? "true" : "false") << endl;
		}

		// minimum distance for node sampling
		if (not this->nh_.getParam(this->ns_ + "/dist_thresh", this->distThresh_)){
			this->distThresh_ = 0.8;
			cout << this->hint_ << ": No distance thresh param. Use default: 0.8" << endl;
		}
		else{
			cout << this->hint_ << ": Distance Thresh: " << this->distThresh_ << endl;
		}

		// safety distance for random sampling in xy
		if (not this->nh_.getParam(this->ns_ + "/safe_distance_xy", this->safeDistXY_)){
			this->safeDistXY_ = 0.3;
			cout << this->hint_ << ": No safe distance in XY param. Use default: 0.3" << endl;
		}
		else{
			cout << this->hint_ << ": Safe distance in XY: " << this->safeDistXY_ << endl;
		}

		// safety distance for random sampling in z
		if (not this->nh_.getParam(this->ns_ + "/safe_distance_z", this->safeDistZ_)){
			this->safeDistZ_ = 0.0;
			cout << this->hint_ << ": No safe distance in Z param. Use default: 0.0" << endl;
		}
		else{
			cout << this->hint_ << ": Safe distance in Z: " << this->safeDistZ_ << endl;
		}

		// safety distance check unknown
		if (not this->nh_.getParam(this->ns_ + "/safe_distance_check_unknown", this->safeDistCheckUnknown_)){
			this->safeDistCheckUnknown_ = true;
			cout << this->hint_ << ": No safe distance check unknown param. Use default: true" << endl;
		}
		else{
			cout << this->hint_ << ": Safe distance check unknown: " << this->safeDistCheckUnknown_ << endl;
		}

		//Camera Parameters	
		if (not this->nh_.getParam(this->ns_ + "/horizontal_FOV", this->horizontalFOV_)){
			this->horizontalFOV_ = 0.8;
			cout << this->hint_ << ": No Horizontal FOV param. Use default: 0.8" << endl;
		}
		else{
			cout << this->hint_ << ": Horizontal FOV: " << this->horizontalFOV_ << endl;
		}
		if (not this->nh_.getParam(this->ns_ + "/vertical_FOV", this->verticalFOV_)){
			this->verticalFOV_ = 0.8;
			cout << this->hint_ << ": No Vertical FOV param. Use default: 0.8" << endl;
		}
		else{
			cout << this->hint_ << ": Vertical FOV: " << this->verticalFOV_ << endl;
		}
		if (not this->nh_.getParam(this->ns_ + "/dmin", this->dmin_)){
			this->dmin_ = 0.0;
			cout << this->hint_ << ": No min depth param. Use default: 0.0" << endl;
		}
		else{
			cout << this->hint_ << ": Min Depth: " << this->dmin_ << endl;
		}
		if (not this->nh_.getParam(this->ns_ + "/dmax", this->dmax_)){
			this->dmax_ = 1.0;
			cout << this->hint_ << ": No max depth param. Use default: 1.0" << endl;
		}
		else{
			cout << this->hint_ << ": Max Depth: " << this->dmax_ << endl;
		}

		// nearest neighbor number
		if (not this->nh_.getParam(this->ns_ + "/nearest_neighbor_number", this->nnNum_)){
			this->nnNum_ = 15;
			cout << this->hint_ << ": No nearest neighbor param. Use default: 15" << endl;
		}
		else{
			cout << this->hint_ << ": Nearest neighbor number is set to: " << this->nnNum_ << endl;
		}

		// frontier nearest neighbor number
		if (not this->nh_.getParam(this->ns_ + "/frontier_nearest_neighbor_number", this->nnNumFrontier_)){
			this->nnNumFrontier_ = 15;
			cout << this->hint_ << ": No frontier nearest neighbor param. Use default: 15" << endl;
		}
		else{
			cout << this->hint_ << ": Frontier nearest neighbor number is set to: " << this->nnNumFrontier_ << endl;
		}

		// node connection max distances
		if (not this->nh_.getParam(this->ns_ + "/max_connect_dist", this->maxConnectDist_)){
			this->maxConnectDist_ = 1.5;
			cout << this->hint_ << ": No max conect distance param. Use default: 1.5m." << endl;
		}
		else{
			cout << this->hint_ << ": Max connect distance is set to: " << this->maxConnectDist_ << endl;
		}

		// number of yaw angles
		int yawNum = 32;
		if (not this->nh_.getParam(this->ns_ + "/num_yaw_angles", yawNum)){
			for (int i=0; i<32; ++i){
				this->yaws_.push_back(i*2*PI_const/32);
			}					
			cout << this->hint_ << ": No number of yaw angles param. Use default: 32." << endl;
		}
		else{
			for (int i=0; i<yawNum; ++i){
				this->yaws_.push_back(i*2*PI_const/32);
			}	
			cout << this->hint_ << ": Number of yaw angles is set to: " << yawNum << endl;
		}

		// minimum threshold of voxels
		if (not this->nh_.getParam(this->ns_ + "/min_voxel_thresh", this->minVoxelThresh_)){
			this->minVoxelThresh_ = 0.1;
			cout << this->hint_ << ": No minimum threshold of voxels param. Use default: 0.1." << endl;
		}
		else{
			cout << this->hint_ << ": Minimum threshold of voxels is set to: " << this->minVoxelThresh_ << endl;
		}

		// minimum number of goal candidates
		if (not this->nh_.getParam(this->ns_ + "/min_goal_candidates", this->minCandidateNum_)){
			this->minCandidateNum_ = 10;
			cout << this->hint_ << ": No minimum number of goal candidates param. Use default: 10." << endl;
		}
		else{
			cout << this->hint_ << ": Minimum number of goal candidates is set to: " << this->minCandidateNum_ << endl;
		}

		// maximum number of goal candidates
		if (not this->nh_.getParam(this->ns_ + "/max_goal_candidates", this->maxCandidateNum_)){
			this->maxCandidateNum_ = 30;
			cout << this->hint_ << ": No maximum number of goal candidates param. Use default: 30." << endl;
		}
		else{
			cout << this->hint_ << ": Maximum number of goal candidates is set to: " << this->maxCandidateNum_ << endl;
		}

		// Information gain update  distance
		if (not this->nh_.getParam(this->ns_ + "/information_gain_update_distance", this->updateDist_)){
			this->updateDist_ = 1.0;
			cout << this->hint_ << ": No information gain update distance param. Use default: 1.0m." << endl;
		}
		else{
			cout << this->hint_ << ": Information gain update distance is set to: " << this->updateDist_ << endl;
		}

		// yaw penalty weight
		if (not this->nh_.getParam(this->ns_ + "/yaw_penalty_weight", this->yawPenaltyWeight_)){
			this->yawPenaltyWeight_ = 2000.0;
			cout << this->hint_ << ": No yaw penalty weight param. Use default: 1.0." << endl;
		}
		else{
			cout << this->hint_ << ": Yaw penalty weight is set to: " << this->yawPenaltyWeight_ << endl;
		}

		// yaw distance scale factor (yaw penalty scales with path length)
		if (not this->nh_.getParam(this->ns_ + "/yaw_dist_scale", this->yawDistScale_)){
			this->yawDistScale_ = 2.0;
			cout << this->hint_ << ": No yaw_dist_scale param. Use default: 3.0m." << endl;
		}
		else{
			cout << this->hint_ << ": Yaw dist scale is set to: " << this->yawDistScale_ << endl;
		}

		// exploration 相关
		use2DMap_ = true;

		// height layers for PRM node Z sampling (indoor multi-level)
		std::vector<double> heightLayersTemp;
		if (not this->nh_.getParam(this->ns_ + "/height_layers", heightLayersTemp)){
			this->height_layers_ = {0.5};
			cout << this->hint_ << ": No height_layers param. Use default: [0.5]" << endl;
		}
		else{
			this->height_layers_ = heightLayersTemp;
			cout << this->hint_ << ": Height layers: ";
			for (auto h : this->height_layers_) cout << h << " ";
			cout << endl;
		}
		// Initialize current height layer index to 0 (first layer)
		this->current_height_layer_idx_ = 0;


		// cross-layer link count per new node (ensures connectivity between height layers)
		if (not this->nh_.getParam(this->ns_ + "/cross_layer_links", this->cross_layer_links_)){
			this->cross_layer_links_ = 5;
			cout << this->hint_ << ": No cross_layer_links param. Use default: 2" << endl;
		}
		else{
			cout << this->hint_ << ": Cross layer links: " << this->cross_layer_links_ << endl;
		}

		// semantic value map parameters
		if (not this->nh_.getParam(this->ns_ + "/use_value_map", this->useValueMap_)){
			this->useValueMap_ = true;
			cout << this->hint_ << ": No use_value_map param. Use default: true" << endl;
		}
		else{
			cout << this->hint_ << ": Use value map: " << this->useValueMap_ << endl;
		}
		if (not this->nh_.getParam(this->ns_ + "/semantic_weight", this->semanticWeight_)){
			this->semanticWeight_ = 7.0;
			cout << this->hint_ << ": No semantic_weight param. Use default: 1.0" << endl;
		}
		else{
			cout << this->hint_ << ": Semantic weight: " << this->semanticWeight_ << endl;
		}
		if (not this->nh_.getParam(this->ns_ + "/candidate_semantic_weight", this->candidateSemanticWeight_)){
			this->candidateSemanticWeight_ = 1.0;
			cout << this->hint_ << ": No candidate_semantic_weight param. Use default: 1.0" << endl;
		}
		else{
			cout << this->hint_ << ": Candidate semantic weight: " << this->candidateSemanticWeight_ << endl;
		}
		if (not this->nh_.getParam(this->ns_ + "/semantic_bypass_thresh", this->semanticBypassThresh_)){
			this->semanticBypassThresh_ = 0.1;
			cout << this->hint_ << ": No semantic_bypass_thresh param. Use default: 0.1" << endl;
		}
		else{
			cout << this->hint_ << ": Semantic bypass threshold: " << this->semanticBypassThresh_ << endl;
		}

		// frontier bonus parameters
		if (not this->nh_.getParam(this->ns_ + "/frontier_weight", this->frontierWeight_)){
			this->frontierWeight_ = 0.0;
			cout << this->hint_ << ": No frontier_weight param. Use default: 0.0" << endl;
		}
		else{
			cout << this->hint_ << ": Frontier weight: " << this->frontierWeight_ << endl;
		}
		if (not this->nh_.getParam(this->ns_ + "/candidate_frontier_weight", this->candidateFrontierWeight_)){
			this->candidateFrontierWeight_ = 0.5;
			cout << this->hint_ << ": No candidate_frontier_weight param. Use default: 0.0" << endl;
		}
		else{
			cout << this->hint_ << ": Candidate frontier weight: " << this->candidateFrontierWeight_ << endl;
		}
		if (not this->nh_.getParam(this->ns_ + "/frontier_bonus_radius", this->frontierBonusRadius_)){
			this->frontierBonusRadius_ = 0.5;
			cout << this->hint_ << ": No frontier_bonus_radius param. Use default: 0.5m" << endl;
		}
		else{
			cout << this->hint_ << ": Frontier bonus radius: " << this->frontierBonusRadius_ << "m" << endl;
		}

		// dynamic distance penalty (ramps down with planning cycles)
		if (not this->nh_.getParam(this->ns_ + "/dist_penalty_max", this->distPenaltyMax_)){
			this->distPenaltyMax_ = 20.0;
			cout << this->hint_ << ": No dist_penalty_max param. Use default: 15.0" << endl;
		}
		else{
			cout << this->hint_ << ": dist_penalty_max: " << this->distPenaltyMax_ << endl;
		}
		if (not this->nh_.getParam(this->ns_ + "/dist_penalty_min", this->distPenaltyMin_)){
			this->distPenaltyMin_ = 1.0;
			cout << this->hint_ << ": No dist_penalty_min param. Use default: 3.0" << endl;
		}
		else{
			cout << this->hint_ << ": dist_penalty_min: " << this->distPenaltyMin_ << endl;
		}
		if (not this->nh_.getParam(this->ns_ + "/dist_penalty_ramp_cycles", this->distPenaltyRampCycles_)){
			this->distPenaltyRampCycles_ = 20;
			cout << this->hint_ << ": No dist_penalty_ramp_cycles param. Use default: 50" << endl;
		}
		else{
			cout << this->hint_ << ": dist_penalty_ramp_cycles: " << this->distPenaltyRampCycles_ << endl;
		}

		// Initial yaw change threshold to trigger replan (degrees, converted to rad)
		double yaw_replan_deg = 90.0;
		if (not this->nh_.getParam(this->ns_ + "/yaw_replan_threshold_deg", yaw_replan_deg)){
			cout << this->hint_ << ": No yaw_replan_threshold_deg param. Use default: 90.0 deg" << endl;
		}
		else{
			cout << this->hint_ << ": yaw_replan_threshold_deg: " << yaw_replan_deg << endl;
		}
		this->yaw_replan_threshold_ = yaw_replan_deg * M_PI / 180.0;
	}

	void DEP::initModules(){
		// initialize roadmap
		this->roadmap_.reset(new PRM::KDTree ());
		// ValueMap2D is created in setMap() after map_ is available
	}

	void DEP::registerPub(){
		// roadmap visualization publisher
		this->roadmapPub_ = this->nh_.advertise<visualization_msgs::MarkerArray>("/dep/roadmap", 10);

		// candidate paths publisher
		this->candidatePathPub_ = this->nh_.advertise<visualization_msgs::MarkerArray>("/dep/candidate_paths", 10);

		// best path publisher
		this->bestPathPub_ = this->nh_.advertise<visualization_msgs::MarkerArray>("/dep/best_paths", 10);

		// fronteir vis publisher
		this->frontierVisPub_ = this->nh_.advertise<visualization_msgs::MarkerArray>("/dep/frontier_regions", 10);

		// value map heatmap publisher (OccupancyGrid for RViz)
		this->valueMapPub_ = this->nh_.advertise<nav_msgs::OccupancyGrid>("/dep/value_map", 10, true);  // latched: 切 episode 时自动清 RViz
	}

	void DEP::registerCallback(){
		// odom subscriber
		this->odomSub_ = this->nh_.subscribe(this->odomTopic_, 1000, &DEP::odomCB, this);
	
		// visualization timer
		this->visTimer_ = this->nh_.createTimer(ros::Duration(0.1), &DEP::visCB, this);

		// valuemap timer
		this->valuemapTimer_ = this->nh_.createTimer(ros::Duration(0.5),&DEP::valuemapCB,this);

		// BLIP2 ITM score subscriber (from blip2_itm_node.py)
		this->itmScoreSub_ = this->nh_.subscribe<std_msgs::Float64>(
			"/blip2/itm_score", 10, &DEP::itmScoreCB, this);

	}



	bool DEP::makePlan(){
		if (!this->odomReceived_){
			ROS_WARN("No odom my dep");
			return false;
		}
		// ROS_WARN("We are going to get value map");

		// ---- ValueMap2D update (random ITM score placeholder) ----
		// if (useValueMap_ && value_map_ && map_->is2DMapReady()) {
		// 	ROS_WARN("get value map");
		// 	value_map_->ensureInitialized();
		// 	ROS_ERROR("ensure init value map");
		// 	if (value_map_->isInitialized()) {
		// 		// TODO: Replace random ITM score with actual BLIP2 ITM score
		// 		// For now, generate a random score in [0, 1] as placeholder
		// 		double random_itm_score = static_cast<double>(rand()) / RAND_MAX;

		// 		// Collect free grids visible in current FOV and update value map
		// 		Eigen::Vector2d sensor_pos(this->position_(0), this->position_(1));
		// 		std::vector<Eigen::Vector2i> free_grids;
		// 		// value_map_->getFreeGrids(free_grids, sensor_pos, this->currYaw_);
		// 		// value_map_->getFreeGrids(free_grids);
		// 		free_grids = map_->get2DFreeGrid();
		// 		value_map_->updateValueMap(sensor_pos, this->currYaw_, free_grids, random_itm_score);

		// 		ROS_DEBUG_THROTTLE(2.0, "[DEP]: ValueMap updated with random ITM=%.3f, free_grids=%zu",
		// 		                   random_itm_score, free_grids.size());
		// 	}
		// }

		// cout << "start detecting frontier" << endl;
		// ros::Time frontierStartTime = ros::Time::now();
		this->detectFrontierRegion(this->frontierPointPairs_);
		//ROS_WARN("let's detect frontier");
		// ros::Time frontierEndTime = ros::Time::now();
		// cout << "frontier detection time: " << (frontierEndTime - frontierStartTime).toSec() << endl;


		// cout << "start building roadmap" << endl;
		// ros::Time buildStartTime = ros::Time::now();
		this->buildRoadMap();
		// Reset the switched-layer flag after first build with new height layer
		this->just_switched_layer_ = false;
		//ROS_WARN("let's build roadmap");
		// ros::Time buildEndTime = ros::Time::now();
		// cout << "build roadmap time: " << (buildEndTime - buildStartTime).toSec() << endl;

		// cout << "start pruning nodes" << endl;
		// ros::Time updateStartTime = ros::Time::now();
		this->pruneNodes();
		//ROS_WARN("let's prune nodes");

		// cout << "start update information gain" << endl;
		this->updateInformationGain();
		//ROS_WARN("let's update IG");
		// ros::Time updateEndTime = ros::Time::now();
		// cout << "update time: " << (updateEndTime - updateStartTime).toSec() << endl;

		// cout << "start get goal candidates" << endl;
		// ros::Time pathStartTime = ros::Time::now();
		this->getBestViewCandidates(this->goalCandidates_);
		//ROS_WARN("let's get best candidates");

		// cout << "finish best view candidate with size: " << this->goalCandidates_.size() << endl;

		bool findCandidatePathSuccess = this->findCandidatePath(this->goalCandidates_, this->candidatePaths_);
		//ROS_WARN("let's get best path");

		// cout << "finish candidate path with size: " << this->candidatePaths_.size() << endl;
		if (!findCandidatePathSuccess){
			// cout << "Find candidate paths fails. need generate more samples." << endl;
			// ROS_WARN("Can not get best path: prmNodeVec_=%zu, roadmap_size=%d, goalCandidates=%zu, position=(%.2f,%.2f,%.2f), height_layer=%d",
			//          this->prmNodeVec_.size(), this->roadmap_->getSize(), this->goalCandidates_.size(),
			//          this->position_(0), this->position_(1), this->position_(2),
			//          this->current_height_layer_idx_);
			// Print each goal candidate position with its numVoxels and adjNodes for diagnosis
			// for (size_t i = 0; i < this->goalCandidates_.size(); ++i) {
			// 	auto& gc = this->goalCandidates_[i];
			// 	// ROS_WARN("[DEP] goalCandidates_[%zu]: pos=(%.2f,%.2f,%.2f) numVoxels=%d adjNodes=%zu",
			// 	//          i, gc->pos(0), gc->pos(1), gc->pos(2), gc->numVoxels, gc->adjNodes.size());
			// }
			// Check if start PRM node has any adjNodes
			std::shared_ptr<PRM::Node> currPos;
			currPos.reset(new PRM::Node(this->position_));
			std::shared_ptr<PRM::Node> start = this->roadmap_->nearestNeighbor(currPos);
			if (start) {
				ROS_WARN("[DEP] start PRM node=(%.2f,%.2f,%.2f), adjNodes=%zu, dist to curr=%.2f",
				         start->pos(0), start->pos(1), start->pos(2), start->adjNodes.size(),
				         (start->pos - this->position_).norm());
			}
			return false;
		}

		this->findBestPath(this->candidatePaths_, this->bestPath_);
		// ros::Time pathEndTime = ros::Time::now();
		// cout << "path time: " << (pathEndTime - pathStartTime).toSec() << endl;
		// cout << "found best path with size: " << this->bestPath_.size() << endl;
		if (this->bestPath_.empty()){
			ROS_WARN("findBestPath produced empty bestPath, returning false");
			return false;
		}
		// Compute initial yaw change: angle between current heading and first path segment
		if (this->bestPath_.size() >= 2) {
			Eigen::Vector3d first_seg = this->bestPath_[1]->pos - this->bestPath_[0]->pos;
			double first_seg_yaw = atan2(first_seg(1), first_seg(0));
			this->initial_yaw_change_ = fabs(globalPlanner::angleDiff(this->currYaw_, first_seg_yaw));
		} else {
			this->initial_yaw_change_ = 0.0;
		}
		depPlanCount_++;
		return true;
	}

	Eigen::Vector3d DEP::getNearestPRMNode(const Eigen::Vector3d& target_point){
		if (this->prmNodeVec_.empty()){
			ROS_WARN("[DEP] getNearestPRMNode: prmNodeVec_ is empty, returning target_point as-is");
			return target_point;
		}
		double min_dist = std::numeric_limits<double>::max();
		Eigen::Vector3d nearest_pos = target_point;
		for (const auto& node : this->prmNodeVec_){
			double dist = (node->pos - target_point).norm();
			if (dist < min_dist){
				min_dist = dist;
				nearest_pos = node->pos;
			}
		}
		ROS_INFO("[DEP] getNearestPRMNode: target=(%.2f,%.2f,%.2f), nearest PRM=(%.2f,%.2f,%.2f), dist=%.2f",
		         target_point.x(), target_point.y(), target_point.z(),
		         nearest_pos.x(), nearest_pos.y(), nearest_pos.z(), min_dist);
		return nearest_pos;
	}

	bool DEP::findPathToPoint(const Eigen::Vector3d& target, std::vector<Eigen::Vector3d>& path_points){
		path_points.clear();
		// If the current roadmap is empty, we cannot find a path
		if (this->prmNodeVec_.empty() || this->roadmap_->getSize() == 0) {
			ROS_WARN("[DEP] findPathToPoint: roadmap is empty, cannot find path");
			return false;
		}

		// Create start and goal nodes
		std::shared_ptr<PRM::Node> currPos;
		currPos.reset(new PRM::Node(this->position_));
		std::shared_ptr<PRM::Node> start = this->roadmap_->nearestNeighbor(currPos);

		std::shared_ptr<PRM::Node> goalNode;
		goalNode.reset(new PRM::Node(target));
		std::shared_ptr<PRM::Node> goal = this->roadmap_->nearestNeighbor(goalNode);

		ROS_INFO("[DEP] findPathToPoint: start=(%.2f,%.2f,%.2f), goal=(%.2f,%.2f,%.2f)",
		         start->pos.x(), start->pos.y(), start->pos.z(),
		         goal->pos.x(), goal->pos.y(), goal->pos.z());

		// A* search on PRM graph
		std::vector<std::shared_ptr<PRM::Node>> path = PRM::AStar(this->roadmap_, start, goal, this->map_);

		if (path.empty()) {
			ROS_WARN("[DEP] findPathToPoint: A* failed to find path to target");
			return false;
		}

		// Insert current position at the beginning
		path.insert(path.begin(), currPos);

		// Shortcut the path (collision-aware)
		std::vector<std::shared_ptr<PRM::Node>> pathSc;
		this->shortcutPath(path, pathSc);

		// Convert to vector of Eigen::Vector3d
		for (const auto& node : pathSc) {
			path_points.push_back(node->pos);
		}

		ROS_INFO("[DEP] findPathToPoint: found path with %zu waypoints (after shortcut: %zu)",
		         path.size(), path_points.size());
		return true;
	}

	void DEP::updateRoadmapOnly(){
		if (!this->odomReceived_){
			ROS_WARN_THROTTLE(1.0,"[DEP] updateRoadmapOnly: No odom");
			return;
		}
		// Only update the PRM graph (build, prune, update IG)
		// Skip frontier detection, candidate scoring, and path search
		this->buildRoadMap();
		this->pruneNodes();
		this->updateInformationGain();
	}

	void DEP::switchHeightLayer(){
		if (this->height_layers_.empty()) {
			ROS_WARN("[DEP] switchHeightLayer: height_layers_ is empty, cannot switch");
			return;
		}
		int prev_idx = this->current_height_layer_idx_;
		this->current_height_layer_idx_ = (this->current_height_layer_idx_ + 1) % this->height_layers_.size();
		ROS_WARN("[DEP] switchHeightLayer: switched from layer[%d]=%.2f to layer[%d]=%.2f",
		         prev_idx, this->height_layers_[prev_idx],
		         this->current_height_layer_idx_, this->height_layers_[this->current_height_layer_idx_]);
		// Set flag so next buildRoadMap only does local sampling for quick escape
		this->just_switched_layer_ = true;
	}

	void DEP::dropCurrentGoalNode(){
		if (this->bestPath_.empty()) {
			ROS_WARN("[DEP] dropCurrentGoalNode: bestPath_ is empty, nothing to drop");
			return;
		}
		auto goal = this->bestPath_.back();

		// Collect goal node and all its adjacent nodes to drop together
		std::unordered_set<std::shared_ptr<PRM::Node>> nodes_to_drop;
		nodes_to_drop.insert(goal);
		for (const auto& adj : goal->adjNodes) {
			nodes_to_drop.insert(adj);
		}

		ROS_WARN("[DEP] dropCurrentGoalNode: dropping goal node at (%.2f, %.2f, %.2f) and %zu adjacent nodes (%zu total)",
				goal->pos.x(), goal->pos.y(), goal->pos.z(),
				nodes_to_drop.size() - 1, nodes_to_drop.size());

		// Remove all collected nodes from PRM node set and KDTree
		for (const auto& node : nodes_to_drop) {
			this->prmNodeVec_.erase(node);
			this->roadmap_->remove(node);
		}

		// Clean up edges pointing to any dropped node from all remaining nodes
		for (auto& n : this->prmNodeVec_){
			for (const auto& dropped : nodes_to_drop) {
				n->adjNodes.erase(dropped);
			}
		}

		// Clear all path/candidate data so next makePlan re-computes from scratch
		this->bestPath_.clear();
		this->candidatePaths_.clear();
		this->goalCandidates_.clear();
	}


	nav_msgs::Path DEP::getBestPath(){
		nav_msgs::Path bestPath;
		if (this->bestPath_.empty()){
			ROS_WARN("getBestPath called with empty bestPath_, returning empty path");
			return bestPath;
		}
		for (int i=0; i<int(this->bestPath_.size()); ++i){
			std::shared_ptr<PRM::Node> currNode = this->bestPath_[i];
			geometry_msgs::PoseStamped p;
			p.pose.position.x = currNode->pos(0);
			p.pose.position.y = currNode->pos(1);
			p.pose.position.z = currNode->pos(2);
			if (i < int(this->bestPath_.size())-1){
				std::shared_ptr<PRM::Node> nextNode = this->bestPath_[i+1];
				Eigen::Vector3d diff = nextNode->pos - currNode->pos;
				double angle = atan2(diff(1), diff(0));
				p.pose.orientation = globalPlanner::quaternion_from_rpy(0, 0, angle);
			}
			bestPath.poses.push_back(p);
		}
		
		// get the best yaw for the last pose
		double bestYaw = this->bestPath_.back()->getBestYaw();
		bestPath.poses.back().pose.orientation = globalPlanner::quaternion_from_rpy(0, 0, bestYaw);

		return bestPath;
	}

	bool DEP::sensorRangeCondition(const shared_ptr<PRM::Node>& n1, const shared_ptr<PRM::Node>& n2){
		Eigen::Vector3d direction = n2->pos - n1->pos;
		Eigen::Vector3d projection;
		projection(0) = direction.x();
		projection(1) = direction.y();
		projection(2) = 0;
		double verticalAngle = angleBetweenVectors(direction, projection);
		if (verticalAngle < this->verticalFOV_/2){
			return true;
		}
		else{
			return false;
		}
	}

	// create sensor check for 
	// vert, horz FOV, collision, and sensor distance range
	// for yaw angles in vector3d:  cos(yaw), sin(yaw), 0
	// horz angle between yaw angle vector and direction (x y 0) vector for horz FOV
	// Vert angle yaw angle vector and yaw angle vector (c s z) z is direction.z()
	bool DEP::sensorFOVCondition(const Eigen::Vector3d& sample, const Eigen::Vector3d& pos){
		Eigen::Vector3d direction = sample - pos;
		double distance = direction.norm();
		if (distance > this->dmax_){
			return false;
		}
		bool hasCollision;
		if (use2DMap_ && map_->is2DMapReady()) {
			hasCollision = map_->is2DInflatedOccupiedLine2D(sample,pos);
		} else {
			hasCollision = this->map_->isInflatedOccupiedLine(sample, pos);
		}
		if (hasCollision == true){
			return false;
		}
		return true;
	}



	bool isNodeRequireUpdate(std::shared_ptr<PRM::Node> n, std::vector<std::shared_ptr<PRM::Node>> path, double& leastDistance){
		double distanceThresh = 2;
		leastDistance = std::numeric_limits<double>::max();
		for (std::shared_ptr<PRM::Node>& waypoint: path){
			double currentDistance = (n->pos - waypoint->pos).norm();
			if (currentDistance < leastDistance){
				leastDistance = currentDistance;
			}
		}
		if (leastDistance <= distanceThresh){
			return true;
		}
		else{
			return false;	
		}
		
	}

	void DEP::detectFrontierRegion(std::vector<std::pair<Eigen::Vector3d, double>>& frontierPointPairs) {
		frontierPointPairs.clear();

		if (!use2DMap_ || !map_->is2DMapReady()) {
			ROS_WARN_THROTTLE(5.0, "[DEP] 2D map not ready for frontier detection");
			return;
		}

		double z_height = this->height_layers_[0];
		if (!height_layers_.empty()) {
			int idx = std::max(0, std::min(current_height_layer_idx_, (int)height_layers_.size() - 1));
			z_height = height_layers_[idx];
		}

		map_->detectFrontierClusters(frontierPointPairs, z_height,
		                             cluster_size_xy_,
		                             min_cluster_cells_,
		                             min_frontier_size_,
		                             min_frontier_center_dist_);

		ROS_INFO_COND(!frontierPointPairs.empty(),
		              "[DEP] Detected %zu frontier clusters at z=%.1f",
		              frontierPointPairs.size(), z_height);
	}


	void DEP::buildRoadMap(){
		bool saturate = false;
		bool regionSaturate = false;
		int countSample = 0;
		std::shared_ptr<PRM::Node> n;
		std::vector<std::shared_ptr<PRM::Node>> newNodes;

		// while does reach sampling threshold (fail time) 
		// 1. sample point from frontier region by weighted sampling
		// 2. find several nearest node
		// 3. for each node, extend random [mindist, maxdist] distance. if success, add new sample 
		std::vector<double> sampleWeights;
		for (int i=0; i<int(this->frontierPointPairs_.size()); ++i){
			double size = this->frontierPointPairs_[i].second;
			//sampleWeights.push_back(pow(size, 2));
			sampleWeights.push_back(1.0);
		}
		int countFrontierFailure = 0;
		while (ros::ok() and countFrontierFailure < this->frontierSampleThresh_
			       and sampleWeights.size() != 0
			       and this->roadmap_->getSize() > 0){   // roadmap 为空则跳过，等 local/global 先建图
			// if(countSample >= 50){
			// 	break;
			// }
			std::shared_ptr<PRM::Node> fn = this->sampleFrontierPoint(sampleWeights);
			// a. find N nearest neighbors
			std::vector<std::shared_ptr<PRM::Node>> fnNeighbors = this->roadmap_->kNearestNeighbor(fn, this->nnNumFrontier_);
			//ROS_WARN("Build from frontier map");
			// b. for each neighbor extend them and check the validity
			if (int(fnNeighbors.size()) > 0){
				int countSampleOnce = 0;
				for (std::shared_ptr<PRM::Node> fnNN : fnNeighbors){
					n = this->extendNode(fnNN, fn);
					bool is2DFreeValid = false;
					if (use2DMap_ && map_->is2DMapReady()) {
						is2DFreeValid = true;
						for (double x = n->pos(0) - this->safeDistXY_; x <= n->pos(0) + this->safeDistXY_; x += map_->getRes()) {
							for (double y = n->pos(1) - this->safeDistXY_; y <= n->pos(1) + this->safeDistXY_; y += map_->getRes()) {
								if (!map_->is2DFree(x, y)) {
									is2DFreeValid = false;
									break;
								}
							}
							if (!is2DFreeValid) break;
						}
					}
					if (is2DFreeValid){
					// if (this->isPosValid(n->pos, this->safeDistXY_, this->safeDistZ_)){
						std::shared_ptr<PRM::Node> nn = this->roadmap_->nearestNeighbor(n);
						double distToNN = (n->pos - nn->pos).norm();
						if (distToNN >= this->distThresh_){
							this->roadmap_->insert(n);
							newNodes.push_back(n);
							// this->prmNodeVec_.push_back(n);
							this->prmNodeVec_.insert(n);
							++countSample;	
							++countSampleOnce;
						}
					}
				}
				if (countSampleOnce == 0){
					++countFrontierFailure;
				}

			}
			else{ // 该采样点附近无 roadmap 节点 → 换一个 frontier 重试
				++countFrontierFailure;
			}
		}
		cout << "added sample from frontier:  " << countSample << endl;
		//ROS_WARN("Frontier sample thresh is %d",this->frontierSampleThresh_);
		while (ros::ok() and not saturate){
			//ROS_WARN("Let us try another sample");
			if (regionSaturate){
				// If just switched layer, skip global sampling to focus on
				// local escape: only build roadmap around current position
				if (this->just_switched_layer_) {
					ROS_WARN("[DEP] buildRoadMap: just switched layer, local-only sampling for quick escape");
					saturate = true;
					break;
				}
				int countFailureGlobal = 0;
				// saturate = true;
				// Generate new node
				while (ros::ok()){
					if (countFailureGlobal > this->globalSampleThresh_){
						saturate = true;
						break;
					}
					//ROS_WARN("Count failure global:%d",countFailureGlobal);
					n = this->randomConfigBBox(this->globalRegionMin_, this->globalRegionMax_);
					// Check how close new node is other nodes
					double distToNN;
					if (this->roadmap_->getSize() != 0){
						shared_ptr<PRM::Node> nn = this->roadmap_->nearestNeighbor(n);
						distToNN = (n->pos - nn->pos).norm();
					}
					else{
						distToNN = this->distThresh_;
					}
					if (distToNN < this->distThresh_){
						++countFailureGlobal;
					}
					else{
						this->roadmap_->insert(n);
						newNodes.push_back(n);
						// this->prmNodeVec_.push_back(n);
						this->prmNodeVec_.insert(n);
						++countSample;
					}
				}
			}
			else{
				if (true){
					int countFailureLocal = 0;
					// Generate new node
					while (ros::ok() and true){
						//cout << "failure number: " << countFailureLocal << endl;
						//ROS_WARN("Count failure local:%d",countFailureLocal);
						if (countFailureLocal > this->localSampleThresh_){
							regionSaturate = true;
							break;
						}
						Eigen::Vector3d localSampleMin = this->position_+this->localRegionMin_;
						Eigen::Vector3d localSampleMax = this->position_+this->localRegionMax_;
						n = this->randomConfigBBox(localSampleMin, localSampleMax);
						// Local sampling: only accept samples in KNOWN FREE space
						if (use2DMap_ && map_->is2DMapReady()) {
							if (!map_->is2DFree(n->pos(0), n->pos(1))) {
								++countFailureLocal;
								continue;
							}
						}
						// Check how close new node is other nodes
						double distToNN;

						if (this->roadmap_->getSize() != 0){
							std::shared_ptr<PRM::Node> nn = this->roadmap_->nearestNeighbor(n);
							distToNN = (n->pos - nn->pos).norm();
						}
						else{
							distToNN = this->distThresh_;
						}
						if (distToNN < this->distThresh_){
							++countFailureLocal;
						}
						else{
							this->roadmap_->insert(n);
							newNodes.push_back(n);
							// this->prmNodeVec_.push_back(n);
							this->prmNodeVec_.insert(n);
							++countSample;
						}
					}
				}
			}
		}
		
		// node connection — same layer freely, limited cross-layer links
		int cross_layer_added = 0;  // global counter across all new nodes this round
		for (std::shared_ptr<PRM::Node>& n : newNodes){
			std::vector<std::shared_ptr<PRM::Node>> knn = this->roadmap_->kNearestNeighbor(n, this->nnNum_);
			for (std::shared_ptr<PRM::Node>& nearestNeighborNode : knn){
				double distance2knn = (n->pos - nearestNeighborNode->pos).norm();
				bool same_layer = (std::abs(n->pos(2) - nearestNeighborNode->pos(2)) < 0.01);
				bool rangeCondition = sensorRangeCondition(n, nearestNeighborNode) and sensorRangeCondition(nearestNeighborNode, n);
				if (distance2knn < this->maxConnectDist_ and rangeCondition == true){
					// Cross-layer: only allow up to cross_layer_links_ edges (skip if already saturated)
					if (!same_layer && cross_layer_added >= this->cross_layer_links_)
						continue;

					bool hasCollision;
					if (use2DMap_ && map_->is2DMapReady()) {
						hasCollision = map_->is2DInflatedOccupiedLine2D(n->pos,nearestNeighborNode->pos);
					} else {
						hasCollision = !this->map_->isInflatedFreeLine(n->pos, nearestNeighborNode->pos);
					}
					if (hasCollision == false){
						n->adjNodes.insert(nearestNeighborNode);
						nearestNeighborNode->adjNodes.insert(n);
						if (!same_layer) ++cross_layer_added;
					}
				}
			}
			n->newNode = true;
		}
	}	 

	void DEP::pruneNodes(){
		// record the invalid nodes
		std::unordered_set<std::shared_ptr<PRM::Node>> invalidSet;
		for (std::shared_ptr<PRM::Node> n : this->prmNodeVec_){ // new nodes
			if (!this->isPosValid(n->pos, this->safeDistXY_, this->safeDistZ_)){// 1. new nodes
			// if (this->map_->isInflatedOccupied(n->pos)){// 1. new nodes
				invalidSet.insert(n);
			}	
		}

		// remove invalid nodes
		for (std::shared_ptr<PRM::Node> in : invalidSet){
			this->prmNodeVec_.erase(in);
			this->roadmap_->remove(in);
		}


		//  remove invalid edges
		for (std::shared_ptr<PRM::Node> n : this->prmNodeVec_){
			std::vector<std::shared_ptr<PRM::Node>> eraseVec;
			for (std::shared_ptr<PRM::Node> neighbor : n->adjNodes){
				if (invalidSet.find(neighbor) != invalidSet.end()){
					eraseVec.push_back(neighbor);
				}
			}

			for (std::shared_ptr<PRM::Node> en : eraseVec){
				n->adjNodes.erase(en);
			}
		}
	}

	void DEP::updateInformationGain(){
		// iterate through all current nodes (ignore update by path now)
		// three types of nodes need update:
		// 1. new nodes
		// 2. nodes close to the historical trajectory
		// 3. nodes along the last executed bestPath (robot just followed this path)
		std::unordered_set<std::shared_ptr<PRM::Node>> updateSet;
		for (std::shared_ptr<PRM::Node> n : this->prmNodeVec_){ // new nodes
			if (n->newNode == true){// 1. new nodes
				updateSet.insert(n);
			}
		}

		for (Eigen::Vector3d& histPos : this->histTraj_){ // traj update nodes
			std::shared_ptr<PRM::Node> histN;
			histN.reset(new PRM::Node(histPos));
			std::vector<std::shared_ptr<PRM::Node>> nns = this->roadmap_->kNearestNeighbor(histN, 10);
			for (std::shared_ptr<PRM::Node>& nn : nns){
				if ((nn->pos - histN->pos).norm() <= this->updateDist_){
					updateSet.insert(nn);
				}
			}
		}

		// 3. nodes along the last bestPath — the robot just traversed this area,
		//    so unknown voxels around these nodes must have decreased significantly
		for (std::shared_ptr<PRM::Node>& pathNode : this->bestPath_){
			std::vector<std::shared_ptr<PRM::Node>> nns = this->roadmap_->kNearestNeighbor(pathNode, 5);
			for (std::shared_ptr<PRM::Node>& nn : nns){
				if ((nn->pos - pathNode->pos).norm() <= this->updateDist_){
					updateSet.insert(nn);
				}
			}
		}

		for (std::shared_ptr<PRM::Node> updateN : updateSet){ // update information gain
			std::unordered_map<double, int> yawNumVoxels;
			int unknownVoxelNum = this->calculateUnknown(updateN, yawNumVoxels);
			updateN->numVoxels = unknownVoxelNum;
			updateN->yawNumVoxels = yawNumVoxels;
			updateN->newNode = false;
		}
		this->histTraj_.clear(); // clear history
	}

	double DEP::getFrontierProximity(const Eigen::Vector3d& pos) const {
		if (frontierPointPairs_.empty()) return 0.0;
		double minDist = std::numeric_limits<double>::max();
		for (const auto& fp : frontierPointPairs_) {
			double d = (Eigen::Vector2d(pos.x(), pos.y()) -
			            Eigen::Vector2d(fp.first.x(), fp.first.y())).norm();
			if (d < minDist) minDist = d;
		}
		if (minDist >= frontierBonusRadius_) return 0.0;
		return 1.0 - minDist / frontierBonusRadius_;
	}

	void DEP::getBestViewCandidates(std::vector<std::shared_ptr<PRM::Node>>& goalCandidates){
		goalCandidates.clear();
		std::priority_queue<std::shared_ptr<PRM::Node>, std::vector<std::shared_ptr<PRM::Node>>, PRM::GainCompareNode> gainPQ;

		// iterate through all points in the roadmap
		for (std::shared_ptr<PRM::Node> n : this->prmNodeVec_){
			gainPQ.push(n);
		}

		// ── Phase 1: Low gain threshold（只过滤已完全探索的区域）──
		int poolSize = std::min(this->maxCandidateNum_ * 3, 100);
		std::vector<std::shared_ptr<PRM::Node>> candidatePool;

		int maxNumVoxel = 0;
		while (ros::ok()){
			if (gainPQ.size() == 0) break;

			std::shared_ptr<PRM::Node> n = gainPQ.top();

			if (maxNumVoxel == 0) maxNumVoxel = n->numVoxels;

			// 只过滤 gain < 10% maxGain（基本完全探索完的），保留语义作用空间
			if (double(n->numVoxels) < double(maxNumVoxel) *0.2) break;

			gainPQ.pop();

			if ((n->pos - this->position_).norm() >= 0.0){
				if (this->isPosValid(n->pos, this->safeDistXY_, this->safeDistZ_)){
					candidatePool.push_back(n);
				}
			}

			if (int(candidatePool.size()) >= poolSize) break;
		}

		// Fill remaining slots if pool too small
		while (int(candidatePool.size()) < this->minCandidateNum_ && gainPQ.size() > 0
		       && int(candidatePool.size()) < poolSize) {
			std::shared_ptr<PRM::Node> n = gainPQ.top();
			gainPQ.pop();
			if ((n->pos - this->position_).norm() >= 0.0){
				if (this->isPosValid(n->pos, this->safeDistXY_, this->safeDistZ_)){
					candidatePool.push_back(n);
				}
			}
		}

		// // ── Semantic bypass: high-value nodes skip the gain filter ──
		// if (useValueMap_ && value_map_ && value_map_->isInitialized()
		//     && semanticBypassThresh_ > 0 && int(candidatePool.size()) < poolSize) {
		// 	std::unordered_set<std::shared_ptr<PRM::Node>> inPool(
		// 		candidatePool.begin(), candidatePool.end());
		// 	for (const auto& n : this->prmNodeVec_) {
		// 		if (inPool.count(n)) continue;
		// 		Eigen::Vector2d pos2d(n->pos(0), n->pos(1));
		// 		if (value_map_->getValue(pos2d) > semanticBypassThresh_) {
		// 			if ((n->pos - this->position_).norm() >= 0.0 &&
		// 			    this->isPosValid(n->pos, this->safeDistXY_, this->safeDistZ_)) {
		// 				candidatePool.push_back(n);
		// 				ROS_WARN("[DEP] Semantic bypass: value=%.3f at (%.1f,%.1f), gain=%d",
		// 				         value_map_->getValue(pos2d), n->pos(0), n->pos(1), n->numVoxels);
		// 				if (int(candidatePool.size()) >= poolSize) break;
		// 			}
		// 		}
		// 	}
		// }

		// ── Phase 2: Re-rank by combined score (info gain + semantic value + frontier) ──
		if (useValueMap_ && value_map_ && value_map_->isInitialized()
		    && candidateSemanticWeight_ > 1e-6 && candidatePool.size() > 1)
		{
			// Normalize info gain to [0, 1] within the pool for fair combination
			double maxGain = candidatePool.front()->numVoxels;
			double minGain = candidatePool.back()->numVoxels;
			double gainRange = (maxGain - minGain > 1e-6) ? (maxGain - minGain) : 1.0;

			// 使用 transformITMScore 的理论范围 [-penalty_scale, bonus_scale] = [-4, 3] 做归一化
			// 避免未观测节点(raw=0)在全是penalty的候选池中被推到 norm=1.0
			double semMin = -ValueMap2D::penalty_scale;  // -4.0
			double semMax =  ValueMap2D::bonus_scale;     //  3.0
			double semRange = semMax - semMin;            //  7.0
			ROS_WARN("[DEP] Semantic range (fixed): min=%.2f max=%.2f", semMin, semMax);

			// Compute combined score for each candidate
			std::vector<std::pair<double, std::shared_ptr<PRM::Node>>> scoredCandidates;
			scoredCandidates.reserve(candidatePool.size());
			for (const auto& n : candidatePool){
				double normalizedGain = (n->numVoxels - minGain) / gainRange;
				Eigen::Vector2d pos2d(n->pos(0), n->pos(1));
				double semanticValue = value_map_->getValue(pos2d);
				double normalizedSemantic = 2.0 * (semanticValue - semMin) / semRange - 1.0;  // [-1, 1]
				double frontierProximity = getFrontierProximity(n->pos);
				// Combined score: gain (0~1) + normalized semantic (-1~1) + frontier proximity bonus (0~1)
				double combinedScore = normalizedGain
					+ candidateSemanticWeight_ * normalizedSemantic
					+ candidateFrontierWeight_ * frontierProximity;
				scoredCandidates.emplace_back(combinedScore, n);
				ROS_WARN("The gain score is %f",normalizedGain);
				ROS_WARN("The sematic score is %f (raw=%.2f, norm=%.2f)", candidateSemanticWeight_ * normalizedSemantic, semanticValue, normalizedSemantic);
				ROS_WARN("The frontier score is %f",candidateFrontierWeight_ * frontierProximity);
			}

			// Sort by combined score descending
			std::sort(scoredCandidates.begin(), scoredCandidates.end(),
			          [](const auto& a, const auto& b) { return a.first > b.first; });

			// Select top K candidates
			for (size_t i = 0; i < scoredCandidates.size()
			     && int(goalCandidates.size()) < this->maxCandidateNum_; ++i)
			{
				goalCandidates.push_back(scoredCandidates[i].second);
			}

			// Fill to minCandidateNum with remaining pool entries (already in order)
			for (size_t i = goalCandidates.size(); i < scoredCandidates.size()
			     && int(goalCandidates.size()) < this->minCandidateNum_; ++i)
			{
				goalCandidates.push_back(scoredCandidates[i].second);
			}
		}
		else
		{
			// Fallback: original info-gain-only ordering (when value map unavailable)
			for (size_t i = 0; i < candidatePool.size()
			     && int(goalCandidates.size()) < this->maxCandidateNum_; ++i)
			{
				goalCandidates.push_back(candidatePool[i]);
			}
		}
	}
	bool DEP::findCandidatePath(const std::vector<std::shared_ptr<PRM::Node>>& goalCandidates, std::vector<std::vector<std::shared_ptr<PRM::Node>>>& candidatePaths){
		bool findPath = false;
		// find nearest node of current location
		std::shared_ptr<PRM::Node> currPos;
		currPos.reset(new PRM::Node (this->position_));
		std::shared_ptr<PRM::Node> start = this->roadmap_->nearestNeighbor(currPos);
		// if(!start){
		// 	ROS_ERROR("[DEP] findCandidatePath: nearestNeighbor returned null! currPos=(%.2f,%.2f,%.2f), roadmap size=%d, prmNodeVec_ size=%zu",
		// 	          this->position_(0), this->position_(1), this->position_(2),
		// 	          this->roadmap_->getSize(), this->prmNodeVec_.size());
		// }
		// else {
		// 	ROS_INFO("[DEP] findCandidatePath: start nearest PRM node=(%.2f,%.2f,%.2f), dist to curr=%.2f, roadmap size=%d, prmNodeVec_ size=%zu",
		// 	         start->pos(0), start->pos(1), start->pos(2),
		// 	         (start->pos - this->position_).norm(),
		// 	         this->roadmap_->getSize(), this->prmNodeVec_.size());
		// }

		// ROS_INFO("[DEP] findCandidatePath: goalCandidates count=%zu, current pos=(%.2f,%.2f,%.2f), curr height layer idx=%d, height=%.2f",
		//          goalCandidates.size(), this->position_(0), this->position_(1), this->position_(2),
		//          this->current_height_layer_idx_,
		//          this->height_layers_.empty() ? -1.0 : this->height_layers_[this->current_height_layer_idx_]);

		candidatePaths.clear();
		int astar_success_count = 0;
		int astar_fail_count = 0;
		for (std::shared_ptr<PRM::Node> goal : goalCandidates){
			std::vector<std::shared_ptr<PRM::Node>> path = PRM::AStar(this->roadmap_, start, goal, this->map_);
			if (int(path.size()) != 0){
				findPath = true;
				++astar_success_count;
				// ROS_INFO("[DEP] findCandidatePath: A* SUCCESS -> goal(%.2f,%.2f,%.2f) path_len=%zu, goal adjNodes=%zu",
				//          goal->pos(0), goal->pos(1), goal->pos(2), path.size(), goal->adjNodes.size());
			}
			else{
				++astar_fail_count;
				// ROS_WARN("[DEP] findCandidatePath: A* FAILED -> goal(%.2f,%.2f,%.2f) adjNodes=%zu, goal numVoxels=%d, start adjNodes=%zu",
				//          goal->pos(0), goal->pos(1), goal->pos(2), goal->adjNodes.size(),
				//          goal->numVoxels, start ? start->adjNodes.size() : 0);
				continue;
			}
			path.insert(path.begin(), currPos);
			std::vector<std::shared_ptr<PRM::Node>> pathSc;
			this->shortcutPath(path, pathSc);
			candidatePaths.push_back(pathSc);
		}
		ROS_INFO("[DEP] findCandidatePath: summary: total_candidates=%zu, astar_success=%d, astar_fail=%d, candidatePaths=%zu, findPath=%s",
		         goalCandidates.size(), astar_success_count, astar_fail_count, candidatePaths.size(), findPath ? "true" : "false");
		return findPath;
	}

	void DEP::findBestPath(const std::vector<std::vector<std::shared_ptr<PRM::Node>>>& candidatePaths, std::vector<std::shared_ptr<PRM::Node>>& bestPath){
		// find path highest unknown
		bestPath.clear();
		goalPathFind_ = false;  // reset each planning cycle
		double highestScore = -10000;
		for (int n=0; n<int(candidatePaths.size()); ++n){
			std::vector<std::shared_ptr<PRM::Node>> path = candidatePaths[n]; 
			// ROS_INFO("CandidatePath size is %d",path.size());
			if (int(path.size()) <= 1) continue;
			if(int(path.size()) == 2){
				if((path[0]->pos-path[1]->pos).norm() < 0.4){
					continue;
				}
			}
			double yawDist = 0;
			double prevYaw = this->currYaw_;
			int unknownVoxel = 0;
			bool skipPath = false;
			double semanticValue = 0.0;  // 累积路径上所有点的语义分数
			for (int i=0; i<int(path.size())-1; ++i){
				std::shared_ptr<PRM::Node> currNode = path[i];
				std::shared_ptr<PRM::Node> nextNode = path[i+1];
				Eigen::Vector3d diff = nextNode->pos - currNode->pos;
				double angle = atan2(diff(1), diff(0));
				//ROS_WARN("The angle is %f",angle);

				// reevaluate the unknowns for intermediate points
				std::unordered_map<double, int> yawNumVoxels;
				int unknownVoxelNum = this->calculateUnknown(currNode, yawNumVoxels);
				currNode->numVoxels = unknownVoxelNum;
				currNode->yawNumVoxels = yawNumVoxels;


				unknownVoxel += currNode->getUnknownVoxels(angle);
				double angleLine = globalPlanner::angleDiff(prevYaw, angle);
				//ROS_WARN("The angleLine is %f",angleLine);
				// if(abs(angleLine) > M_PI){
				// 	skipPath = true;
				// 	break;
				// }
				yawDist += angleLine;
				prevYaw = angle;

				// 累积中间节点的语义分数
				if (useValueMap_ && value_map_ && value_map_->isInitialized()) {
					Eigen::Vector2d pos2d(currNode->pos(0), currNode->pos(1));
					semanticValue += 0.3*value_map_->getValue(pos2d);
				}
			}
			// if(skipPath)continue;
			// reevaluate the goal node
			std::unordered_map<double, int> yawNumVoxels;
			int unknownVoxelNum = this->calculateUnknown(path.back(), yawNumVoxels);
			path.back()->numVoxels = unknownVoxelNum;
			path.back()->yawNumVoxels = yawNumVoxels;
			unknownVoxel += path.back()->getBestYawVoxel();
			yawDist += globalPlanner::angleDiff(prevYaw, path.back()->getBestYaw());

			double distance = this->calculatePathLength(path);
			// cout << "total is distance is: " << distance << " total yaw distance is: " << yawDist << " voxel: " << path.back()->numVoxels << endl;
			double progress = std::min(1.0, double(depPlanCount_) / double(distPenaltyRampCycles_));
			double dWeight = distPenaltyMax_ - progress * (distPenaltyMax_ - distPenaltyMin_);
			double effectiveYawWeight = this->yawPenaltyWeight_ * (1.0 + distance / this->yawDistScale_);
			double pathTime = dWeight * distance / this->vel_ + effectiveYawWeight * yawDist / this->angularVel_;
			
			// 累积目标节点的语义分数
			if (useValueMap_ && value_map_ && value_map_->isInitialized()) {
				Eigen::Vector2d pos2d(path.back()->pos(0), path.back()->pos(1));
				semanticValue += value_map_->getValue(pos2d);
			}
			ROS_INFO("PathTime = %f",pathTime);
			double score = double(unknownVoxel)/pathTime;
			ROS_INFO("Before score = %f",score);
			// Apply semantic value / pathTime: 与 info gain 一致，都是 per-unit-time 收益
			if (useValueMap_ && value_map_ && value_map_->isInitialized()) {
				double sv = semanticWeight_ * semanticValue / pathTime;
				score += sv;
				ROS_INFO("c %.2f (raw=%.3f, /pathTime=%.3f)", sv, semanticValue, semanticValue/pathTime);
			}
			// Frontier proximity bonus for the goal node
			if (this->frontierWeight_ > 1e-6 && !frontierPointPairs_.empty()) {
				double frontierBonus = this->frontierWeight_ * this->getFrontierProximity(path.back()->pos);
				score += frontierBonus;
			}
			// score = (score + unknownVoxel)/pathTime;
			// cout << "unknown for path: " << n <<  " is: " << unknownVoxel << " score: " << score << " distance: " << distance << " Time: " << pathTime <<  " Last total unknown: " << path.back()->numVoxels << " last best: " << path.back()->getBestYawVoxel() << endl;
			if (score > highestScore){
				highestScore = score;
				bestPath = path;
			}
		}
		if (highestScore == 0){
			cout << "[DEP]: Current score is 0. The exploration might complete." << endl;
		}
	}

	void DEP::odomCB(const nav_msgs::OdometryConstPtr& odom){
		this->odom_ = *odom;
		this->position_ = Eigen::Vector3d (this->odom_.pose.pose.position.x, this->odom_.pose.pose.position.y, this->odom_.pose.pose.position.z);
		this->currYaw_ = globalPlanner::rpy_from_quaternion(this->odom_.pose.pose.orientation);
		this->odomReceived_ = true;

		if (this->histTraj_.size() == 0){
			this->histTraj_.push_back(this->position_);
		}
		else{
			Eigen::Vector3d lastPos = this->histTraj_.back();
			double dist = (this->position_ - lastPos).norm();
			if (dist >= 0.5){
				if (this->histTraj_.size() >= 100){
					this->histTraj_.pop_front();
					this->histTraj_.push_back(this->position_);
				}
				else{
					this->histTraj_.push_back(this->position_);
				}
			}
		}
	}

	void DEP::itmScoreCB(const std_msgs::Float64ConstPtr& msg){
		latestItmScore_ = msg->data;
		itmScoreReceived_ = true;
	}

	void DEP::valuemapCB(const ros::TimerEvent&){
		// ---- ValueMap2D update with BLIP2 ITM score ----
		if (useValueMap_ && value_map_ && map_->is2DMapReady()) {
			value_map_->ensureInitialized();
			if (value_map_->isInitialized()) {
				// Use real BLIP2 ITM score; fall back to random if not yet received
				double itm_score;
				if (itmScoreReceived_) {
					itm_score = latestItmScore_;
				} else {
					// Fallback: random score until first real ITM score arrives
					itm_score = static_cast<double>(rand()) / RAND_MAX;
					ROS_WARN_THROTTLE(5.0, "[DEP]: No BLIP2 ITM score received yet, using random=%.3f",
					                  itm_score);
				}

				// Collect free grids visible in current FOV and update value map
				Eigen::Vector2d sensor_pos(this->position_(0), this->position_(1));
				std::vector<Eigen::Vector2i> free_grids;
				free_grids = map_->get2DFreeGrid();
				value_map_->updateValueMap(sensor_pos, this->currYaw_, free_grids, itm_score);

				ROS_DEBUG_THROTTLE(2.0, "[DEP]: ValueMap updated with ITM=%.3f (from %s), free_grids=%zu",
				                   itm_score,
				                   itmScoreReceived_ ? "BLIP2" : "random",
				                   free_grids.size());
			}
		}
	}

	void DEP::visCB(const ros::TimerEvent&){
		if (this->prmNodeVec_.size() != 0){
			this->publishRoadmap();
		}

		if (this->candidatePaths_.size() != 0){
			this->publishCandidatePaths();
		}

		if (this->bestPath_.size() != 0){
			this->publishBestPath();
		}

		if (this->frontierPointPairs_.size() != 0){
			this->publishFrontier();
		}

		// Publish value map heatmap (OccupancyGrid for RViz + OpenCV window)
		if (useValueMap_ && value_map_ && value_map_->isInitialized()) {
			this->publishValueMap();
		}
		if(!useValueMap_){
			ROS_ERROR("Do not choose to use value map !");
		}
		if(! value_map_){
			ROS_WARN_THROTTLE(5.0, "No value map yet (map_ not set?)");
		}
		else if(! value_map_->isInitialized()){
			ROS_WARN_THROTTLE(5.0, "Value map not initialized (2D map not ready?)");
		}
	}


	bool DEP::isPosValid(const Eigen::Vector3d& p){
		for (double x=p(0)-this->safeDistXY_; x<=p(0)+this->safeDistXY_; x+=this->map_->getRes()){
			for (double y=p(1)-this->safeDistXY_; y<=p(1)+this->safeDistXY_; y+=this->map_->getRes()){
				for (double z=p(2)-this->safeDistZ_; z<=p(2)+this->safeDistZ_; z+=this->map_->getRes()){
					if (this->safeDistCheckUnknown_){
						Eigen::Vector3i vec;
						map_->posToIndex(Eigen::Vector3d (x, y, z),vec);
						if (!this->map_->isKnownFree(vec)){
							return false;
						}
					}
					else{
						if (this->map_->isKnownOccupied(Eigen::Vector3d (x, y, z))){
							return false;
						}
					}
				}
			}
		}
		return true;			
	}

	bool DEP::isPosValid(const Eigen::Vector3d& p, double safeDistXY, double safeDistZ){
 		if (use2DMap_ && map_->is2DMapReady()) {
		// 只检查2D平面，忽略Z
		//ROS_WARN_THROTTLE(1.0,"USING the 2d prm check");
		for (double x = p(0)-safeDistXY; x <= p(0)+safeDistXY; x += map_->getRes()) {
			for (double y = p(1)-safeDistXY; y <= p(1)+safeDistXY; y += map_->getRes()) {
				//if(!map_->is2DFree(x,y))return false;
				if (!this->safeDistCheckUnknown_) {
					if (map_->is2DFree(x, y))  // 原逻辑：已知自由=无效（探索点要选未知区）
						return false;
				} else {
					if (map_->is2DOccupied(x, y))
						return false;
				}
			}
		}
       		return true;
    	}

	// fallback: 2D地图未就绪时使用3D检测
	for (double x=p(0)-safeDistXY; x<=p(0)+safeDistXY; x+=this->map_->getRes()){
		for (double y=p(1)-safeDistXY; y<=p(1)+safeDistXY; y+=this->map_->getRes()){
			for (double z=p(2)-safeDistZ; z<=p(2)+safeDistZ; z+=this->map_->getRes()){
				if (this->safeDistCheckUnknown_){
					Eigen::Vector3i vec;
					map_->posToIndex(Eigen::Vector3d (x, y, z),vec);
					if (this->map_->isKnownFree(vec)){
						return false;
					}
				}
				else{
					if (this->map_->isKnownOccupied(Eigen::Vector3d (x, y, z))){
						return false;
					}
				}
			}
		}
	}
	return true;
	}

	std::shared_ptr<PRM::Node> DEP::randomConfigBBox(const Eigen::Vector3d& minRegion, const Eigen::Vector3d& maxRegion){
		Eigen::Vector3d mapMinRegion, mapMaxRegion, minSampleRegion, maxSampleRegion;
		this->map_->getCurrMapRange(mapMinRegion, mapMaxRegion);
		// cout << "current map range is: " << mapMinRegion.transpose() << ", " << mapMaxRegion.transpose() << endl;
		minSampleRegion(0) = std::max(mapMinRegion(0), minRegion(0));
		minSampleRegion(1) = std::max(mapMinRegion(1), minRegion(1));
		minSampleRegion(2) = std::max(mapMinRegion(2), minRegion(2));
		maxSampleRegion(0) = std::min(mapMaxRegion(0), maxRegion(0));
		maxSampleRegion(1) = std::min(mapMaxRegion(1), maxRegion(1));
		maxSampleRegion(2) = std::min(mapMaxRegion(2), maxRegion(2));

		minSampleRegion(0) = std::max(minSampleRegion(0), this->globalRegionMin_(0));
		minSampleRegion(1) = std::max(minSampleRegion(1), this->globalRegionMin_(1));
		minSampleRegion(2) = std::max(minSampleRegion(2), this->globalRegionMin_(2));
		maxSampleRegion(0) = std::min(maxSampleRegion(0), this->globalRegionMax_(0));
		maxSampleRegion(1) = std::min(maxSampleRegion(1), this->globalRegionMax_(1));
		maxSampleRegion(2) = std::min(maxSampleRegion(2), this->globalRegionMax_(2));


		bool valid = false;
		int sampleAttempts = 0;
		const int maxSampleAttempts = 500;
		Eigen::Vector3d p;
		while (valid == false && sampleAttempts < maxSampleAttempts){	
			p(0) = globalPlanner::randomNumber(minSampleRegion(0), maxSampleRegion(0));
			p(1) = globalPlanner::randomNumber(minSampleRegion(1), maxSampleRegion(1));
			// Use only the current active height layer (switched by FSM signal)
			if (!this->height_layers_.empty()) {
				int idx = std::max(0, std::min(this->current_height_layer_idx_, (int)this->height_layers_.size() - 1));
				p(2) = this->height_layers_[idx];
			} else {
				p(2) = this->height_layers_[0];  // fallback
			}
			if (use2DMap_ && map_->is2DMapReady()) {
				valid = map_->is2DFree(p(0), p(1));
			} else {
				valid = this->isPosValid(p, this->safeDistXY_, this->safeDistZ_);
			}
			++sampleAttempts;
		}
		// If failed to find a valid point, clamp to current position as fallback
		if (!valid) {
			p = this->position_;
		}

		std::shared_ptr<PRM::Node> newNode (new PRM::Node(p));
		return newNode;
	}

	// __计算 PRM 节点 `n` 处，无人机传感器能看到多少未知体素（信息增益）__，同时记录每个偏航角能看到多少未知体素
	int DEP::calculateUnknown(const shared_ptr<PRM::Node>& n, std::unordered_map<double, int>& yawNumVoxels){
		for (double yaw : this->yaws_){
			yawNumVoxels[yaw] = 0;
		}
		// Position:
		Eigen::Vector3d p = n->pos;
		double h = this->position_(2);
		h = std::max(this->globalRegionMin_(2), std::min(h, this->globalRegionMax_(2)));
		//double zRange = this->dmax_ * tan(this->verticalFOV_/2.0);
		int countTotalUnknown = 0;
		if (use2DMap_ && map_->is2DMapReady()) {
			for (double y = p(1) - this->dmax_; y <= p(1)+ this->dmax_; y += this->map_->getRes()){
				for (double x = p(0) - this->dmax_; x <= p(0) + this->dmax_; x += this->map_->getRes()){
					Eigen::Vector3d nodePoint (x, y, h);
					if (nodePoint(0) < this->globalRegionMin_(0) or nodePoint(0) > this->globalRegionMax_(0) or
						nodePoint(1) < this->globalRegionMin_(1) or nodePoint(1) > this->globalRegionMax_(1) )
						{
						// not in global range
						continue;
					}

					if (map_->is2DUnknown(nodePoint(0),nodePoint(1)) && !map_->is2DOccupied(nodePoint(0),nodePoint(1))){
						if (this->sensorFOVCondition(nodePoint, p)){
							++countTotalUnknown;
							for (double yaw: this->yaws_){
								Eigen::Vector3d yawDirection (cos(yaw), sin(yaw), 0);
								Eigen::Vector3d direction = nodePoint - p;
								Eigen::Vector3d face (direction(0), direction(1), 0);
								double angleToYaw = angleBetweenVectors(face, yawDirection);
								if (angleToYaw <= this->horizontalFOV_/2){
									yawNumVoxels[yaw] += 1;
								}
							}
						}
					}
				}
			}
			return countTotalUnknown;
		}else{
			ROS_ERROR("No Blob !");
		return 0;
		}
	}

	double DEP::calculatePathLength(const std::vector<shared_ptr<PRM::Node>>& path){
		int idx1 = 0;
		double length = 0;
		for (size_t idx2=1; idx2<=path.size()-1; ++idx2){
			length += (path[idx2]->pos - path[idx1]->pos).norm();
			++idx1;
		}
		return length;
	}

	void DEP::shortcutPath(const std::vector<std::shared_ptr<PRM::Node>>& path, std::vector<std::shared_ptr<PRM::Node>>& pathSc){
		size_t ptr1 = 0; size_t ptr2 = 2;
		pathSc.push_back(path[ptr1]);

		if (path.size() == 1){
			return;
		}

		if (path.size() == 2){
			pathSc.push_back(path[1]);
			return;
		}

		while (ros::ok()){
			if (ptr2 > path.size()-1){
				break;
			}
			std::shared_ptr<PRM::Node> p1 = path[ptr1];
			std::shared_ptr<PRM::Node> p2 = path[ptr2];
			Eigen::Vector3d pos1 = p1->pos;
			Eigen::Vector3d pos2 = p2->pos;
			bool lineValidCheck;
			if (use2DMap_ && map_->is2DMapReady()) {
				lineValidCheck = !map_->is2DInflatedOccupiedLine2D(pos1, pos2);
			} else {
				lineValidCheck = this->map_->isInflatedFreeLine(pos1, pos2);
			}
			// double maxDistance = std::numeric_limits<double>::max();
			// double maxDistance = 3.0;
			// if (lineValidCheck and (pos1 - pos2).norm() <= maxDistance){
			if (lineValidCheck){
				if (ptr2 == path.size()-1){
					pathSc.push_back(p2);
					break;
				}
				++ptr2;
			}
			else{
				pathSc.push_back(path[ptr2-1]);
				if (ptr2 == path.size()-1){
					pathSc.push_back(p2);
					break;
				}
				ptr1 = ptr2-1;
				ptr2 = ptr1+2;
			}
		}		
	}

	int DEP::weightedSample(const std::vector<double>& weights){
		double total = std::accumulate(weights.begin(), weights.end(), 0.0);
		std::vector<double> normalizedWeights;

		for (const double weight : weights){
		 	normalizedWeights.push_back(weight/total);
		}

		std::random_device rd;
		std::mt19937 gen(rd());
		std::discrete_distribution<int> distribution(normalizedWeights.begin(), normalizedWeights.end());
		return distribution(gen);
	}


	std::shared_ptr<PRM::Node> DEP::sampleFrontierPoint(const std::vector<double>& sampleWeights){
		// choose the frontier region (random sample by frontier area) 
		int idx = weightedSample(sampleWeights);

		// sample a frontier point in the region
		Eigen::Vector3d frontierCenter = this->frontierPointPairs_[idx].first;
		double frontierSize = this->frontierPointPairs_[idx].second;
		double xmin = std::max(frontierCenter(0) - frontierSize/sqrt(2), this->globalRegionMin_(0));
		double xmax = std::min(frontierCenter(0) + frontierSize/sqrt(2), this->globalRegionMax_(0));
		double ymin = std::max(frontierCenter(1) - frontierSize/sqrt(2), this->globalRegionMin_(1));
		double ymax = std::min(frontierCenter(1) + frontierSize/sqrt(2), this->globalRegionMax_(1));
		// double zmin = frontierCenter(2);
		// double zmax = frontierCenter(2);
		Eigen::Vector3d frontierPoint;
		frontierPoint(0) = globalPlanner::randomNumber(xmin, xmax);
		frontierPoint(1) = globalPlanner::randomNumber(ymin, ymax);
		frontierPoint(2) = this->height_layers_[0];
		std::shared_ptr<PRM::Node> frontierNode (new PRM::Node(frontierPoint));
		return frontierNode;
	}

	std::shared_ptr<PRM::Node> DEP::extendNode(const std::shared_ptr<PRM::Node>& n, const std::shared_ptr<PRM::Node>& target){
		double extendDist = randomNumber(this->distThresh_, this->maxConnectDist_);
		Eigen::Vector3d p = n->pos + (target->pos - n->pos)/(target->pos - n->pos).norm() * extendDist;
		p(0) = std::max(this->globalRegionMin_(0), std::min(p(0), this->globalRegionMax_(0)));
		p(1) = std::max(this->globalRegionMin_(1), std::min(p(1), this->globalRegionMax_(1)));
		//p(2) = std::max(this->globalRegionMin_(2), std::min(p(2), this->globalRegionMax_(2)));
		p(2) = this->height_layers_[0];
		std::shared_ptr<PRM::Node> extendedNode (new PRM::Node(p));
		return extendedNode;
	}

	void DEP::publishRoadmap(){
		visualization_msgs::MarkerArray roadmapMarkers;

		// PRM nodes and edges
		int countPointNum = 0;
		int countEdgeNum = 0;
		int countVoxelNumText = 0;
		for (std::shared_ptr<PRM::Node> n : this->prmNodeVec_){
			// std::shared_ptr<PRM::Node> n = this->prmNodeVec_[i];

			// Node point
			visualization_msgs::Marker point;
			point.header.frame_id = "map";
			point.header.stamp = ros::Time::now();
			point.ns = "prm_point";
			point.id = countPointNum;
			point.type = visualization_msgs::Marker::SPHERE;
			point.action = visualization_msgs::Marker::ADD;
			point.pose.position.x = n->pos(0);
			point.pose.position.y = n->pos(1);
			point.pose.position.z = n->pos(2);
			point.lifetime = ros::Duration(0.5);
			point.scale.x = 0.1;
			point.scale.y = 0.1;
			point.scale.z = 0.1;
			point.color.a = 1.0;
			point.color.r = 1.0;
			point.color.g = 0.0;
			point.color.b = 0.0;
			++countPointNum;
			roadmapMarkers.markers.push_back(point);

			// number of voxels for each node
			visualization_msgs::Marker voxelNumText;
			voxelNumText.ns = "num_voxel_text";
			voxelNumText.header.frame_id = "map";
			voxelNumText.id = countVoxelNumText;
			voxelNumText.header.stamp = ros::Time::now();
			voxelNumText.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
			voxelNumText.action = visualization_msgs::Marker::ADD;
			voxelNumText.pose.position.x = n->pos(0);
			voxelNumText.pose.position.y = n->pos(1);
			voxelNumText.pose.position.z = n->pos(2)+0.1;
			voxelNumText.scale.x = 0.1;
			voxelNumText.scale.y = 0.1;
			voxelNumText.scale.z = 0.1;
			voxelNumText.color.a = 1.0;
			// 信息增益 + 语义增益 + 前沿增益 加权显示
			double displayScore = static_cast<double>(n->numVoxels);
			Eigen::Vector2d pos2d(n->pos(0), n->pos(1));
			if (useValueMap_ && value_map_ && value_map_->isInitialized()) {
				displayScore += semanticWeight_ * value_map_->getValue(pos2d);
			}
			displayScore += frontierWeight_ * getFrontierProximity(n->pos);
			char buf[16];
			snprintf(buf, sizeof(buf), "%.0f", displayScore);
			voxelNumText.text = buf;
			voxelNumText.lifetime = ros::Duration(0.5);
			++countVoxelNumText;
			roadmapMarkers.markers.push_back(voxelNumText);


			// Edges
			visualization_msgs::Marker line;
			line.ns = "edge";
			line.header.frame_id = "map";
			line.type = visualization_msgs::Marker::LINE_LIST;
			line.header.stamp = ros::Time::now();
			for (std::shared_ptr<PRM::Node> adjNode : n->adjNodes){
				geometry_msgs::Point p1, p2;
				p1.x = n->pos(0);
				p1.y = n->pos(1);
				p1.z = n->pos(2);
				p2.x = adjNode->pos(0);
				p2.y = adjNode->pos(1);
				p2.z = adjNode->pos(2);				
				line.points.push_back(p1);
				line.points.push_back(p2);
				line.id = countEdgeNum;
				line.scale.x = 0.05;
				line.scale.y = 0.05;
				line.scale.z = 0.05;
				line.color.r = 0.0;
				line.color.g = 1.0;
				line.color.b = 0.0;
				line.color.a = 1.0;
				line.lifetime = ros::Duration(0.5);
				++countEdgeNum;
				roadmapMarkers.markers.push_back(line);
			}
		}

		int countGoalCandidateNum = 0;
		for (size_t i=0; i<this->goalCandidates_.size(); ++i){
			std::shared_ptr<PRM::Node> n = this->goalCandidates_[i];

			// Goal candidates
			visualization_msgs::Marker goalCandidatePoint;
			goalCandidatePoint.ns = "goal_candidate";
			goalCandidatePoint.header.frame_id = "map";
			goalCandidatePoint.header.stamp = ros::Time::now();
			goalCandidatePoint.id = countGoalCandidateNum;
			goalCandidatePoint.type = visualization_msgs::Marker::SPHERE;
			goalCandidatePoint.action = visualization_msgs::Marker::ADD;
			goalCandidatePoint.pose.position.x = n->pos(0);
			goalCandidatePoint.pose.position.y = n->pos(1);
			goalCandidatePoint.pose.position.z = n->pos(2);
			goalCandidatePoint.lifetime = ros::Duration(0.5);
			goalCandidatePoint.scale.x = 0.2;
			goalCandidatePoint.scale.y = 0.2;
			goalCandidatePoint.scale.z = 0.2;
			goalCandidatePoint.color.a = 1.0;
			goalCandidatePoint.color.r = 1.0;
			goalCandidatePoint.color.g = 0.0;
			goalCandidatePoint.color.b = 1.0;
			++countGoalCandidateNum;
			roadmapMarkers.markers.push_back(goalCandidatePoint);
		}

		this->roadmapPub_.publish(roadmapMarkers);
	}

	void DEP::publishCandidatePaths(){
		visualization_msgs::MarkerArray candidatePathMarkers;
		int countNodeNum = 0;
		int countLineNum = 0;
		for (std::vector<std::shared_ptr<PRM::Node>>& path : this->candidatePaths_){
			for (size_t i=0; i<path.size(); ++i){
				std::shared_ptr<PRM::Node> n = path[i];
				visualization_msgs::Marker point;
				point.header.frame_id = "map";
				point.header.stamp = ros::Time::now();
				point.ns = "candidate_path_node";
				point.id = countNodeNum;
				point.type = visualization_msgs::Marker::SPHERE;
				point.action = visualization_msgs::Marker::ADD;
				point.pose.position.x = n->pos(0);
				point.pose.position.y = n->pos(1);
				point.pose.position.z = n->pos(2);
				point.lifetime = ros::Duration(0.1);
				point.scale.x = 0.15;
				point.scale.y = 0.15;
				point.scale.z = 0.15;
				point.color.a = 1.0;
				point.color.r = 1.0;
				point.color.g = 1.0;
				point.color.b = 0.0;
				++countNodeNum;
				candidatePathMarkers.markers.push_back(point);

				if (i<path.size()-1){
					std::shared_ptr<PRM::Node> nNext = path[i+1];
					visualization_msgs::Marker line;
					line.ns = "candidate_path";
					line.header.frame_id = "map";
					line.type = visualization_msgs::Marker::LINE_LIST;
					line.header.stamp = ros::Time::now();
					geometry_msgs::Point p1, p2;
					p1.x = n->pos(0);
					p1.y = n->pos(1);
					p1.z = n->pos(2);
					p2.x = nNext->pos(0);
					p2.y = nNext->pos(1);
					p2.z = nNext->pos(2);				
					line.points.push_back(p1);
					line.points.push_back(p2);
					line.id = countLineNum;
					line.scale.x = 0.1;
					line.scale.y = 0.1;
					line.scale.z = 0.1;
					line.color.r = 0.0;
					line.color.g = 0.0;
					line.color.b = 0.0;
					line.color.a = 1.0;
					line.lifetime = ros::Duration(0.5);
					++countLineNum;
					candidatePathMarkers.markers.push_back(line);				
				}
			}
		}
		this->candidatePathPub_.publish(candidatePathMarkers);		
	}
	
	void DEP::publishBestPath(){
		visualization_msgs::MarkerArray bestPathMarkers;
		int countNodeNum = 0;
		int countLineNum = 0;
		for (size_t i=0; i<this->bestPath_.size(); ++i){
			std::shared_ptr<PRM::Node> n = this->bestPath_[i];
			visualization_msgs::Marker point;
			point.header.frame_id = "map";
			point.header.stamp = ros::Time::now();
			point.ns = "best_path_node";
			point.id = countNodeNum;
			point.type = visualization_msgs::Marker::SPHERE;
			point.action = visualization_msgs::Marker::ADD;
			point.pose.position.x = n->pos(0);
			point.pose.position.y = n->pos(1);
			point.pose.position.z = n->pos(2);
			point.lifetime = ros::Duration(0.5);
			point.scale.x = 0.2;
			point.scale.y = 0.2;
			point.scale.z = 0.2;
			point.color.a = 1.0;
			point.color.r = 1.0;
			point.color.g = 1.0;
			point.color.b = 1.0;
			++countNodeNum;
			bestPathMarkers.markers.push_back(point);

			if (i<this->bestPath_.size()-1){
				std::shared_ptr<PRM::Node> nNext = this->bestPath_[i+1];
				visualization_msgs::Marker line;
				line.ns = "best_path";
				line.header.frame_id = "map";
				line.type = visualization_msgs::Marker::LINE_LIST;
				line.header.stamp = ros::Time::now();
				geometry_msgs::Point p1, p2;
				p1.x = n->pos(0);
				p1.y = n->pos(1);
				p1.z = n->pos(2);
				p2.x = nNext->pos(0);
				p2.y = nNext->pos(1);
				p2.z = nNext->pos(2);				
				line.points.push_back(p1);
				line.points.push_back(p2);
				line.id = countLineNum;
				line.scale.x = 0.2;
				line.scale.y = 0.2;
				line.scale.z = 0.2;
				line.color.r = 1.0;
				line.color.g = 0.0;
				line.color.b = 0.0;
				line.color.a = 1.0;
				line.lifetime = ros::Duration(0.5);
				++countLineNum;
				bestPathMarkers.markers.push_back(line);				
			}
		}
		this->bestPathPub_.publish(bestPathMarkers);		
	}

	void DEP::publishFrontier(){
		visualization_msgs::MarkerArray frontierMarkers;
		int frontierRangeCount = 0;
		for (int i=0; i<int(this->frontierPointPairs_.size()); ++i){
			visualization_msgs::Marker range;

			Eigen::Vector3d p = this->frontierPointPairs_[i].first;
			double dist = this->frontierPointPairs_[i].second;

			range.header.frame_id = "map";
			range.header.stamp = ros::Time::now();
			range.ns = "frontier range";
			range.id = frontierRangeCount;
			range.type = visualization_msgs::Marker::SPHERE;
			range.action = visualization_msgs::Marker::ADD;
			range.pose.position.x = p(0);
			range.pose.position.y = p(1);
			range.pose.position.z = p(2);
			range.lifetime = ros::Duration(0.5);
			range.scale.x = dist;
			range.scale.y = dist;
			range.scale.z = 0.1;
			range.color.a = 0.4;
			range.color.r = 0.0;
			range.color.g = 0.0;
			range.color.b = 1.0;
			++frontierRangeCount;
			frontierMarkers.markers.push_back(range);			
		}
		this->frontierVisPub_.publish(frontierMarkers);
	}

	void DEP::publishValueMap(){
		if (!value_map_) return;

		// 未初始化时发布空地图清空 RViz 残留
		if (!value_map_->isInitialized()) {
			nav_msgs::OccupancyGrid empty_grid;
			empty_grid.header.frame_id = "map";
			empty_grid.header.stamp = ros::Time::now();
			empty_grid.info.resolution = 0.05;
			empty_grid.info.width = 1;
			empty_grid.info.height = 1;
			empty_grid.info.origin.orientation.w = 1.0;
			empty_grid.data.resize(1, -1);
			this->valueMapPub_.publish(empty_grid);
			return;
		}

		int w = value_map_->getWidth();
		int h = value_map_->getHeight();
		double res = value_map_->getResolution();
		double ox = value_map_->getOriginX();
		double oy = value_map_->getOriginY();
		const auto& values = value_map_->getValueBuffer();

		// // ---- 1. Publish as OccupancyGrid for RViz ----
		nav_msgs::OccupancyGrid grid;
		grid.header.frame_id = "map";
		grid.header.stamp = ros::Time::now();
		grid.info.resolution = res;
		grid.info.width = w;
		grid.info.height = h;
		grid.info.origin.position.x = ox;
		grid.info.origin.position.y = oy;
		grid.info.origin.position.z = 0.0;
		grid.info.origin.orientation.w = 1.0;

		// 使用 transformITMScore 的理论范围作为固定色阶 (penalty_scale=-4.0, bonus_scale=3.0)
		double vmin = -ValueMap2D::penalty_scale;  // -4.0
		double vmax =  ValueMap2D::bonus_scale;     //  3.0
		if (vmax - vmin < 1e-6) { vmax = vmin + 1.0; }
		const auto& conf = value_map_->getConfidenceBuffer();

		grid.data.resize(w * h, -1);  // default: unknown
		for (int i = 0; i < w * h; ++i) {
			if (conf[i] > 1e-6) {
				// Normalize value to [0, 100] for OccupancyGrid display
				// double normalized = (values[i] - 0) / (10);
				double normalized = (values[i] - vmin) / (vmax - vmin);
				grid.data[i] = static_cast<int8_t>(normalized * 100.0);
			}
			// Cells with zero confidence remain -1 (unknown/transparent in RViz)
		}
		this->valueMapPub_.publish(grid);

		// ---- 2. OpenCV JET heatmap window ----
		// Create grayscale image from normalized values
		cv::Mat gray(h, w, CV_8UC1, cv::Scalar(0));
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				int idx = y * w + x;
				if (conf[idx] > 1e-6) {
					// double normalized = (values[idx] - 0) / (10);
					double normalized = (values[idx] - vmin) / (vmax - vmin);
					gray.at<uchar>(y, x) = static_cast<uchar>(normalized * 255.0);
				}
				// else remains 0 (black = no data)
			}
		}

		// Apply JET colormap: blue(low) -> green -> yellow -> red(high)
		cv::Mat colorMap;
		cv::applyColorMap(gray, colorMap, cv::COLORMAP_JET);

		// Mark no-data pixels as dark gray for clarity
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				int idx = y * w + x;
				if (conf[idx] <= 1e-6) {
					colorMap.at<cv::Vec3b>(y, x) = cv::Vec3b(40, 40, 40);  // dark gray
				}
			}
		}

		// Scale up for better visibility if the map is small
		double scale = std::max(1.0, 400.0 / std::max(w, h));
		cv::Mat display;
		cv::resize(colorMap, display, cv::Size(), scale, scale, cv::INTER_NEAREST);

		// Draw drone position marker on the heatmap
		int px = static_cast<int>((this->position_(0) - ox) / res * scale);
		int py = static_cast<int>((this->position_(1) - oy) / res * scale);
		if (px >= 0 && px < display.cols && py >= 0 && py < display.rows) {
			cv::circle(display, cv::Point(px, py), static_cast<int>(5 * scale),
			           cv::Scalar(255, 255, 255), 2);
			// Draw yaw direction line
			int dx = static_cast<int>(15 * scale * std::cos(this->currYaw_));
			int dy = static_cast<int>(15 * scale * std::sin(this->currYaw_));
			cv::arrowedLine(display, cv::Point(px, py),
			                cv::Point(px + dx, py + dy),
			                cv::Scalar(255, 255, 255), 2);
		}

		cv::imshow("Value Map Heatmap", display);
		cv::waitKey(1);
	}

}
