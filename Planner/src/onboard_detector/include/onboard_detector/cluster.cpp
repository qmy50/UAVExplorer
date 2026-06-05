/*
    FILE: cluster.cpp
    ------------------
    Standalone node that subscribes to yolo_detector/mask_image (BGR8 segmentation mask)
    and /iris_0/realsense/depth_camera/depth/image_raw (depth image), extracts 3D points
    from masked regions, clusters them using pcl::EuclideanClusterExtraction with KD-tree
    acceleration, and publishes cluster centers as visualization_msgs::MarkerArray.

    References:
    - yolo_detector.py for mask format, body_to_camera transform, depth handling
    - map_ros.cpp for EuclideanClusterExtraction + KD-tree clustering pattern
*/

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/search/kdtree.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>

#include <vector>
#include <cmath>

typedef pcl::PointXYZ Point3D;

class ClusterDetector {
private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;  // private namespace for params

    // Subscribers
    ros::Subscriber mask_sub_;
    ros::Subscriber depth_sub_;
    ros::Subscriber camera_info_sub_;
    ros::Subscriber pose_sub_;

    // Publishers
    ros::Publisher cluster_marker_pub_;
    ros::Publisher cluster_cloud_pub_;
    ros::Publisher cluster_pos_pub_;

    // Camera intrinsics (default from Realsense, updated by camera_info)
    double fx_, fy_, cx_, cy_;
    bool camera_info_received_;

    // Camera pose in map frame (computed from body pose + body_to_camera)
    Eigen::Vector3d camera_position_;
    Eigen::Matrix3d camera_orientation_;
    bool pose_received_;

    // Body to camera transform (from detector_param.yaml)
    // [0,  0,  1,  0.09,
    //  -1, 0,  0,  0.0 ,
    //   0, -1,  0,  0.095,
    //   0,  0,  0,  1.0]
    Eigen::Matrix4d body_to_camera_;

    // Latest sensor data
    cv::Mat mask_image_;
    cv::Mat depth_image_;
    bool mask_received_;
    bool depth_received_;
    ros::Time last_mask_stamp_;
    double max_mask_age_;  // seconds, skip processing if mask is too old

    // Parameters
    double depth_min_value_;      // meters
    double depth_max_value_;      // meters
    double depth_scale_factor_;   // 1000 for Intel Realsense (mm -> m)
    int skip_pixel_;              // downsample factor
    double cluster_tolerance_;    // Euclidean cluster tolerance (meters)
    int min_cluster_size_;        // min points per cluster
    double voxel_leaf_size_;      // voxel grid leaf size (meters)
    bool have_send_cluster_center_;

public:
    ClusterDetector()
        : nh_()
        , pnh_("~")
        , camera_info_received_(false)
        , pose_received_(false)
        , mask_received_(false)
        , depth_received_(false)
        , max_mask_age_(0.5) {

        // Default camera intrinsics (Realsense, overwritten by camera_info)
        fx_ = 608.08740234375;
        fy_ = 608.08740234375;
        cx_ = 317.48284912109375;
        cy_ = 234.11557006835938;

        camera_position_ = Eigen::Vector3d::Zero();
        camera_orientation_ = Eigen::Matrix3d::Identity();

        // Body to camera transform (same as yolo_detector.py and detector_param.yaml)
        body_to_camera_ <<  0.0,  0.0,  1.0,  0.09,
                           -1.0,  0.0,  0.0,  0.0,
                            0.0, -1.0,  0.0,  0.095,
                            0.0,  0.0,  0.0,  1.0;

        loadParams();

        // Subscribers
        mask_sub_ = nh_.subscribe("/yolo_detector/mask_image", 1,
                                  &ClusterDetector::maskCallback, this);
        depth_sub_ = nh_.subscribe("/iris_0/realsense/depth_camera/depth/image_raw", 1,
                                   &ClusterDetector::depthCallback, this);
        camera_info_sub_ = nh_.subscribe("/iris_0/realsense/depth_camera/color/camera_info", 1,
                                         &ClusterDetector::cameraInfoCallback, this);
        pose_sub_ = nh_.subscribe("/iris_0/mavros/vision_pose/pose", 1,
                                  &ClusterDetector::poseCallback, this);

        // Publishers
        cluster_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>(
            "cluster_centers_marker", 10);
        cluster_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(
            "cluster_cloud", 10);
        cluster_pos_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/cluster_target", 10);

        ROS_INFO("[ClusterDetector]: Initialized. Waiting for mask + depth + pose...");

        have_send_cluster_center_ = false;
    }

    void loadParams() {
        pnh_.param("depth_min_value", depth_min_value_, 0.5);
        pnh_.param("depth_max_value", depth_max_value_, 5.0);
        pnh_.param("depth_scale_factor", depth_scale_factor_, 1000.0);
        pnh_.param("skip_pixel", skip_pixel_, 2);
        pnh_.param("cluster_tolerance", cluster_tolerance_, 0.3);
        pnh_.param("min_cluster_size", min_cluster_size_, 100);
        pnh_.param("voxel_leaf_size", voxel_leaf_size_, 0.05);
        pnh_.param("max_mask_age", max_mask_age_, 1.5);

        ROS_INFO("[ClusterDetector]: Parameters loaded:");
        ROS_INFO("  depth_min_value = %.2f m", depth_min_value_);
        ROS_INFO("  depth_max_value = %.2f m", depth_max_value_);
        ROS_INFO("  depth_scale_factor = %.1f", depth_scale_factor_);
        ROS_INFO("  skip_pixel = %d", skip_pixel_);
        ROS_INFO("  cluster_tolerance = %.2f m", cluster_tolerance_);
        ROS_INFO("  min_cluster_size = %d", min_cluster_size_);
        ROS_INFO("  voxel_leaf_size = %.2f m", voxel_leaf_size_);
        ROS_INFO("  max_mask_age = %.2f s", max_mask_age_);
    }

    // ---- Callbacks ----

    void maskCallback(const sensor_msgs::ImageConstPtr& msg) {
        try {
            mask_image_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
            mask_received_ = true;
            last_mask_stamp_ = msg->header.stamp;
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("[ClusterDetector]: cv_bridge mask exception: %s", e.what());
        }
    }

    void depthCallback(const sensor_msgs::ImageConstPtr& msg) {
        // Handle both 32FC1 (meters) and 16UC1 (mm) depth encodings
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("[ClusterDetector]: cv_bridge depth exception: %s", e.what());
            return;
        }

        if (msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
            // Convert float32 (meters) -> uint16 (mm) for uniform processing
            cv_ptr->image.convertTo(depth_image_, CV_16UC1, depth_scale_factor_);
        } else if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
            depth_image_ = cv_ptr->image;
        } else {
            ROS_ERROR_THROTTLE(5.0, "[ClusterDetector]: Unsupported depth encoding: %s",
                               msg->encoding.c_str());
            return;
        }
        depth_received_ = true;

        // Process whenever new depth arrives (mask is typically at >= depth rate)
        processIfReady();
    }

    void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
        if (!camera_info_received_) {
            fx_ = msg->K[0];
            fy_ = msg->K[4];
            cx_ = msg->K[2];
            cy_ = msg->K[5];
            camera_info_received_ = true;
            ROS_INFO("[ClusterDetector]: Camera intrinsics from camera_info: "
                     "fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f", fx_, fy_, cx_, cy_);
        }
    }

    void poseCallback(const geometry_msgs::PoseStampedConstPtr& msg) {
        // Body position and orientation in map frame
        Eigen::Vector3d body_pos(msg->pose.position.x,
                                 msg->pose.position.y,
                                 msg->pose.position.z);
        Eigen::Quaterniond body_quat(msg->pose.orientation.w,
                                     msg->pose.orientation.x,
                                     msg->pose.orientation.y,
                                     msg->pose.orientation.z);
        Eigen::Matrix3d R_mb = body_quat.toRotationMatrix();  // body -> map

        // Apply body_to_camera transform: camera = body * body_to_camera
        Eigen::Matrix3d R_bc = body_to_camera_.block<3, 3>(0, 0);
        Eigen::Vector3d t_bc = body_to_camera_.block<3, 1>(0, 3);

        camera_orientation_ = R_mb * R_bc;          // camera -> map rotation
        camera_position_ = R_mb * t_bc + body_pos;  // camera position in map
        pose_received_ = true;
    }

    // ---- Core Processing ----

    void processIfReady() {
        if (!mask_received_ || !depth_received_) {
            return;
        }
        if (!pose_received_) {
            ROS_WARN_THROTTLE(2.0, "[ClusterDetector]: Waiting for pose...");
            return;
        }
        // Skip if mask is stale
        // ros::Duration mask_age = ros::Time::now() - last_mask_stamp_;
        // if (mask_age.toSec() > max_mask_age_) {
        //     ROS_WARN_THROTTLE(3.0, "[ClusterDetector]: Mask too old (%.2f s), skipping.",
        //                       mask_age.toSec());
        //     return;
        // }

        processData();
    }

    void processData() {
        if(have_send_cluster_center_){
            return;
        }
        ros::WallTime t_start = ros::WallTime::now();

        // Step 1: Extract 3D points from masked pixels in depth image
        pcl::PointCloud<Point3D>::Ptr raw_cloud(new pcl::PointCloud<Point3D>);
        extractMaskPoints(raw_cloud);

        if (raw_cloud->empty()) {
            ROS_DEBUG("[ClusterDetector]: No points extracted from mask.");
            publishEmptyMarkers();
            return;
        }

        ROS_DEBUG("[ClusterDetector]: Extracted %lu raw points from mask.", raw_cloud->size());

        // Step 2: Voxel grid downsampling for speed
        pcl::PointCloud<Point3D>::Ptr filtered_cloud(new pcl::PointCloud<Point3D>);
        pcl::VoxelGrid<Point3D> voxel;
        voxel.setInputCloud(raw_cloud);
        voxel.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
        voxel.filter(*filtered_cloud);

        ROS_DEBUG("[ClusterDetector]: After voxel filter: %lu points.", filtered_cloud->size());

        if (filtered_cloud->size() < static_cast<size_t>(min_cluster_size_)) {
            ROS_DEBUG("[ClusterDetector]: Too few points (%lu < %d) for clustering.",
                      filtered_cloud->size(), min_cluster_size_);
            publishEmptyMarkers();
            return;
        }

        // Step 3: Build KD-tree for accelerated neighbor search
        pcl::search::KdTree<Point3D>::Ptr kdtree(new pcl::search::KdTree<Point3D>);
        kdtree->setInputCloud(filtered_cloud);

        // Step 4: Euclidean Cluster Extraction (DBSCAN-like density clustering)
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<Point3D> ec;
        ec.setClusterTolerance(cluster_tolerance_);
        ec.setMinClusterSize(min_cluster_size_);
        ec.setMaxClusterSize(filtered_cloud->size());  // no upper limit
        ec.setSearchMethod(kdtree);
        ec.setInputCloud(filtered_cloud);
        ec.extract(cluster_indices);

        ROS_DEBUG("[ClusterDetector]: Found %lu clusters.", cluster_indices.size());

        // Step 5: Compute AABB nearest-face center for each cluster and collect clustered points
        std::vector<Eigen::Vector3d> centers;
        pcl::PointCloud<Point3D>::Ptr all_cluster_cloud(new pcl::PointCloud<Point3D>);

        for (size_t i = 0; i < cluster_indices.size(); ++i) {
            const auto& indices = cluster_indices[i].indices;
            // Compute AABB
            Eigen::Vector3d aabb_min(std::numeric_limits<double>::max(),
                                     std::numeric_limits<double>::max(),
                                     std::numeric_limits<double>::max());
            Eigen::Vector3d aabb_max(-std::numeric_limits<double>::max(),
                                     -std::numeric_limits<double>::max(),
                                     -std::numeric_limits<double>::max());
            for (int idx : indices) {
                const auto& pt = filtered_cloud->points[idx];
                aabb_min = aabb_min.cwiseMin(Eigen::Vector3d(pt.x, pt.y, pt.z));
                aabb_max = aabb_max.cwiseMax(Eigen::Vector3d(pt.x, pt.y, pt.z));
                all_cluster_cloud->push_back(pt);
            }
            Eigen::Vector3d aabb_center = (aabb_min + aabb_max) * 0.5;

            // Compute 6 face centers and find the nearest one to drone
            std::vector<Eigen::Vector3d> face_centers = {
                Eigen::Vector3d(aabb_min.x(), aabb_center.y(), aabb_center.z()),  // -X face
                Eigen::Vector3d(aabb_max.x(), aabb_center.y(), aabb_center.z()),  // +X face
                Eigen::Vector3d(aabb_center.x(), aabb_min.y(), aabb_center.z()),  // -Y face
                Eigen::Vector3d(aabb_center.x(), aabb_max.y(), aabb_center.z()),  // +Y face
                Eigen::Vector3d(aabb_center.x(), aabb_center.y(), aabb_min.z()),  // -Z face
                Eigen::Vector3d(aabb_center.x(), aabb_center.y(), aabb_max.z())   // +Z face
            };
            double min_dist = std::numeric_limits<double>::max();
            Eigen::Vector3d nearest_face_center = aabb_center;  // fallback
            for (const auto& fc : face_centers) {
                double d = (fc - camera_position_).norm();
                if (d < min_dist) {
                    min_dist = d;
                    nearest_face_center = fc;
                }
            }
            centers.push_back(nearest_face_center);

            ROS_DEBUG("[ClusterDetector]:   Cluster %lu: nearest_face=(%.2f,%.2f,%.2f), aabb=[%.2f~%.2f, %.2f~%.2f, %.2f~%.2f], size=%lu",
                      i, nearest_face_center.x(), nearest_face_center.y(), nearest_face_center.z(),
                      aabb_min.x(), aabb_max.x(), aabb_min.y(), aabb_max.y(), aabb_min.z(), aabb_max.z(),
                      indices.size());
        }

        // Step 6: Publish results
        publishClusterMarkers(centers);
        have_send_cluster_center_ = true;
        publishClusterCloud(all_cluster_cloud);

        double elapsed = (ros::WallTime::now() - t_start).toSec() * 1000.0;
        ROS_DEBUG("[ClusterDetector]: Processing time: %.1f ms, %lu clusters.",
                  elapsed, centers.size());
    }

    // ---- Point Extraction ----

    void extractMaskPoints(pcl::PointCloud<Point3D>::Ptr cloud) {
        int H = mask_image_.rows;
        int W = mask_image_.cols;

        if (H == 0 || W == 0) return;

        // Ensure depth image matches mask dimensions
        if (depth_image_.rows != H || depth_image_.cols != W) {
            ROS_WARN_THROTTLE(10.0, "[ClusterDetector]: Depth size (%d,%d) != mask size (%d,%d), resizing.",
                              depth_image_.rows, depth_image_.cols, H, W);
            cv::resize(depth_image_, depth_image_, cv::Size(W, H), 0, 0, cv::INTER_NEAREST);
        }

        double inv_fx = 1.0 / fx_;
        double inv_fy = 1.0 / fy_;
        double inv_scale = 1.0 / depth_scale_factor_;

        // Pre-compute camera pose matrices for fast transform
        Eigen::Matrix3d R = camera_orientation_;
        Eigen::Vector3d t = camera_position_;

        cloud->reserve((H / skip_pixel_) * (W / skip_pixel_) / 4);  // rough estimate

        for (int v = 0; v < H; v += skip_pixel_) {
            const cv::Vec3b* mask_row = mask_image_.ptr<cv::Vec3b>(v);
            const uint16_t* depth_row = depth_image_.ptr<uint16_t>(v);
            for (int u = 0; u < W; u += skip_pixel_) {
                // Check if pixel belongs to a mask (non-black pixel)
                const cv::Vec3b& pixel = mask_row[u];
                if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0) {
                    continue;  // background, skip
                }

                // Get depth value and convert to meters
                double depth_m = static_cast<double>(depth_row[u]) * inv_scale;

                // Filter depth range
                if (depth_m < depth_min_value_ || depth_m > depth_max_value_) {
                    continue;
                }

                // Pixel -> camera coordinates
                double x_cam = (static_cast<double>(u) - cx_) * depth_m * inv_fx;
                double y_cam = (static_cast<double>(v) - cy_) * depth_m * inv_fy;
                double z_cam = depth_m;

                // Camera -> map coordinates
                // pt_map = R * pt_cam + t
                Point3D pt;
                pt.x = R(0, 0) * x_cam + R(0, 1) * y_cam + R(0, 2) * z_cam + t(0);
                pt.y = R(1, 0) * x_cam + R(1, 1) * y_cam + R(1, 2) * z_cam + t(1);
                pt.z = R(2, 0) * x_cam + R(2, 1) * y_cam + R(2, 2) * z_cam + t(2);

                cloud->push_back(pt);
            }
        }
    }

    // ---- Publishers ----

    void publishClusterMarkers(const std::vector<Eigen::Vector3d>& centers) {
        visualization_msgs::MarkerArray marker_array;

        // First, clear all previous markers
        if (centers.empty()) {
            visualization_msgs::Marker clear;
            clear.action = visualization_msgs::Marker::DELETEALL;
            marker_array.markers.push_back(clear);
            cluster_marker_pub_.publish(marker_array);
            return;
        }

        for (size_t i = 0; i < centers.size(); ++i) {
            visualization_msgs::Marker marker;
            marker.header.frame_id = "map";
            marker.header.stamp = ros::Time::now();
            marker.ns = "cluster_centers";
            marker.id = static_cast<int>(i);
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;
            marker.pose.position.x = centers[i].x();
            marker.pose.position.y = centers[i].y();
            marker.pose.position.z = centers[i].z();
            marker.pose.orientation.w = 1.0;
            marker.scale.x = 0.2;
            marker.scale.y = 0.2;
            marker.scale.z = 0.2;
            marker.color.a = 1.0;
            marker.color.r = 1.0;
            marker.color.g = 0.0;
            marker.color.b = 0.0;
            marker.lifetime = ros::Duration(0.5);  // auto-expire after 0.5s
            marker_array.markers.push_back(marker);

            // Also publish as PoseStamped (for clicked_point subscriber / cluster target)
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.frame_id = "map";
            pose_stamped.header.stamp = ros::Time::now();
            pose_stamped.pose.position.x = centers[i].x();
            pose_stamped.pose.position.y = centers[i].y();
            pose_stamped.pose.position.z = centers[i].z();
            pose_stamped.pose.orientation.w = 1.0;
            cluster_pos_pub_.publish(pose_stamped);
        }

        cluster_marker_pub_.publish(marker_array);
    }

    void publishEmptyMarkers() {
        visualization_msgs::MarkerArray marker_array;
        visualization_msgs::Marker clear;
        clear.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(clear);
        cluster_marker_pub_.publish(marker_array);
    }

    void publishClusterCloud(const pcl::PointCloud<Point3D>::Ptr& cloud) {
        if (cloud->empty()) return;
        sensor_msgs::PointCloud2 msg;
        pcl::toROSMsg(*cloud, msg);
        msg.header.frame_id = "map";
        msg.header.stamp = ros::Time::now();
        cluster_cloud_pub_.publish(msg);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "cluster_detector");
    ClusterDetector detector;
    ros::spin();
    return 0;
}