#!/usr/bin/env python

#!/usr/bin/env python
import rospy
import cv2
import numpy as np
import os
import threading
from sensor_msgs.msg import Image
import ros_numpy
from ultralytics import YOLO

# 路径与参数
path_curr = os.path.dirname(__file__)
img_topic = "/iris_0/realsense/depth_camera/color/image_raw"
weight = "weights/yolo26s-seg.pt"
class_names = "config/coco.names"
thresh = 0.5

class yolo_detector:
    def __init__(self):
        print("[onboardDetector]: yolo detector init (only RGB + mask)")

        self.img_received = False
        self.img_detected = False
        self.lock = threading.Lock()

        # 加载YOLO分割模型
        self.model = YOLO(os.path.join(path_curr, weight))

        # 加载类别名称
        self.LABEL_NAMES = []
        with open(os.path.join(path_curr, class_names), 'r') as f:
            for line in f.readlines():
                self.LABEL_NAMES.append(line.strip())

        # 订阅RGB图像
        self.img_sub = rospy.Subscriber(img_topic, Image, self.image_callback)

        # 只发布掩码图像
        self.mask_pub = rospy.Publisher("/yolo_detector/mask_image", Image, queue_size=10)

        # 定时器：检测与可视化（检测降低到10Hz，避免YOLO推理积压）
        rospy.Timer(rospy.Duration(0.1), self.detect_callback)
        rospy.Timer(rospy.Duration(0.05), self.vis_callback)

    def image_callback(self, msg):
        self.img = ros_numpy.numpify(msg)
        self.img_received = True

    def detect_callback(self, event):
        if self.img_received:
            try:
                output = self.inference(self.img)
                mask = self.postprocess(self.img, output)
                with self.lock:
                    self.mask_img = mask
                    self.img_detected = True
            except Exception as e:
                rospy.logerr("[yolo_detector] detect_callback error: %s", str(e))

    def vis_callback(self, event):
        try:
            with self.lock:
                if not self.img_detected:
                    return
                mask_copy = self.mask_img.copy()
            # 发布掩码图像
            self.mask_pub.publish(ros_numpy.msgify(Image, mask_copy, encoding='bgr8'))
            # 可选：本地显示掩码
            cv2.imshow("segmentation_mask", mask_copy)
            cv2.waitKey(1)
        except Exception as e:
            rospy.logerr("[yolo_detector] vis_callback error: %s", str(e))

    def inference(self, ori_img):
        results = self.model(ori_img, conf=thresh, verbose=False)
        return results

    def postprocess(self, ori_img, results):
        H, W, _ = ori_img.shape
        # 创建纯黑背景的掩码图（BGR格式）
        mask_img = np.zeros_like(ori_img)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            num_boxes = len(boxes)
            has_masks = result.masks is not None and len(result.masks) == num_boxes

            for i in range(num_boxes):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                category_id = int(box.cls[0].cpu().numpy())
                category = self.LABEL_NAMES[category_id]

                # 为每个类别生成固定颜色（保证同一类别颜色一致）
                rng = np.random.RandomState(category_id)
                color = rng.randint(50, 255, (3,), dtype=np.uint8)
                color_list = color.tolist()

                if has_masks and category == "car":
                    mask_data = result.masks.data[i].cpu().numpy()  # shape: (h, w)
                    mask_resized = cv2.resize(mask_data, (W, H), interpolation=cv2.INTER_NEAREST)
                    mask_bool = mask_resized > 0.5

                    # 半透明填充掩码区域
                    mask_img[mask_bool] = (mask_img[mask_bool] * 0.3 +
                                           np.array(color_list) * 0.7).astype(np.uint8)

                    # 绘制白色轮廓
                    contours, _ = cv2.findContours(mask_bool.astype(np.uint8),
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(mask_img, contours, -1, (255, 255, 255), 2)

                    # 在掩码图上添加类别标签
                    cv2.putText(mask_img, category, (x1, y1 - 5), 0, 0.7, color_list, 2)

        return mask_img

if __name__ == "__main__":
    rospy.init_node("yolo_detector")
    detector = yolo_detector()
    rospy.spin()




# import rospy
# import cv2
# import numpy as np
# import os
# import std_msgs
# from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
# from sensor_msgs import point_cloud2
# from vision_msgs.msg import Detection2DArray, Detection2D
# from nav_msgs.msg import Odometry
# import ros_numpy
# from ultralytics import YOLO
# from geometry_msgs.msg import PoseStamped
# from visualization_msgs.msg import Marker, MarkerArray
# from sklearn.cluster import DBSCAN

# target_classes = ["person"]

# path_curr = os.path.dirname(__file__)
# img_topic = "/iris_0/realsense/depth_camera/color/image_raw"
# depth_topic = "/iris_0/realsense/depth_camera/depth/image_raw"
# camera_info_topic = "/iris_0/realsense/depth_camera/color/camera_info"
# pose_topic = "/iris_0/mavros/vision_pose/pose"
# weight = "weights/yolo26n-seg.pt"
# class_names = "config/coco.names"
# thresh = 0.5

# # Depth parameters (consistent with detector_param.yaml)
# depth_scale_factor = 1000.0  # 1000 for Intel Realsense
# depth_min_value = 0.5       # meters
# depth_max_value = 8.0       # meters
# depth_skip_pixel = 2        # downsample factor for point cloud

# # Body to camera transform (from detector_param.yaml)
# # [0,  0,  1,  0.09,
# #  -1, 0,  0,  0.0 ,
# #   0, -1,  0,  0.095,
# #   0,  0,  0,  1.0]
# body_to_camera = np.array([
#     [0.0,  0.0,  1.0,  0.09],
#     [-1.0, 0.0,  0.0,  0.0 ],
#     [0.0, -1.0,  0.0,  0.095],
#     [0.0,  0.0,  0.0,  1.0 ]
# ])

# class yolo_detector:
#     def __init__(self):
#         print("[onboardDetector]: yolo detector init...")

#         self.img_received = False
#         self.depth_received = False
#         self.img_detected = False

#         # Camera intrinsics (will be overwritten by camera_info callback if available)
#         self.fx = 608.08740234375
#         self.fy = 608.08740234375
#         self.cx = 317.48284912109375
#         self.cy = 234.11557006835938
#         self.camera_info_received = False

#         # Camera pose (from odom)
#         self.camera_position = np.zeros(3)      # position in map frame
#         self.camera_orientation = np.eye(3)      # rotation matrix (body frame in map)
#         self.odom_received = False

#         # DBSCAN parameters (tunable via ROS param)
#         self.dbscan_eps = rospy.get_param("~dbscan_eps", 0.3)          # meters
#         self.dbscan_min_samples = rospy.get_param("~dbscan_min_samples", 300)

#         # load ultralytics YOLO segmentation model
#         self.model = YOLO(os.path.join(path_curr, weight))

#         # load class names
#         self.LABEL_NAMES = []
#         with open(os.path.join(path_curr, class_names), 'r') as f:
#             for line in f.readlines():
#                 self.LABEL_NAMES.append(line.strip())

#         # subscriber
#         self.img_sub = rospy.Subscriber(img_topic, Image, self.image_callback)
#         self.depth_sub = rospy.Subscriber(depth_topic, Image, self.depth_callback)
#         self.camera_info_sub = rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_callback)
#         self.pose_sub = rospy.Subscriber(pose_topic, PoseStamped, self.pose_callback)

#         # publisher
#         self.img_pub = rospy.Publisher("yolo_detector/detected_image", Image, queue_size=10)
#         self.mask_pub = rospy.Publisher("yolo_detector/mask_image", Image, queue_size=10)
#         self.bbox_pub = rospy.Publisher("yolo_detector/detected_bounding_boxes", Detection2DArray, queue_size=10)
#         self.time_pub = rospy.Publisher("yolo_detector/yolo_time", std_msgs.msg.Float64, queue_size=1)
#         self.mask_cloud_pub = rospy.Publisher("yolo_detector/mask_pointcloud", PointCloud2, queue_size=10)
#         # new publisher for cluster centers as markers
#         self.cluster_marker_pub = rospy.Publisher("yolo_detector/cluster_centers", MarkerArray, queue_size=10)

#         # timer
#         rospy.Timer(rospy.Duration(0.033), self.detect_callback)
#         rospy.Timer(rospy.Duration(0.033), self.vis_callback)
#         rospy.Timer(rospy.Duration(0.033), self.bbox_callback)
    
#     def image_callback(self, msg):
#         self.img = ros_numpy.numpify(msg)
#         self.img_received = True

#     def depth_callback(self, msg):
#         # depth image: typically 16UC1 (mm) or 32FC1 (meters)
#         depth_img = ros_numpy.numpify(msg)
#         if depth_img.dtype == np.float32:
#             # convert float32 (meters) to uint16 (mm) for uniform processing
#             depth_img = (depth_img * depth_scale_factor).astype(np.uint16)
#         self.depth_img = depth_img
#         self.depth_received = True

#     def camera_info_callback(self, msg):
#         if not self.camera_info_received:
#             self.fx = msg.K[0]
#             self.fy = msg.K[4]
#             self.cx = msg.K[2]
#             self.cy = msg.K[5]
#             self.camera_info_received = True
#             print("[onboardDetector]: Camera intrinsics updated from camera_info: fx={}, fy={}, cx={}, cy={}".format(
#                 self.fx, self.fy, self.cx, self.cy))

#     def pose_callback(self, msg):
#         """从 pose 话题获取机体在 map 坐标系下的位姿，再通过 body_to_camera 计算相机位姿"""
#         # 机体在 map 下的位置和姿态
#         x = msg.pose.position.x
#         y = msg.pose.position.y
#         z = msg.pose.position.z

#         qx = msg.pose.orientation.x
#         qy = msg.pose.orientation.y
#         qz = msg.pose.orientation.z
#         qw = msg.pose.orientation.w

#         # body 在 map 下的位姿
#         R_mb = self.quaternion_to_rotation_matrix(qw, qx, qy, qz)  # body to map rotation
#         t_mb = np.array([x, y, z])                                  # body position in map

#         # body_to_camera: 将 body 坐标变换到 camera 坐标
#         # T_map_cam = T_map_body @ T_body_cam
#         R_bc = body_to_camera[:3, :3]   # body to camera rotation
#         t_bc = body_to_camera[:3, 3]    # body to camera translation

#         # 相机在 map 下的位姿
#         self.camera_orientation = R_mb @ R_bc                  # camera to map rotation
#         self.camera_position = R_mb @ t_bc + t_mb              # camera position in map
#         self.odom_received = True

#     @staticmethod
#     def quaternion_to_rotation_matrix(w, x, y, z):
#         """四元数转3x3旋转矩阵"""
#         R = np.array([
#             [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
#             [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
#             [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
#         ])
#         return R

#     def detect_callback(self, event):
#         startTime = rospy.Time.now()
#         if self.img_received:
#             output = self.inference(self.img)
#             depth = self.depth_img if self.depth_received else None
#             self.detected_img, self.mask_img, self.detected_bboxes, self.mask_pointcloud = self.postprocess(self.img, output, depth)
#             self.img_detected = True
#             # rospy.logwarn_throttle(1.0,"We have set the img_detected true !!!")
#         endTime = rospy.Time.now()
#         self.time_pub.publish((endTime-startTime).to_sec())

#     def vis_callback(self, event):
#         if self.img_detected:
#             # self.img_pub.publish(ros_numpy.msgify(self.detected_img, "bgr8"))
#             self.mask_pub.publish(ros_numpy.msgify(Image, self.mask_img, encoding='bgr8'))

#             # 单独显示分割掩码图
#             cv2.imshow("segmentation_mask", self.mask_img)
#             cv2.waitKey(1)

#     def bbox_callback(self, event):
#         if self.img_detected:
#             bboxes_msg = Detection2DArray()
#             for detected_box in self.detected_bboxes:
#                 if detected_box[4] in target_classes:
#                     bbox_msg = Detection2D()
#                     bbox_msg.bbox.center.x = int(detected_box[0])
#                     bbox_msg.bbox.center.y = int(detected_box[1])
#                     bbox_msg.bbox.size_x = abs(detected_box[2] - detected_box[0]) 
#                     bbox_msg.bbox.size_y = abs(detected_box[3] - detected_box[1])

#                     bboxes_msg.detections.append(bbox_msg)
#                 bboxes_msg.header.stamp = rospy.Time.now()
#             self.bbox_pub.publish(bboxes_msg)

#         # publish mask point cloud
#         if self.img_detected and self.mask_pointcloud is not None:
#             # rospy.logwarn_throttle(1.0,"We have send the pcl2 msg !!!")
#             self.mask_cloud_pub.publish(self.mask_pointcloud)

#     def inference(self, ori_img):
#         # use ultralytics API for inference
#         results = self.model(ori_img, conf=thresh, verbose=False)
#         return results

#     def postprocess(self, ori_img, results, depth_img=None):
#         H, W, _ = ori_img.shape

#         detected_boxes = []
#         # 创建纯黑背景的掩码图（单独显示分割掩码用）
#         mask_img = np.zeros_like(ori_img)
#         # 存储所有掩码区域的点云（map坐标系）
#         all_mask_points = []

#         for result in results:
#             boxes = result.boxes
#             if boxes is None:
#                 continue

#             num_boxes = len(boxes)
#             has_masks = result.masks is not None and len(result.masks) == num_boxes

#             for i in range(num_boxes):
#                 box = boxes[i]
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
#                 obj_score = box.conf[0].cpu().numpy()
#                 category_id = int(box.cls[0].cpu().numpy())
#                 category = self.LABEL_NAMES[category_id]

#                 detected_box = [x1, y1, x2, y2, category]
#                 detected_boxes.append(detected_box)

#                 # draw bounding box on detected image
#                 cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
#                 cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)  
#                 cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

#                 # generate a fixed random color for this mask (seed by category for consistency)
#                 rng = np.random.RandomState(category_id)
#                 color = rng.randint(50, 255, (3,), dtype=np.uint8)
#                 color_list = color.tolist()

#                 if has_masks:
#                     # 使用 masks.data 获取第 i 个掩码
#                     mask_data = result.masks.data[i].cpu().numpy()  # shape: (h, w)
#                     # resize mask to original image size
#                     mask_resized = cv2.resize(mask_data, (W, H), interpolation=cv2.INTER_NEAREST)
#                     mask_bool = mask_resized > 0.5

#                     # 在 mask_img 上绘制半透明掩码（纯色填充）
#                     mask_img[mask_bool] = (mask_img[mask_bool] * 0.3 + 
#                                            np.array(color_list) * 0.7).astype(np.uint8)

#                     # draw mask contour on mask_img
#                     contours, _ = cv2.findContours(
#                         mask_bool.astype(np.uint8), 
#                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#                     )
#                     cv2.drawContours(mask_img, contours, -1, (255, 255, 255), 2)

#                     # draw mask contour on detected image
#                     cv2.drawContours(ori_img, contours, -1, color_list, 2)

#                     # add label on mask image
#                     cv2.putText(mask_img, category, (x1, y1 - 5), 0, 0.7, color_list, 2)

#                     # ===== 提取掩码区域的点云 =====
#                     if depth_img is not None:
#                         # rospy.loginfo_throttle(1.0,"We got the depth_img !!!")
#                         mask_points = self.extract_mask_pointcloud(
#                             mask_bool, depth_img, category, category_id, color
#                         )
#                         all_mask_points.extend(mask_points)

#         # ========= 新增 DBSCAN 聚类并发布 Marker 中心 =========
#         if len(all_mask_points) > 0:
#             # 提取坐标点 (x, y, z)
#             pts_xyz = np.array([[p[0], p[1], p[2]] for p in all_mask_points])  # (N,3)
#             if pts_xyz.shape[0] > self.dbscan_min_samples:
#                 clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples,algorithm='kd_tree').fit(pts_xyz)
#                 labels = clustering.labels_
#                 n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

#                 # 计算每个簇的中心（均值）
#                 cluster_centers = []
#                 for k in range(n_clusters):
#                     mask = (labels == k)
#                     cluster_points = pts_xyz[mask]
#                     center = np.mean(cluster_points, axis=0)   # (3,)
#                     cluster_centers.append(center)

#                 # 发布 MarkerArray
#                 self.publish_cluster_markers(cluster_centers)

#                 # 打印调试信息
#                 rospy.logdebug_throttle(2.0, "DBSCAN: %d clusters found, %d noise points", n_clusters, np.sum(labels == -1))
#             else:
#                 rospy.logdebug_throttle(5.0, "Too few points for DBSCAN: %d", pts_xyz.shape[0])
#         else:
#             # 无点云时发布空数组
#             self.publish_cluster_markers([])
#         # ====================================================

#         # 构建 PointCloud2 消息（map坐标系）
#         cloud_msg = self.create_pointcloud2_msg(all_mask_points) if all_mask_points else None
#         if cloud_msg is None:
#             rospy.logdebug_throttle(1.0,"cloud_msg is none")
#         # rospy.logerr_throttle(1.0,"We have create the cloud msg !!!")

#         return ori_img, mask_img, detected_boxes, cloud_msg

#     def extract_mask_pointcloud(self, mask_bool, depth_img, category, category_id, color):
#         """
#         从掩码区域和深度图中提取3D点云，并转换到map坐标系
        
#         Args:
#             mask_bool: (H, W) bool数组，True表示掩码区域
#             depth_img: (H, W) uint16深度图，单位mm
#             category: 类别名称
#             category_id: 类别ID
#             color: RGB颜色 (numpy array, 3,)
        
#         Returns:
#             list of (x, y, z, rgb) 点云数据（map坐标系）
#         """
#         H, W = mask_bool.shape
        
#         # 确保深度图和掩码尺寸一致
#         if depth_img.shape[:2] != (H, W):
#             depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)

#         inv_fx = 1.0 / self.fx
#         inv_fy = 1.0 / self.fy
#         inv_scale = 1.0 / depth_scale_factor

#         # 按skip_pixel下采样，减少点云数量
#         v_indices = np.arange(0, H, depth_skip_pixel)
#         u_indices = np.arange(0, W, depth_skip_pixel)
#         uu, vv = np.meshgrid(u_indices, v_indices)
        
#         # 只取掩码区域的像素
#         mask_sub = mask_bool[vv, uu]
#         vv_mask = vv[mask_sub]
#         uu_mask = uu[mask_sub]

#         # 获取深度值
#         depth_values = depth_img[vv_mask, uu_mask].astype(np.float64) * inv_scale  # 转换为米

#         # 过滤无效深度
#         valid = (depth_values >= depth_min_value) & (depth_values <= depth_max_value)
#         vv_valid = vv_mask[valid]
#         uu_valid = uu_mask[valid]
#         depth_valid = depth_values[valid]

#         if len(depth_valid) == 0:
#             rospy.logwarn_throttle(2.0, "no valid depth in mask")
#             return []

#         # 像素坐标 → 相机坐标系3D点
#         # 相机光轴坐标系: x向右, y向下, z向前(深度方向)
#         x_cam = (uu_valid - self.cx) * depth_valid * inv_fx
#         y_cam = (vv_valid - self.cy) * depth_valid * inv_fy
#         z_cam = depth_valid

#         # 相机坐标系点云 (3, N)
#         pts_cam = np.stack([x_cam, y_cam, z_cam], axis=0)  # (3, N)

#         # ===== 变换到 map 坐标系 =====
#         # pose_callback 中已通过 body_to_camera 将 body 位姿转换为 camera 位姿:
#         #   camera_orientation = R_mb @ R_bc  (camera to map rotation)
#         #   camera_position = R_mb @ t_bc + t_mb  (camera position in map)
#         # 因此直接: pts_map = camera_orientation * pts_cam + camera_position
#         # 与 dynamicDetector.cpp 中 currPointMap = orientation_ * currPointCam + position_ 一致
#         R_mc = self.camera_orientation    # camera to map rotation
#         t_mc = self.camera_position       # camera position in map

#         # 批量变换 (3, N)
#         pts_map = R_mc @ pts_cam + t_mc.reshape(3, 1)

#         # 打包为 (x, y, z, rgb) 格式
#         rgb_int = self.pack_rgb(int(color[2]), int(color[1]), int(color[0]))  # BGR → RGB

#         points = []
#         for j in range(pts_map.shape[1]):
#             points.append((pts_map[0, j], pts_map[1, j], pts_map[2, j], rgb_int))

#         return points

#     @staticmethod
#     def pack_rgb(r, g, b):
#         """将RGB颜色打包为一个uint32整数"""
#         return (r << 16) | (g << 8) | b

#     def create_pointcloud2_msg(self, points):
#         """
#         将点云列表转换为 sensor_msgs/PointCloud2 消息
        
#         Args:
#             points: list of (x, y, z, rgb)
        
#         Returns:
#             PointCloud2 消息
#         """
#         header = std_msgs.msg.Header()
#         header.stamp = rospy.Time.now()
#         header.frame_id = "map"

#         fields = [
#             PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
#             PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
#             PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
#             PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
#         ]

#         cloud = point_cloud2.create_cloud(header, fields, points)
#         return cloud

#     def publish_cluster_markers(self, centers):
#         """将聚类中心点以 MarkerArray (SPHERE) 形式发布"""
#         marker_array = MarkerArray()
#         for i, center in enumerate(centers):
#             marker = Marker()
#             marker.header.frame_id = "map"
#             marker.header.stamp = rospy.Time.now()
#             marker.ns = "cluster_centers"
#             marker.id = i
#             marker.type = Marker.SPHERE
#             marker.action = Marker.ADD
#             marker.pose.position.x = center[0]
#             marker.pose.position.y = center[1]
#             marker.pose.position.z = center[2]
#             marker.scale.x = 0.2   # 半径 0.2 米
#             marker.scale.y = 0.2
#             marker.scale.z = 0.2
#             marker.color.a = 1.0
#             marker.color.r = 1.0
#             marker.color.g = 0.0
#             marker.color.b = 0.0
#             marker.lifetime = rospy.Duration(0.2)  # 短暂保留，避免堆积
#             marker_array.markers.append(marker)
#         self.cluster_marker_pub.publish(marker_array)


# if __name__ == "__main__":
#     rospy.init_node("yolo_detector")
#     detector = yolo_detector()
#     rospy.spin()