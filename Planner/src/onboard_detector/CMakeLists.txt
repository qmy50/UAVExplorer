cmake_minimum_required(VERSION 3.0.2)
project(onboard_detector)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  cv_bridge
  sensor_msgs
  plan_env
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs plan_env
)

catkin_install_python(PROGRAMS
  scripts/yolo_detector/flask_yolo_server.py
  scripts/yolo_detector/ros_yolo_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  scripts/yolo_detector/config
  scripts/yolo_detector/weights
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
