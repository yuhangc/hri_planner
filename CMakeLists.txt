cmake_minimum_required(VERSION 2.8.3)
project(hri_planner)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        geometry_msgs
        sensor_msgs
        message_generation)

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  HumanStat.msg
  TrackedHumans.msg
)

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs  # Or other packages containing msgs
)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
    CATKIN_DEPENDS message_runtime
)

###########
## Build ##
###########
## Your package locations should be listed before other locations
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
)

## specify Eigen3 package and directories
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

## specify openCV package and directories
# force using the system opencv
set(OpenCV_DIR "/usr/local/share/OpenCV")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("${OpenCV_INCLUDE_DIRS}")

## specify the json package and directories
find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)
include_directories(${JSONCPP_INCLUDE_DIRS})

# specify ARUCO package and directories
find_package(aruco REQUIRED)
include_directories(aruco_INCLUDE_DIRS)

## Declare C++ libraries
add_library(social_force
        include/social_force/social_force.h
        include/social_force/social_force_sim.h
        src/social_force/social_force.cpp
        src/social_force/social_force_sim.cpp)
target_link_libraries(social_force ${catkin_LIBRARIES} ${JSONCPP_LIBRARIES})

#add_library(hri_planner
#        include/social_force/comm_planner.h
#        src/social_force/comm_planner.cpp)
#target_link_libraries(hri_planner ${catkin_LIBRARIES} ${JSONCPP_LIBRARIES} social_force)
add_library(hri_planner
        include/hri_planner/shared_config.h
        include/hri_planner/human_belief_model.h
        src/hri_planner/shared_config.cpp
        src/hri_planner/human_belief_model.cpp)
target_link_libraries(hri_planner ${catkin_LIBRARIES})

## Declare C++ executables
# social force simulation
add_executable(social_force_sim
        src/social_force/social_force_sim_node.cpp)
target_link_libraries(social_force_sim ${catkin_LIBRARIES} social_force)

# fake tracker of human pose
add_executable(fake_tracker
        src/hri_planner/fake_tracker.cpp)
target_link_libraries(fake_tracker ${catkin_LIBRARIES})

# communication planner
#add_executable(comm_planner
#        src/hri_planner/comm_planner_node.cpp)
#target_link_libraries(comm_planner ${catkin_LIBRARIES} hri_planner)

# video processor
add_executable(video_processor
        include/top_view_tracker/hat_tracker.h
        include/top_view_tracker/video_processor.h
        src/top_view_tracker/hat_tracker.cpp
        src/top_view_tracker/video_processor.cpp)
target_link_libraries(video_processor ${catkin_LIBRARIES}
        ${OpenCV_LIBRARIES} ${JSONCPP_LIBRARIES} aruco)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_hri_planner.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
