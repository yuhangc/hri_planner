//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 4/4/2018
// Last revision: 4/4/2018
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#ifndef HRI_PLANNER_PLANNER_NODE_H
#define HRI_PLANNER_PLANNER_NODE_H

#include <string>
#include <memory>

#include <Eigen/Dense>

#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Bool.h>

#include <hri_planner/planner.h>

// enum for state machine
enum PlannerStates {
    Idle,
    Planning,
    PlanningNoHuman,
    Pausing
};

// the planner node class
class PlannerNode {
public:
    // constructor
    PlannerNode(ros::NodeHandle &nh, ros::NodeHandle &pnh);

    void run();

private:
    // planner
    std::shared_ptr<hri_planner::PlannerBase> planner_interactive_;
    std::shared_ptr<hri_planner::PlannerBase> planner_simple_;

    // planner state
    PlannerStates state_;

    // control flags
    bool flag_start_planning_;
    bool flag_pause_planning_;

    // goals
    Eigen::VectorXd xr_goal_;
    Eigen::VectorXd xh_goal_;
    int intent_;

    // measurements
    Eigen::VectorXd xr_meas_;
    Eigen::VectorXd ur_meas_;
    Eigen::VectorXd xh_meas_;

    // rates
    double planning_rate_;
    double state_machine_rate_;

    // mode
    std::string mode_;

    // node handler
    ros::NodeHandle nh_;

    // publishers and subscribers
    ros::Subscriber goal_sub_;
    ros::Subscriber planner_ctrl_sub_;

    ros::Subscriber robot_state_sub_;
    ros::Subscriber robot_odom_sub_;
    ros::Subscriber human_state_sub_;

    // callback functions
    void goal_callback(const std_msgs::Float64MultiArrayConstPtr& goal_msg);
    void pause_planner_callback(const std_msgs::BoolConstPtr& msg);

    void robot_state_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg);
    void robot_odom_callback(const nav_msgs::OdometryConstPtr& odom_msg);
    void human_state_callback(const std_msgs::Float64MultiArrayConstPtr& state_msg);
};


#endif //HRI_PLANNER_PLANNER_NODE_H
