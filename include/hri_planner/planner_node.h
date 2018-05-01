//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 4/4/2018
// Last revision: 4/23/2018
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
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>

#include <tf/transform_listener.h>

#include "people_msgs/People.h"
#include "people_msgs/PositionMeasurementArray.h"

#include "hri_planner/planner.h"

// enum for state machine
enum PlannerStates {
    Idle,
    Planning,
    GoalReaching,
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
    bool flag_stop_planning_;
    bool flag_human_detected_;
    bool flag_human_detected_frame_;
    bool flag_human_tracking_lost_;

    int human_tracking_lost_frames_;
    int tracking_lost_th_;

    // human tracking
    double t_meas_last_;

    // goals
    int goal_dim_;
    Eigen::VectorXd xr_goal_;
    Eigen::VectorXd xh_goal_;
    Eigen::VectorXd xh_init_;
    int intent_;

    // measurements
    Eigen::VectorXd xr_meas_;
    Eigen::VectorXd ur_meas_;
    Eigen::VectorXd xh_meas_;

    // rates
    double planning_rate_;
    double controller_rate_;
    double state_machine_rate_;

    double dt_planning_;

    // goal reaching threshold
    double goal_reaching_th_planner_;
    double goal_reaching_th_controller_;

    // human filter parameters
    double human_filter_dist_th_;

    // mode
    std::string mode_;
    bool flag_allow_explicit_comm_;

    // node handler
    ros::NodeHandle nh_;

    // tf listener to update robot pose
    tf::TransformListener tf_listener_;

    // publishers and subscribers
    ros::Subscriber goal_sub_;
    ros::Subscriber planner_ctrl_sub_;

    ros::Subscriber robot_state_sub_;
    ros::Subscriber robot_odom_sub_;
    ros::Subscriber human_tracking_sub_;

    ros::Publisher goal_reached_pub_;
    ros::Publisher robot_ctrl_pub_;
    ros::Publisher robot_human_state_pub_;

    // helper functions
    void plan(const std::shared_ptr<hri_planner::PlannerBase>& planenr);
    void compute_and_publish_control();

    void reset_state_machine();

    double point_line_dist(const Eigen::VectorXd& p, const Eigen::VectorXd& a, const Eigen::VectorXd& b);

    // callback functions
    void goal_callback(const std_msgs::Float64MultiArrayConstPtr& goal_msg);
    void planner_ctrl_callback(const std_msgs::StringConstPtr& msg);

    void robot_state_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg);
    void robot_odom_callback(const nav_msgs::OdometryConstPtr& odom_msg);
    void human_tracking_callback(const people_msgs::PeopleConstPtr& people_msg);
    void human_detection_callback(const people_msgs::PositionMeasurementArrayConstPtr& pos_arr_msg);
};


#endif //HRI_PLANNER_PLANNER_NODE_H
