//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 4/9/2018
// Last revision: 4/15/2018
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#ifndef HRI_PLANNER_CONTROLLER_H
#define HRI_PLANNER_CONTROLLER_H

#include <Eigen/Dense>

#include <ros/ros.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <std_msgs/Bool.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Twist.h>

#include <tf/transform_listener.h>

enum ControllerState {
    Idle,
    Running
};

class ControllerNode {
public:
    ControllerNode(ros::NodeHandle &nh, ros::NodeHandle &pnh);

    void run();

private:
    // node rate
    double controller_rate_;

    // robot state and goal
    Eigen::VectorXd xr_;
    Eigen::VectorXd x_goal_;

    // controller parameters
    double k_rho_;
    double k_v_;
    double k_alp_;
    double k_phi_;

    double v_max_;
    double om_max_;

    double goal_reaching_th_controller_;

    // control flags
    bool flag_start_controller_;

    // tf listener to update robot pose
    tf::TransformListener tf_listener_;

    // node handler
    ros::NodeHandle nh_;

    // subscribers and publishers
    ros::Subscriber robot_state_sub_;
    ros::Subscriber goal_sub_;
    ros::Subscriber start_controller_sub_;

    ros::Publisher goal_reached_pub_;
    ros::Publisher robot_ctrl_pub_;

    void compute_and_publish_control();

    void goal_callback(const std_msgs::Float64MultiArrayConstPtr& goal_msg);
    void robot_state_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg);
    void start_controller_callback(const std_msgs::BoolConstPtr& msg);
};

#endif //HRI_PLANNER_CONTROLLER_H
