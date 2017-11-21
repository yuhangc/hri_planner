//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 11/20/2017
// Last revision: 11/20/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#ifndef HRI_PLANNER_COMM_PLANNER_H
#define HRI_PLANNER_COMM_PLANNER_H

#include <vector>
#include <string>
#include <iostream>

#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/Pose2D.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Float64MultiArray.h"

#include "Eigen/Dense"

#include "hri_planner_sim/social_force_sim.h"

// hri planner namespace
namespace HRIPlanner {

class CommPlanner {
public:
    // constructor
    CommPlanner();

private:
    // node handler
    ros::NodeHandle nh_;

    // subscribers and publishers
    ros::Subscriber human_pose_vel_sub_;
    ros::Subscriber robot_pose_vel_sub_;

    // POMDP model parameters
    int num_actions_;
    int num_states_;

    // use one set of social force parameters for all humans
    SocialForce::SFParam social_force_param_;

    // awareness level transition probabilities
    // state_trans_model_[a](l, l') represents probability of going to state l'
    // from l when robot takes action a
    std::vector<Eigen::MatrixXd> state_trans_model_;

    // human states
    std::vector<Eigen::Vector3d> pose_human_;
    std::vector<Eigen::Vector3d> vel_human_;
    std::vector<Eigen::Vector2d> goal_human_;
    std::vector<std::vector<double> > blief_human_;

    // robot states
    Eigen::Vector3d pose_robot_;
    Eigen::Vector2d vel_robot_;

    // covariance of social force model
    Eigen::Matrix2d cov_sf_model_;

    // callbacks

    // functions
    void belief_update();
    double stoch_sf_tansition();

    void load_config(const std::string &config_file_path);
};

}

#endif //HRI_PLANNER_COMM_PLANNER_H
