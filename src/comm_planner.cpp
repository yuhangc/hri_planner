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
#include <fstream>

#include "hri_planner/comm_planner.h"

namespace HRIPlanner {

//----------------------------------------------------------------------------------
CommPlanner::CommPlanner(ros::NodeHandle &nh, ros::NodeHandle &pnh): nh_(nh)
{
    std::string config_file_path;

    // get the parameters
    pnh.param<std::string>("config_file", config_file_path, "../resources/sim_setting/default.json");

    // load configuration
    load_config(config_file_path);

    // setup publishers and subscribers

}

//----------------------------------------------------------------------------------
void CommPlanner::load_config(const std::string &config_file_path)
{
    // load json file
    std::ifstream config_doc(config_file_path, std::ifstream::binary);

    Json::Value root;
    config_doc >> root;

    // number of action and state values
    num_actions_ = root["num_actions"].asInt();
    num_states_ = root["num_states"].asInt();

    // load the transition table
    for (int a = 0; a < num_actions_; a++) {
        Eigen::MatrixXd m(num_states_, num_states_);
        for (int l = 0; l < num_states_; l++) {
            for (int lp = 0; lp < num_states_; lp++) {
                m(l, lp) = root["state_transition"][a][l][lp].asDouble();
            }
        }
        state_trans_model_.push_back(m);
    }

    // load the human parameters
    SocialForce::SocialForceSimGazebo::load_human_param(root["social_force_params"],
                                                        social_force_param_);

    // load goals for each human
    num_human_ = root["num_human"].asInt();
    for (int id = 0; id < num_human_; id++) {
        Eigen::Vector2d goal;
        goal << root["goal_human"][id][0].asDouble(), root["goal_human"][id][1].asDouble();
        goal_human_.push_back(goal);
    }
}

//----------------------------------------------------------------------------------
void CommPlanner::sf_tansition(const Eigen::Vector3d &pose_agent, Eigen::Vector3d &pose_new)
{

}

//----------------------------------------------------------------------------------
double CommPlanner::stoch_sf_prob(int agent_id, Eigen::Vector3d agent_pose)
{

}

//----------------------------------------------------------------------------------
double CommPlanner::belief_update()
{

}

//----------------------------------------------------------------------------------
void CommPlanner::human_pose_vel_callback(const std_msgs::Float64MultiArrayConstPtr &human_pose_vel)
{
    // TODO: need identification of each human, make up a new type of message?
}

//----------------------------------------------------------------------------------
void CommPlanner::robot_pose_vel_callback(const std_msgs::Float64MultiArrayConstPtr &robot_pose_vel)
{
    pose_robot_ << robot_pose_vel->data[0],
            robot_pose_vel->data[1],
            robot_pose_vel->data[2];

    vel_robot_ << robot_pose_vel->data[3], robot_pose_vel->data[4];
}

}