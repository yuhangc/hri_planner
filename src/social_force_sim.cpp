//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 11/19/2017
// Last revision: 11/19/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------
#include <fstream>
#include <sstream>
#include <hri_planner_sim/social_force_sim.h>
#include "json/json.h"

#include "hri_planner_sim/social_force_sim.h"

namespace SocialForce {

//----------------------------------------------------------------------------------
SocialForceSimGazebo::SocialForceSimGazebo(ros::NodeHandle &nh, ros::NodeHandle &pnh): nh_(nh)
{
    std::string config_file_path;

    // get the parameters
    pnh.param<std::string>("config_file", config_file_path, "../../resources/sim_setting/default.json");

    // load configuration
    load_config(config_file_path);

    // initialize all agents
    init_agents();

    // setup the callbacks
    model_state_sub_ = nh_.subscribe<gazebo_msgs::ModelStates>("/gazebo/GetModelStates", 1,
                                                               &SocialForceSimGazebo::gazebo_model_states_callback,
                                                               this);
    model_state_pub_ = nh_.advertise<gazebo_msgs::ModelStates>("/gazebo/SetModelStates", 1);
}

//----------------------------------------------------------------------------------
void SocialForceSimGazebo::update(const double dt)
{
    // loop through all agents
    for (int id = 0; id < num_agents_; id++) {
        vec2d force = social_force_goal(pose_agent_[id], vel_agent_[id],
                                        params_agent_[id].pose_goal, params_agent_[id].k, params_agent_[id].vd);

        // loop through all other agents
        for (int j = 0; j < num_agents_; j++) {
            if (j != id) {
                force += social_force_interact(pose_agent_[id], vel_agent_[id],
                                               pose_agent_[j], vel_agent_[j], params_agent_[id].hh_param);
            }
        }

        // interact with robot
        force += social_force_hri(pose_agent_[id], vel_agent_[id],
                                  pose_robot_, vel_robot_, params_agent_[id].hr_param[state_agent_[id]]);
    }
}

//----------------------------------------------------------------------------------
void SocialForceSimGazebo::gazebo_model_states_callback(const gazebo_msgs::ModelStatesConstPtr &states_msg)
{
    // loop through to find turtlebot states
    for (int i = 0; i < states_msg->name.size(); i++) {
        if (states_msg->name[i] == "Turtlebot") {
            //TODO: need to convert quaternion to 2D rotation
            double th = std::atan2(states_msg->pose[i].orientation.y,
                                   states_msg->pose[i].orientation.z) * 2.0;
            pose_robot_ << states_msg->pose[i].position.x,
                           states_msg->pose[i].position.y,
                           th;

            vel_robot_ << states_msg->twist[i].linear.x,
                          states_msg->twist[i].linear.y,
                          states_msg->twist[i].angular.z;

            break;
        }
    }
}

//----------------------------------------------------------------------------------
void SocialForceSimGazebo::load_config(const std::string &config_file_path)
{
    // load json file
    std::ifstream config_doc(config_file_path, std::ifstream::binary);

    Json::Value root;
    config_doc >> root;

    // number of humans
    num_agents_ = root["num_humans"].asInt();

    // load parameters for all simulated humans
    for (int id = 0; id < num_agents_; id++) {
        std::stringstream ss;
        ss << "human" << id;

        std::string agent_id = ss.str();

        // get goal
        params_agent_[id].pose_goal << root[agent_id]["goal_x"].asDouble(),
                                       root[agent_id]["goal_y"].asDouble(), 0.0;

        // get start pose
        vec3d pose_start;
        pose_start << root[agent_id][0].asDouble(),
                      root[agent_id][1].asDouble(),
                      root[agent_id][2].asDouble();
        pose_agent_.push_back(pose_start);

        // k and vd
        params_agent_[id].k = root[agent_id]["k"].asDouble();
        params_agent_[id].vd = root[agent_id]["vd"].asDouble();

        // social force parameter for human-human interaction
        params_agent_[id].hh_param.push_back(root[agent_id]["hh_param"][0].asDouble());
        params_agent_[id].hh_param.push_back(root[agent_id]["hh_param"][1].asDouble());
        params_agent_[id].hh_param.push_back(root[agent_id]["hh_param"][2].asDouble());

        // social force parameters for human-robot interaction
        for (int k = 0; k < 3; k++) {
            std::vector<double> hr_param;
            hr_param.push_back(root[agent_id]["hr_param"][k][0].asDouble());
            hr_param.push_back(root[agent_id]["hr_param"][k][1].asDouble());
            hr_param.push_back(root[agent_id]["hr_param"][k][2].asDouble());
            hr_param.push_back(root[agent_id]["hr_param"][k][3].asDouble());
            params_agent_[id].hr_param.push_back(hr_param);
        }

        // maximum velocity/acceleration limit
        params_agent_[id].max_v = root[agent_id]["max_v"].asDouble();
        params_agent_[id].max_acc = root[agent_id]["max_acc"].asDouble();
    }
}

//----------------------------------------------------------------------------------
void SocialForceSimGazebo::init_agents()
{
    // set all agents stationary initially
    for (int id = 0; id < num_agents_; id++) {
        vec3d vel_start = vec3d::Zero();
        vel_agent_.push_back(vel_start);

        // set all agents to state 1
        state_agent_.push_back(1);
    }

    flag_start_sim_ = false;
}

}