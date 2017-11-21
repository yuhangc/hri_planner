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

#ifndef HRI_PLANNER_SOCIAL_FORCE_SIM_H
#define HRI_PLANNER_SOCIAL_FORCE_SIM_H

#include <vector>
#include <string>
#include <iostream>

#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/Pose.h"
#include "std_msgs/Bool.h"
#include "gazebo_msgs/ModelStates.h"
#include "gazebo_msgs/ModelState.h"

#include "Eigen/Dense"
#include "json/json.h"

#include "hri_planner_sim/social_force.h"

//! namespace for all social force functions
namespace SocialForce {

// structure for parameters
typedef struct {
    vec3d pose_goal;
    double k;
    double vd;
    std::vector<double> hh_param;
    std::vector<std::vector<double> > hr_param;
    double max_v;
    double max_acc;
    double height;
} SFParam;

// simulation class
class SocialForceSimGazebo {
public:
    // constructor
    SocialForceSimGazebo(ros::NodeHandle &nh, ros::NodeHandle &pnh);

    // main update function
    void update(const double dt);

private:
    // node handler
    ros::NodeHandle nh_;

    // subscribers and publishers
    ros::Subscriber model_state_sub_;
    ros::Subscriber start_sim_sub_;
    ros::Publisher model_state_pub_;

    // agent states
    int num_agents_;
    std::vector<vec3d> pose_agent_;
    std::vector<vec3d> vel_agent_;
    std::vector<int> state_agent_;
    std::vector<SFParam> params_agent_;

    // robot state
    vec3d pose_robot_;
    vec2d vel_robot_;

    // flag to control start/stop simulation
    bool flag_start_sim_;

    // callbacks
    void gazebo_model_states_callback(const gazebo_msgs::ModelStatesConstPtr &states_msg);
    void start_sim_callback(const std_msgs::BoolConstPtr &start_sim_msg);

    // functions
    static void load_human_param(Json::Value &root, SFParam &new_param);
    void load_config(const std::string &config_file_path);
    void init_agents();
    void publish_states();
};

}

#endif //HRI_PLANNER_SOCIAL_FORCE_SIM_H
