//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 11/20/2017
// Last revision: 11/22/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#ifndef HRI_PLANNER_COMM_PLANNER_H
#define HRI_PLANNER_COMM_PLANNER_H

#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>

#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/Pose2D.h"
#include "std_msgs/Bool.h"
#include "std_msgs/Float64MultiArray.h"
#include "hri_planner/TrackedHumans.h"

#include "Eigen/Dense"
#include "json/json.h"

#include "hri_planner_sim/social_force_sim.h"

// hri planner namespace
namespace HRIPlanner {

typedef struct {
    Eigen::Vector3d pose;
    Eigen::Vector3d vel;
} AgentPhysicalState;

class CommPlanner {
public:
    // constructor
    CommPlanner(ros::NodeHandle &nh, ros::NodeHandle &pnh);

    // main function for planning
    void run();

private:
    // node handler
    ros::NodeHandle nh_;

    // subscribers and publishers
    ros::Subscriber human_pose_vel_sub_;
    ros::Subscriber robot_pose_vel_sub_;

    ros::Publisher belief_pub_;
    ros::Publisher action_pub_;

    // POMDP model parameters
    int num_actions_;
    int num_states_;

    // plan update rate
    double dt_plan_;
    double dt_sim_;

    // initial belief over states
    Eigen::VectorXd init_belief_;

    // awareness level transition probabilities
    // state_trans_model_[a](l, l') represents probability of going to state l'
    // from l when robot takes action a
    std::vector<Eigen::MatrixXd> state_trans_model_;

    // human states
    int num_human_;
    std::vector<int> id_tracked_human_;
    std::vector<AgentPhysicalState> pose_vel_tracked_human_;

    std::unordered_map<int, AgentPhysicalState> physical_state_human_;
    std::unordered_map<int, Eigen::Vector3d> goal_human_;
    std::unordered_map<int, Eigen::VectorXd> belief_human_;

    // robot states
    Eigen::Vector3d pose_robot_;
    Eigen::Vector2d vel_robot_;

    // use one set of social force parameters for all humans
    SocialForce::SFParam social_force_param_;

    // covariance of social force model
    Eigen::Matrix2d cov_sf_;
    Eigen::Matrix2d inv_cov_sf_;

    // callbacks
    void human_pose_vel_callback(const hri_planner::TrackedHumansConstPtr &tracked_humans);
    void robot_pose_vel_callback(const std_msgs::Float64MultiArrayConstPtr &robot_pose_vel);

    // functions
    void load_config(const std::string &config_file_path);

    void belief_update_measurement();
    void belief_update_action(const int &comm_action,
                              std::unordered_map<int, Eigen::VectorXd> &new_belief);
    void init_tracked_human(const int &agent_id, const AgentPhysicalState &agent_pose_vel);

    void sf_transition(const int &agent_id,
                      const int &agent_state,
                      const std::unordered_map<int, AgentPhysicalState> &pose_vels,
                      const double &dt,
                      AgentPhysicalState &new_pose_vel);
    double stoch_sf_prob(const int &agent_id, const int &agent_state,
                         const Eigen::Vector3d &agent_pose);
};

}

#endif //HRI_PLANNER_COMM_PLANNER_H
