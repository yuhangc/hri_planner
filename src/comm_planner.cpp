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
#include <fstream>
#include <hri_planner/comm_planner.h>
#include <hri_planner_sim/social_force_sim.h>

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

    // TODO: cheating for now - store all the goal positions ahead
    // load goals for each human
    num_human_ = root["num_human"].asInt();
    for (int id = 0; id < num_human_; id++) {
        Eigen::Vector3d goal;
        goal << root["goal_human"][id][0].asDouble(), root["goal_human"][id][1].asDouble(), 0.0;
        goal_human_.insert({id, goal});
    }
}

//----------------------------------------------------------------------------------
void CommPlanner::sf_transition(const int &agent_id,
                                const int &agent_state,
                                const std::unordered_map<int, AgentPhysicalState> &pose_vels,
                                const double &dt,
                                AgentPhysicalState &new_pose_vel)
{
    // get the state of the specified agent
    AgentPhysicalState &state = physical_state_human_[agent_id];

    // calculate force for goal
    Eigen::Vector2d force = SocialForce::social_force_goal(state.pose, state.vel,
                                                           goal_human_[agent_id],
                                                           social_force_param_.k,
                                                           social_force_param_.vd);

    // calculate interaction forces
    for (auto &agent: pose_vels) {
        if (agent.first != agent_id) {
            force += SocialForce::social_force_interact(state.pose, state.vel,
                                                        agent.second.pose, agent.second.vel,
                                                        social_force_param_.hh_param);
        }
    }

    // interaction force with the robot
    force += SocialForce::social_force_hri(state.pose, state.vel,
                                           pose_robot_, vel_robot_,
                                           social_force_param_.hr_param[agent_state]);
    // add in damping
    force -= SocialForce::social_force_damping_factor * state.vel.head(2);

    // update vel and pose
    Eigen::Vector3d vel_new;
    vel_new << state.vel(0) + force(0) * dt, state.vel(1) + force(1) * dt, 0.0;

    new_pose_vel.pose = state.pose + 0.5 * dt * (state.vel + vel_new);
    new_pose_vel.pose(2) = std::atan2(vel_new(1), vel_new(0));

    new_pose_vel.vel = vel_new;
}

//----------------------------------------------------------------------------------
double CommPlanner::stoch_sf_prob(const int &agent_id, const int &agent_state,
                                  const Eigen::Vector3d &agent_pose)
{
    // calculate the expected pose
    AgentPhysicalState pose_vel_exp;

    sf_transition(agent_id, agent_state, physical_state_human_, dt_plan_, pose_vel_exp);

    // calculate the probability
    Eigen::Vector3d pose_diff = agent_pose - pose_vel_exp.pose;
    Eigen::RowVector3d pose_diff_t = pose_diff.transpose();

    return std::exp(-0.5 * pose_diff_t * inv_cov_sf_ * pose_diff);
}

//----------------------------------------------------------------------------------
void CommPlanner::belief_update(const int &comm_action)
{
    // TODO: remove undetected human
    // TODO: for now, just track all humans that were detected
    const auto num_detected_human = (int) id_tracked_human_.size();

    // to store the new belief
    std::unordered_map<int, Eigen::VectorXd> new_belief;

    for (int i = 0; i < num_detected_human; i++) {
        const int agent_id = id_tracked_human_[i];

        // add & initialize a new human
        if (!belief_human_.count(agent_id)) {
            init_tracked_human(agent_id, pose_vel_tracked_human_[i]);
            continue;
        }

        Eigen::VectorXd belief_item(num_states_);
        const Eigen::VectorXd &old_belief = belief_human_[agent_id];

        // update belief for each state
        double normalizer = 0.0;

        for (int s = 0; s < num_states_; s++) {
            // forward transition model
            for (int s_last = 0; s_last < num_states_; s_last++)
                belief_item(s) += old_belief[s_last] * state_trans_model_[comm_action](s_last, s);

            // measurement model
            belief_item(s) *= stoch_sf_prob(agent_id, s, pose_vel_tracked_human_[i].pose);

            normalizer += belief_item(s);
        }

        // normalize the belief
        belief_item /= normalizer;

        // update belief for tracked human
        belief_human_[agent_id] = belief_item;
    }
}

//----------------------------------------------------------------------------------
void CommPlanner::init_tracked_human(const int &agent_id, const AgentPhysicalState &agent_pose_vel)
{

}

//----------------------------------------------------------------------------------
void CommPlanner::human_pose_vel_callback(const hri_planner::TrackedHumansConstPtr &tracked_humans)
{
    // clear the observation
    id_tracked_human_.clear();
    pose_vel_tracked_human_.clear();

    // extract the tracking data
    for (const auto &human_data : tracked_humans->tracking_data) {
        // id
        id_tracked_human_.push_back((int)human_data.id);

        // pose and vel
        AgentPhysicalState new_state;
        new_state.pose << human_data.pose_vel[0], human_data.pose_vel[1], human_data.pose_vel[2];

        // vel
        new_state.vel << human_data.pose_vel[3], human_data.pose_vel[4], human_data.pose_vel[5];
        pose_vel_tracked_human_.push_back(new_state);
    }
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