//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 11/20/2017
// Last revision: 11/27/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------
#include <fstream>
#include <social_force/comm_planner.h>
#include <social_force/social_force_sim.h>

#include "social_force/comm_planner.h"

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
    robot_pose_vel_sub_ = nh_.subscribe<std_msgs::Float64MultiArray>("/localization/robot_pose_vel", 1,
                                                                     &CommPlanner::robot_pose_vel_callback,
                                                                     this);
    human_pose_vel_sub_ = nh_.subscribe<hri_planner::TrackedHumans>("/tracking/tracked_humans", 1,
                                                                   &CommPlanner::human_pose_vel_callback,
                                                                   this);

    belief_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("/planning/human_state_belief", 1);
    action_pub_ = nh_.advertise<std_msgs::Int32>("/planning/opt_action", 1);
}

//----------------------------------------------------------------------------------
void CommPlanner::run()
{
    const double planning_rate = 1.0 / dt_plan_;
    ros::Rate rate(planning_rate);

    while (!ros::isShuttingDown()) {
        // TODO: may want to queue up the callbacks since the planning loop rate
        // TODO: is lower than sensing loop rate
        ros::spinOnce();

        // update belief based on measurement
        belief_update_measurement();

        // for debug purpose, publish the updated belief
        publish_belief();

        // TODO: plan for the optimal communication action
        int a_opt = 0;

        // update belief based on action
        std::unordered_map<int, Eigen::VectorXd> new_belief;
        belief_update_action(a_opt, new_belief);

        // copy belief over
        for (auto &it: new_belief)
            belief_human_[it.first] = it.second;

        rate.sleep();
    }
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

    // discrete time intervals for planning
    dt_plan_ = root["dt_planning"].asDouble();
    dt_sim_ = root["dt_forward_sim"].asDouble();

    // initial belief
    // FIXME: use only one set of fixed values for now
    init_belief_.resize(num_states_);
    for (int s = 0; s < num_states_; s++) {
        init_belief_(s) = root["init_belief"][s].asDouble();
    }

    // load the human parameters
    SocialForce::SocialForceSimGazebo::load_human_param(root["social_force_params"],
                                                        social_force_param_);

    // TODO: cheating for now - store all the goal positions ahead
    // TODO: in the future this should be estimated online?
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
    const AgentPhysicalState &state = physical_state_human_[agent_id];

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

//    Eigen::MatrixXd prod = pose_diff_t.head(2) * inv_cov_sf_ * pose_diff.head(2);
    return std::exp(-0.5 * pose_diff_t.head(2) * inv_cov_sf_ * pose_diff.head(2));
}

//----------------------------------------------------------------------------------
void CommPlanner::belief_update_measurement(){
    // TODO: remove undetected human
    // TODO: for now, just track all humans that were detected
    const auto num_detected_human = (int) id_tracked_human_.size();

    // to store the new belief
//    std::unordered_map<int, Eigen::VectorXd> new_belief;

    for (int i = 0; i < num_detected_human; i++) {
        const int agent_id = id_tracked_human_[i];

        // add & initialize a new human
        if (!belief_human_.count(agent_id)) {
            init_tracked_human(agent_id, pose_vel_tracked_human_[i]);
            continue;
        }

//        Eigen::VectorXd belief_item(num_states_);
        Eigen::VectorXd &belief = belief_human_[agent_id];

        // update belief for each state
        double normalizer = 0.0;

        for (int s = 0; s < num_states_; s++) {
//            // forward transition model
//            for (int s_last = 0; s_last < num_states_; s_last++)
//                belief_item(s) += old_belief[s_last] * state_trans_model_[comm_action](s_last, s);

            // measurement model
            belief(s) *= stoch_sf_prob(agent_id, s, pose_vel_tracked_human_[i].pose);

            normalizer += belief(s);
        }

        // normalize the belief
        belief /= normalizer;

//        // update belief for tracked human
//        belief_human_[agent_id] = belief_item;
    }
}

//----------------------------------------------------------------------------------
void CommPlanner::belief_update_action(const int &comm_action,
                                       std::unordered_map<int, Eigen::VectorXd> &new_belief)
{
    const auto num_detected_human = (int) id_tracked_human_.size();

    // loop through all humans
    for (int i = 0; i < num_detected_human; i++) {
        const int agent_id = id_tracked_human_[i];

        Eigen::VectorXd belief_item(num_states_);
        Eigen::VectorXd &belief = belief_human_[agent_id];

        // update belief for each state
        for (int s = 0; s < num_states_; s++) {
            belief_item(s) = 0.0;

            // forward transition model
            for (int s_last = 0; s_last < num_states_; s_last++)
                belief_item(s) += belief(s_last) * state_trans_model_[comm_action](s_last, s);
        }

        // insert into new belief
        new_belief.insert({agent_id, belief_item});
    }
}

//----------------------------------------------------------------------------------
void CommPlanner::init_tracked_human(const int &agent_id, const AgentPhysicalState &agent_pose_vel)
{
    // TODO: initialize differently based on observation
    physical_state_human_.insert({agent_id, agent_pose_vel});
    belief_human_.insert({agent_id, init_belief_});
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

//----------------------------------------------------------------------------------
void CommPlanner::publish_belief()
{
    std_msgs::Float64MultiArray belief_data;

    // belief data is a vector of length n_human * (n_state + 1)
    // the format is (id, belief, id, belief, ...)
    for (auto &belief_iter: belief_human_) {
        belief_data.data.push_back((double)belief_iter.first);
        for (int s = 0; s < num_states_; s++) {
            belief_data.data.push_back(belief_iter.second(s));
        }
    }

    belief_pub_.publish(belief_data);
}

}