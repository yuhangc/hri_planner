//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 2/27/2017
// Last revision: 2/27/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include <memory>

#include "ros/ros.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"

#include "hri_planner/shared_config.h"
#include "hri_planner/human_belief_model.h"

#include "hri_planner/BeliefUpdate.h"

// function for testing the belief update
bool test_belief_update(hri_planner::BeliefUpdate::Request& req,
                        hri_planner::BeliefUpdate::Response& res) {
    // extract the messages
    Eigen::Map<Eigen::VectorXd> xr(req.xr.data(), req.xr.size());
    Eigen::Map<Eigen::VectorXd> ur(req.ur.data(), req.ur.size());
    Eigen::Map<Eigen::VectorXd> xh0(req.xh0.data(), req.xh0.size());

    // construct the shared configuration
    auto config = std::make_shared<hri_planner::SharedConfig>();

    // construct the belief update object
    double ratio;
    double decay_rate;
    std::vector<double> fcorrection(2, 0);

    ros::param::param<double>("~explicit_comm/ratio", ratio, 100.0);
    ros::param::param<double>("~explicit_comm/decay_rate", decay_rate, 2.5);
    ros::param::param<double>("~explicit_comm/fcorrection_hp", fcorrection[hri_planner::HumanPriority], 3.0);
    ros::param::param<double>("~explicit_comm/fcorrection_rp", fcorrection[hri_planner::RobotPriority], 30.0);

    hri_planner::BeliefModelExponential belief_model(config, ratio, decay_rate, fcorrection);

    ROS_WARN("Belief model initialized");

    // compute all the beliefs
    for (int t = 0; t < req.t_total - config->T; ++t) {
        Eigen::Vector2d belief;
        belief_model.belief_update(xr.segment(t*config->nXr, config->nXr*config->T),
                                   ur.segment(t*config->nUr, config->nUr*config->T),
                                   xh0.segment(t*config->nXh, config->nXh),
                                   req.acomm, req.tcomm, (t+config->T) * config->dt, belief);
        res.belief.push_back(belief(hri_planner::HumanPriority));
    }

    return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "component_test_service");
    ros::NodeHandle n;

    ros::ServiceServer service = n.advertiseService("update_belief", test_belief_update);
    ROS_INFO("Ready to calculate new beliefs.");
    ros::spin();

    return 0;
}