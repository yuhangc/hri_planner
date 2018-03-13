//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 2/27/2017
// Last revision: 3/13/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include <string>
#include <fstream>
#include <memory>

#include "ros/ros.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"

#include "hri_planner/shared_config.h"
#include "hri_planner/human_belief_model.h"
#include "hri_planner/cost_feature_bases.h"
#include "hri_planner/cost_features.h"
#include "hri_planner/optimizer.h"

#include "hri_planner/BeliefUpdate.h"
#include "hri_planner/TestComponent.h"

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

bool test_cost_features(hri_planner::TestComponent::Request& req,
                        hri_planner::TestComponent::Response& res) {
    // extract the messages
    Eigen::Map<Eigen::VectorXd> xr(req.xr.data(), req.xr.size());
    Eigen::Map<Eigen::VectorXd> ur(req.ur.data(), req.ur.size());
    Eigen::Map<Eigen::VectorXd> xh(req.xh.data(), req.xh.size());
    Eigen::Map<Eigen::VectorXd> uh(req.uh.data(), req.uh.size());
    Eigen::Map<Eigen::VectorXd> xr0(req.xr0.data(), req.xr0.size());
    Eigen::Map<Eigen::VectorXd> xh0(req.xh0.data(), req.xh0.size());

    std::string log_path = req.log_path;

    // create all the cost functions
    using namespace hri_planner;
    HumanVelCost human_vel_cost;
    HumanAccCost human_acc_cost;

    Eigen::VectorXd x_goal(2);
    x_goal << 0.73216, 6.00955;
    HumanGoalCost human_goal_cost(x_goal);

    CollisionCost collision_cost(0.5);
    DynCollisionCost dyn_collision_cost(0.3, 0.5, 0.5);

    RobotControlCost robot_control_cost;

    x_goal << 5.0, 3.0;
    RobotGoalCost robot_goal_cost(x_goal);

    // create a log file to store the result
    std::ofstream logger(log_path + "/log_feature_costs.txt");
    std::ofstream traj_logger(log_path + "/log_traj.txt");

    // create the trajectories
    Trajectory human_traj(CONST_ACC_MODEL, 10, 0.5);
    Trajectory robot_traj(DIFFERENTIAL_MODEL, 10, 0.5);

    logger << "Start logging..." << std::endl;

    human_traj.update(xh0, uh);
    robot_traj.update(xr0, ur);
    human_traj.compute_jacobian();
    robot_traj.compute_jacobian();

    // compare the trajectories and log
    Eigen::VectorXd x_diff = human_traj.x - xh;
    logger << "Human trajectory error: " << x_diff.norm() << std::endl;

    x_diff = robot_traj.x - xr;
    logger << "Robot trajectory error: " << x_diff.norm() << std::endl;

    traj_logger << robot_traj.x << std::endl;
    traj_logger << human_traj.x << std::endl;

    // set robot trajectory back to the recorded value
    robot_traj.x = xr;

    // log trajectory Jacobians
    logger << std::endl;
    logger << "Human trajectory Jacobian:" << std::endl;
    logger << human_traj.Ju << std::endl;
    logger << "Robot trajectory Jacobian:" << std::endl;
    logger << robot_traj.Ju << std::endl;

    // compute the value, gradient and hessian (if applicable) for each cost feature
    double val;
    Eigen::VectorXd grad_uh(human_traj.traj_control_size());
    Eigen::VectorXd grad_ur(robot_traj.traj_control_size());
    Eigen::MatrixXd hess_uh(human_traj.traj_control_size(), human_traj.traj_control_size());
    Eigen::MatrixXd hess_uh_ur(human_traj.traj_control_size(), robot_traj.traj_control_size());

    // velocity feature
    logger << "Velocity feature:" << std::endl;
    logger << "value: " << human_vel_cost(robot_traj, human_traj) << std::endl;

    human_vel_cost.grad_uh(robot_traj, human_traj, grad_uh);
    logger << "gradient:" << std::endl;
    logger << grad_uh.transpose() << std::endl;

    human_vel_cost.hessian_uh(robot_traj, human_traj, hess_uh);
    logger << "hessian:" << std::endl;
    logger << hess_uh << std::endl;

    // acceleration feature
    logger << "Acceleration feature:" << std::endl;
    logger << "value: " << human_acc_cost(robot_traj, human_traj) << std::endl;

    human_acc_cost.grad_uh(robot_traj, human_traj, grad_uh);
    logger << "gradient:" << std::endl;
    logger << grad_uh.transpose() << std::endl;

    human_acc_cost.hessian_uh(robot_traj, human_traj, hess_uh);
    logger << "hessian:" << std::endl;
    logger << hess_uh << std::endl;

    // goal feature
    logger << "Human goal feature:" << std::endl;
    logger << "value: " << human_goal_cost(robot_traj, human_traj) << std::endl;

    human_goal_cost.grad_uh(robot_traj, human_traj, grad_uh);
    logger << "gradient:" << std::endl;
    logger << grad_uh.transpose() << std::endl;

    human_goal_cost.hessian_uh(robot_traj, human_traj, hess_uh);
    logger << "hessian:" << std::endl;
    logger << hess_uh << std::endl;

    // collision feature
    logger << "Collision feature:" << std::endl;
    logger << "value: " << collision_cost(robot_traj, human_traj) << std::endl;

    collision_cost.grad_uh(robot_traj, human_traj, grad_uh);
    logger << "gradient w.r.t. uh:" << std::endl;
    logger << grad_uh.transpose() << std::endl;

    collision_cost.grad_ur(robot_traj, human_traj, grad_ur);
    logger << "gradient w.r.t. ur:" << std::endl;
    logger << grad_ur.transpose() << std::endl;

    collision_cost.hessian_uh(robot_traj, human_traj, hess_uh);
    logger << "hessian w.r.t. uh:" << std::endl;
    logger << hess_uh << std::endl;

    collision_cost.hessian_uh_ur(robot_traj, human_traj, hess_uh_ur);
    logger << "hessian w.r.t. uh ur:" << std::endl;
    logger << hess_uh_ur << std::endl;

    // dynamic collision feature
    logger << "Dynamic collision feature:" << std::endl;
    logger << "value: " << dyn_collision_cost(robot_traj, human_traj) << std::endl;

    dyn_collision_cost.grad_uh(robot_traj, human_traj, grad_uh);
    logger << "gradient w.r.t. uh:" << std::endl;
    logger << grad_uh.transpose() << std::endl;

    dyn_collision_cost.grad_ur(robot_traj, human_traj, grad_ur);
    logger << "gradient w.r.t. ur:" << std::endl;
    logger << grad_ur.transpose() << std::endl;

    dyn_collision_cost.hessian_uh(robot_traj, human_traj, hess_uh);
    logger << "hessian w.r.t. uh:" << std::endl;
    logger << hess_uh << std::endl;

    dyn_collision_cost.hessian_uh_ur(robot_traj, human_traj, hess_uh_ur);
    logger << "hessian w.r.t. uh ur:" << std::endl;
    logger << hess_uh_ur << std::endl;

    // robot control feature
    logger << "Robot control feature:" << std::endl;
    logger << "value:" << robot_control_cost(robot_traj, human_traj) << std::endl;

    robot_control_cost.grad_ur(robot_traj, human_traj, grad_ur);
    logger << "gradient:" << std::endl;
    logger << grad_ur.transpose() << std::endl;

    // robot goal feature
    logger << "Robot goal feature:" << std::endl;
    logger << "value:" << robot_goal_cost(robot_traj, human_traj) << std::endl;

    robot_goal_cost.grad_ur(robot_traj, human_traj, grad_ur);
    logger << "gradient:" << std::endl;
    logger << grad_ur.transpose() << std::endl;

    logger.close();
    traj_logger.close();

    res.succeeded = true;

    return true;
}

bool test_simple_optimizer(hri_planner::TestComponent::Request& req,
                           hri_planner::TestComponent::Response& res) {
    // extract the messages
    Eigen::Map<Eigen::VectorXd> xr(req.xr.data(), req.xr.size());
    Eigen::Map<Eigen::VectorXd> ur(req.ur.data(), req.ur.size());
    Eigen::Map<Eigen::VectorXd> xh(req.xh.data(), req.xh.size());
    Eigen::Map<Eigen::VectorXd> uh(req.uh.data(), req.uh.size());
    Eigen::Map<Eigen::VectorXd> xr0(req.xr0.data(), req.xr0.size());
    Eigen::Map<Eigen::VectorXd> xh0(req.xh0.data(), req.xh0.size());

    std::string log_path = req.log_path;

    // create a log file to store the result
    std::ofstream logger(log_path + "/log_optimizer_test.txt");
    logger << "start logging..." << std::endl;

    // create all the cost functions
    using namespace hri_planner;
    std::vector<std::shared_ptr<FeatureBase> > features;

    // problem dimensions
    int T = 10;
    int nXh = 4;
    int nUh = 2;
    int nXr = 3;
    int nUr = 2;
    double dt = 0.5;

    // velocity and acceleration feature
    features.push_back(std::make_shared<HumanVelCost>());
    features.push_back(std::make_shared<HumanAccCost>());

    // goal feature
    Eigen::VectorXd x_goal(2);
    x_goal << 0.73216, 6.00955;
    features.push_back(std::make_shared<HumanGoalCost>(x_goal));

    // avoiding robot
    features.push_back(std::make_shared<CollisionCost>(0.5));
    features.push_back(std::make_shared<DynCollisionCost>(0.5, 0.5, 0.5));

    logger << "cost features created!" << std::endl;

    // create cost function to optimize
    auto cost_human = std::make_shared<SingleTrajectoryCostHuman>(req.weights, features);

    // set const data for the cost function
    Trajectory robot_traj(DIFFERENTIAL_MODEL, T, dt);
    robot_traj.update(xr0, ur);

    cost_human->set_trajectory_data(robot_traj);

    // create the optimizer
    unsigned int dim = 10 * 2;
    TrajectoryOptimizer optimizer(dim, nlopt::LD_LBFGS);
    optimizer.set_cost_function(cost_human);

    logger << "optimizer created!" << std::endl;

    // set bounds for the optimizer
    Eigen::VectorXd lb(dim);
    Eigen::VectorXd ub(dim);

    lb.setOnes();
    lb *= -10.0;
    ub.setOnes();
    ub *= 10.0;

    optimizer.set_bounds(lb, ub);

    // set initial condition
    Trajectory traj_init(CONST_ACC_MODEL, T, dt);
    traj_init.update(xh0, uh);

    // optimize!
    ros::Time t_start = ros::Time::now();
    Trajectory traj_opt(CONST_ACC_MODEL, T, dt);
    optimizer.optimize(traj_init, traj_opt);

    ros::Duration t_elapse = ros::Time::now() - t_start;
    logger << "optimization finished successfully, took " << t_elapse.toSec() << " seconds." << std::endl;

    std::ofstream traj_logger(log_path + "/log_traj.txt");

    traj_logger << traj_opt.x << std::endl;
    traj_logger << traj_opt.u << std::endl;

    logger.close();
    traj_logger.close();

    res.succeeded = true;

    return true;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "component_test_service");
    ros::NodeHandle n;

    ros::ServiceServer belief_service = n.advertiseService("update_belief", test_belief_update);
    ros::ServiceServer feature_service = n.advertiseService("test_cost_features", test_cost_features);
    ros::ServiceServer simple_optimizer_service = n.advertiseService("test_simple_optimizer", test_simple_optimizer);

    ROS_INFO("Services are ready!");
    ros::spin();

    return 0;
}