//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 2/27/2017
// Last revision: 3/29/2017
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
#include "hri_planner/cost_features.h"
#include "hri_planner/cost_features_vectorized.h"
#include "hri_planner/cost_probabilistic.h"
#include "hri_planner/optimizer.h"

#include "hri_planner/BeliefUpdate.h"
#include "hri_planner/TestComponent.h"


//! helper functions to reuse code
void create_belief_model(std::shared_ptr<hri_planner::BeliefModelBase>& belief_model)
{
    int T_hist;
    double ratio;
    double decay_rate;
    std::vector<double> fcorrection(2, 0);

    ros::param::param<int>("~explicit_comm/history_length", T_hist, 10);
    ros::param::param<double>("~explicit_comm/ratio", ratio, 100.0);
    ros::param::param<double>("~explicit_comm/decay_rate", decay_rate, 2.5);
    ros::param::param<double>("~explicit_comm/fcorrection_hp", fcorrection[hri_planner::HumanPriority], 2.0);
    ros::param::param<double>("~explicit_comm/fcorrection_rp", fcorrection[hri_planner::RobotPriority], 20.0);

    belief_model = std::make_shared<hri_planner::BeliefModelExponential>(T_hist, fcorrection, ratio, decay_rate);
    belief_model->reset_hist(Eigen::Vector2d::Zero());
}

void create_human_costs(std::vector<std::shared_ptr<hri_planner::FeatureBase> >& features,
                        const Eigen::VectorXd& x_goal)
{
    // velocity and acceleration feature
    features.push_back(std::make_shared<hri_planner::HumanVelCost>());
    features.push_back(std::make_shared<hri_planner::HumanAccCost>());

    // goal feature
    features.push_back(std::make_shared<hri_planner::HumanGoalCost>(x_goal));

    // avoiding robot
    features.push_back(std::make_shared<hri_planner::CollisionCost>(0.5));
    features.push_back(std::make_shared<hri_planner::DynCollisionCost>(0.5, 0.5, 0.5));
}

void create_robot_costs(std::vector<std::shared_ptr<hri_planner::FeatureBase> >& f_non_int,
                        std::vector<std::shared_ptr<hri_planner::FeatureVectorizedBase> >& f_int,
                        const Eigen::VectorXd& x_goal)
{
    // create non-interactive features
    // control
    f_non_int.push_back(std::make_shared<hri_planner::RobotControlCost>());

    // goal feature
    f_non_int.push_back(std::make_shared<hri_planner::RobotGoalCost>(x_goal));

    // create interactive costs
    // human effort
    f_int.push_back(std::make_shared<hri_planner::HumanAccCostVec>());

    // collision
    f_int.push_back(std::make_shared<hri_planner::CollisionCostVec>(0.5));
}

void create_probabilistic_cost(std::shared_ptr<hri_planner::NestedTrajectoryOptimizer> optimizer)
{

}

// function for testing the belief update
bool test_belief_update(hri_planner::TestComponent::Request& req,
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
    std::ofstream logger(log_path + "/log_belief_model_test.txt");
    std::ofstream belief_logger(log_path + "/log_belief.txt");

    // construct the belief update object
    std::shared_ptr<hri_planner::BeliefModelBase> belief_model;
    create_belief_model(belief_model);

    logger << "Belief model initialized..." << std::endl;

    int nXr = 3;
    int nUr = 2;
    int nXh = 4;
    int nUh = 2;
    double dt = 0.5;

    int T = (int)req.xr.size() / nXr;

    // extract acomm and tcomm from xr0
    int acomm = req.acomm;
    double tcomm = req.tcomm;

    // test the simple update
    Eigen::VectorXd prob_hp(T+1);
    prob_hp(0) = belief_model->get_belief();

    for (int t = 0; t < T; ++t) {
        Eigen::VectorXd xr_t = xr.segment(t * nXr, nXr);
        Eigen::VectorXd ur_t = ur.segment(t * nUr, nUr);
        Eigen::VectorXd xh_t = xh.segment(t * nXh, nXh);

        Eigen::VectorXd ur_d(2);
        Eigen::VectorXd xr_goal(2);
        xr_goal << 6.0, 6.0;
        double rho = (xr_goal - xr_t.head(2)).norm();
        double phi = 0.0;   //! don't care about orientation for now
        double th_z = std::atan2(xr_goal(1) - xr_t(1), xr_goal(0) - xr_t(0));
        double alpha = utils::wrap_to_pi(th_z - xr_t(2));
        std::cout << "alpha: " << alpha << std::endl;

        ur_d(0) = utils::clamp(1.0 * std::tanh(3.8 * rho), -0.5, 0.5);
        ur_d(1) = utils::clamp(6.0 * alpha - 1.0 * phi, -2.0, 2.0);

        belief_model->set_ur_nav(ur_d);

        prob_hp(t+1) = belief_model->update_belief(xr_t, ur_t, xh_t, acomm, tcomm, (t+1) * dt);
    }
    belief_logger << prob_hp.transpose() << std::endl;

    // test full update
    // using the exact trajectory
    /*
    hri_planner::Trajectory robot_traj(hri_planner::DIFFERENTIAL_MODEL, T, dt);
    hri_planner::Trajectory human_traj(hri_planner::CONST_ACC_MODEL, T, dt);

    robot_traj.update(xr0, ur);
    robot_traj.compute_jacobian();
    human_traj.update(xh0, uh);
    human_traj.compute_jacobian();

    Eigen::MatrixXd Jur(T, robot_traj.traj_control_size());

    logger << "Now performing full belief update..." << std::endl;

    belief_model->reset_hist(Eigen::Vector2d::Zero());
    belief_model->update_belief(robot_traj, human_traj, acomm, tcomm, 0.0, prob_hp, Jur);

    // log belief and jacobian
    belief_logger << prob_hp.transpose() << std::endl;

    logger << "Jacobian of belief w.r.t. ur:" << std::endl;
    logger << Jur << std::endl;
     */

    logger.close();
    belief_logger.close();

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

    Eigen::VectorXd x_goal(2);
    x_goal << 0.73216, 6.00955;

    create_human_costs(features, x_goal);
    logger << "cost features created!" << std::endl;

    // create cost function to optimize
    auto cost_human = std::make_shared<SingleTrajectoryCostHuman>(req.weights, features);

    // set const data for the cost function
    Trajectory robot_traj(DIFFERENTIAL_MODEL, T, dt);
    robot_traj.update(xr0, ur);

    // create the optimizer
    int dim = T * nUr;
    TrajectoryOptimizer optimizer(static_cast<unsigned int>(dim), nlopt::LD_LBFGS);
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
    optimizer.optimize(traj_init, robot_traj, traj_opt);

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

bool test_probabilistic_cost(hri_planner::TestComponent::Request& req,
                            hri_planner::TestComponent::Response& res)
{
    // extract the messages
    Eigen::Map<Eigen::VectorXd> xr(req.xr.data(), req.xr.size());
    Eigen::Map<Eigen::VectorXd> ur(req.ur.data(), req.ur.size());
    Eigen::Map<Eigen::VectorXd> xh(req.xh.data(), req.xh.size());
    Eigen::Map<Eigen::VectorXd> uh(req.uh.data(), req.uh.size());
    Eigen::Map<Eigen::VectorXd> xr0(req.xr0.data(), req.xr0.size());
    Eigen::Map<Eigen::VectorXd> xh0(req.xh0.data(), req.xh0.size());

    std::string log_path = req.log_path;

    // create a log file to store the result
    std::ofstream logger(log_path + "/log_cost_prob.txt");
    logger << "start logging..." << std::endl;

    // create all the cost functions
    using namespace hri_planner;
    std::vector<std::shared_ptr<FeatureBase> > f_non_int;
    std::vector<std::shared_ptr<FeatureVectorizedBase> > f_int;

    // problem dimensions
    int T = 10;
    int nXh = 4;
    int nUh = 2;
    int nXr = 3;
    int nUr = 2;
    double dt = 0.5;

    // create cost features
    Eigen::VectorXd x_goal(2);
    x_goal << 4., 4.;

    create_robot_costs(f_non_int, f_int, x_goal);

    logger << "cost features created..." << std::endl;

    // create a belief model
    std::shared_ptr<BeliefModelBase> belief_model;
    create_belief_model(belief_model);

    logger << "belief model created..." << std::endl;

    // create the probabilistic cost component
    ProbabilisticCost cost(belief_model);

    int n_f_non_int = 2;
    int n_f_int = 2;

    std::vector<double> w_non_int(req.weights.begin(), req.weights.begin() + n_f_non_int);
    std::vector<double> w_int(req.weights.begin()+n_f_non_int, req.weights.begin()+n_f_non_int+n_f_int);

    cost.set_features_non_int(w_non_int, f_non_int);
    cost.set_features_int(w_int, f_int);

    logger << "probabilistic cost function created..." << std::endl;

    // create trajectories
    Trajectory robot_traj(DIFFERENTIAL_MODEL, T, dt);
    robot_traj.update(xr0, ur);

    Trajectory human_traj_hp(CONST_ACC_MODEL, T, dt);
    human_traj_hp.update(xh0, uh);

    Trajectory human_traj_rp(CONST_ACC_MODEL, T, dt);
    human_traj_rp.update(xh0, uh);

    Trajectory human_traj_pred(CONST_ACC_MODEL, T, dt);
    human_traj_pred.update(xh0, Eigen::VectorXd::Zero(ur.size()));

    // compute
    Eigen::VectorXd grad_ur;
    Eigen::VectorXd grad_uh_hp;
    Eigen::VectorXd grad_uh_rp;

    cost.update_human_pred(human_traj_pred);
    double val = cost.compute(robot_traj, human_traj_hp, human_traj_rp,
                              req.acomm, req.tcomm, grad_ur, grad_uh_hp, grad_uh_rp);

    // log the results
    logger << "computation finished, cost is: " << val << std::endl;
    logger << "gradient w.r.t. ur is: " << std::endl;
    logger << grad_ur.transpose() << std::endl;
    logger << "gradient w.r.t. uh_hp is: " << std::endl;
    logger << grad_uh_hp.transpose() << std::endl;
    logger << "gradient w.r.t. uh_rp is: " << std::endl;
    logger << grad_uh_rp.transpose() << std::endl;

    res.succeeded = true;

    return true;
}

bool test_nested_optimizer(hri_planner::TestComponent::Request& req,
                           hri_planner::TestComponent::Response& res)
{
    // flag that controls the type of optimizer to test
    bool flag_naive_optimizer = true;

    // extract the messages
    Eigen::Map<Eigen::VectorXd> xr(req.xr.data(), req.xr.size());
    Eigen::Map<Eigen::VectorXd> ur(req.ur.data(), req.ur.size());
    Eigen::Map<Eigen::VectorXd> xh(req.xh.data(), req.xh.size());
    Eigen::Map<Eigen::VectorXd> uh(req.uh.data(), req.uh.size());
    Eigen::Map<Eigen::VectorXd> xr0(req.xr0.data(), req.xr0.size());
    Eigen::Map<Eigen::VectorXd> xh0(req.xh0.data(), req.xh0.size());

    std::string log_path = req.log_path;

    // create a log file to store the result
    std::ofstream logger(log_path + "/log_nested_optimizer.txt");
    std::ofstream human_traj_logger(log_path + "/log_human_traj.txt");
    std::ofstream robot_traj_logger(log_path + "/log_robot_traj.txt");
    logger << "start logging..." << std::endl;

    using namespace hri_planner;

    // create human cost features for human priority
    std::vector<std::shared_ptr<FeatureBase> > features_hp;
    std::vector<std::shared_ptr<FeatureBase> > features_rp;

    Eigen::VectorXd x_goal(2);
    x_goal << 0.73216, 6.00955;

    create_human_costs(features_hp, x_goal);
    create_human_costs(features_rp, x_goal);

    logger << "human cost features created..." << std::endl;

    // create human cost functions
    int nf_human = 5;
    std::vector<double> w_hp(req.weights.begin(), req.weights.begin()+nf_human);
    std::vector<double> w_rp(req.weights.begin()+nf_human, req.weights.begin()+nf_human*2);

    auto cost_human_hp = std::make_shared<HumanCost>(w_hp, features_hp);
    auto cost_human_rp = std::make_shared<HumanCost>(w_rp, features_rp);
    auto single_cost_hp = std::make_shared<SingleTrajectoryCostHuman>(w_hp, features_hp);
    auto single_cost_rp = std::make_shared<SingleTrajectoryCostHuman>(w_rp, features_rp);

    logger << "human cost functions created..." << std::endl;

    // create robot cost features
    std::vector<std::shared_ptr<FeatureBase> > f_non_int;
    std::vector<std::shared_ptr<FeatureVectorizedBase> > f_int;

    x_goal << 4., 4.;
    create_robot_costs(f_non_int, f_int, x_goal);

    logger << "robot cost features created..." << std::endl;

    // create a belief model
    std::shared_ptr<BeliefModelBase> belief_model;
    create_belief_model(belief_model);

    logger << "belief model created..." << std::endl;

    // create the probabilistic cost component
    std::shared_ptr<ProbabilisticCostBase> robot_cost;
//    robot_cost = std::make_shared<ProbabilisticCost>(belief_model);
    robot_cost = std::make_shared<ProbabilisticCostSimplified>(belief_model);

    int n_f_non_int = 2;
    int n_f_int = 2;

    std::vector<double> w_robot(req.weights.begin()+nf_human*2, req.weights.end());
    std::vector<double> w_non_int(w_robot.begin(), w_robot.begin() + n_f_non_int);
    std::vector<double> w_int(w_robot.begin()+n_f_non_int, w_robot.end());

    robot_cost->set_features_non_int(w_non_int, f_non_int);
    robot_cost->set_features_int(w_int, f_int);

    logger << "probabilistic cost function created..." << std::endl;

    // create the nested optimizer
    // problem dimensions
    int T = 10;
    int nXh = 4;
    int nUh = 2;
    int nXr = 3;
    int nUr = 2;
    double dt = 0.5;

    int dim = T * (nUr + 2 * nUh);

    std::shared_ptr<NestedOptimizerBase> optimizer;

    if (flag_naive_optimizer) {
        int dim_r = T * nUr;
        int dim_h = T * nUh;

        optimizer = std::make_shared<NaiveNestedOptimizer>(static_cast<unsigned int>(dim_r),
                                                           static_cast<unsigned int>(dim_r),
                                                           nlopt::LD_SLSQP, nlopt::LD_SLSQP);
        optimizer->set_human_cost(single_cost_hp, single_cost_rp);
    }
    else {
        optimizer = std::make_shared<NestedTrajectoryOptimizer>(static_cast<unsigned int>(dim),
                                                                nlopt::LD_SLSQP);
        optimizer->set_human_cost(cost_human_hp, cost_human_rp);
    }

    // set robot cost func
    optimizer->set_robot_cost(robot_cost);

    // set bounds
    Eigen::VectorXd lb_ur(T * nUr);
    Eigen::VectorXd ub_ur(T * nUr);
    Eigen::VectorXd lb_uh(T * nUh);
    Eigen::VectorXd ub_uh(T * nUh);

    for (int t = 0; t < T; ++t) {
        int stu = t * nUr;
        lb_ur(stu) = -0.55;
        ub_ur(stu) = 0.55;
        lb_ur(stu+1) = -2.0;
        ub_ur(stu+1) = 2.0;
    }

    lb_uh.setOnes(); lb_uh *= -10.0;
    ub_uh.setOnes(); ub_uh *= 10.0;

    optimizer->set_bounds(lb_ur, ub_ur, lb_uh, ub_uh);
    logger << "nested optimizer created..." << std::endl;

    // set start trajectories
    //add in some offset for the robot inintial condition and perturb the control
//    Eigen::VectorXd xr0_offset(nXr);
//    xr0_offset << 0.3, 0.3, 0.0;
//    xr0 += xr0_offset;
    for (int t = 0; t < T; ++t) {
        ur(t * 2) = ur(2);
        ur(t * 2 + 1) = 0.0;
    }

    logger << "initial control is: " << std::endl;
    logger << ur.transpose() << std::endl;

    Trajectory robot_traj(DIFFERENTIAL_MODEL, T, dt);
    robot_traj.update(xr0, ur);

//    robot_traj_logger << robot_traj.x.transpose() << std::endl;

    Trajectory human_traj_hp(CONST_ACC_MODEL, T, dt);
    human_traj_hp.update(xh0, uh);

    Trajectory human_traj_rp(CONST_ACC_MODEL, T, dt);
    human_traj_rp.update(xh0, uh);

    // first generate an initial guess of the robot trajectory
    // here just use the recorded trajectory
    // use this to generate feasible starting points
    // create a simple optimizer
    // first need to create single trajectory costs
    ros::Time t_start;
    Trajectory robot_traj_opt(DIFFERENTIAL_MODEL, T ,dt);
    Trajectory human_traj_hp_new(CONST_ACC_MODEL, T, dt);
    Trajectory human_traj_rp_new(CONST_ACC_MODEL, T, dt);

    if (!flag_naive_optimizer) {
        dim = T * nUh;
        TrajectoryOptimizer human_optimizer_hp(static_cast<unsigned int>(dim), nlopt::LD_MMA);
        TrajectoryOptimizer human_optimizer_rp(static_cast<unsigned int>(dim), nlopt::LD_MMA);

        human_optimizer_hp.set_cost_function(single_cost_hp);
        human_optimizer_rp.set_cost_function(single_cost_rp);

        human_optimizer_hp.set_bounds(lb_uh, ub_uh);
        human_optimizer_rp.set_bounds(lb_uh, ub_uh);

        logger << "created optimizers for initializing human trajectories..." << std::endl;

        // optimize to get initial human trajectories
        Trajectory human_traj_hp_opt(CONST_ACC_MODEL, T, dt);
        Trajectory human_traj_rp_opt(CONST_ACC_MODEL, T, dt);

        human_optimizer_hp.optimize(human_traj_hp, robot_traj, human_traj_hp_opt);
        human_traj_logger << human_traj_hp_opt.x.transpose() << std::endl;

        human_optimizer_rp.optimize(human_traj_rp, robot_traj, human_traj_rp_opt);
        human_traj_logger << human_traj_rp_opt.x.transpose() << std::endl;

        logger << "initialized human trajectories!" << std::endl;

        logger << "initial robot pose is: " << xr0.transpose() << std::endl;
        logger << "initial human pose is: " << xh0.transpose() << std::endl;

        // check if the initial trajectories are feasible
        double err = std::dynamic_pointer_cast<NestedTrajectoryOptimizer>(optimizer)->
                check_constraint(robot_traj, human_traj_hp_opt, human_traj_rp_opt);
        logger << "constraint error: " << err << std::endl;

        // perform the full optimization!
        t_start = ros::Time::now();
        optimizer->optimize(robot_traj, human_traj_hp_opt, human_traj_rp_opt, req.acomm, req.tcomm,
                            robot_traj_opt, &human_traj_hp_new, &human_traj_rp_new);
    }
    else {
        // perform the full optimization!
        t_start = ros::Time::now();
        optimizer->optimize(robot_traj, human_traj_hp, human_traj_rp, req.acomm, req.tcomm,
                            robot_traj_opt, &human_traj_hp_new, &human_traj_rp_new);
    }

    // log the data
    ros::Duration t_elapse = ros::Time::now() - t_start;
    logger << "optimization finished! time taken: " << t_elapse.toSec() << std::endl;
    logger << "optimized robot trajectory is:" << std::endl;
    logger << robot_traj_opt.x.transpose() << std::endl;
    logger << "optimized robot control is:" << std::endl;
    logger << robot_traj_opt.u.transpose() << std::endl;
    logger << "difference from initial control: " << std::endl;
    logger << robot_traj_opt.u.transpose() - robot_traj.u.transpose() << std::endl;

    robot_traj_logger << robot_traj_opt.x.transpose() << std::endl;
    human_traj_logger << human_traj_hp_new.x.transpose() << std::endl;
    human_traj_logger << human_traj_rp_new.x.transpose() << std::endl;

    // close all loggers
    logger.close();
    human_traj_logger.close();
    robot_traj_logger.close();

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
    ros::ServiceServer prob_cost_service = n.advertiseService("test_prob_cost", test_probabilistic_cost);
    ros::ServiceServer nested_optimizer_service = n.advertiseService("test_nested_optimizer", test_nested_optimizer);

    ROS_INFO("Services are ready!");
    ros::spin();

    return 0;
}