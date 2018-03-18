//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 2/25/2017
// Last revision: 3/17/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include <cmath>

#include "hri_planner/human_belief_model.h"

namespace hri_planner {

//----------------------------------------------------------------------------------
double BeliefModelBase::update_belief(const Eigen::VectorXd &xr, const Eigen::VectorXd &ur, const Eigen::VectorXd &xh,
                                      int acomm, double tcomm, double t0)
{
    // add in new costs
    cost_hist_hp_.push_back(implicit_cost_simple(HumanPriority, xr, ur, xh));
    cost_hist_rp_.push_back(implicit_cost_simple(RobotPriority, xr, ur, xh));

    // pop out old ones
    if (cost_hist_hp_.size() > T_hist_)
        cost_hist_hp_.pop_front();
    if (cost_hist_rp_.size() > T_hist_)
        cost_hist_rp_.pop_front();

    // compute cost sum
    double cost_hp = std::accumulate(cost_hist_hp_.begin(), cost_hist_hp_.end(), 0.0);
    double cost_rp = std::accumulate(cost_hist_rp_.begin(), cost_hist_rp_.end(), 0.0);

    // update belief
    double normalizer = 1.0;

    double p_hp = std::exp(-cost_hp * fcorrection_[HumanPriority]) *
            belief_explicit(HumanPriority, t0, acomm, tcomm);
    double p_rp = std::exp(-cost_rp * fcorrection_[RobotPriority]) *
            belief_explicit(RobotPriority, t0, acomm, tcomm);

    return p_hp / (p_hp + p_rp);
}

//----------------------------------------------------------------------------------
void BeliefModelBase::update_belief(const Trajectory &robot_traj, const Trajectory &human_traj, int acomm,
                                    double tcomm, double t0, Eigen::VectorXd &belief, Eigen::MatrixXd &jacobian)
{

}

//----------------------------------------------------------------------------------
BeliefModelExponential::BeliefModelExponential(int T_hist, const std::vector<double>& fcorrection,
                                               double ratio, double decay_rate):
        BeliefModelBase(T_hist, fcorrection), ratio_(ratio), decay_rate_(decay_rate)
{
    // TODO: pre-compute the normalization factors?
}

//----------------------------------------------------------------------------------
double BeliefModelExponential::belief_explicit(const int intent, const double tcurr,
                                               const int acomm, const double tcomm)
{
    if (intent != acomm)
        return 1.0;

    return 1.0 + ratio_ * std::exp((tcomm - tcurr) / decay_rate_);
}

//----------------------------------------------------------------------------------
double BeliefModelExponential::implicit_cost_simple(const int intent, const Eigen::VectorXd &xr,
                                                  const Eigen::VectorXd &ur, const Eigen::VectorXd &xh)
{
    double cost = 0.0;

    Eigen::Vector2d x_rel(xh(0) - xr(0), xh(1) - xr(1));
    Eigen::Vector2d u_rel(ur(0) * std::cos(xr(2)), ur(0) * std::sin(xr(2)));

    double prod = u_rel.dot(x_rel);

    if (prod <= 0)
        return cost;

    if (intent == HumanPriority) {
        cost += prod / std::max(1.0, x_rel.squaredNorm());
    }
    else {
        if (ur_last_.size() > 0) {
            double v_inc = ur(0) - ur_last_(0);
            cost += v_inc * v_inc;
        }

        ur_last_ = ur;
    }

    return cost;
}

//----------------------------------------------------------------------------------
void BeliefModelExponential::implicit_cost_hp(const Trajectory &robot_traj, const Trajectory &human_traj,
                                              Eigen::VectorXd &costs, Eigen::MatrixXd &im_jacobian)
{
    costs.setZero(robot_traj.traj_control_size());
    im_jacobian.setZero(robot_traj.horizon(), robot_traj.traj_control_size());
}

//----------------------------------------------------------------------------------
void BeliefModelExponential::implicit_cost_rp(const Trajectory &robot_traj, const Trajectory &human_traj,
                                              Eigen::VectorXd &costs, Eigen::MatrixXd &im_jacobian)
{

}

}