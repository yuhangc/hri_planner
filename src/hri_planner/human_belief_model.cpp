//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 2/25/2017
// Last revision: 3/22/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include <iostream>
#include <cmath>

#include "hri_planner/human_belief_model.h"

namespace hri_planner {

//----------------------------------------------------------------------------------
double BeliefModelBase::update_belief(int acomm, double tcomm, double tcurr)
{
    // compute cost sum
    double cost_hp = std::accumulate(cost_hist_hp_.begin(), cost_hist_hp_.end(), 0.0);
    double cost_rp = std::accumulate(cost_hist_rp_.begin(), cost_hist_rp_.end(), 0.0);

    // update belief
    double p_hp = std::exp(-cost_hp * fcorrection_[HumanPriority]) *
                  belief_explicit(HumanPriority, tcurr, acomm, tcomm);
    double p_rp = std::exp(-cost_rp * fcorrection_[RobotPriority]) *
                  belief_explicit(RobotPriority, tcurr, acomm, tcomm);

    return p_hp / (p_hp + p_rp);
}

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
    double p_hp = std::exp(-cost_hp * fcorrection_[HumanPriority]) *
            belief_explicit(HumanPriority, t0, acomm, tcomm);
    double p_rp = std::exp(-cost_rp * fcorrection_[RobotPriority]) *
            belief_explicit(RobotPriority, t0, acomm, tcomm);

    prob_hp_ = p_hp / (p_hp + p_rp);
    return prob_hp_;
}

//----------------------------------------------------------------------------------
void BeliefModelBase::update_belief(const Trajectory &robot_traj, const Trajectory &human_traj, int acomm,
                                    double tcomm, double t0, Eigen::VectorXd &belief, Eigen::MatrixXd &jacobian)
{
    // compute the implicit costs
    Eigen::VectorXd costs_hp;
    Eigen::VectorXd costs_rp;
    Eigen::MatrixXd jacobian_hp;
    Eigen::MatrixXd jacobian_rp;

    implicit_cost(robot_traj, human_traj, costs_hp, jacobian_hp, costs_rp, jacobian_rp);

    // compute the probabilities and jacobians
    double dt = robot_traj.dt();
    double tcurr = t0;
    for (int t = 0; t < robot_traj.horizon(); ++t) {
        // explicit part
        tcurr += dt;
        double p_ex_hp = belief_explicit(HumanPriority, tcurr, acomm, tcomm);
        double p_ex_rp = belief_explicit(RobotPriority, tcurr, acomm, tcomm);

        // implicit part
        double p_im_hp = std::exp(-fcorrection_[HumanPriority] * costs_hp(t));
        double p_im_rp = std::exp(-fcorrection_[RobotPriority] * costs_rp(t));

        // probability
        double den_inv = 1.0 / (p_ex_hp * p_im_hp + p_ex_rp * p_im_rp);
        belief(t) = p_ex_hp * p_im_hp * den_inv;

        // gradient/jacobian
        jacobian.row(t) = den_inv * den_inv * p_ex_hp * p_im_hp * p_ex_rp * p_im_rp *
                (fcorrection_[RobotPriority] * jacobian_rp.row(t) - fcorrection_[HumanPriority] * jacobian_hp.row(t));
    }
}

//----------------------------------------------------------------------------------
void BeliefModelBase::reset_hist(const Eigen::VectorXd &ur0)
{
    ur_last_ = ur0;

    cost_hist_hp_.clear();
    cost_hist_rp_.clear();
}

//----------------------------------------------------------------------------------
void BeliefModelBase::update_cost_hist(double ct, std::deque<double> &ct_hist, double &cost)
{
    cost += ct;

    ct_hist.push_back(ct);
    if (ct_hist.size() > T_hist_) {
        cost -= ct_hist.front();
        ct_hist.pop_front();
    }
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
        double v_inc = ur(0) - ur_last_(0);
        cost += v_inc * v_inc;

        ur_last_ = ur;
    }

    return cost;
}

//----------------------------------------------------------------------------------
void BeliefModelExponential::implicit_cost(const Trajectory& robot_traj, const Trajectory& human_traj,
                                           Eigen::VectorXd& costs_hp, Eigen::MatrixXd& jacobian_hp,
                                           Eigen::VectorXd& costs_rp, Eigen::MatrixXd& jacobian_rp)
{
    int T = robot_traj.horizon();
    costs_hp.setZero(T);
    costs_rp.setZero(T);
    jacobian_hp.setZero(T, robot_traj.traj_control_size());
    jacobian_rp.setZero(T, robot_traj.traj_control_size());

    Eigen::VectorXd grad_u(robot_traj.traj_control_size());
    Eigen::VectorXd grad_x(robot_traj.traj_state_size());

    Eigen::VectorXd grad_u_rp(robot_traj.traj_control_size());

    Eigen::VectorXd u_last = ur_last_;

    std::deque<double> cost_hist_hp(cost_hist_hp_.begin(), cost_hist_hp_.end());
    std::deque<double> cost_hist_rp(cost_hist_rp_.begin(), cost_hist_rp_.end());
    double cost_hp = std::accumulate(cost_hist_hp.begin(), cost_hist_hp.end(), 0.0);
    double cost_rp = std::accumulate(cost_hist_rp.begin(), cost_hist_rp.end(), 0.0);

    int nXr = robot_traj.state_size();
    int nXh = human_traj.state_size();
    int nUr = robot_traj.control_size();
    for (int t = 0; t < T; ++t) {
        // compute the cost
        int str = t * nXr;
        int sth = t * nXh;
        int stu = t * nUr;
        double vr = robot_traj.u(stu);

        Eigen::Vector2d x_rel(human_traj.x(sth) - robot_traj.x(str),
                              human_traj.x(sth+1) - robot_traj.x(str+1));

        double th = robot_traj.x(str+2);
        Eigen::Vector2d u_rel(vr * std::cos(th), vr * std::sin(th));

        double prod = u_rel.dot(x_rel);

        double ct_hp, ct_rp;
        Eigen::Vector2d gradu_hp;
        Eigen::VectorXd gradu_rp(robot_traj.traj_control_size());
        Eigen::Vector2d gradx_hp;

        // only non-zero when prod is greater than 0
        if (prod > 0) {
            //! compute cost for robot priority
            double v_inc = 0.0;
            v_inc = vr - u_last(0);

            ct_rp = v_inc * v_inc;

            if (t > 0) {
                gradu_rp(stu-2) = -2.0 * v_inc;
            }
            gradu_rp(stu) = 2.0 * v_inc;

            u_last(0) = robot_traj.u(stu);

            //! compute cost for human priority
            double d = x_rel.squaredNorm();

            if (d > 1.0) {
                // compute the cost value
                ct_hp = prod / d;

                // gradient w.r.t. u
                gradu_hp << ct_hp / vr, 0.0;

                // gradient w.r.t. x
                double dd = (x_rel(0) * x_rel(0) - x_rel(1) * x_rel(1)) / (d * d);
                gradx_hp << dd * u_rel(0), -dd * u_rel(1);
            } else {
                ct_hp = prod;

                // gradient w.r.t. u
                gradu_hp << ct_hp / vr, 0.0;

                // gradient w.r.t. x
                gradx_hp << -u_rel(0), -u_rel(1);
            }
        }
        else {
            ct_hp = 0.0;
            ct_rp = 0.0;
            gradu_hp.setZero();
            gradu_rp.setZero();
            gradx_hp.setZero();
        }

        // update total cost and cost history
        update_cost_hist(ct_hp, cost_hist_hp, cost_hp);
        update_cost_hist(ct_rp, cost_hist_rp, cost_rp);

        // assign to output cost
        costs_hp(t) = cost_hp;
        costs_rp(t) = cost_rp;

        // update gradients
        if (t > 0) {
            jacobian_hp.row(t) = jacobian_hp.row(t-1);
            jacobian_rp.row(t) = jacobian_rp.row(t-1);
        }

        // human priority
        jacobian_hp.block(t, stu, 1, 2) = gradu_hp.transpose();
        jacobian_hp.row(t) = jacobian_hp.row(t) + gradx_hp.transpose() * robot_traj.Ju.block(str, 0, 2, robot_traj.Ju.cols());

        // robot priority
        jacobian_rp.row(t) = jacobian_rp.row(t) + gradu_rp.transpose();
    }
}

}