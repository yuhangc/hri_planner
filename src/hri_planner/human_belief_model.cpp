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

    std::deque<double> cost_hist(cost_hist_hp_.begin(), cost_hist_hp_.end());
    double cost_hp = std::accumulate(cost_hist.begin(), cost_hist.end(), 0.0);

    int nXr = robot_traj.state_size();
    int nXh = human_traj.state_size();
    int nUr = robot_traj.control_size();
    for (int t = 0; t < T; ++t) {
        // compute the cost
        int str = t * nXr;
        int sth = t * nXh;
        int stu = t * nUr;

        Eigen::Vector2d x_rel(human_traj.x(sth) - robot_traj.x(str),
                              human_traj.x(sth+1) - robot_traj.x(str+1));

        double th = robot_traj.x(str+2);
        Eigen::Vector2d u_rel(robot_traj.u(stu) * std::cos(th), robot_traj.u(stu) * std::sin(th));

        double prod = u_rel.dot(x_rel);

        // only non-zero when prod is greater than 0
        if (prod > 0) {
            double d = x_rel.squaredNorm();

            if (d > 1.0) {
                // compute the cost value
                double cost = prod / d;
                cost_hp += cost;

                cost_hist.push_back(cost);
                if (cost_hist.size() > T_hist_) {
                    cost_hp -= cost_hist.front();
                    cost_hist.pop_front();
                }

                costs(t) = cost_hp;

                // compute gradient
                grad_u(stu) = cost / robot_traj.u(stu);

                double dd = (x_rel(0) * x_rel(0) - x_rel(1) * x_rel(1)) / (d*d);
                grad_x(str) = dd * u_rel(0);
                grad_x(str+1) = -dd * u_rel(1);

                im_jacobian.block(t, 0, 1, stu+nUr) = grad_u.segment(0, stu+nUr).transpose() +
                        grad_x.segment(0, str+nXr) * robot_traj.Ju.topLeftCorner(str+nXr, stu+nUr);
            }
            else {
                // compute the cost value
                double cost = prod / 1.0;
                cost_hp += cost;

                cost_hist.push_back(cost);
                if (cost_hist.size() > T_hist_) {
                    cost_hp -= cost_hist.front();
                    cost_hist.pop_front();
                }

                costs(t) = cost_hp;

                // compute gradient
                grad_u(stu) = cost / robot_traj.u(stu);

                grad_x(str) = -u_rel(0);
                grad_x(str+1) = -u_rel(1);

                im_jacobian.block(t, 0, 1, stu+nUr) = grad_u.segment(0, stu+nUr).transpose() +
                        grad_x.segment(0, str+nXr) * robot_traj.Ju.topLeftCorner(str+nXr, stu+nUr);
            }
        }
    }
}

}