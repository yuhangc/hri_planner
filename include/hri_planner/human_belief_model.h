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

#ifndef HRI_PLANNER_HUMAN_BELIEF_MODEL_H
#define HRI_PLANNER_HUMAN_BELIEF_MODEL_H

#include <iostream>
#include <utility>
#include <vector>
#include <deque>
#include <memory>

#include <Eigen/Dense>

#include "hri_planner/shared_config.h"
#include "hri_planner/trajectory.h"
#include "hri_planner/cost_feature_bases.h"

namespace hri_planner {

class BeliefModelBase {
public:
    // requires the history length
    explicit BeliefModelBase(int T_hist, const std::vector<double>& fcorrection):
            T_hist_(T_hist), fcorrection_(fcorrection) {ur_last_.setZero(0);};

    // virtual destructor
    virtual ~BeliefModelBase() = default;

    // over loading compute belief
    double update_belief(const Eigen::VectorXd& xr, const Eigen::VectorXd& ur, const Eigen::VectorXd& xh,
                         int acomm, double tcomm, double t0);
    void update_belief(const Trajectory& robot_traj, const Trajectory& human_traj,
                       int acomm, double tcomm, double t0, Eigen::VectorXd& belief, Eigen::MatrixXd& jacobian);

protected:
    // dimensions
    int T_hist_;

    // history of implicit costs
    std::deque<double> cost_hist_hp_;
    std::deque<double> cost_hist_rp_;

    Eigen::VectorXd ur_last_;

    // implicit belief update parameters
    std::vector<double> fcorrection_;

    // calculate effect of implicit communication
    virtual double implicit_cost_simple(const int intent, const Eigen::VectorXd& xr,
                                        const Eigen::VectorXd& ur, const Eigen::VectorXd& xh) = 0;
    virtual void implicit_cost_hp(const Trajectory& robot_traj, const Trajectory& human_traj,
                                  Eigen::VectorXd& costs, Eigen::MatrixXd& im_jacobian) = 0;
    virtual void implicit_cost_rp(const Trajectory& robot_traj, const Trajectory& human_traj,
                                  Eigen::VectorXd& costs, Eigen::MatrixXd& im_jacobian) = 0;

    virtual double belief_explicit(const int intent, const double tcurr,
                                   const int acomm, const double tcomm) = 0;

};

class BeliefModelExponential: public BeliefModelBase {
public:
    BeliefModelExponential(int T_hist, const std::vector<double>& fcorrection, double ratio, double decay_rate);

protected:
    double implicit_cost_simple(const int intent, const Eigen::VectorXd& xr,
                                const Eigen::VectorXd& ur, const Eigen::VectorXd& xh) override;
    void implicit_cost_hp(const Trajectory& robot_traj, const Trajectory& human_traj,
                          Eigen::VectorXd& costs, Eigen::MatrixXd& im_jacobian) override;
    void implicit_cost_rp(const Trajectory& robot_traj, const Trajectory& human_traj,
                          Eigen::VectorXd& costs, Eigen::MatrixXd& im_jacobian) override;

    double belief_explicit(const int intent, const double tcurr,
                           const int acomm, const double tcomm) override;

private:
    // explicit belief update parameters
    double norm_factor_;
    double decay_rate_;
    double ratio_;
};

}

#endif //HRI_PLANNER_HUMAN_BELIEF_MODEL_H
