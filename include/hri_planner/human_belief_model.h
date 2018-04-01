//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 2/25/2017
// Last revision: 3/31/2017
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
            T_hist_(T_hist), fcorrection_(fcorrection) {prob_hp_ = 0.5;};

    // virtual destructor
    virtual ~BeliefModelBase() = default;

    // overloading compute belief
    double update_belief(int acomm, double tcomm, double tcurr);
    double update_belief(const Eigen::VectorXd& xr, const Eigen::VectorXd& ur, const Eigen::VectorXd& xh,
                         int acomm, double tcomm, double t0);
    void update_belief(const Trajectory& robot_traj, const Trajectory& human_traj,
                       int acomm, double tcomm, double t0, Eigen::VectorXd& belief, Eigen::MatrixXd& jacobian);

    // reset
    void reset_hist(const Eigen::VectorXd& ur0);

    // get latest belief
    double get_belief() const {
        return prob_hp_;
    }

    // set velocity from navigation function/control
    void set_ur_nav(const Eigen::VectorXd& ur_nav) {
        ur_nav_ = ur_nav;
    }

protected:
    // dimensions
    int T_hist_;

    // current belief
    double prob_hp_;

    // history of implicit costs
    std::deque<double> cost_hist_hp_;
    std::deque<double> cost_hist_rp_;

    // accumulated costs
    double cost_hp_;
    double cost_rp_;

    Eigen::VectorXd ur_last_;
    Eigen::VectorXd ur_nav_;

    // implicit belief update parameters
    std::vector<double> fcorrection_;

    // calculate effect of implicit communication
    virtual double implicit_cost_simple(const int intent, const Eigen::VectorXd& xr,
                                        const Eigen::VectorXd& ur, const Eigen::VectorXd& xh) = 0;
    virtual void implicit_cost(const Trajectory& robot_traj, const Trajectory& human_traj,
                               Eigen::VectorXd& costs_hp, Eigen::MatrixXd& jacobian_hp,
                               Eigen::VectorXd& costs_rp, Eigen::MatrixXd& jacobian_rp) = 0;

    virtual double belief_explicit(const int intent, const double tcurr,
                                   const int acomm, const double tcomm) = 0;

    // helper functions
    void update_cost_hist(double ct, std::deque<double>& ct_hist, double& cost);

};

class BeliefModelExponential: public BeliefModelBase {
public:
    BeliefModelExponential(int T_hist, const std::vector<double>& fcorrection, double ratio, double decay_rate);

protected:
    double implicit_cost_simple(const int intent, const Eigen::VectorXd& xr,
                                const Eigen::VectorXd& ur, const Eigen::VectorXd& xh) override;
    void implicit_cost(const Trajectory& robot_traj, const Trajectory& human_traj,
                       Eigen::VectorXd& costs_hp, Eigen::MatrixXd& jacobian_hp,
                       Eigen::VectorXd& costs_rp, Eigen::MatrixXd& jacobian_rp) override;

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
