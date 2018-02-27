//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 2/25/2017
// Last revision: 2/25/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#ifndef HRI_PLANNER_HUMAN_BELIEF_MODEL_H
#define HRI_PLANNER_HUMAN_BELIEF_MODEL_H

#include <iostream>
#include <utility>
#include <vector>
#include <memory>

#include <Eigen/Dense>

#include "hri_planner/shared_config.h"

namespace hri_planner {

using Eigen::VectorXd;
using Eigen::Vector2d;

class BeliefModelBase {
public:
    // just use default constructor?
    explicit BeliefModelBase(std::shared_ptr<SharedConfig> config): config_(std::move(config)) {};

    // virtual destructor
    virtual ~BeliefModelBase() = default;

    // main update
    void belief_update(const VectorXd& xr, const VectorXd& ur, const VectorXd& xh0,
                       const int acomm, const double tcomm, const double tcurr, Vector2d& belief);

protected:
    // dimensions
    std::shared_ptr<SharedConfig> config_;

    // calculate effect of implicit communication
    virtual double belief_implicit(const int intent, const VectorXd& xr, const VectorXd& ur,
                                   const VectorXd& xh0) = 0;

    virtual double belief_explicit(const int intent, const double tcurr,
                                   const int acomm, const double tcomm) = 0;

};

class BeliefModelExponential: public BeliefModelBase {
public:
    BeliefModelExponential(std::shared_ptr<SharedConfig> config, double ratio, double decay_rate);

protected:
    double belief_implicit(const int intent, const VectorXd& xr,
                           const VectorXd& ur, const VectorXd& xh0) override;

    double belief_explicit(const int intent, const double tcurr,
                           const int acomm, const double tcomm) override;

    // implicit belief update helper functions
    virtual double implicit_cost(const int intent, const VectorXd& xr,
                                 const VectorXd& ur, const VectorXd& xh0);

private:
    // explicit belief update parameters
    double norm_factor_;
    double decay_rate_;
    double ratio_;

    // implicit belief update parameters

};

}

#endif //HRI_PLANNER_HUMAN_BELIEF_MODEL_H
