//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 2/25/2017
// Last revision: 2/27/2017
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

using Eigen::VectorXf;
using Eigen::Vector2f;

class BeliefModelBase {
public:
    // just use default constructor?
    explicit BeliefModelBase(std::shared_ptr<SharedConfig> config): config_(std::move(config)) {};

    // virtual destructor
    virtual ~BeliefModelBase() = default;

    // main update
    void belief_update(const VectorXf& xr, const VectorXf& ur, const VectorXf& xh0,
                       int acomm, float tcomm, float tcurr, Vector2f& belief);

protected:
    // dimensions
    std::shared_ptr<SharedConfig> config_;

    // calculate effect of implicit communication
    virtual float belief_implicit(const int intent, const VectorXf& xr, const VectorXf& ur,
                                   const VectorXf& xh0) = 0;

    virtual float belief_explicit(const int intent, const float tcurr,
                                   const int acomm, const float tcomm) = 0;

};

class BeliefModelExponential: public BeliefModelBase {
public:
    BeliefModelExponential(std::shared_ptr<SharedConfig> config, float ratio,
                           float decay_rate, const std::vector<float>& fcorrection);

protected:
    float belief_implicit(const int intent, const VectorXf& xr,
                           const VectorXf& ur, const VectorXf& xh0) override;

    float belief_explicit(const int intent, const float tcurr,
                           const int acomm, const float tcomm) override;

    // implicit belief update helper functions
    virtual float implicit_cost(const int intent, const VectorXf& xr,
                                 const VectorXf& ur, const VectorXf& xh0);

private:
    // explicit belief update parameters
    float norm_factor_;
    float decay_rate_;
    float ratio_;

    // implicit belief update parameters
    std::vector<float> fcorrection_;
};

}

#endif //HRI_PLANNER_HUMAN_BELIEF_MODEL_H
