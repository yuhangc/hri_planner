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

#include <cmath>

#include "hri_planner/human_belief_model.h"

namespace hri_planner {

//----------------------------------------------------------------------------------
void BeliefModelBase::belief_update(const VectorXd &xr, const VectorXd &ur, const VectorXd &xh0,
                                    const int acomm, const double tcomm, const double tcurr, Vector2d& belief)
{
    // simple Bayes update
    belief(HumanPriority) = belief_implicit(HumanPriority, xr, ur, xh0) *
            belief_explicit(HumanPriority, tcurr, acomm, tcomm);
    belief(RobotPriority) = belief_implicit(RobotPriority, xr, ur, xh0) *
                            belief_explicit(RobotPriority, tcurr, acomm, tcomm);

    // normalize belief
    belief.normalize();
}

//----------------------------------------------------------------------------------
BeliefModelExponential::BeliefModelExponential(std::shared_ptr<SharedConfig> config,
                                               double ratio, double decay_rate):
        BeliefModelBase(std::move(config)), ratio_(ratio), decay_rate_(decay_rate)
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
double BeliefModelExponential::belief_implicit(const int intent, const VectorXd &xr,
                                               const VectorXd &ur, const VectorXd &xh0)
{
    // TODO: calculate normalization factor
    // TODO: this may depend on the hessian
    double normalizer = 1.0;

    return std::exp(-implicit_cost(intent, xr, ur, xh0)) * normalizer;
}

//----------------------------------------------------------------------------------
double BeliefModelExponential::implicit_cost(const int intent, const VectorXd &xr,
                                             const VectorXd &ur, const VectorXd &xh0)
{
    double cost = 0;

    if (intent == HumanPriority) {
        // human priority - penalize large velocity towards human
        for (int i = 0; i < config_->T; ++i) {
            cost +=
        }
    }
    else {
        // robot priority - penalize velocity changes
        for (int i = 1; i < config_->T; ++i) {
            int st = i * config_->nUr;
            int st_last = st - config_->nUr;
            cost += ur.segment(st, config_->nUr) - ur.segment(st_last, config_->nUr);
        }
    }

    return cost;
}

}