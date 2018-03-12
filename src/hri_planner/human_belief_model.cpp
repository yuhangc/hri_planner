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

//    std::cout << belief.transpose() << std::endl;

    // normalize belief
    belief /= (belief(HumanPriority) + belief(RobotPriority));
}

//----------------------------------------------------------------------------------
BeliefModelExponential::BeliefModelExponential(std::shared_ptr<SharedConfig> config, double ratio,
                                               double decay_rate, const std::vector<double>& fcorrection):
        BeliefModelBase(std::move(config)), ratio_(ratio), decay_rate_(decay_rate), fcorrection_(fcorrection)
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

    return std::exp(-implicit_cost(intent, xr, ur, xh0) * fcorrection_[intent]) * normalizer;
}

//----------------------------------------------------------------------------------
double BeliefModelExponential::implicit_cost(const int intent, const VectorXd &xr,
                                             const VectorXd &ur, const VectorXd &xh0)
{
    double cost = 0;

    if (intent == HumanPriority) {
        // human priority - penalize large velocity towards human
        for (int i = 0; i < config_->T; ++i) {
            double vr = ur(i * config_->nUr);
            double th = xr(i * config_->nXr + 2);
            Vector2d ui(vr * std::cos(th), vr * std::sin(th));
            Vector2d xrel = const_cast<VectorXd&>(xh0).head(2) -
                    const_cast<VectorXd&>(xr).segment(i * config_->nXr, 2);

//            std::cout << "robot linear v: " << vr << ",  robot orientation: " << th << std::endl;
//            std::cout << "robot vel: " << ui.transpose() << std::endl;
//            std::cout << "relative position: " << xrel.transpose() << std::endl;

            double prod = ui.dot(xrel);
            if (prod > 0)
                cost += prod / std::max(1.0, xrel.squaredNorm());
        }
    }
    else {
        // robot priority - penalize velocity changes
//        VectorXd udiff = const_cast<VectorXd&>(ur).head(ur.size() - config_->nUr) -
//                const_cast<VectorXd&>(ur).tail(ur.size() - config_->nUr);
//        cost = udiff.dot(udiff);
        for (int i = 1; i < config_->T; ++i) {
            double v_inc = ur(i * config_->nUr) - ur((i-1) * config_->nUr);
            cost += v_inc * v_inc;
        }
    }

    std::cout << "intent: " << intent << ",  cost: " << cost << std::endl;
    return cost;
}

}