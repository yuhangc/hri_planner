//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/7/2017
// Last revision: 3/8/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#ifndef HRI_PLANNER_COSTS_H
#define HRI_PLANNER_COSTS_H

#include <vector>
#include <memory>
#include <Eigen/Dense>

#include "hri_planner/cost_feature_bases.h"

namespace hri_planner {

//! a linear cost/reward function class
class LinearCost: public FeatureBase {
public:
    // default constructor
    LinearCost() {
        nfeatures_ = 0;
    }

    // constructor with weights and features
    LinearCost(const std::vector<float>& weights, std::vector<std::shared_ptr<FeatureBase> >& features):
            weights_(weights), features_(features)
    {
        nfeatures_ = (int) weights.size();
    }

    // virtual destructor
    virtual ~LinearCost(){};

    virtual void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad);
    virtual void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad);

    // incrementally add in features
    void add_feature(float weight, FeatureBase* feature);
    void add_feature(float weight, const std::shared_ptr<FeatureBase> feature);

protected:
    virtual float compute(const Trajectory& robot_traj, const Trajectory& human_traj);

    int nfeatures_;
    std::vector<float> weights_;
    std::vector<std::shared_ptr<FeatureBase> > features_;
};

//! human cost class - extend linear cost to calculate hessians
class HumanCost: public LinearCost {
public:
    virtual void hessian_uh(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess);
    virtual void hessian_uh_ur(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess);
};

} // namespace

#endif //HRI_PLANNER_COSTS_H
