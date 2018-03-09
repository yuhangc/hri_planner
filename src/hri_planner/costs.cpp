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

#include "hri_planner/costs.h"

namespace hri_planner {

//----------------------------------------------------------------------------------
float LinearCost::compute(const Trajectory &robot_traj, const Trajectory &human_traj)
{
    float cost = 0.0;

    for (int i = 0; i < nfeatures_; ++i) {
        cost += weights_[i] * features_[i]->compute(robot_traj, human_traj);
    }

    return cost;
}

//----------------------------------------------------------------------------------
void LinearCost::grad_ur(const Trajectory &robot_traj, const Trajectory &human_traj, VecRef grad)
{
    grad.setZero();

    int len = human_traj.traj_control_size();
    for (int i = 0; i < nfeatures_; ++i) {
        Eigen::VectorXf grad_f(len);
        features_[i]->grad_ur(robot_traj, human_traj, grad_f);

        grad += weights_[i] * grad_f;
    }
}

//----------------------------------------------------------------------------------
void LinearCost::grad_uh(const Trajectory &robot_traj, const Trajectory &human_traj, VecRef grad)
{
    grad.setZero();

    int len = human_traj.traj_control_size();
    for (int i = 0; i < nfeatures_; ++i) {
        Eigen::VectorXf grad_f(len);
        features_[i]->grad_uh(robot_traj, human_traj, grad_f);

        grad += weights_[i] * grad_f;
    }
}

//----------------------------------------------------------------------------------
void LinearCost::add_feature(float weight, FeatureBase *feature)
{
    weights_.push_back(weight);
    features_.push_back(std::shared_ptr<FeatureBase>(feature));
}

//----------------------------------------------------------------------------------
void LinearCost::add_feature(float weight, const std::shared_ptr<FeatureBase> feature)
{
    weights_.push_back(weight);
    features_.push_back(feature);
}

//----------------------------------------------------------------------------------
void HumanCost::hessian_uh(const Trajectory &robot_traj, const Trajectory &human_traj, MatRef hess)
{
    hess.setZero();

    int len = human_traj.traj_control_size();
    for (int i = 0; i < nfeatures_; ++i) {
        Eigen::MatrixXf hess_f(len, len);
        static_cast<FeatureHumanCost*>(features_[i].get())->hessian_uh(robot_traj, human_traj, hess_f);

        hess += weights_[i] * hess_f;
    }
}

//----------------------------------------------------------------------------------
void HumanCost::hessian_uh_ur(const Trajectory &robot_traj, const Trajectory &human_traj, MatRef hess)
{
    hess.setZero();

    int len = human_traj.traj_control_size();
    for (int i = 0; i < nfeatures_; ++i) {
        Eigen::MatrixXf hess_f(len, len);
        static_cast<FeatureHumanCost*>(features_[i].get())->hessian_uh_ur(robot_traj, human_traj, hess_f);

        hess += weights_[i] * hess_f;
    }
}

} // namespace