//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/18/2017
// Last revision: 3/18/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include "hri_planner/cost_probabilistic.h"

namespace hri_planner {

//----------------------------------------------------------------------------------
void ProbabilisticCostBase::set_features_non_int(const std::vector<double> &w,
                                                 const std::vector<std::shared_ptr<FeatureBase> > &f)
{
    w_non_int_ = w;
    f_non_int_ = f;
}

//----------------------------------------------------------------------------------
void ProbabilisticCostBase::set_features_int(const std::vector<double> &w,
                                             const std::vector<std::shared_ptr<FeatureVectorizedBase> > &f)
{
    w_int_ = w;
    f_int_ = f;
}

//----------------------------------------------------------------------------------
double ProbabilisticCost::compute(const Trajectory& robot_traj, const Trajectory& human_traj_hp,
                                  const Trajectory& human_traj_rp, int acomm, double tcomm,
                                  Eigen::VectorXd& grad_ur, Eigen::VectorXd& grad, Eigen::VectorXd& grad_rp)
{
    double cost = 0.0;

    // first compute non-interactive costs
    // doesn't matter which human trajectory to use
    for (int i = 0; i < w_non_int_.size(); ++i)
        cost += w_non_int_[i] * f_non_int_[i]->compute(robot_traj, human_traj_hp);

    // compute the interactive costs weighted by beliefs
    Eigen::VectorXd prob_hp;

    return cost;
}


}
