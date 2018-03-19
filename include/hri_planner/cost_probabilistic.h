//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/18/2017
// Last revision: 3/19/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#ifndef HRI_PLANNER_COST_PROBABILISTIC_H
#define HRI_PLANNER_COST_PROBABILISTIC_H

#include <vector>
#include <memory>

#include "hri_planner/trajectory.h"
#include "hri_planner/cost_features_vectorized.h"
#include "hri_planner/cost_features.h"
#include "hri_planner/human_belief_model.h"

namespace hri_planner {

class ProbabilisticCostBase {
public:
    virtual ~ProbabilisticCostBase() = default;

    // 3-in-1 computes everything
    virtual double compute(const Trajectory& robot_traj, const Trajectory& human_traj_hp,
                           const Trajectory& human_traj_rp, int acomm, double tcomm,
                           Eigen::VectorXd& grad_ur, Eigen::VectorXd& grad, Eigen::VectorXd& grad_rp) = 0;

    void set_features_non_int(const std::vector<double>& w, const std::vector<std::shared_ptr<FeatureBase> >& f);
    void set_features_int(const std::vector<double>& w, const std::vector<std::shared_ptr<FeatureVectorizedBase> >& f);

protected:
    // a belief model
    std::shared_ptr<BeliefModelBase> belief_model_;

    // weights for non-interactive features
    std::vector<double> w_non_int_;
    std::vector<std::shared_ptr<FeatureBase> > f_non_int_;

    // weights for interactive features
    std::vector<double> w_int_;
    std::vector<std::shared_ptr<FeatureVectorizedBase> > f_int_;
};


//! naive implementation
class ProbabilisticCost: public ProbabilisticCostBase {
public:
    double compute(const Trajectory& robot_traj, const Trajectory& human_traj_hp,
                   const Trajectory& human_traj_rp, int acomm, double tcomm,
                   Eigen::VectorXd& grad_ur, Eigen::VectorXd& grad_hp, Eigen::VectorXd& grad_rp) override;
};

}

#endif //HRI_PLANNER_COST_PROBABILISTIC_H
