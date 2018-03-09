//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/7/2017
// Last revision: 3/7/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#ifndef HRI_PLANNER_COST_FEATURE_H
#define HRI_PLANNER_COST_FEATURE_H

#include <Eigen/Dense>

#include "hri_planner/trajectory.h"

namespace hri_planner {

class FeatureBase {
protected:
    typedef Eigen::Ref<Eigen::VectorXf> VecRef;
    typedef Eigen::Ref<Eigen::MatrixXf> MatRef;
    typedef const Eigen::Ref<const Eigen::VectorXf> ConstVecRef;
    typedef const Eigen::Ref<const Eigen::MatrixXf> ConstMatRef;
public:
    // overloading the () operator
    float operator()(const Trajectory& robot_traj, const Trajectory& human_traj);
    
    virtual void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) = 0;
    virtual void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) = 0;
protected:
    virtual float compute(const Trajectory& robot_traj, const Trajectory& human_traj) = 0;

};

}

#endif //HRI_PLANNER_COST_FEATURE_H
