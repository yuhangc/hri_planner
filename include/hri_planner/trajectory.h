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

#ifndef HRI_PLANNER_TRAJECTORY_H
#define HRI_PLANNER_TRAJECTORY_H

#include <memory>
#include <Eigen/Dense>

#include "hri_planner/dynamics.h"

namespace hri_planner {

class Trajectory {
public:
    Trajectory(DynamicsModel dyn_type, int T, float dt);

    void update(const Eigen::VectorXf& x0, const Eigen::VectorXf& u_new);
    void compute_jacobian(const Eigen::VectorXf& x0);

    Eigen::VectorXf x;
    Eigen::VectorXf u;
    Eigen::MatrixXf Ju;

private:
    int nX_;
    int nU_;
    int T_;
    float dt_;

    std::shared_ptr<DynamicsBase> dyn_;
};

}

#endif //HRI_PLANNER_TRAJECTORY_H
