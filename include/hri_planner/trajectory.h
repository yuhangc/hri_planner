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

#ifndef HRI_PLANNER_TRAJECTORY_H
#define HRI_PLANNER_TRAJECTORY_H

#include <memory>
#include <Eigen/Dense>

#include "hri_planner/dynamics.h"

namespace hri_planner {

class Trajectory {
public:
    Trajectory(DynamicsModel dyn_type, int T, double dt);

    void update(const Eigen::VectorXd& x0, const Eigen::VectorXd& u_new);
    void compute_jacobian(const Eigen::VectorXd& x0);

    // overloading the = operator
    Trajectory& operator=(const Trajectory& traj);

    inline int state_size() const {
        return nX_;
    }

    inline int control_size() const {
        return nU_;
    }

    inline int horizon() const {
        return T_;
    }

    inline double dt() const {
        return dt_;
    }

    inline int traj_state_size() const {
        return nXt_;
    }

    inline int traj_control_size() const {
        return nUt_;
    }

    Eigen::VectorXd x;
    Eigen::VectorXd u;
    Eigen::MatrixXd Ju;

    DynamicsModel dyn_type;

private:
    int nX_;
    int nU_;
    int T_;
    double dt_;

    int nXt_;
    int nUt_;

    std::shared_ptr<DynamicsBase> dyn_;
};

}

#endif //HRI_PLANNER_TRAJECTORY_H
