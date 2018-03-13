//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/7/2017
// Last revision: 3/12/2017
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
    typedef Eigen::Ref<Eigen::VectorXd> VecRef;
    typedef Eigen::Ref<Eigen::MatrixXd> MatRef;
    typedef const Eigen::Ref<const Eigen::VectorXd> ConstVecRef;
    typedef const Eigen::Ref<const Eigen::MatrixXd> ConstMatRef;
public:
    // virtual destructor
    virtual ~FeatureBase(){};

    // overloading the () operator
    double operator()(const Trajectory& robot_traj, const Trajectory& human_traj) {
        return compute(robot_traj, human_traj);
    };

    virtual double compute(const Trajectory& robot_traj, const Trajectory& human_traj) = 0;
    virtual void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) = 0;
    virtual void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) = 0;
};

class FeatureTwiceDiff: public FeatureBase {
public:
    // virtual destructor
    virtual ~FeatureTwiceDiff(){};

    virtual void hessian_uh(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) = 0;
    virtual void hessian_ur(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) = 0;
    virtual void hessian_uh_ur(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) = 0;
};

// human cost features requires hessian to u_h and (u_h, u_r)
class FeatureHumanCost: public FeatureBase {
public:
    // virtual destructor
    virtual ~FeatureHumanCost(){};

    virtual void hessian_uh(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) = 0;
    virtual void hessian_uh_ur(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) = 0;
};

// robot features only require first-order derivatives
typedef FeatureBase FeatureRobotCost;

// commonly used Gaussian feature for collision avoidance
class GaussianCost {
public:
    static double compute(const Eigen::VectorXd& x, const int nX, const int T, const double a, const double b);
    static void grad(const Eigen::VectorXd& x, const int nX, const int T,
                     const double a, const double b, Eigen::VectorXd& grad);
    static void hessian(const Eigen::VectorXd& x, const int nX1, const int nX2, const int T,
                        const double a, const double b, Eigen::MatrixXd& hess);
};

}

#endif //HRI_PLANNER_COST_FEATURE_H
