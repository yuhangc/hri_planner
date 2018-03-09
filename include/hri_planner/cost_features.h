//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/8/2017
// Last revision: 3/8/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#ifndef HRI_PLANNER_COST_FEATURES_H
#define HRI_PLANNER_COST_FEATURES_H

#include "hri_planner/cost_feature_bases.h"

namespace hri_planner {

class FeatureHumanCostNonInt: public FeatureHumanCost {
public:
    // virtual destructor
    virtual ~FeatureHumanCostNonInt(){};

    virtual void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) {
        grad.setZero();
    };

    virtual void hessian_uh_ur(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) {
        hess.setZero();
    };
};

//! human cost functions
class HumanVelCost: public FeatureHumanCostNonInt {
public:
    virtual void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad);
    virtual void hessian_uh(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess);

    virtual float compute(const Trajectory& robot_traj, const Trajectory& human_traj);
};

class HumanAccCost: public FeatureHumanCostNonInt {
public:
    virtual void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad);
    virtual void hessian_uh(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess);

    virtual float compute(const Trajectory& robot_traj, const Trajectory& human_traj);
};

class HumanGoalCost: public FeatureHumanCostNonInt {
public:
    HumanGoalCost(const Eigen::VectorXf& x_goal, float reg=1e-2): x_goal_(x_goal), reg_(reg) {};

    virtual void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad);
    virtual void hessian_uh(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess);

    virtual float compute(const Trajectory& robot_traj, const Trajectory& human_traj);

private:
    Eigen::VectorXf x_goal_;
    float reg_;
};

class HumanObsCost: public FeatureHumanCostNonInt {
public:
    HumanObsCost(const Eigen::VectorXf& x_obs): x_obs_(x_obs) {};
    virtual void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad);
    virtual void hessian_uh(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess);

    virtual float compute(const Trajectory& robot_traj, const Trajectory& human_traj);

private:
    Eigen::VectorXf x_obs_;
};

//! cost for both human and/or robot
class CollisionCost: public FeatureHumanCost {
public:
    CollisionCost(float R): R_(R) {};

    virtual void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad);
    virtual void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad);
    virtual void hessian_uh(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess);
    virtual void hessian_uh_ur(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess);

    virtual float compute(const Trajectory& robot_traj, const Trajectory& human_traj);

private:
    float R_;
};

class DynCollisionCost: public FeatureHumanCost {
public:
    virtual void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad);
    virtual void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad);
    virtual void hessian_uh(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess);
    virtual void hessian_uh_ur(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess);

    virtual float compute(const Trajectory& robot_traj, const Trajectory& human_traj);
};

//! robot cost functions
class RobotControlCost: public FeatureRobotCost {
public:
    virtual void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad);
    virtual void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad);

    virtual float compute(const Trajectory& robot_traj, const Trajectory& human_traj);
};

class RobotGoalCost: public FeatureRobotCost {
public:
    RobotGoalCost(const Eigen::VectorXf& x_goal): x_goal_(x_goal) {};

    virtual void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad);
    virtual void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad);

    virtual float compute(const Trajectory& robot_traj, const Trajectory& human_traj);

private:
    Eigen::VectorXf x_goal_;
};


}

#endif //HRI_PLANNER_COST_FEATURES_H
