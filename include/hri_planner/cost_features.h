//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/8/2017
// Last revision: 3/11/2017
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
    virtual ~FeatureHumanCostNonInt() = default;

    void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override {
        grad.setZero();
    };

    void hessian_uh_ur(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) override
    {
        hess.setZero();
    };
};

//! human cost functions
class HumanVelCost: public FeatureHumanCostNonInt {
public:
    void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;
    void hessian_uh(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) override;

    double compute(const Trajectory& robot_traj, const Trajectory& human_traj) override;
};

class HumanAccCost: public FeatureHumanCostNonInt {
public:
    void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;
    void hessian_uh(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) override;

    double compute(const Trajectory& robot_traj, const Trajectory& human_traj) override;
};

class HumanGoalCost: public FeatureHumanCostNonInt {
public:
    explicit HumanGoalCost(const Eigen::VectorXd& x_goal, double reg=1e-2): x_goal_(x_goal), reg_(reg) {};

    void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;
    void hessian_uh(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) override;

    double compute(const Trajectory& robot_traj, const Trajectory& human_traj) override;

private:
    Eigen::VectorXd x_goal_;
    double reg_;
};

class HumanObsCost: public FeatureHumanCostNonInt {
public:
    explicit HumanObsCost(const Eigen::VectorXd& x_obs): x_obs_(x_obs) {};
    void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;
    void hessian_uh(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) override;

    double compute(const Trajectory& robot_traj, const Trajectory& human_traj) override;

private:
    Eigen::VectorXd x_obs_;
};

//! cost for both human and/or robot
class CollisionCost: public FeatureHumanCost {
public:
    explicit CollisionCost(double R): R_(R) {};

    void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;
    void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;
    void hessian_uh(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) override;
    void hessian_uh_ur(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) override;

    double compute(const Trajectory& robot_traj, const Trajectory& human_traj) override;

private:
    double R_;
};

class DynCollisionCost: public FeatureHumanCost {
public:
    DynCollisionCost(double Rx, double Ry, double dt_pred=1.0): Rx_(Rx), Ry_(Ry), dt_pred_(dt_pred) {};
    void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;
    void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;
    void hessian_uh(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) override;
    void hessian_uh_ur(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) override;

    double compute(const Trajectory& robot_traj, const Trajectory& human_traj) override;

private:
    double Rx_;
    double Ry_;
    double dt_pred_;
};

//! robot cost functions
class RobotControlCost: public FeatureRobotCost {
public:
    void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;
    void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;

    double compute(const Trajectory& robot_traj, const Trajectory& human_traj) override;
};

class RobotGoalCost: public FeatureRobotCost {
public:
    explicit RobotGoalCost(const Eigen::VectorXd& x_goal): x_goal_(x_goal) {};

    void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;
    void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;

    double compute(const Trajectory& robot_traj, const Trajectory& human_traj) override;

private:
    Eigen::VectorXd x_goal_;
};


}

#endif //HRI_PLANNER_COST_FEATURES_H
