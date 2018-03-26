//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/8/2017
// Last revision: 3/25/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#ifndef HRI_PLANNER_COST_FEATURES_H
#define HRI_PLANNER_COST_FEATURES_H

#include <string>
#include <memory>

#include "hri_planner/cost_feature_bases.h"

namespace hri_planner {

// human cost features requires hessian to u_h and (u_h, u_r)
class FeatureHumanCost: public FeatureBase {
public:
    // virtual destructor
    virtual ~FeatureHumanCost() = default;

    virtual void hessian_uh(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) = 0;
    virtual void hessian_uh_ur(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) = 0;

    // FIXME: right now assuming double-type args only
    static std::shared_ptr<FeatureHumanCost> create(const std::string& feature_type,
                                                    const std::vector<double>& args);
};

// robot features only require first-order derivatives
class FeatureRobotCost: public FeatureBase {
public:
    // virtual destructor
    virtual ~FeatureRobotCost() = default;

    // Factory creation method
    static std::shared_ptr<FeatureRobotCost> create(const std::string& feature_type,
                                                    const std::vector<double>& args);
};

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

    void set_data(const void* data) override {};
};

class HumanAccCost: public FeatureHumanCostNonInt {
public:
    void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;
    void hessian_uh(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) override;

    double compute(const Trajectory& robot_traj, const Trajectory& human_traj) override;

    void set_data(const void* data) override {};
};

class HumanGoalCost: public FeatureHumanCostNonInt {
public:
    explicit HumanGoalCost(const Eigen::VectorXd& x_goal, double reg=1e-2): x_goal_(x_goal), reg_(reg) {};

    void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;
    void hessian_uh(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) override;

    double compute(const Trajectory& robot_traj, const Trajectory& human_traj) override;

    void set_data(const void* data) override {
        x_goal_ = *static_cast<const Eigen::VectorXd*>(data);
    };

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

    void set_data(const void* data) override {
        x_obs_ = *static_cast<const Eigen::VectorXd*>(data);
    };

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

    void set_data(const void* data) override {
        R_ = *static_cast<const double*>(data);
    };

private:
    double R_;
};

class DynCollisionCost: public FeatureHumanCost {
public:
    DynCollisionCost(double Rx, double Ry, double d): Rx_(Rx), Ry_(Ry), d_(d) {};
    void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;
    void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;
    void hessian_uh(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) override;
    void hessian_uh_ur(const Trajectory& robot_traj, const Trajectory& human_traj, MatRef hess) override;

    double compute(const Trajectory& robot_traj, const Trajectory& human_traj) override;

    void set_data(const void* data) override {
        auto data_vec = static_cast<const double*>(data);
        Rx_ = data_vec[0];
        Ry_ = data_vec[1];
        d_ = data_vec[2];
    };

private:
    double Rx_;
    double Ry_;
    double d_;
};

//! robot cost functions
class RobotControlCost: public FeatureRobotCost {
public:
    void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;
    void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;

    double compute(const Trajectory& robot_traj, const Trajectory& human_traj) override;

    void set_data(const void* data) override {};
};

class RobotGoalCost: public FeatureRobotCost {
public:
    explicit RobotGoalCost(const Eigen::VectorXd& x_goal, double reg=1e-2): x_goal_(x_goal), reg_(reg) {};

    void grad_uh(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;
    void grad_ur(const Trajectory& robot_traj, const Trajectory& human_traj, VecRef grad) override;

    double compute(const Trajectory& robot_traj, const Trajectory& human_traj) override;

    void set_data(const  void* data) override {
        x_goal_ = *static_cast<const Eigen::VectorXd*>(data);
    };

private:
    Eigen::VectorXd x_goal_;
    double reg_;
};


}

#endif //HRI_PLANNER_COST_FEATURES_H
