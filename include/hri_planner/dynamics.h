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

#ifndef HRI_PLANNER_KINEMATICS_H
#define HRI_PLANNER_KINEMATICS_H

#include <Eigen/Dense>

namespace hri_planner {

typedef enum {
    CONST_ACC_MODEL,
    DIFFERENTIAL_MODEL
} DynamicsModel;

class DynamicsBase {
protected:
    typedef Eigen::Ref<Eigen::VectorXd> VecRef;
    typedef Eigen::Ref<Eigen::MatrixXd> MatRef;
    typedef const Eigen::Ref<const Eigen::VectorXd> ConstVecRef;
    typedef const Eigen::Ref<const Eigen::MatrixXd> ConstMatRef;
public:
    DynamicsBase(int nX, int nU, double dt): nX_(nX), nU_(nU), dt_(dt) {};
    virtual void forward_dyn(ConstVecRef x, ConstVecRef u, VecRef x_new) = 0;
    virtual void grad_x(ConstVecRef x, ConstVecRef u, MatRef Jx) = 0;
    virtual void grad_u(ConstVecRef x, ConstVecRef u, MatRef Ju) = 0;

protected:
    int nX_;
    int nU_;
    double dt_;
};


class ConstAccDynamics: public DynamicsBase {
public:
    ConstAccDynamics(double dt);

    virtual void forward_dyn(ConstVecRef x, ConstVecRef u, VecRef x_new);
    virtual void grad_x(ConstVecRef x, ConstVecRef u, MatRef Jx);
    virtual void grad_u(ConstVecRef x, ConstVecRef u, MatRef Ju);

private:
    Eigen::MatrixXd A_;
    Eigen::MatrixXd B_;
};


class DifferentialDynamics: public DynamicsBase {
public:
    DifferentialDynamics(double dt, double om_tol=1e-3): DynamicsBase(3, 2, dt), om_tol_(om_tol) {};

    virtual void forward_dyn(ConstVecRef x, ConstVecRef u, VecRef x_new);
    virtual void grad_x(ConstVecRef x, ConstVecRef u, MatRef Jx);
    virtual void grad_u(ConstVecRef x, ConstVecRef u, MatRef Ju);

private:
    double om_tol_;
};

} // namespace
#endif //HRI_PLANNER_KINEMATICS_H
