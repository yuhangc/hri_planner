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
    typedef Eigen::Ref<Eigen::VectorXf> VecRef;
    typedef Eigen::Ref<Eigen::MatrixXf> MatRef;
    typedef const Eigen::Ref<const Eigen::VectorXf> ConstVecRef;
    typedef const Eigen::Ref<const Eigen::MatrixXf> ConstMatRef;
public:
    DynamicsBase(int nX, int nU, float dt): nX_(nX), nU_(nU), dt_(dt) {};
    virtual void forward_dyn(ConstVecRef x, ConstVecRef u, VecRef x_new) = 0;
    virtual void grad_x(ConstVecRef x, ConstVecRef u, MatRef Jx) = 0;
    virtual void grad_u(ConstVecRef x, ConstVecRef u, MatRef Ju) = 0;

protected:
    int nX_;
    int nU_;
    float dt_;
};


class ConstAccDynamics: public DynamicsBase {
public:
    ConstAccDynamics(float dt);

    virtual void forward_dyn(ConstVecRef x, ConstVecRef u, VecRef x_new);
    virtual void grad_x(ConstVecRef x, ConstVecRef u, MatRef Jx);
    virtual void grad_u(ConstVecRef x, ConstVecRef u, MatRef Ju);

private:
    Eigen::MatrixXf A_;
    Eigen::MatrixXf B_;
};


class DifferentialDynamics: public DynamicsBase {
public:
    DifferentialDynamics(float dt, float om_tol=1e-3): DynamicsBase(3, 2, dt), om_tol_(om_tol) {};

    virtual void forward_dyn(ConstVecRef x, ConstVecRef u, VecRef x_new);
    virtual void grad_x(ConstVecRef x, ConstVecRef u, MatRef Jx);
    virtual void grad_u(ConstVecRef x, ConstVecRef u, MatRef Ju);

private:
    float om_tol_;
};

} // namespace
#endif //HRI_PLANNER_KINEMATICS_H
