//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/9/2017
// Last revision: 3/21/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include <utility>

#include "hri_planner/optimizer.h"

namespace hri_planner {

//----------------------------------------------------------------------------------
void EigenToVector(const Eigen::VectorXd& eigen_vec, std::vector<double>& std_vec)
{
    std_vec.assign(eigen_vec.data(), eigen_vec.data() + eigen_vec.size());
}

//----------------------------------------------------------------------------------
void EigenToVector3(const Eigen::VectorXd& vec1, const Eigen::VectorXd& vec2,
                    const Eigen::VectorXd& vec3, std::vector<double>& std_vec)
{
    std_vec.assign(vec1.data(), vec1.data() + vec1.size());
    std_vec.insert(std_vec.end(), vec2.data(), vec2.data() + vec2.size());
    std_vec.insert(std_vec.end(), vec3.data(), vec3.data() + vec3.size());
}

//----------------------------------------------------------------------------------
void VectorToEigen(const std::vector<double>& std_vec, Eigen::VectorXd& eigen_vec)
{
    eigen_vec = Eigen::Map<Eigen::VectorXd>(const_cast<double*>(std_vec.data()), std_vec.size());
}

//----------------------------------------------------------------------------------
TrajectoryOptimizer::TrajectoryOptimizer(unsigned int dim, const nlopt::algorithm& alg)
{
    optimizer_ = nlopt::opt(alg, dim);
}

//----------------------------------------------------------------------------------
void TrajectoryOptimizer::set_cost_function(SingleTrajectoryCost *cost)
{
    cost_ = std::shared_ptr<SingleTrajectoryCost>(cost);
}

//----------------------------------------------------------------------------------
void TrajectoryOptimizer::set_cost_function(std::shared_ptr<SingleTrajectoryCost> cost)
{
    cost_ = std::move(cost);
}

//----------------------------------------------------------------------------------
void TrajectoryOptimizer::set_bounds(const Eigen::VectorXd &lb, const Eigen::VectorXd &ub)
{
    lb_ = lb;
    ub_ = ub;
}

//----------------------------------------------------------------------------------
bool TrajectoryOptimizer::optimize(const Trajectory &traj_init, Trajectory &traj_opt)
{
    // recreate the trajectory object with trajectory data
    traj_.reset(new Trajectory(traj_init.dyn_type, traj_init.horizon(), traj_init.dt()));
    traj_->x0 = traj_init.x0;

    // set lower and upper bounds
    std::vector<double> lb;
    std::vector<double> ub;

    for (int t = 0; t < traj_init.horizon(); ++t) {
        for (int i = 0; i < traj_init.control_size(); ++i) {
            lb.push_back(lb_(i));
            ub.push_back(ub_(i));
        }
    }

    optimizer_.set_lower_bounds(lb);
    optimizer_.set_upper_bounds(ub);

    // set cost function
    optimizer_.set_min_objective(cost_wrapper, this);

    // set tolerance
    optimizer_.set_xtol_abs(1e-5);

    // initial condition
    std::vector<double> u_opt;
    EigenToVector(traj_init.u, u_opt);

    // optimizer!
    double min_cost;
    optimizer_.optimize(u_opt, min_cost);

    // send result back
    traj_opt.x0 = traj_init.x0;
    VectorToEigen(u_opt, traj_opt.u);
    traj_opt.compute();

    return true;
}

//----------------------------------------------------------------------------------
double TrajectoryOptimizer::cost_func(const std::vector<double> &u, std::vector<double> &grad)
{
    // re-compute the trjectory
    Eigen::Map<const Eigen::VectorXd> u_new(u.data(), u.size());

    traj_->update(u_new);
    traj_->compute_jacobian();

    // compute the cost and gradient
    Eigen::VectorXd grad_vec(u_new.size());
    cost_->grad(*traj_, grad_vec);

    // covert eigen vec to std vector
    EigenToVector(grad_vec, grad);

    // return the cost
    return cost_->compute(*traj_);
}

//----------------------------------------------------------------------------------
double TrajectoryOptimizer::cost_wrapper(const std::vector<double> &u, std::vector<double> &grad, void *data)
{
    return reinterpret_cast<TrajectoryOptimizer *>(data)->cost_func(u, grad);
}

//----------------------------------------------------------------------------------
NestedTrajectoryOptimizer::NestedTrajectoryOptimizer(unsigned int dim, const nlopt::algorithm &alg)
{
    optimizer_ = nlopt::opt(alg, dim);
}

//----------------------------------------------------------------------------------
void NestedTrajectoryOptimizer::set_human_cost(HumanCost* cost_hp, HumanCost* cost_rp)
{
    human_cost_hp_ = std::shared_ptr<HumanCost>(cost_hp);
    human_cost_rp_ = std::shared_ptr<HumanCost>(cost_rp);
}

//----------------------------------------------------------------------------------
void NestedTrajectoryOptimizer::set_human_cost(std::shared_ptr<HumanCost> cost_hp,
                                               std::shared_ptr<HumanCost> cost_rp)
{
    human_cost_hp_ = cost_hp;
    human_cost_rp_ = cost_rp;
}

//----------------------------------------------------------------------------------
void NestedTrajectoryOptimizer::set_robot_cost(ProbabilisticCostBase *cost)
{
    robot_cost_ = std::shared_ptr<ProbabilisticCostBase>(cost);
}

//----------------------------------------------------------------------------------
void NestedTrajectoryOptimizer::set_robot_cost(std::shared_ptr<ProbabilisticCostBase> cost)
{
    robot_cost_ = cost;
}

//----------------------------------------------------------------------------------
void NestedTrajectoryOptimizer::set_bounds(const Eigen::VectorXd &lb_ur, const Eigen::VectorXd &ub_ur,
                                           const Eigen::VectorXd &lb_uh, const Eigen::VectorXd &ub_uh)
{
    lb_ur_ = lb_ur;
    ub_ur_ = ub_ur;
    lb_uh_ = lb_uh;
    ub_uh_ = ub_uh;
}

//----------------------------------------------------------------------------------
bool NestedTrajectoryOptimizer::optimize(const Eigen::VectorXd &xr0, const Eigen::VectorXd &xh0,
                                         const Trajectory &robot_traj_init, const Trajectory &human_traj_hp_init,
                                         const Trajectory& human_traj_rp_init, int acomm, double tcomm,
                                         Trajectory& robot_traj_opt, Trajectory* human_traj_hp_opt,
                                         Trajectory* human_traj_rp_opt)
{
    // set communication action
    acomm_ = acomm;
    tcomm_ = tcomm;

    // recreate the trajectory object with trajectory data
    int T = robot_traj_init.horizon();
    double dt = robot_traj_init.dt();
    robot_traj_.reset(new Trajectory(DIFFERENTIAL_MODEL, T, dt));
    robot_traj_->x0 = robot_traj_init.x0;

    human_traj_hp_.reset(new Trajectory(CONST_ACC_MODEL, T, dt));
    human_traj_hp_->x0 = human_traj_hp_init.x0;

    human_traj_rp_.reset(new Trajectory(CONST_ACC_MODEL, T, dt));
    human_traj_rp_->x0 = human_traj_rp_init.x0;

    // a constant speed prediction of human motion
    Trajectory human_traj_pred(CONST_ACC_MODEL, T, dt);
    human_traj_pred.update(human_traj_hp_init.x0, Eigen::VectorXd::Zero(human_traj_hp_init.traj_control_size()));

    robot_cost_->update_human_pred(human_traj_pred);

    // set lower and upper bounds
    std::vector<double> lb;
    std::vector<double> ub;

    EigenToVector3(lb_ur_, lb_uh_, lb_uh_, lb);
    EigenToVector3(ub_ur_, ub_uh_, ub_uh_, ub);

    optimizer_.set_lower_bounds(lb);
    optimizer_.set_upper_bounds(ub);

    // set cost function
    optimizer_.set_min_objective(cost_wrapper, this);

    // set constraint
    optimizer_.add_equality_constraint(constraint_wrapper, this, 1e-3);

    // set tolerance
    optimizer_.set_xtol_abs(1e-3);

    // initial condition
    std::vector<double> u_opt;
    EigenToVector3(robot_traj_init.u, human_traj_hp_init.u, human_traj_rp_init.u, u_opt);

    // optimizer!
    double min_cost;
    nlopt::result result = optimizer_.optimize(u_opt, min_cost);
    std::cout << "result is: " << result << std::endl;

    // print cost and constraint error
    std::cout << "min cost is: " << min_cost << std::endl;

    std::vector<double> grad_opt;
    std::cout << "constraint error is: " << constraint(u_opt, grad_opt) << std::endl;

    // send result back
    int len_ur = robot_traj_opt.traj_control_size();
    robot_traj_opt.x0 = robot_traj_init.x0;
    robot_traj_opt.u = Eigen::Map<Eigen::VectorXd>(u_opt.data(), len_ur);

    robot_traj_opt.compute();

    // compute "optimal" human trajectory if not null
    if (human_traj_hp_opt != nullptr) {
        int len_uh = human_traj_hp_init.traj_control_size();
        human_traj_hp_opt->update(human_traj_hp_init.x0,
                                  Eigen::Map<Eigen::VectorXd>(u_opt.data()+len_ur, len_uh));
        human_traj_rp_opt->update(human_traj_rp_init.x0,
                                  Eigen::Map<Eigen::VectorXd>(u_opt.data()+len_ur+len_uh, len_uh));
    }

    return true;
}

//----------------------------------------------------------------------------------
double NestedTrajectoryOptimizer::check_constraint(const Trajectory &robot_traj, const Trajectory &human_traj_hp,
                                                   const Trajectory &human_traj_rp)
{
    // convert to std vector
    std::vector<double> u;
    EigenToVector3(robot_traj.u, human_traj_hp.u, human_traj_rp.u, u);

    std::vector<double> grad;

    // construct the trajectories
    int T = robot_traj.horizon();
    double dt = robot_traj.dt();
    robot_traj_.reset(new Trajectory(DIFFERENTIAL_MODEL, T, dt));
    robot_traj_->x0 = robot_traj.x0;

    human_traj_hp_.reset(new Trajectory(CONST_ACC_MODEL, T, dt));
    human_traj_hp_->x0 = human_traj_hp.x0;

    human_traj_rp_.reset(new Trajectory(CONST_ACC_MODEL, T, dt));
    human_traj_rp_->x0 = human_traj_rp.x0;

    return constraint(u, grad);
}

//----------------------------------------------------------------------------------
double NestedTrajectoryOptimizer::cost_func(const std::vector<double> &u, std::vector<double> &grad)
{
    // update robot and human trajectories
    int len = robot_traj_->traj_control_size();
    Eigen::Map<const Eigen::VectorXd> ur(u.data(), len);
    robot_traj_->update(ur);
    robot_traj_->compute_jacobian();

    Eigen::Map<const Eigen::VectorXd> uh_hp(u.data()+len, human_traj_hp_->traj_control_size());
    human_traj_hp_->update(uh_hp);
    human_traj_hp_->compute_jacobian();

    len += human_traj_hp_->traj_control_size();
    Eigen::Map<const Eigen::VectorXd> uh_rp(u.data()+len, human_traj_rp_->traj_control_size());
    human_traj_rp_->update(uh_rp);
    human_traj_rp_->compute_jacobian();

    // compute cost and gradients
    Eigen::VectorXd grad_ur;
    Eigen::VectorXd grad_uh_hp;
    Eigen::VectorXd grad_uh_rp;

    double cost = robot_cost_->compute(*robot_traj_, *human_traj_hp_, *human_traj_rp_, acomm_, tcomm_,
                                       grad_ur, grad_uh_hp, grad_uh_rp);

    // convert to single gradient vector
    EigenToVector3(grad_ur, grad_uh_hp, grad_uh_rp, grad);

    return cost;
}

//----------------------------------------------------------------------------------
double NestedTrajectoryOptimizer::cost_wrapper(const std::vector<double> &u, std::vector<double> &grad,
                                              void *cost_func_data)
{
    return reinterpret_cast<NestedTrajectoryOptimizer *>(cost_func_data)->cost_func(u, grad);
}

//----------------------------------------------------------------------------------
double NestedTrajectoryOptimizer::constraint(const std::vector<double> &u, std::vector<double> &grad)
{
    // update robot and human trajectories
    int len = robot_traj_->traj_control_size();
    Eigen::Map<const Eigen::VectorXd> ur(u.data(), len);
    robot_traj_->update(ur);
    robot_traj_->compute_jacobian();

    Eigen::Map<const Eigen::VectorXd> uh_hp(u.data()+len, human_traj_hp_->traj_control_size());
    human_traj_hp_->update(uh_hp);
    human_traj_hp_->compute_jacobian();

    len += human_traj_hp_->traj_control_size();
    Eigen::Map<const Eigen::VectorXd> uh_rp(u.data()+len, human_traj_rp_->traj_control_size());
    human_traj_rp_->update(uh_rp);
    human_traj_rp_->compute_jacobian();

    // find gradients of the human cost functions
    int len_uh = human_traj_rp_->traj_control_size();
    int len_ur = robot_traj_->traj_control_size();
    Eigen::VectorXd grad_uh_hp(len_uh);
    Eigen::VectorXd grad_uh_rp(len_uh);

    human_cost_hp_->grad_uh(*robot_traj_, *human_traj_hp_, grad_uh_hp);
    human_cost_rp_->grad_uh(*robot_traj_, *human_traj_rp_, grad_uh_rp);

    // compute the constraint
    double constraint_val = grad_uh_hp.squaredNorm() + grad_uh_rp.squaredNorm();

    //! compute the constraint gradient
    Eigen::MatrixXd Ju_hp(len_uh, u.size());
    Eigen::MatrixXd Ju_rp(len_uh, u.size());

    human_cost_hp_->hessian_uh_ur(*robot_traj_, *human_traj_hp_, Ju_hp.block(0, 0, len_uh, len_ur));
    human_cost_hp_->hessian_uh(*robot_traj_, *human_traj_hp_, Ju_hp.block(0, len_ur, len_uh, len_uh));
    Ju_hp.block(0, len_ur+len_uh, len_uh, len_uh).setZero();

    human_cost_rp_->hessian_uh_ur(*robot_traj_, *human_traj_rp_, Ju_rp.block(0, 0, len_uh, len_ur));
    Ju_hp.block(0, len_ur, len_uh, len_uh).setZero();
    human_cost_rp_->hessian_uh(*robot_traj_, *human_traj_rp_, Ju_rp.block(0, len_ur+len_uh, len_uh, len_uh));

    grad_uh_hp = Ju_hp.transpose() * grad_uh_hp * 2.0;
    grad_uh_rp = Ju_rp.transpose() * grad_uh_rp * 2.0;

    // convert to std vector
    EigenToVector(grad_uh_hp + grad_uh_rp, grad);

    return constraint_val;
}

//----------------------------------------------------------------------------------
double NestedTrajectoryOptimizer::constraint_wrapper(const std::vector<double> &u, std::vector<double> &grad,
                                                    void *constraint_data)
{
    return reinterpret_cast<NestedTrajectoryOptimizer *>(constraint_data)->constraint(u, grad);
}

} // namespace