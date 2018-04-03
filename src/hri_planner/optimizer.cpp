//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/9/2017
// Last revision: 3/31/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include <utility>

#include "hri_planner/optimizer.h"

namespace hri_planner {

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
bool TrajectoryOptimizer::optimize(const Trajectory& traj_init, const Trajectory& traj_const, Trajectory& traj_opt)
{
    // recreate the trajectory object with trajectory data
    traj_.reset(new Trajectory(traj_init.dyn_type, traj_init.horizon(), traj_init.dt()));
    traj_->x0 = traj_init.x0;

    // set the const trajectory data
    cost_->set_trajectory_data(traj_const);

    // set lower and upper bounds
    std::vector<double> lb;
    std::vector<double> ub;

    utils::EigenToVector(lb_, lb);
    utils::EigenToVector(ub_, ub);

    optimizer_.set_lower_bounds(lb);
    optimizer_.set_upper_bounds(ub);

    // set cost function
    optimizer_.set_min_objective(cost_wrapper, this);

    // set tolerance
    optimizer_.set_xtol_abs(1e-2);

    // initial condition
    std::vector<double> u_opt;
    utils::EigenToVector(traj_init.u, u_opt);

    // optimizer!
    double min_cost;
    optimizer_.optimize(u_opt, min_cost);

    // send result back
    traj_opt.x0 = traj_init.x0;
    utils::VectorToEigen(u_opt, traj_opt.u);
    traj_opt.compute();
    traj_opt.compute_jacobian();

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
    utils::EigenToVector(grad_vec, grad);

    // return the cost
    return cost_->compute(*traj_);
}

//----------------------------------------------------------------------------------
double TrajectoryOptimizer::cost_wrapper(const std::vector<double> &u, std::vector<double> &grad, void *data)
{
    return reinterpret_cast<TrajectoryOptimizer *>(data)->cost_func(u, grad);
}

//----------------------------------------------------------------------------------
NestedOptimizerBase::NestedOptimizerBase(unsigned int dim, const nlopt::algorithm &alg)
{
    optimizer_ = nlopt::opt(alg, dim);
}

//----------------------------------------------------------------------------------
void NestedOptimizerBase::set_robot_cost(ProbabilisticCostBase *cost)
{
    robot_cost_ = std::shared_ptr<ProbabilisticCostBase>(cost);
}

//----------------------------------------------------------------------------------
void NestedOptimizerBase::set_robot_cost(std::shared_ptr<ProbabilisticCostBase> cost)
{
    robot_cost_ = cost;
}

//----------------------------------------------------------------------------------
double NestedOptimizerBase::cost_wrapper(const std::vector<double> &u, std::vector<double> &grad,
                                         void *cost_func_data)
{
    return reinterpret_cast<NestedOptimizerBase *>(cost_func_data)->cost_func(u, grad);
}

//----------------------------------------------------------------------------------
void NestedTrajectoryOptimizer::set_human_cost(LinearCost* cost_hp, LinearCost* cost_rp)
{
    human_cost_hp_ = std::shared_ptr<LinearCost>(cost_hp);
    human_cost_rp_ = std::shared_ptr<LinearCost>(cost_rp);
}

//----------------------------------------------------------------------------------
void NestedTrajectoryOptimizer::set_human_cost(const std::shared_ptr<LinearCost>& cost_hp,
                                               const std::shared_ptr<LinearCost>& cost_rp)
{
    human_cost_hp_ = cost_hp;
    human_cost_rp_ = cost_rp;
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
double NestedTrajectoryOptimizer::optimize(const Trajectory &robot_traj_init, const Trajectory &human_traj_hp_init,
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

    utils::EigenToVector3(lb_ur_, lb_uh_, lb_uh_, lb);
    utils::EigenToVector3(ub_ur_, ub_uh_, ub_uh_, ub);

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
    utils::EigenToVector3(robot_traj_init.u, human_traj_hp_init.u, human_traj_rp_init.u, u_opt);

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

    return min_cost;
}

//----------------------------------------------------------------------------------
double NestedTrajectoryOptimizer::check_constraint(const Trajectory &robot_traj, const Trajectory &human_traj_hp,
                                                   const Trajectory &human_traj_rp)
{
    // convert to std vector
    std::vector<double> u;
    utils::EigenToVector3(robot_traj.u, human_traj_hp.u, human_traj_rp.u, u);

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
    utils::EigenToVector3(grad_ur, grad_uh_hp, grad_uh_rp, grad);

    return cost;
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

    // cast the pointers first
    HumanCost* cost_hp_cast = dynamic_cast<HumanCost*>(human_cost_hp_.get());
    HumanCost* cost_rp_cast = dynamic_cast<HumanCost*>(human_cost_rp_.get());

    cost_hp_cast->hessian_uh_ur(*robot_traj_, *human_traj_hp_, Ju_hp.block(0, 0, len_uh, len_ur));
    cost_hp_cast->hessian_uh(*robot_traj_, *human_traj_hp_, Ju_hp.block(0, len_ur, len_uh, len_uh));
    Ju_hp.block(0, len_ur+len_uh, len_uh, len_uh).setZero();

    cost_rp_cast->hessian_uh_ur(*robot_traj_, *human_traj_rp_, Ju_rp.block(0, 0, len_uh, len_ur));
    Ju_hp.block(0, len_ur, len_uh, len_uh).setZero();
    cost_rp_cast->hessian_uh(*robot_traj_, *human_traj_rp_, Ju_rp.block(0, len_ur+len_uh, len_uh, len_uh));

    grad_uh_hp = Ju_hp.transpose() * grad_uh_hp * 2.0;
    grad_uh_rp = Ju_rp.transpose() * grad_uh_rp * 2.0;

    // convert to std vector
    utils::EigenToVector(grad_uh_hp + grad_uh_rp, grad);

    return constraint_val;
}

//----------------------------------------------------------------------------------
double NestedTrajectoryOptimizer::constraint_wrapper(const std::vector<double> &u, std::vector<double> &grad,
                                                    void *constraint_data)
{
    return reinterpret_cast<NestedTrajectoryOptimizer *>(constraint_data)->constraint(u, grad);
}

//----------------------------------------------------------------------------------
NaiveNestedOptimizer::NaiveNestedOptimizer(unsigned int dim_r, unsigned int dim_h, const nlopt::algorithm &alg,
                                           const nlopt::algorithm &sub_alg): NestedOptimizerBase(dim_r, alg)
{
    optimizer_hp_.reset(new TrajectoryOptimizer(dim_h, sub_alg));
    optimizer_rp_.reset(new TrajectoryOptimizer(dim_h, sub_alg));
}

//----------------------------------------------------------------------------------
void NaiveNestedOptimizer::set_human_cost(LinearCost *cost_hp, LinearCost *cost_rp)
{
    human_cost_hp_ = std::shared_ptr<LinearCost>(cost_hp);
    human_cost_rp_ = std::shared_ptr<LinearCost>(cost_rp);
    optimizer_hp_->set_cost_function(reinterpret_cast<SingleTrajectoryCost*>(cost_hp));
    optimizer_rp_->set_cost_function(reinterpret_cast<SingleTrajectoryCost*>(cost_rp));
}

//----------------------------------------------------------------------------------
void NaiveNestedOptimizer::set_human_cost(const std::shared_ptr<LinearCost>& cost_hp,
                                          const std::shared_ptr<LinearCost>& cost_rp)
{
    human_cost_hp_ = cost_hp;
    human_cost_rp_ = cost_rp;
    optimizer_hp_->set_cost_function(std::dynamic_pointer_cast<SingleTrajectoryCost>(cost_hp));
    optimizer_rp_->set_cost_function(std::dynamic_pointer_cast<SingleTrajectoryCost>(cost_rp));
}

//----------------------------------------------------------------------------------
void NaiveNestedOptimizer::set_bounds(const Eigen::VectorXd &lb_ur, const Eigen::VectorXd &ub_ur,
                                      const Eigen::VectorXd &lb_uh, const Eigen::VectorXd &ub_uh)
{
    // robot bounds
    lb_ur_ = lb_ur;
    ub_ur_ = ub_ur;

    // human bounds
    optimizer_hp_->set_bounds(lb_uh, ub_uh);
    optimizer_rp_->set_bounds(lb_uh, ub_uh);
}

//----------------------------------------------------------------------------------
double NaiveNestedOptimizer::optimize(const Trajectory &robot_traj_init, const Trajectory &human_traj_hp_init,
                                      const Trajectory &human_traj_rp_init, int acomm, double tcomm,
                                      Trajectory &robot_traj_opt, Trajectory *human_traj_hp_opt,
                                      Trajectory *human_traj_rp_opt)
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
    human_traj_hp_->u = human_traj_hp_init.u;

    human_traj_rp_.reset(new Trajectory(CONST_ACC_MODEL, T, dt));
    human_traj_rp_->x0 = human_traj_rp_init.x0;
    human_traj_rp_->u = human_traj_rp_init.u;

    // set lower and upper bounds
    std::vector<double> lb;
    std::vector<double> ub;

    utils::EigenToVector(lb_ur_, lb);
    utils::EigenToVector(ub_ur_, ub);

    optimizer_.set_lower_bounds(lb);
    optimizer_.set_upper_bounds(ub);

    // set cost function
    optimizer_.set_min_objective(cost_wrapper, this);

    // set tolerance
    optimizer_.set_xtol_abs(1e-2);
//    optimizer_.set_xtol_rel(1e-3);

    // initial condition
    std::vector<double> u_opt;
    utils::EigenToVector(robot_traj_init.u, u_opt);

//    *human_traj_hp_ = human_traj_hp_init;
//    *human_traj_rp_ = human_traj_rp_init;

    // optimizer!
    double min_cost;
    nlopt::result result = optimizer_.optimize(u_opt, min_cost);
    std::cout << "result is: " << result << ", min cost is: " << min_cost << std::endl;

    // send result back
    robot_traj_opt.x0 = robot_traj_init.x0;
    robot_traj_opt.u = Eigen::Map<Eigen::VectorXd>(u_opt.data(), u_opt.size());

    robot_traj_opt.compute();

    // compute "optimal" human trajectory if not null
    if (human_traj_hp_opt != nullptr) {
        *human_traj_hp_opt = *human_traj_hp_;
        *human_traj_rp_opt = *human_traj_rp_;
    }

    return min_cost;
}

//----------------------------------------------------------------------------------
double NaiveNestedOptimizer::cost_func(const std::vector<double> &u, std::vector<double> &grad)
{
    double cost = 0.0;

    // first need to compute the optimal human paths
    Eigen::Map<const Eigen::VectorXd> ur(u.data(), u.size());
    robot_traj_->update(ur);
    robot_traj_->compute_jacobian();

    int T = human_traj_hp_->horizon();
    double dt = human_traj_hp_->dt();
    Trajectory human_traj_hp_opt(CONST_ACC_MODEL, T, dt);
    Trajectory human_traj_rp_opt(CONST_ACC_MODEL, T, dt);

    // use two parallel threads to run optimization
    std::thread th1(&TrajectoryOptimizer::optimize, optimizer_hp_.get(), std::ref(*human_traj_hp_),
                    std::ref(*robot_traj_), std::ref(human_traj_hp_opt));
    std::thread th2(&TrajectoryOptimizer::optimize, optimizer_rp_.get(), std::ref(*human_traj_rp_),
                    std::ref(*robot_traj_), std::ref(human_traj_rp_opt));
//    optimizer_hp_->optimize(*human_traj_hp_, *robot_traj_, human_traj_hp_opt);
//    optimizer_rp_->optimize(*human_traj_rp_, *robot_traj_, human_traj_rp_opt);
    th1.join();
    th2.join();

    *human_traj_hp_ = human_traj_hp_opt;
    *human_traj_rp_ = human_traj_rp_opt;

    // compute the robot cost and gradients
    Eigen::VectorXd grad_ur;
    Eigen::VectorXd grad_uh_hp;
    Eigen::VectorXd grad_uh_rp;

    cost = robot_cost_->compute(*robot_traj_, human_traj_hp_opt, human_traj_rp_opt, acomm_, tcomm_,
                                grad_ur, grad_uh_hp, grad_uh_rp);

    robot_cost_->get_partial_cost(cost_hp_, cost_rp_, costs_non_int_);

    // compute the hessians for human cost functions
//    int len_uh = human_traj_hp_->traj_control_size();
//    int len_ur = robot_traj_->traj_control_size();
//    Eigen::MatrixXd hess_uh_hp(len_uh, len_uh);
//    Eigen::MatrixXd hess_uh_ur_hp(len_uh, len_ur);
//    Eigen::MatrixXd hess_uh_rp(len_uh, len_uh);
//    Eigen::MatrixXd hess_uh_ur_rp(len_uh, len_ur);
//
    auto cost_hp_cast = dynamic_cast<SingleTrajectoryCostHuman*>(human_cost_hp_.get());
    auto cost_rp_cast = dynamic_cast<SingleTrajectoryCostHuman*>(human_cost_rp_.get());
//
//    cost_hp_cast->hessian_uh(*robot_traj_, human_traj_hp_opt, hess_uh_hp);
//    cost_hp_cast->hessian_uh_ur(*robot_traj_, human_traj_hp_opt, hess_uh_ur_hp);
//    cost_rp_cast->hessian_uh(*robot_traj_, human_traj_rp_opt, hess_uh_rp);
//    cost_rp_cast->hessian_uh_ur(*robot_traj_, human_traj_rp_opt, hess_uh_ur_rp);
//
//    // compute the full gradient of ur
//    Eigen::VectorXd sol_hp = hess_uh_hp.colPivHouseholderQr().solve(grad_uh_hp);
//    Eigen::VectorXd sol_rp = hess_uh_rp.colPivHouseholderQr().solve(grad_uh_rp);
//
//    Eigen::VectorXd grad_inc = hess_uh_ur_hp.transpose() * sol_hp + hess_uh_ur_rp.transpose() * sol_rp;
//
//    grad_ur -= grad_inc;

    Eigen::VectorXd sub_grad_hp;
    Eigen::VectorXd sub_grad_rp;

    std::thread th_hp(&NaiveNestedOptimizer::cost_func_subroutine, this, std::ref(cost_hp_cast),
                      std::ref(human_traj_hp_opt), std::ref(grad_uh_hp), std::ref(sub_grad_hp));
    std::thread th_rp(&NaiveNestedOptimizer::cost_func_subroutine, this, std::ref(cost_rp_cast),
                      std::ref(human_traj_rp_opt), std::ref(grad_uh_rp), std::ref(sub_grad_rp));
    th_hp.join();
    th_rp.join();

    grad_ur -= sub_grad_hp + sub_grad_rp;


//    std::cout << "-------------" << std::endl;
//    std::cout << grad_inc.transpose() << std::endl;
//    std::cout << "-------------" << std::endl;
//    std::cout << hess_uh_ur_hp << std::endl;
//    std::cout << "-------------" << std::endl;
//    std::cout << hess_uh_ur_rp << std::endl;
//    std::cout << "-------------" << std::endl;
//    grad_ur -= hess_uh_ur_hp.transpose() * hess_uh_hp.colPivHouseholderQr().solve(grad_uh_hp)
//               + hess_uh_ur_rp.transpose() * hess_uh_rp.colPivHouseholderQr().solve(grad_uh_rp);

    utils::EigenToVector(grad_ur, grad);

//    static int counter = 0;
//    ++counter;
//    std::cout << "at iteration " << counter << " cost is: " << cost << std::endl;
//    std::cout << "gradient is: " << std::endl;
//    std::cout << grad_ur.transpose() << std::endl;
//    std::cout << std::endl;

    return cost;
}

//----------------------------------------------------------------------------------
void NaiveNestedOptimizer::cost_func_subroutine(SingleTrajectoryCostHuman *&cost, const Trajectory& human_traj,
                                                const Eigen::VectorXd &grad_uh, Eigen::VectorXd &sub_grad)
{
    int len_uh = human_traj_hp_->traj_control_size();
    int len_ur = robot_traj_->traj_control_size();
    Eigen::MatrixXd hess_uh(len_uh, len_uh);
    Eigen::MatrixXd hess_uh_ur(len_uh, len_ur);

    cost->hessian_uh(*robot_traj_, human_traj, hess_uh);
    cost->hessian_uh_ur(*robot_traj_, human_traj, hess_uh_ur);

    Eigen::VectorXd sol = hess_uh.colPivHouseholderQr().solve(grad_uh);
    sub_grad = hess_uh_ur.transpose() * sol;
}

} // namespace