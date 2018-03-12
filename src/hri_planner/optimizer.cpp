//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/9/2017
// Last revision: 3/9/2017
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
bool TrajectoryOptimizer::optimize(const Eigen::VectorXd& x0, const Trajectory &traj_init, Trajectory &traj_opt)
{
    // update initial state
    x0_ = x0;

    // recreate the trajectory object with trajectory data
    traj_.reset(new Trajectory(traj_init.dyn_type, traj_init.horizon(), traj_init.dt()));

    // set lower and upper bounds
    std::vector<double> lb;
    std::vector<double> ub;

    EigenToVector(lb_, lb);
    EigenToVector(ub_, ub);

    optimizer_.set_lower_bounds(lb);
    optimizer_.set_upper_bounds(ub);

    return true;
}

//----------------------------------------------------------------------------------
double TrajectoryOptimizer::cost_func(const std::vector<double> &u, std::vector<double> &grad)
{
    // re-compute the trjectory
    Eigen::Map<const Eigen::VectorXd> u_new(u.data(), u.size());

    traj_->update(x0_, u_new);
    traj_->compute_jacobian(x0_);

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

}

//----------------------------------------------------------------------------------
void NestedTrajectoryOptimizer::set_human_cost(HumanCost *cost)
{

}

//----------------------------------------------------------------------------------
void NestedTrajectoryOptimizer::set_human_cost(std::shared_ptr<HumanCost> cost)
{

}

//----------------------------------------------------------------------------------
void NestedTrajectoryOptimizer::set_robot_cost(LinearCost *cost)
{

}

//----------------------------------------------------------------------------------
void NestedTrajectoryOptimizer::set_robot_cost(std::shared_ptr<LinearCost> cost)
{

}

//----------------------------------------------------------------------------------
bool NestedTrajectoryOptimizer::optimize(const Eigen::VectorXd &xr0, const Eigen::VectorXd &xh0,
                                         const Trajectory &robot_traj_init, const Trajectory &human_traj_init,
                                         Trajectory &robot_traj_opt, Trajectory &human_traj_opt)
{

}

//----------------------------------------------------------------------------------
double NestedTrajectoryOptimizer::cost_func(const std::vector<double> &u, std::vector<double> &grad)
{

}

//----------------------------------------------------------------------------------
double NestedTrajectoryOptimizer::cost_wrapper(const std::vector<double> &u, std::vector<double> &grad,
                                              void *cost_func_data)
{

}

//----------------------------------------------------------------------------------
double NestedTrajectoryOptimizer::constraint(const std::vector<double> &u, std::vector<double> &grad)
{

}

//----------------------------------------------------------------------------------
double NestedTrajectoryOptimizer::constraint_wrapper(const std::vector<double> &u, std::vector<double> &grad,
                                                    void *constraint_data)
{

}

} // namespace