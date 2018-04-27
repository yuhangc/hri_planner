//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/9/2018
// Last revision: 4/26/2018
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#ifndef HRI_PLANNER_OPTIMIZER_H
#define HRI_PLANNER_OPTIMIZER_H

#include <vector>
#include <memory>
#include <thread>

#include <Eigen/Dense>
#include <nlopt.hpp>

#include "hri_planner/trajectory.h"
#include "hri_planner/costs.h"
#include "hri_planner/cost_probabilistic.h"
#include "utils/utils.h"

namespace hri_planner {

class TrajectoryOptimizer {
public:
    // constructor
    TrajectoryOptimizer(unsigned int dim, const nlopt::algorithm& alg=nlopt::LD_MMA);

    // set cost function
    void set_cost_function(SingleTrajectoryCost* cost);
    void set_cost_function(std::shared_ptr<SingleTrajectoryCost> cost);

    void set_bounds(const Eigen::VectorXd& lb, const Eigen::VectorXd& ub);

    // set optimization time limit
    void set_time_limit(const double t_max);

    // set max iterations
    void set_max_iter(const int max_iter) {
        optimizer_.set_maxeval(max_iter);
    }

    // optimize!
    bool optimize(const Trajectory& traj_init, const Trajectory& traj_const, Trajectory& traj_opt);

    // get iteration
    int get_niter() {
        int neval = optimizer_.get_numevals() - neval_last_;
        neval_last_ = optimizer_.get_numevals();
        return neval;
    }

private:
    // non-linear optimizer
    nlopt::opt optimizer_;

    int neval_last_;

    // pointer to cost function
    std::shared_ptr<SingleTrajectoryCost> cost_;

    // an trajectory object to facilitate cost computation
    std::unique_ptr<Trajectory> traj_;

    // lower and upper bounds
    Eigen::VectorXd lb_;
    Eigen::VectorXd ub_;

    // wrapper function for using nlopt interface
    double cost_func(const std::vector<double>& u, std::vector<double>& grad);
    static double cost_wrapper(const std::vector<double>& u, std::vector<double>& grad, void* data);
};

class NestedOptimizerBase {
public:
    // constructor
    explicit NestedOptimizerBase(unsigned int dim, const nlopt::algorithm& alg=nlopt::LD_MMA);

    // set cost functions
    void set_robot_cost(ProbabilisticCostBase* cost);
    void set_robot_cost(std::shared_ptr<ProbabilisticCostBase> cost);

    virtual void set_human_cost(LinearCost* cost_hp, LinearCost* cost_rp) = 0;
    virtual void set_human_cost(const std::shared_ptr<LinearCost>& cost_hp,
                                const std::shared_ptr<LinearCost>& cost_rp) = 0;

    virtual void set_bounds(const Eigen::VectorXd& lb_ur, const Eigen::VectorXd& ub_ur,
                            const Eigen::VectorXd& lb_uh, const Eigen::VectorXd& ub_uh) = 0;

    // set optimization time limit
    virtual void set_time_limit(const double t_max) {
        optimizer_.set_maxtime(t_max);
    }

    // set iteration limit
    virtual void set_max_iter(const int max_iter) {
        optimizer_.set_maxeval(max_iter);
    }

    // optimize!
    virtual double optimize(const Trajectory& robot_traj_init, const Trajectory& human_traj_hp_init,
                            const Trajectory& human_traj_rp_init, int acomm, double tcomm,
                            Trajectory& robot_traj_opt, Trajectory* human_traj_hp_opt=nullptr,
                            Trajectory* human_traj_rp_opt=nullptr) = 0;

    void optimize_nr(const Trajectory& robot_traj_init, const Trajectory& human_traj_hp_init,
                     const Trajectory& human_traj_rp_init, int acomm, double tcomm, double& min_cost,
                     Trajectory& robot_traj_opt, Trajectory* human_traj_hp_opt=nullptr,
                     Trajectory* human_traj_rp_opt=nullptr) {
        min_cost = optimize(robot_traj_init, human_traj_hp_init, human_traj_rp_init, acomm, tcomm,
                            robot_traj_opt, human_traj_hp_opt, human_traj_rp_opt);
    }

    // get partial costs
    void get_partial_cost(double& cost_hp, double& cost_rp, std::vector<double>& cost_non_int) {
        cost_hp = cost_hp_;
        cost_rp = cost_rp_;
        cost_non_int = costs_non_int_;
    }

    int get_niter() {
        int n_eval = optimizer_.get_numevals() - neval_last_;
        neval_last_ = optimizer_.get_numevals();
        return n_eval;
    }

    void get_niter_nested(int& neval_hp, int& neval_rp) const {
        neval_hp = neval_nested_hp_;
        neval_rp = neval_nested_rp_;
    }

protected:
    // non-linear optimizer
    nlopt::opt optimizer_;

    int neval_last_;
    int neval_nested_hp_;
    int neval_nested_rp_;

    // pointer to cost function
    std::shared_ptr<ProbabilisticCostBase> robot_cost_;
    std::shared_ptr<LinearCost> human_cost_hp_;
    std::shared_ptr<LinearCost> human_cost_rp_;

    // trajectory objects to facilitate cost computation
    std::unique_ptr<Trajectory> robot_traj_;
    std::unique_ptr<Trajectory> human_traj_hp_;
    std::unique_ptr<Trajectory> human_traj_rp_;

    // lower and upper bounds
    Eigen::VectorXd lb_ur_;
    Eigen::VectorXd ub_ur_;
    Eigen::VectorXd lb_uh_;
    Eigen::VectorXd ub_uh_;

    // explicit communication action and time
    int acomm_;
    double tcomm_;

    // keep track of the partial costs
    double cost_hp_;
    double cost_rp_;
    std::vector<double> costs_non_int_;

    // wrapper cost function
    virtual double cost_func(const std::vector<double>& u, std::vector<double>& grad) = 0;
    static double cost_wrapper(const std::vector<double>& u, std::vector<double>& grad, void *cost_func_data);
};

class NestedTrajectoryOptimizer: public NestedOptimizerBase {
public:
    // constructor
    explicit NestedTrajectoryOptimizer(unsigned int dim, const nlopt::algorithm& alg=nlopt::LD_SLSQP):
            NestedOptimizerBase(dim, alg) {};

    void set_human_cost(LinearCost* cost_hp, LinearCost* cost_rp) override;
    void set_human_cost(const std::shared_ptr<LinearCost>& cost_hp,
                        const std::shared_ptr<LinearCost>& cost_rp) override;

    void set_bounds(const Eigen::VectorXd& lb_ur, const Eigen::VectorXd& ub_ur,
                    const Eigen::VectorXd& lb_uh, const Eigen::VectorXd& ub_uh) override;

    // optimize!
    double optimize(const Trajectory& robot_traj_init, const Trajectory& human_traj_hp_init,
                    const Trajectory& human_traj_rp_init, int acomm, double tcomm,
                    Trajectory& robot_traj_opt, Trajectory* human_traj_hp_opt=nullptr,
                    Trajectory* human_traj_rp_opt=nullptr) override;

    // check constraint
    double check_constraint(const Trajectory& robot_traj, const Trajectory& human_traj_hp,
                            const Trajectory& human_traj_rp);

private:
    // wrapper cost function
    double cost_func(const std::vector<double>& u, std::vector<double>& grad) override;

    // wrapper constraint function
    double constraint(const std::vector<double>& u, std::vector<double>& grad);
    static double constraint_wrapper(const std::vector<double>& u, std::vector<double>& grad, void *constraint_data);
};

//! the naive nested optimizer
class NaiveNestedOptimizer: public NestedOptimizerBase {
public:
    // contructor
    explicit NaiveNestedOptimizer(unsigned int dim_r, unsigned int dim_h,
                                  const nlopt::algorithm& alg, const nlopt::algorithm& sub_alg);

    void set_human_cost(LinearCost* cost_hp, LinearCost* cost_rp) override;
    void set_human_cost(const std::shared_ptr<LinearCost>& cost_hp,
                        const std::shared_ptr<LinearCost>& cost_rp) override;

    void set_bounds(const Eigen::VectorXd& lb_ur, const Eigen::VectorXd& ub_ur,
                    const Eigen::VectorXd& lb_uh, const Eigen::VectorXd& ub_uh) override;

    void set_time_limit(const double t_max) override {
        optimizer_.set_maxtime(t_max);

        // heuristically set the time limit for the follower optimizers
        optimizer_hp_->set_time_limit(t_max * 0.08);
        optimizer_rp_->set_time_limit(t_max * 0.08);
    }

    void set_max_iter(const int max_iter) override {
        optimizer_.set_maxeval(max_iter);

        // FIXME: hard-coded limit
        optimizer_hp_->set_max_iter(20);
        optimizer_rp_->set_max_iter(20);
    }

    // optimize!
    double optimize(const Trajectory& robot_traj_init, const Trajectory& human_traj_hp_init,
                    const Trajectory& human_traj_rp_init, int acomm, double tcomm,
                    Trajectory& robot_traj_opt, Trajectory* human_traj_hp_opt=nullptr,
                    Trajectory* human_traj_rp_opt=nullptr) override;

private:
    // optimizers for obtaining human trajectory
    std::unique_ptr<TrajectoryOptimizer> optimizer_hp_;
    std::unique_ptr<TrajectoryOptimizer> optimizer_rp_;

    double cost_func(const std::vector<double>& u, std::vector<double>& grad) override;
    void cost_func_subroutine(SingleTrajectoryCostHuman*& cost, const Trajectory& human_traj,
                              const Eigen::VectorXd& grad_uh, Eigen::VectorXd& sub_grad);
};

} // namespace

#endif //HRI_PLANNER_OPTIMIZER_H
