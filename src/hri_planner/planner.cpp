//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/24/2017
// Last revision: 3/26/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include <string>

#include "hri_planner/planner.h"

namespace hri_planner {

//----------------------------------------------------------------------------------
Planner::Planner(ros::NodeHandle &nh, ros::NodeHandle &pnh): nh_(nh)
{
    // load the dimensions
    ros::param::param<int>("~dimension/T", T_, 10);
    ros::param::param<int>("~dimension/nXh", nXh_, 4);
    ros::param::param<int>("~dimension/nUh", nUh_, 2);
    ros::param::param<int>("~dimension/nXr", nXr_, 3);
    ros::param::param<int>("~dimension/nUr", nUr_, 2);
    ros::param::param<double>("~dimension/dt", dt_, 0.5);

    // communication cost
    ros::param::param<double>("~planner/comm_cost", comm_cost_, 5.0);

    // create the optimizer
    create_optimizer();

    // assuming no communication at the beginning
    tcomm_ = -20;
    acomm_ = 0;

    // initialize the optimal trajectory
    robot_traj_opt_ = Trajectory(DIFFERENTIAL_MODEL, T_, dt_);
    human_traj_hp_opt_ = Trajectory(CONST_ACC_MODEL, T_, dt_);
    human_traj_rp_opt_ = Trajectory(CONST_ACC_MODEL, T_, dt_);

    robot_traj_init_ = Trajectory(DIFFERENTIAL_MODEL, T_, dt_);
    human_traj_hp_init_ = Trajectory(CONST_ACC_MODEL, T_, dt_);
    human_traj_rp_init_ = Trajectory(CONST_ACC_MODEL, T_, dt_);

    // flags
    ros::param::param<bool>("~planner/publish_full_plan", flag_publish_full_plan_, false);
    flag_gen_init_guesses_ = true;

    // measurements
    xr_meas_.setZero(nXr_);
    ur_meas_.setZero(nUr_);
    xh_meas_.setZero(nXh_);

    // create subscribers and publishers
    robot_state_sub_ = nh_.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/amcl_pose", 1,
                                                                               &Planner::robot_state_callback, this);

    robot_odom_sub_ = nh_.subscribe<nav_msgs::Odometry>("/odom", 1, &Planner::robot_odom_callback, this);

    human_state_sub_ = nh_.subscribe<std_msgs::Float64MultiArray>("/tracked_human", 1,
                                                                  &Planner::human_state_callback, this);

    robot_ctrl_pub_ = nh_.advertise<geometry_msgs::Twist>("/planner/cmd_vel", 1);
    comm_pub_ = nh_.advertise<std_msgs::Int32>("/planner/communication", 1);
    plan_pub_ = nh_.advertise<hri_planner::PlannedTrajectories>("/planner/full_plan", 1);

    ROS_INFO("All subscribers and publishers initialized!");
}

//----------------------------------------------------------------------------------
void Planner::create_belief_model(std::shared_ptr<BeliefModelBase> &belief_model)
{
    int T_hist;
    double ratio;
    double decay_rate;
    std::vector<double> fcorrection(2, 0);

    ros::param::param<int>("~explicit_comm/history_length", T_hist, 10);
    ros::param::param<double>("~explicit_comm/ratio", ratio, 100.0);
    ros::param::param<double>("~explicit_comm/decay_rate", decay_rate, 2.5);
    ros::param::param<double>("~explicit_comm/fcorrection_hp", fcorrection[HumanPriority], 3.0);
    ros::param::param<double>("~explicit_comm/fcorrection_rp", fcorrection[RobotPriority], 30.0);

    belief_model = std::make_shared<hri_planner::BeliefModelExponential>(T_hist, fcorrection, ratio, decay_rate);
    belief_model->reset_hist(Eigen::Vector2d::Zero());
}

//----------------------------------------------------------------------------------
void Planner::create_human_costs(std::shared_ptr<HumanCost>& human_cost_hp,
                                 std::shared_ptr<HumanCost>& human_cost_rp,
                                 std::shared_ptr<SingleTrajectoryCostHuman>& single_cost_hp,
                                 std::shared_ptr<SingleTrajectoryCostHuman>& single_cost_rp)
{
    std::vector<std::shared_ptr<FeatureBase> > features_hp;
    std::vector<std::shared_ptr<FeatureBase> > features_rp;
    std::vector<double> weights_hp;
    std::vector<double> weights_rp;

    // first create the cost features
    for (int type = 0; type < 2; ++type) {
        std::string type_str;
        type_str = (type == HumanPriority) ? "hp" : "rp";

        int n_features;
        ros::param::param<int>("~human_cost/n_features", n_features, 5);

        for (int i = 0; i < n_features; ++i) {
            std::string feature_str = "~human_cost_" + type_str + "/feature" + std::to_string(i);

            ROS_INFO("now loading feature %s ...", feature_str.c_str());

            std::string feature_name;
            ros::param::param<std::string>(feature_str + "/name", feature_name, "");

            int n_args;
            std::vector<double> args;

            ros::param::param<int>(feature_str + "/nargs", n_args, 0);
            if (n_args > 0)
                ros::param::get(feature_str + "/args", args);

            double w;
            ros::param::param<double>(feature_str + "/weight", w, 1.0);

            // add to feature and weight list
            std::shared_ptr<FeatureBase> feature = FeatureHumanCost::create(feature_name, args);
            features_human_.insert({feature_name + "_" + type_str, feature});

            if (type == HumanPriority) {
                features_hp.push_back(feature);
                weights_hp.push_back(w);
            }
            else {
                features_rp.push_back(feature);
                weights_rp.push_back(w);
            }
        }
    }

    // create the cost functions
    human_cost_hp = std::make_shared<HumanCost>(weights_hp, features_hp);
    human_cost_rp = std::make_shared<HumanCost>(weights_rp, features_rp);
    single_cost_hp = std::make_shared<SingleTrajectoryCostHuman>(weights_hp, features_hp);
    single_cost_rp = std::make_shared<SingleTrajectoryCostHuman>(weights_rp, features_rp);
}

//----------------------------------------------------------------------------------
void Planner::create_robot_costs(std::shared_ptr<ProbabilisticCostBase>& robot_cost)
{
    std::vector<std::shared_ptr<FeatureBase> > f_non_int;
    std::vector<std::shared_ptr<FeatureVectorizedBase> > f_int;
    std::vector<double> w_non_int;
    std::vector<double> w_int;

    // non interactive features
    int n_non_int;
    ros::param::param<int>("~robot_cost/n_features_non_int", n_non_int, 2);

    for (int i = 0; i < n_non_int; ++i) {
        std::string feature_str = "~robot_cost_non_int/feature" + std::to_string(i);
        ROS_INFO("now loading feature %s ...", feature_str.c_str());

        std::string feature_name;
        ros::param::param<std::string>(feature_str + "/name", feature_name, "");

        int n_args;
        std::vector<double> args;

        ros::param::param<int>(feature_str + "/nargs", n_args, 0);
        if (n_args > 0)
            ros::param::get(feature_str + "/args", args);

        double w;
        ros::param::param<double>(feature_str + "/weight", w, 1.0);

        // add to feature and weight list
        std::shared_ptr<FeatureBase> feature = FeatureRobotCost::create(feature_name, args);
        features_robot_.insert({feature_name, feature});

        f_non_int.push_back(feature);
        w_non_int.push_back(w);
    }

    // interactive features
    int n_int;
    ros::param::param<int>("~robot_cost/n_features_int", n_int, 2);

    for (int i = 0; i < n_int; ++i) {
        std::string feature_str = "~robot_cost_int/feature" + std::to_string(i);
        ROS_INFO("now loading feature %s ...", feature_str.c_str());

        std::string feature_name;
        ros::param::param<std::string>(feature_str + "/name", feature_name, "");

        int n_args;
        std::vector<double> args;

        ros::param::param<int>(feature_str + "/nargs", n_args, 0);
        if (n_args > 0)
            ros::param::get(feature_str + "/args", args);

        double w;
        ros::param::param<double>(feature_str + "/weight", w, 1.0);

        // add to feature and weight list
        f_int.push_back(FeatureVectorizedBase::create(feature_name, args));
        w_int.push_back(w);
    }

    // create a belief model
    create_belief_model(belief_model_);
    ROS_INFO("Belief model created...");

    // create the robot cost function and set cost features
    robot_cost = std::make_shared<ProbabilisticCostSimplified>(belief_model_);

    robot_cost->set_features_non_int(w_non_int, f_non_int);
    robot_cost->set_features_int(w_int, f_int);
}

//----------------------------------------------------------------------------------
void Planner::create_optimizer()
{
    // create human cost functions
    std::shared_ptr<HumanCost> human_cost_hp;
    std::shared_ptr<HumanCost> human_cost_rp;
    std::shared_ptr<SingleTrajectoryCostHuman> single_cost_hp;
    std::shared_ptr<SingleTrajectoryCostHuman> single_cost_rp;

    create_human_costs(human_cost_hp, human_cost_rp, single_cost_hp, single_cost_rp);

    ROS_INFO("Human cost func created...");

    // create the robot cost functions
    std::shared_ptr<ProbabilisticCostBase> robot_cost;
    create_robot_costs(robot_cost);

    ROS_INFO("Robot cost func created...");

    // load optimizer configuration
//    std::string optimizer_type;
//    ros::param::param<std::string>("~optimizer/type", optimizer_type, "NestedNaive");

    int dim_r = T_ * nUr_;
    int dim_h = T_ * nUh_;

    // FIXME: only use the naive nested optimizer with SLSQP for now
    optimizer_ = std::make_shared<NaiveNestedOptimizer>(static_cast<unsigned int>(dim_r),
                                                        static_cast<unsigned int>(dim_r),
                                                        nlopt::LD_SLSQP, nlopt::LD_SLSQP);

    ROS_INFO("Optimizer created...");

    // set costs
    optimizer_->set_robot_cost(robot_cost);
    optimizer_->set_human_cost(single_cost_hp, single_cost_rp);

    // load and set bounds
    Eigen::VectorXd lb_ur(dim_r);
    Eigen::VectorXd ub_ur(dim_r);
    Eigen::VectorXd lb_uh(dim_h);
    Eigen::VectorXd ub_uh(dim_h);

    ros::param::get("~optimizer/bounds/lb_ur", lb_ur_vec_);
    ros::param::get("~optimizer/bounds/ub_ur", ub_ur_vec_);
    ros::param::get("~optimizer/bounds/lb_uh", lb_uh_vec_);
    ros::param::get("~optimizer/bounds/ub_uh", ub_uh_vec_);

    for (int t = 0; t < T_; ++t) {
        for (int i = 0; i < nUr_; ++i) {
            lb_ur(t*nUr_+i) = lb_ur_vec_[i];
            ub_ur(t*nUr_+i) = ub_ur_vec_[i];
        }

        for (int i = 0; i < nUh_; ++i) {
            lb_uh(t*nUh_+i) = lb_uh_vec_[i];
            ub_uh(t*nUh_+i) = ub_uh_vec_[i];
        }
    }

    std::cout << "bounds:" << std::endl;
    std::cout << lb_ur.transpose() << std::endl;
    std::cout << ub_ur.transpose() << std::endl;
    std::cout << lb_uh.transpose() << std::endl;
    std::cout << ub_uh.transpose() << std::endl;

    optimizer_->set_bounds(lb_ur, ub_ur, lb_uh, ub_uh);
}

//----------------------------------------------------------------------------------
void Planner::compute_plan()
{
    ROS_INFO("Start to compute plan...");
    // copy the current state measurements
    xr_ = xr_meas_;
    ur_ = ur_meas_;
    xh_ = xh_meas_;

    // first update current belief
    // FIXME: decrease tcomm each time, t_curr is always 0
    tcomm_ -= dt_;
    belief_model_->update_belief(xr_, ur_, xh_, acomm_, tcomm_, 0.0);

    // update initial guesses
    update_init_guesses();

    // optimize for no communication
    double cost_no_comm;
    Trajectory robot_traj_opt_n(DIFFERENTIAL_MODEL, T_, dt_);
    Trajectory human_traj_hp_opt_n(CONST_ACC_MODEL, T_, dt_);
    Trajectory human_traj_rp_opt_n(CONST_ACC_MODEL, T_, dt_);

    std::cout << "before optimization" << std::endl;
    cost_no_comm = optimizer_->optimize(robot_traj_init_, human_traj_hp_init_, human_traj_rp_init_, acomm_, tcomm_,
                                        robot_traj_opt_n, &human_traj_hp_opt_n, &human_traj_rp_opt_n);
    std::cout << "after optimization" << std::endl;

    // optimize for communication
    double cost_comm;
    Trajectory robot_traj_opt(DIFFERENTIAL_MODEL, T_, dt_);
    Trajectory human_traj_hp_opt(CONST_ACC_MODEL, T_, dt_);
    Trajectory human_traj_rp_opt(CONST_ACC_MODEL, T_, dt_);

    cost_comm = optimizer_->optimize(robot_traj_init_, human_traj_hp_init_, human_traj_rp_init_, intent_, 0.0,
                                     robot_traj_opt, &human_traj_hp_opt, &human_traj_rp_opt);
    cost_comm += comm_cost_;

    // compare the cost and choose optimal actions
    if (cost_comm < cost_no_comm) {
        robot_traj_opt_ = robot_traj_opt_n;
        human_traj_hp_opt_ = human_traj_hp_opt_n;
        human_traj_rp_opt_ = human_traj_hp_opt_n;

        acomm_ = intent_;
        tcomm_ = 0.0;
    }
    else {
        robot_traj_opt_ = robot_traj_opt;
        human_traj_hp_opt_ = human_traj_hp_opt;
        human_traj_rp_opt_ = human_traj_hp_opt;
    }

    ROS_INFO("Got plan!");

    // publish communicative action if any
    if (tcomm_ == 0.0) {
        std_msgs::Int32 comm;
        comm.data = acomm_;
        comm_pub_.publish(comm);
    }

    // publish robot control
    geometry_msgs::Twist cmd_vel;
    cmd_vel.linear.x = robot_traj_opt_.u(0);
    cmd_vel.angular.z = robot_traj_opt_.u(1);
    robot_ctrl_pub_.publish(cmd_vel);

    // publish full plan if specified
    if (flag_publish_full_plan_) {
        PlannedTrajectories trajectories;
        trajectories.T = T_;
        trajectories.nXr = nXr_;
        trajectories.nXh = nXh_;
        utils::EigenToVector(robot_traj_opt_.x, trajectories.robot_traj_opt);
        utils::EigenToVector(human_traj_hp_opt_.x, trajectories.human_traj_hp_opt);
        utils::EigenToVector(human_traj_rp_opt_.x, trajectories.human_traj_rp_opt);

        plan_pub_.publish(trajectories);
    }
}

//----------------------------------------------------------------------------------
void Planner::reset_planner(const Eigen::VectorXd &xr_goal, const Eigen::VectorXd &xh_goal)
{
    // update the goals for robot and human
    features_robot_["Goal"]->set_data(&xr_goal);
    features_human_["Goal_hp"]->set_data(&xh_goal);
    features_human_["Goal_rp"]->set_data(&xh_goal);

    xr_goal_ = xr_goal;
    xh_goal_ = xh_goal;

    // reset other things
    // assuming no communication at the beginning
    tcomm_ = -20;
    acomm_ = 0;

    // flags
    flag_gen_init_guesses_ = true;

    // reset belief model
    belief_model_->reset_hist(Eigen::Vector2d::Zero());
}

//----------------------------------------------------------------------------------
void Planner::generate_init_guesses(Trajectory &robot_traj, Trajectory &human_traj_hp, Trajectory &human_traj_rp)
{
    // create initial guesses for robot control and human trajectory
    //! use a closed-loop control law to generate initial control for the robot
    Eigen::VectorXd ur;
    generate_steer_posq(xr_, xr_goal_, ur);

    robot_traj.update(xr_, ur);

    //! for human initial guess also use a control-based alg to generate
    Eigen::VectorXd uh;
    generate_steer_acc(xh_, xh_goal_, uh);

    human_traj_hp.update(xh_, uh);
    human_traj_rp.update(xh_, uh);
}

//----------------------------------------------------------------------------------
void Planner::update_init_guesses()
{
    if (flag_gen_init_guesses_) {
        generate_init_guesses(robot_traj_init_, human_traj_hp_init_, human_traj_rp_init_);
    }
    else {
        // use the optimal plan from previous time step as initial guess
        shift_control(robot_traj_opt_.u, robot_traj_init_.u, nUr_, false);
        shift_control(human_traj_hp_opt_.u, human_traj_hp_init_.u, nUh_, true);
        shift_control(human_traj_rp_opt_.u, human_traj_rp_init_.u, nUh_, true);

        robot_traj_init_.x0 = xr_;
        human_traj_hp_init_.x0 = xh_;
        human_traj_rp_init_.x0 = xh_;

        robot_traj_init_.compute();
        robot_traj_init_.compute_jacobian();
        human_traj_hp_init_.compute();
        human_traj_hp_init_.compute_jacobian();
        human_traj_rp_init_.compute();
        human_traj_rp_init_.compute_jacobian();
    }
}

//----------------------------------------------------------------------------------
void Planner::shift_control(const Eigen::VectorXd &u_in, Eigen::VectorXd &u_out, int dim, bool pad_zero)
{
    long len = u_in.size();
    u_out.segment(0, len-dim) = const_cast<Eigen::VectorXd&>(u_in).segment(dim, len-dim);

    if (pad_zero) {
        // set control for new time step as 0
        u_out.segment(len-dim, dim).setZero();
    }
    else {
        // set control for new time step the same as last time step
        u_out.segment(len-dim, dim) = const_cast<Eigen::VectorXd&>(u_in).segment(len-dim, dim);
    }
}

//----------------------------------------------------------------------------------
void Planner::generate_steer_posq(const Eigen::VectorXd &x0, const Eigen::VectorXd &x_goal, Eigen::VectorXd &ur)
{
    // get parameters
    double k_rho, k_v, k_alp, k_phi, gamma;
    ros::param::param<double>("~steer_posq/k_rho", k_rho, 1.0);
    ros::param::param<double>("~steer_posq/k_v", k_v, 3.8);
    ros::param::param<double>("~steer_posq/k_alp", k_alp, 6.0);
    ros::param::param<double>("~steer_posq/k_phi", k_phi, -1.0);
    ros::param::param<double>("~steer_posq/gamma", gamma, 0.15);

    Eigen::VectorXd xr(T_ * nXr_);
    ur.resize(T_ * nUr_);

    // create a differential dynamics object
    DifferentialDynamics dyn(dt_);

    Eigen::VectorXd x_last = x0;
    for (int t = 0; t < T_; ++t) {
        // compute desired control
        double rho = (x_goal - x_last.head(2)).norm();
        double phi = 0.0;   //! don't care about orientation for now
        double th_z = std::atan2(x_goal(1) - x_last(1), x_goal(0) - x_last(0));
        double alpha = utils::wrap_to_pi(th_z - x_last(2));

        ur(t*nUr_) = utils::clamp(k_rho * std::tanh(k_v * rho), lb_ur_vec_[0], ub_ur_vec_[0]);
        ur(t*nUr_+1) = utils::clamp(k_alp * alpha + k_phi * phi, lb_ur_vec_[1], ub_ur_vec_[1]);

        dyn.forward_dyn(x_last, ur.segment(t*nUr_, nUr_), xr.segment(t*nXr_, nXr_));
        x_last = xr.segment(t*nXr_, nXr_);
    }
}

//----------------------------------------------------------------------------------
void Planner::generate_steer_acc(const Eigen::VectorXd &x0, const Eigen::VectorXd &x_goal, Eigen::VectorXd &uh)
{
    double v_max, a_max;
    double k_v;
    ros::param::param<double>("~steer_acc/v_max", v_max, 1.0);
    ros::param::param<double>("~steer_acc/a_max", a_max, 0.5);
    ros::param::param<double>("~steer_acc/k_v", k_v, 1.0);

    Eigen::VectorXd xh(T_ * nXh_);
    uh.resize(T_ * nUh_);

    // create a const acc dynamics object
    ConstAccDynamics dyn(dt_);
    Eigen::VectorXd x_last = x0;
    for (int t = 0; t < T_; ++t) {
        // compute desired velocity
        Eigen::VectorXd x_diff = x_goal - x_last.head(2);
        Eigen::VectorXd vd = k_v * x_diff;
        double v_mag = utils::clamp(vd.norm(), 0.0, v_max);

        vd.normalize();
        vd *= v_mag;

        // compute the desired acceleration
        Eigen::VectorXd ad = (vd - x_last.tail(2)) / dt_;
        double a_mag = utils::clamp(ad.norm(), 0.0, a_max);

        ad.normalize();
        ad *= a_mag;

        // update
        uh.segment(t*nUh_, nUh_) = ad;
        dyn.forward_dyn(x_last, ad, xh.segment(t*nXh_, nXh_));
        x_last = xh.segment(t*nXh_, nXh_);
    }
}

//----------------------------------------------------------------------------------
void Planner::robot_state_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr &pose_msg)
{
    xr_meas_(0) = pose_msg->pose.pose.position.x;
    xr_meas_(1) = pose_msg->pose.pose.position.y;

    // find rotation
    auto &q = pose_msg->pose.pose.orientation;
    double siny = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);

    xr_meas_(2) = std::atan2(siny, cosy);
}

//----------------------------------------------------------------------------------
void Planner::robot_odom_callback(const nav_msgs::OdometryConstPtr &odom_msg)
{
    ur_meas_(0) = odom_msg->twist.twist.linear.x;
    ur_meas_(1) = odom_msg->twist.twist.angular.z;
}

//----------------------------------------------------------------------------------
void Planner::human_state_callback(const std_msgs::Float64MultiArrayConstPtr& state_msg)
{
    xh_meas_(0) = state_msg->data[0];
    xh_meas_(1) = state_msg->data[1];
    xh_meas_(2) = state_msg->data[2];
    xh_meas_(3) = state_msg->data[3];
}

} // namespace