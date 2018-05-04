//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/24/2018
// Last revision: 4/23/2018
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include <ctime>
#include <chrono>

#include "hri_planner/planner_node.h"

//----------------------------------------------------------------------------------
PlannerNode::PlannerNode(ros::NodeHandle &nh, ros::NodeHandle &pnh): nh_(nh)
{
    // get parameters
    ros::param::param<double>("~planner/planner_rate", planning_rate_, 2.0);
    ros::param::param<double>("~planner/dt_planning", dt_planning_, 1.0 / planning_rate_);
    ros::param::param<double>("~planner/controller_rate", controller_rate_, 10);
    ros::param::param<double>("~planner/state_machine_rate", state_machine_rate_, 1000);
    ros::param::param<std::string>("~planner/planner_mode", mode_, "simulation");
    ros::param::param<bool>("~planner/allow_explicit_comm", flag_allow_explicit_comm_, true);
    ros::param::param<double>("~planner/goal_reaching_th_planner", goal_reaching_th_planner_, 0.5);
    ros::param::param<double>("~planner/goal_reaching_th_controller", goal_reaching_th_controller_, 0.1);

    ros::param::param<double>("~planner/human_filter_dist_th", human_filter_dist_th_, 1.0);

    ros::param::param<int >("~planner/human_tracking_lost_th", tracking_lost_th_, 2);

    int nXh, nUh, nXr, nUr;
    ros::param::param<int>("~dimension/nXh", nXh, 4);
    ros::param::param<int>("~dimension/nUh", nUh, 2);
    ros::param::param<int>("~dimension/nXr", nXr, 3);
    ros::param::param<int>("~dimension/nUr", nUr, 2);

    // create planner
    planner_interactive_ = std::make_shared<hri_planner::Planner>(nh, pnh);
    planner_simple_ = std::make_shared<hri_planner::PlannerSimple>(nh, pnh);

    // measurements
    xr_meas_.setZero(nXr);
    ur_meas_.setZero(nUr);
    xh_meas_.setZero(nXh);

    ros::param::param<int>("~dimension/dim_goal", goal_dim_, 2);

    xr_goal_.resize(goal_dim_);
    xr_goal_ << 0.0, 3.0;
    xh_goal_.resize(goal_dim_);
    xh_init_.resize(goal_dim_);

    // create subscribers
    goal_sub_ = nh.subscribe<std_msgs::Float64MultiArray>("/planner/set_goal", 1,
                                                          &PlannerNode::goal_callback, this);
    planner_ctrl_sub_ = nh.subscribe<std_msgs::String>("/planner/ctrl", 1,
                                                       &PlannerNode::planner_ctrl_callback, this);

//    robot_state_sub_ = nh_.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/amcl_pose", 1,
//                                                            &PlannerNode::robot_state_callback, this);

    robot_odom_sub_ = nh_.subscribe<nav_msgs::Odometry>("/odom", 1, &PlannerNode::robot_odom_callback, this);

    if (mode_ == "simulation") {
        human_tracking_sub_ = nh_.subscribe<people_msgs::People>("/people", 1,
                                                                 &PlannerNode::human_tracking_callback, this);
    }
    else {
        human_tracking_sub_ = nh_.subscribe<people_msgs::PositionMeasurementArray>("/people_tracker_measurements", 1,
                                                                       &PlannerNode::human_detection_callback, this);
    }

    // publishers
    goal_reached_pub_ = nh_.advertise<std_msgs::Bool>("/planner/goal_reached", 1);
    robot_ctrl_pub_ = nh_.advertise<geometry_msgs::Twist>("/planner/cmd_vel", 1);
    robot_human_state_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("/planner/robot_human_state", 1);
}

//----------------------------------------------------------------------------------
void PlannerNode::run()
{
    // loop and a simple state machine
    PlannerStates planner_state = Idle;
    reset_state_machine();

    // two rates
    ros::Rate rate_fast(state_machine_rate_);
    ros::Rate rate_slow(planning_rate_);
    ros::Rate rate_controller(controller_rate_);

    while (!ros::isShuttingDown()) {
        switch (planner_state) {
            case Idle:
                ROS_INFO("In state Idle");

                rate_fast.reset();
                while (!flag_start_planning_ && !ros::isShuttingDown()) {
                    ros::spinOnce();
                    rate_fast.sleep();
                }

                flag_start_planning_ = false;

                std::cout << "Robot goal is: " << xr_goal_.transpose() << std::endl;

                planner_state = Planning;

                break;
            case Pausing:
                ROS_INFO("In state Pausing");

                rate_fast.sleep();
                while (!ros::isShuttingDown()) {
                    ros::spinOnce();

                    if (flag_stop_planning_) {
                        flag_stop_planning_ = false;
                        planner_state = Idle;
                        reset_state_machine();
                        break;
                    }

                    if (!flag_pause_planning_) {
                        flag_pause_planning_ = true;
                        planner_state = Planning;
                        break;
                    }

                    rate_fast.sleep();
                }

                break;
            case Planning:
                ROS_INFO("In state Planning");

                rate_slow.reset();
                while (!ros::isShuttingDown()) {
                    ros::spinOnce();

                    // if human is detected, then use interactive planner
                    if (flag_human_detected_) {
                        flag_human_detected_frame_ = true;

                        if (flag_human_tracking_lost_) {
                            flag_human_tracking_lost_ = false;
                            human_tracking_lost_frames_ = 0;
                            planner_interactive_->reset_planner();
                        }

                        flag_human_detected_ = false;

                        ROS_INFO("Using interactive planner...");
                        plan(planner_interactive_);
                    }
                    else {
                        flag_human_detected_frame_ = false;

                        // if human not detected, increase count and check
                        if (!flag_human_tracking_lost_) {
                            ++human_tracking_lost_frames_;
                            if (human_tracking_lost_frames_ > tracking_lost_th_) {
                                flag_human_tracking_lost_ = true;
                                planner_simple_->reset_planner();
                            }
                        }

                        // if tracking not lost, use a prediction
                        if (!flag_human_tracking_lost_) {
                            dynamic_cast<hri_planner::Planner*>(planner_interactive_.get())->
                                    get_human_pred(0, intent_, xh_meas_);

                            ROS_INFO("Using interactive planner...");
                            plan(planner_interactive_);
                        }
                        else {
                            // otherwise use the simple planner
                            ROS_INFO("Using simple planner since no human detected...");
                            plan(planner_simple_);
                        }
                    }

                    rate_slow.sleep();

                    if (mode_ == "simulation") {
                        planner_state = Pausing;
                        break;
                    }
                    else {
                        // check for stop flag and goal reached
                        if (flag_stop_planning_) {
                            flag_stop_planning_ = false;
                            planner_state = Idle;
                            reset_state_machine();

                            break;
                        }

                        Eigen::VectorXd x_diff = xr_goal_ - xr_meas_.head(2);
                        if (x_diff.norm() < goal_reaching_th_planner_) {
//                            planner_state = GoalReaching;
                            // publish goal reaching message
                            ROS_INFO("Now switching to closed-loop controller...");
                            std_msgs::Bool goal_reach_data;
                            goal_reach_data.data = static_cast<uint8_t>(true);
                            goal_reached_pub_.publish(goal_reach_data);

                            // planner back to idle
                            planner_state = Idle;
                            reset_state_machine();

                            break;
                        }
                    }
                }

                break;

            case GoalReaching:
                ROS_INFO("In state Goal Reaching");

                rate_controller.reset();
                while (!ros::isShuttingDown()) {
                    ros::spinOnce();

                    // check for stop signal
                    if (flag_stop_planning_) {
                        flag_stop_planning_ = false;
                        planner_state = Idle;
                        break;
                    }

                    // check for goal reaching
                    Eigen::VectorXd x_diff = xr_goal_ - xr_meas_.head(2);
                    if (x_diff.norm() < goal_reaching_th_controller_) {
                        planner_state = Idle;

                        // publish to goal reached if so
                        std_msgs::Bool goal_reach_data;
                        goal_reach_data.data = static_cast<uint8_t>(true);
                        goal_reached_pub_.publish(goal_reach_data);

                        break;
                    }

                    // compute the desired actions
                    compute_and_publish_control();

                    rate_controller.sleep();
                }

                break;
        }
    }
}

//----------------------------------------------------------------------------------
void PlannerNode::plan(const std::shared_ptr<hri_planner::PlannerBase> &planner)
{
    // update robot pose using tf listener
    tf::StampedTransform transform;
    try {
        tf_listener_.lookupTransform("/map", "/base_footprint", ros::Time(0), transform);
    }
    catch (tf::TransformException &ex) {
        ROS_ERROR("%s", ex.what());
    }

    auto pos = transform.getOrigin();
    xr_meas_(0) = pos.x();
    xr_meas_(1) = pos.y();

    auto q = transform.getRotation();
    double siny = 2.0 * (q.w() * q.z() + q.x() * q.y());
    double cosy = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
    xr_meas_(2) = std::atan2(siny, cosy);

    // use simple prediction to take into account the planning time
    if (mode_ != "simulation") {
        // FIXME: this doesn't seem to be helpful?
        Eigen::VectorXd xr_pred = xr_meas_;
        Eigen::VectorXd xh_pred = xh_meas_;

        // predict robot pose approximately with linear model
        double th = xr_meas_(2);
        xr_meas_(0) += ur_meas_(0) * std::cos(th) * dt_planning_;
        xr_meas_(1) += ur_meas_(0) * std::sin(th) * dt_planning_;
        xr_meas_(2) += ur_meas_(1) * dt_planning_;

        th = xr_meas_(2);
        xr_meas_(0) += ur_meas_(0) * std::cos(th) * dt_planning_;
        xr_meas_(1) += ur_meas_(0) * std::sin(th) * dt_planning_;
        xr_meas_(2) += ur_meas_(1) * dt_planning_;

        // predict human pose with steer model
        Eigen::VectorXd xh_pred0(4);
        planner->propagate_steer_acc(xh_meas_, xh_goal_, xh_pred0, 0.5);
        planner->propagate_steer_acc(xh_pred0, xh_goal_, xh_pred, 0.3);

        // set planning initial condition with the predicted states
        planner->set_robot_state(xr_pred, ur_meas_);
        planner->set_human_state(xh_pred);
//        planner->set_robot_state(xr_meas_, ur_meas_);
//        planner->set_human_state(xh_meas_);
    }
    else {
        // update planner measurements
        planner->set_robot_state(xr_meas_, ur_meas_);
        planner->set_human_state(xh_meas_);
    }

    // set the time limit for the optimizer to be 85% of the desired planner rate
    double t_max_planning = 0.85 * (1.0 / planning_rate_);

    // compute plan
    using namespace std::chrono;
    steady_clock::time_point t1 = steady_clock::now();

    if (flag_allow_explicit_comm_ || flag_human_tracking_lost_) {
        planner->compute_plan(t_max_planning);
    }
    else {
        dynamic_cast<hri_planner::Planner*>(planner.get())->compute_plan_no_comm(t_max_planning);
    }

    steady_clock::time_point t2 = steady_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "time spent for planning is: " << time_span.count() << "s" << std::endl;

    // publish plan
    planner->publish_plan(flag_human_detected_frame_);

    // publish the real measurement
    std_msgs::Float64MultiArray state_data;
    utils::EigenToVector(xr_meas_, state_data.data);
    state_data.data.insert(state_data.data.end(), xh_meas_.data(), xh_meas_.data() + xh_meas_.size());

    robot_human_state_pub_.publish(state_data);
}

//----------------------------------------------------------------------------------
void PlannerNode::compute_and_publish_control()
{
    Eigen::VectorXd ur(ur_meas_.size());
    planner_interactive_->compute_steer_posq(xr_meas_, xr_goal_, ur);

    // publish
    geometry_msgs::Twist cmd_vel;
    cmd_vel.linear.x = ur(0);
    cmd_vel.angular.z = ur(1);

    robot_ctrl_pub_.publish(cmd_vel);
}

//----------------------------------------------------------------------------------
void PlannerNode::reset_state_machine()
{
    flag_start_planning_ = false;
    flag_pause_planning_ = true;
    flag_stop_planning_ = false;
    flag_human_detected_ = false;

    flag_human_tracking_lost_ = true;
    human_tracking_lost_frames_ = 0;

    t_meas_last_ = -1;
}

//----------------------------------------------------------------------------------
void PlannerNode::goal_callback(const std_msgs::Float64MultiArrayConstPtr& goal_msg)
{
    xr_goal_(0) = goal_msg->data[0];
    xr_goal_(1) = goal_msg->data[1];
    xh_goal_(0) = goal_msg->data[3];
    xh_goal_(1) = goal_msg->data[4];
    xh_init_(0) = goal_msg->data[5];
    xh_init_(1) = goal_msg->data[6];


    // the last value of goal is intent
    intent_ = static_cast<int>(goal_msg->data[7]);

    ROS_INFO("Received new goal, reset planner...");

    // reset the planners
    std::string ns;
    if (intent_ == hri_planner::HumanPriority)
        ns = "hp/";
    else
        ns = "rp/";

    planner_interactive_->reset_planner(xr_goal_, xh_goal_, intent_, ns);
    planner_simple_->reset_planner(xr_goal_, xh_goal_, intent_, ns);
}

//----------------------------------------------------------------------------------
void PlannerNode::planner_ctrl_callback(const std_msgs::StringConstPtr& msg)
{
    const std::string& ctrl = msg->data;

    if (ctrl == "pause") {
        flag_pause_planning_ = true;
    }
    else if (ctrl == "resume") {
        flag_pause_planning_ = false;
    }
    else if (ctrl == "stop") {
        flag_stop_planning_ = true;
    }
    else if (ctrl == "start") {
        flag_start_planning_ = true;
    }
}

//----------------------------------------------------------------------------------
void PlannerNode::robot_state_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr &pose_msg)
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
void PlannerNode::robot_odom_callback(const nav_msgs::OdometryConstPtr &odom_msg)
{
    ur_meas_(0) = odom_msg->twist.twist.linear.x;
    ur_meas_(1) = odom_msg->twist.twist.angular.z;
}

//----------------------------------------------------------------------------------
void PlannerNode::human_tracking_callback(const people_msgs::PeopleConstPtr &people_msg)
{
    double min_dist = human_filter_dist_th_;
    int person_id = -1;

    // filter out detections that are too far away from "desired path"
    for (int i = 0; i < people_msg->people.size(); ++i) {
        Eigen::VectorXd pos(goal_dim_);
        pos << people_msg->people[i].position.x, people_msg->people[i].position.y;

        double dist = point_line_dist(pos, xh_init_, xh_goal_);
//        std::cout << dist << ", ";
        if (dist < min_dist) {
            min_dist = dist;
            person_id = i;
        }
    }
//    std::cout << std::endl;

    if (person_id == -1) {
        flag_human_detected_ = false;
    }
    else {
        flag_human_detected_ = true;
        xh_meas_(0) = people_msg->people[person_id].position.x;
        xh_meas_(1) = people_msg->people[person_id].position.y;
        xh_meas_(2) = people_msg->people[person_id].velocity.x;
        xh_meas_(3) = people_msg->people[person_id].velocity.y;
    }
}

//----------------------------------------------------------------------------------
void PlannerNode::human_detection_callback(const people_msgs::PositionMeasurementArrayConstPtr &pos_arr_msg)
{
    double min_dist = human_filter_dist_th_;
    int person_id = -1;

    // filter out detections that are too far away from "desired path"
    for (int i = 0; i < pos_arr_msg->people.size(); ++i) {
        Eigen::VectorXd pos(goal_dim_);
        pos << pos_arr_msg->people[i].pos.x, pos_arr_msg->people[i].pos.y;

        double dist = point_line_dist(pos, xh_init_, xh_goal_);
        if (dist < min_dist) {
            min_dist = dist;
            person_id = i;
        }
    }

    if (person_id == -1) {
        flag_human_detected_ = false;
    }
    else {
        flag_human_detected_ = true;

        // if first detection
        if (t_meas_last_ < 0) {
            xh_meas_(2) = 0.0;
            xh_meas_(3) = 0.0;
        }
        else {
            double dt = pos_arr_msg->header.stamp.toSec() - t_meas_last_;
            xh_meas_(2) = (pos_arr_msg->people[person_id].pos.x - xh_meas_(0)) / dt;
            xh_meas_(3) = (pos_arr_msg->people[person_id].pos.y - xh_meas_(1)) / dt;
        }
        xh_meas_(0) = pos_arr_msg->people[person_id].pos.x;
        xh_meas_(1) = pos_arr_msg->people[person_id].pos.y;

        t_meas_last_ = pos_arr_msg->header.stamp.toSec();

        std::cout << "measurement at t = " << t_meas_last_ << " is: " << xh_meas_.transpose() << std::endl;
    }
}

//----------------------------------------------------------------------------------
double PlannerNode::point_line_dist(const Eigen::VectorXd &p, const Eigen::VectorXd &a, const Eigen::VectorXd &b)
{
    Eigen::VectorXd n = (b - a).normalized();
    Eigen::VectorXd ap = p - a;
    Eigen::VectorXd pt = a + n.dot(ap) * n;

//    std::cout << "p: " << p.transpose() << ", a: " << a.transpose() << ", b: " << b.transpose() << std::endl;

    if ((a - pt).dot(b - pt) < 0) {
        return (p - pt).norm();
    }
    else {
        return std::min(ap.norm(), (p - b).norm());
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "hri_planner");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // create the planner and run
    PlannerNode planner_node(nh, pnh);
    planner_node.run();

    return 0;
}