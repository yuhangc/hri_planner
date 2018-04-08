//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 3/24/2018
// Last revision: 4/6/2018
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include "hri_planner/planner_node.h"

//----------------------------------------------------------------------------------
PlannerNode::PlannerNode(ros::NodeHandle &nh, ros::NodeHandle &pnh): nh_(nh)
{
    // get parameters
    ros::param::param<double>("~planner/planner_rate", planning_rate_, 2.0);
    ros::param::param<double>("~planner/controller_rate", controller_rate_, 10);
    ros::param::param<double>("~planner/state_machine_rate", state_machine_rate_, 1000);
    ros::param::param<std::string>("~planner/planner_mode", mode_, "simulation");
    ros::param::param<double>("~planner/goal_reaching_th_planner", goal_reaching_th_planner_, 0.5);
    ros::param::param<double>("~planner/goal_reaching_th_controller", goal_reaching_th_controller_, 0.1);

    ros::param::param<double>("~human_filter/dist_threshold", human_filter_dist_th_, 1.0);

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

    ROS_INFO("Received new goal, reset planner...");

    xr_goal_.resize(goal_dim_);
    xr_goal_ << 0.0, 3.0;
    xh_goal_.resize(goal_dim_);
    xh_init_.resize(goal_dim_);

    // create subscribers
    goal_sub_ = nh.subscribe<std_msgs::Float64MultiArray>("/planner/set_goal", 1,
                                                          &PlannerNode::goal_callback, this);
    planner_ctrl_sub_ = nh.subscribe<std_msgs::String>("/planner/ctrl", 1,
                                                       &PlannerNode::planner_ctrl_callback, this);

    robot_state_sub_ = nh_.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/amcl_pose", 1,
                                                            &PlannerNode::robot_state_callback, this);

    robot_odom_sub_ = nh_.subscribe<nav_msgs::Odometry>("/odom", 1, &PlannerNode::robot_odom_callback, this);

    human_tracking_sub_ = nh_.subscribe<people_msgs::People>("/people", 1,
                                                             &PlannerNode::human_tracking_callback, this);

    // publishers
    goal_reached_pub_ = nh_.advertise<std_msgs::Bool>("/planner/goal_reached", 1);
    robot_ctrl_pub_ = nh_.advertise<geometry_msgs::Twist>("/planner/cmd_vel", 1);
}

//----------------------------------------------------------------------------------
void PlannerNode::run()
{
    // loop and a simple state machine
    PlannerStates planner_state = Idle;
    flag_start_planning_ = false;
    flag_pause_planning_ = true;
    flag_stop_planning_ = false;
    flag_human_detected_ = false;

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
                planner_interactive_->reset_planner(xr_goal_, xh_goal_, intent_);
                planner_simple_->reset_planner(xr_goal_, xh_goal_, intent_);

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

                    // check if human is detected
                    if (flag_human_detected_) {
                        ROS_INFO("Using interactive planner...");
                        // FIXME: reset the simple planner?
                        plan(planner_interactive_);
                        flag_human_detected_ = false;
                        planner_simple_->reset_planner(xr_goal_, xh_goal_, intent_);
                    }
                    else {
                        ROS_INFO("Using simple planner since no human detected...");
                        // FIXME: reset the interactive planner?
                        plan(planner_simple_);
                        planner_interactive_->reset_planner(xr_goal_, xh_goal_, intent_);
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

                            break;
                        }

                        Eigen::VectorXd x_diff = xr_goal_ - xr_meas_.head(2);
                        if (x_diff.norm() < goal_reaching_th_planner_) {
                            planner_state = GoalReaching;
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
    // update planner measurements
    planner->set_robot_state(xr_meas_, ur_meas_);
    planner->set_human_state(xh_meas_);

    // compute plan
    auto t_s = ros::Time::now();
    planner->compute_plan();
    ros::Duration t_plan = ros::Time::now() - t_s;
    std::cout << "time spent for planning is: " << t_plan.toSec() << "s" << std::endl;

    // publish plan
    planner->publish_plan();
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
void PlannerNode::goal_callback(const std_msgs::Float64MultiArrayConstPtr& goal_msg)
{
    for (int i = 0; i < goal_dim_; ++i) {
        xr_goal_(i) = goal_msg->data[i];
        xh_goal_(i) = goal_msg->data[i+goal_dim_];
        xh_init_(i) = goal_msg->data[i+goal_dim_*2];
    }

    // the last value of goal is intent
    intent_ = static_cast<int>(goal_msg->data[goal_dim_*3]);
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
        xh_meas_(0) = people_msg->people[person_id].position.x;
        xh_meas_(1) = people_msg->people[person_id].position.y;
        xh_meas_(2) = people_msg->people[person_id].velocity.x;
        xh_meas_(3) = people_msg->people[person_id].velocity.y;
    }
}

//----------------------------------------------------------------------------------
double PlannerNode::point_line_dist(const Eigen::VectorXd &p, const Eigen::VectorXd &a, const Eigen::VectorXd &b)
{
    Eigen::VectorXd n = (b - a).normalized();
    Eigen::VectorXd ap = p - a;
    Eigen::VectorXd pt = a + n.dot(ap) * n;

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