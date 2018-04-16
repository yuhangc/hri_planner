//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 4/9/2018
// Last revision: 4/15/2018
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include "hri_planner/controller_node.h"
#include "utils/utils.h"

//----------------------------------------------------------------------------------
ControllerNode::ControllerNode(ros::NodeHandle &nh, ros::NodeHandle &pnh): nh_(nh)
{
    // parameters for initializing robot trajectory
    ros::param::param<double>("~k_rho", k_rho_, 0.5);
    ros::param::param<double>("~k_v", k_v_, 2.0);
    ros::param::param<double>("~k_alp", k_alp_, 3.0);
    ros::param::param<double>("~k_phi", k_phi_, -0.5);

    ros::param::param<double>("~controller_rate", controller_rate_, 20);

    ros::param::param<double>("~goal_reaching_th_controller", goal_reaching_th_controller_, 0.15);

    // control bounds
    ros::param::param<double>("~v_max", v_max_, 0.5);
    ros::param::param<double>("~om_max", om_max_, 2.0);

    // set the vector sizes
    int nXr = 3;
    xr_.setZero(nXr);
    x_goal_.setZero(nXr);

    // subscribers and publishers
    goal_sub_ = nh.subscribe<std_msgs::Float64MultiArray>("/controller/set_goal", 1,
                                                          &ControllerNode::goal_callback, this);
//    robot_state_sub_ = nh_.subscribe<geometry_msgs::PoseWithCovarianceStamped>("/amcl_pose", 1,
//                                                                 &ControllerNode::robot_state_callback, this);
    start_controller_sub_ = nh_.subscribe<std_msgs::Bool>("controller/start_controller", 1,
                                                          &ControllerNode::start_controller_callback, this);

    goal_reached_pub_ = nh_.advertise<std_msgs::Bool>("/controller/goal_reached", 1);
    robot_ctrl_pub_ = nh_.advertise<geometry_msgs::Twist>("/controller/cmd_vel", 1);
}

//----------------------------------------------------------------------------------
void ControllerNode::run()
{
    flag_start_controller_ = false;

    ControllerState state = Idle;

    ros::Rate rate_controller(controller_rate_);
    while (!ros::isShuttingDown()) {
        ros::spinOnce();

        switch (state) {
            case Idle:
                // check for start flag
                if (flag_start_controller_) {
                    flag_start_controller_ = false;
                    state = Running;
                    ROS_INFO("Controller started to run...");
                }

                break;
            case Running:
                // update pose
                tf::StampedTransform transform;
                try {
                    tf_listener_.lookupTransform("/map", "/base_footprint", ros::Time(0), transform);
                }
                catch (tf::TransformException &ex) {
                    ROS_ERROR("%s", ex.what());
                    ros::Duration(0.01).sleep();
                    continue;
                }

                auto pos = transform.getOrigin();
                xr_(0) = pos.x();
                xr_(1) = pos.y();

                auto q = transform.getRotation();
                double siny = 2.0 * (q.w() * q.z() + q.x() * q.y());
                double cosy = 1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());
                xr_(2) = std::atan2(siny, cosy);

                // compute control
                compute_and_publish_control();

                // check for goal reaching
                Eigen::VectorXd x_diff = x_goal_ - xr_;
                x_diff(2) = utils::wrap_to_pi(x_diff(2));
                if (x_diff.norm() < goal_reaching_th_controller_) {
                    // send a zero velocity command
                    geometry_msgs::Twist ur;
                    robot_ctrl_pub_.publish(ur);

                    // publish to tell that goal is reached
                    std_msgs::Bool goal_reach_data;
                    goal_reach_data.data = static_cast<uint8_t>(true);
                    goal_reached_pub_.publish(goal_reach_data);

                    state = Idle;
                    ROS_INFO("Goal reached! Now switching back to Idle...");
                }

                break;
        }

        rate_controller.sleep();
    }
}

//----------------------------------------------------------------------------------
void ControllerNode::compute_and_publish_control()
{
    double rho = (x_goal_.head(2) - xr_.head(2)).norm();
    double phi = utils::wrap_to_pi(x_goal_(2) - xr_(2));
    double th_z = std::atan2(x_goal_(1) - xr_(1), x_goal_(0) - xr_(0));
    double alpha = utils::wrap_to_pi(th_z - xr_(2));
    double k_phi_rho = std::min(1.0, rho * 5.0);

    std::cout << "robot pose is: " << xr_.transpose() << std::endl;
    std::cout << "rho: " << rho << ", phi: " << phi << ", th: " << th_z << ", alpha: " << alpha << std::endl;

    geometry_msgs::Twist ur;
    ur.linear.x = utils::clamp(k_rho_ * std::tanh(k_v_ * rho), -v_max_, v_max_);
    ur.angular.z = utils::clamp(k_alp_ * alpha + k_phi_ * k_phi_rho * phi, -om_max_, om_max_);

    std::cout << "control is: " << ur.linear.x << ", " << ur.angular.z << std::endl;

    // publish
    robot_ctrl_pub_.publish(ur);
}

//----------------------------------------------------------------------------------
void ControllerNode::goal_callback(const std_msgs::Float64MultiArrayConstPtr &goal_msg)
{
    x_goal_(0) = goal_msg->data[0];
    x_goal_(1) = goal_msg->data[1];
    x_goal_(2) = goal_msg->data[2];
}

//----------------------------------------------------------------------------------
void ControllerNode::robot_state_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr &pose_msg)
{
    xr_(0) = pose_msg->pose.pose.position.x;
    xr_(1) = pose_msg->pose.pose.position.y;

    // find rotation
    auto &q = pose_msg->pose.pose.orientation;
    double siny = 2.0 * (q.w * q.z + q.x * q.y);
    double cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);

    xr_(2) = std::atan2(siny, cosy);
}

//----------------------------------------------------------------------------------
void ControllerNode::start_controller_callback(const std_msgs::BoolConstPtr &msg)
{
    ROS_INFO("received message!");
    flag_start_controller_ = msg->data;
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "hri_controller");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // create the planner and run
    ControllerNode controller(nh, pnh);
    controller.run();

    return 0;
}