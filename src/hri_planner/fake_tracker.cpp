//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 11/20/2018
// Last revision: 11/27/2018
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include <vector>
#include <string>
#include <sstream>

#include "ros/ros.h"
#include "std_msgs/Float64MultiArray.h"
#include "gazebo_msgs/ModelStates.h"

#include "Eigen/Dense"

#include "hri_planner/TrackedHumans.h"

class FakeTracker {
public:
    // constructor
    FakeTracker(ros::NodeHandle &nh, ros::NodeHandle &pnh) {
        // publisher and subscriber
        model_state_sub_ = nh_.subscribe<gazebo_msgs::ModelStates>("/gazebo/model_states", 1,
                                                                   &FakeTracker::model_state_callback,
                                                                   this);
        robot_pose_vel_pub_ = nh_.advertise<std_msgs::Float64MultiArray>("localization/robot_pose_vel", 1);
        human_pose_vel_pub_ = nh_.advertise<hri_planner::TrackedHumans>("tracking/tracked_humans", 1);

        // get parameters
        pnh.param<double>("dist_detection", dist_detection, 5.0);
        pnh.param<double>("ang_detection", ang_detection, 2.61799);
    }

    // fake detection and localization update
    void update() {
        // publish robot state
        std_msgs::Float64MultiArray robot_pose_vel;
        robot_pose_vel.data.push_back(pose_robot_(0));
        robot_pose_vel.data.push_back(pose_robot_(1));
        robot_pose_vel.data.push_back(pose_robot_(2));
        robot_pose_vel.data.push_back(vel_robot_(0));
        robot_pose_vel.data.push_back(vel_robot_(1));

        robot_pose_vel_pub_.publish(robot_pose_vel);

        // find human within detection range
        hri_planner::TrackedHumans tracked_humans;
        for (int i = 0; i < id_human_.size(); i++) {
            const Eigen::Vector2d pose_diff = pose_human_[i].head(2) - pose_robot_.head(2);

            // check distance
            const double dist = pose_diff.norm();
            if (dist > dist_detection)
                continue;

            // check angle
            const double ang_rel = std::atan2(pose_diff(1), pose_diff(0));
            if (ang_rel < -ang_detection || ang_rel > ang_detection)
                continue;

            // add for publish
            hri_planner::HumanStat pose_vel;
            pose_vel.id = id_human_[i];
            pose_vel.pose_vel.push_back(pose_human_[i](0));
            pose_vel.pose_vel.push_back(pose_human_[i](1));
            pose_vel.pose_vel.push_back(pose_human_[i](2));
            pose_vel.pose_vel.push_back(vel_human_[i](0));
            pose_vel.pose_vel.push_back(vel_human_[i](1));
            pose_vel.pose_vel.push_back(vel_human_[i](2));

            tracked_humans.tracking_data.push_back(pose_vel);
        }

        human_pose_vel_pub_.publish(tracked_humans);
    }

private:
    ros::NodeHandle nh_;

    // subscriber and publisher
    ros::Subscriber model_state_sub_;
    ros::Publisher robot_pose_vel_pub_;
    ros::Publisher human_pose_vel_pub_;

    Eigen::Vector3d pose_robot_;
    Eigen::Vector2d vel_robot_;

    std::vector<short> id_human_;
    std::vector<Eigen::Vector3d> pose_human_;
    std::vector<Eigen::Vector3d> vel_human_;

    // fake detection parameters
    double dist_detection;
    double ang_detection;

    // callback function
    void model_state_callback(const gazebo_msgs::ModelStatesConstPtr &states_msg) {
        pose_human_.clear();
        vel_human_.clear();

        // loop through all models
        for (int i = 0; i < states_msg->name.size(); i++) {
            if (states_msg->name[i] == "turtlebot") {
                double th = std::atan2(states_msg->pose[i].orientation.y,
                                       states_msg->pose[i].orientation.z) * 2.0;
                pose_robot_ << states_msg->pose[i].position.x,
                               states_msg->pose[i].position.y,
                               th;

                vel_robot_ << states_msg->twist[i].linear.x,
                              states_msg->twist[i].angular.z;
            } else {
                const std::string &model_name = states_msg->name[i];
                if (model_name.find("human")) {
                    double th = std::atan2(states_msg->pose[i].orientation.y,
                                           states_msg->pose[i].orientation.z) * 2.0;

                    Eigen::Vector3d pose(states_msg->pose[i].position.x,
                                         states_msg->pose[i].position.y,
                                         th);
                    pose_human_.push_back(pose);

                    Eigen::Vector3d vel(states_msg->twist[i].linear.x,
                                        states_msg->twist[i].linear.y,
                                        states_msg->twist[i].angular.z);
                    vel_human_.push_back(vel);

                    short id;
                    std::stringstream ss(model_name.substr(5, model_name.length()-5));
                    ss >> id;
                    id_human_.push_back(id);
                }
            }
        }
    }
};

//----------------------------------------------------------------------------------
int main(int argc, char** argv)
{
    ros::init(argc, argv, "fake_tracker");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    FakeTracker fake_tracker(nh, pnh);

    const double sim_rate = 20;
    ros::Rate rate(sim_rate);
    while (!ros::isShuttingDown()) {
        ros::spinOnce();

        fake_tracker.update();

        rate.sleep();
    }
}