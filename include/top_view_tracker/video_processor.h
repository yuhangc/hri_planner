//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 12/14/2017
// Last revision: 01/30/2018
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#ifndef VIDEO_PROCESSOR_H
#define VIDEO_PROCESSOR_H

#include <fstream>
#include <iostream>
#include <vector>
#include <map>

#include "ros/ros.h"

#include "aruco/aruco.h"

#include "top_view_tracker/hat_tracker.h"

namespace tracking {

class VideoProcessor {
public:
    // constructor
    VideoProcessor(ros::NodeHandle &nh, ros::NodeHandle &pnh);

    // perform calibration
    bool extrensic_calibration(std::string &figure_path);

    // process video
    void process(std::string &video_path, std::string &save_path);

    // bulk process
    void process_all(std::string &path);

private:
    // node handler
    ros::NodeHandle nh_;

    // trackers
    tracking::HatTracker human_tracker_;
    aruco::MarkerDetector robot_tracker_;

    // tracking parameters
    float marker_size_;
    int marker_id_robot_;

    double fps_;

    // camera parameters
    aruco::CameraParameters cam_param_;
    cv::Mat cam_rvec_;
    cv::Mat cam_rmat_;
    cv::Mat cam_tvec_;

    // human heights
    std::map<int, double> human_heights_;

    // human and robot poses/velocities
    std::vector<cv::Mat> human_pose_;
    std::vector<cv::Mat> human_vel_;
    cv::Mat robot_pose_;
    cv::Mat robot_vel_;

    // use a simple Kalman filter to handle occasional lost of robot tracking
    cv::Ptr<cv::KalmanFilter> robot_pose_filter_;
    bool flag_filter_initialized_;

    // helper functions
    void calculate_pose_world(const cv::Mat &pose_im, const double z0, cv::Mat &pose_world);
};

} // namespace

#endif //VIDEO_PROCESSOR_H
