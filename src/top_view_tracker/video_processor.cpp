//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 12/14/2017
// Last revision: 01/09/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include <sstream>

#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include "top_view_tracker/video_processor.h"

namespace tracking {

//----------------------------------------------------------------------------------
VideoProcessor::VideoProcessor(ros::NodeHandle &nh, ros::NodeHandle &pnh): nh_(nh)
{
    // get ros parameters
    std::string dict;
    std::string camera_info_file;
    std::string hat_tracker_config_file;

    // get parameters
    ros::param::param<std::string>("~dictionary", dict, "ARUCO");
    ros::param::param<std::string>("~camera_info_file", camera_info_file, "gopro.yml");
    ros::param::param<std::string>("~hat_tracker_config", hat_tracker_config_file, "config.json");

    ros::param::param<int>("~marker_id_robot", marker_id_robot_, 11);
    ros::param::param<float>("~marker_size", marker_size_, 0.19);
    ros::param::param<double>("~frame_rate", fps_, 60);

    // initialize the trackers
    human_tracker_.load_config(hat_tracker_config_file);

    robot_tracker_.setDictionary(dict);

    cam_param_.readFromXMLFile(camera_info_file);
}

//----------------------------------------------------------------------------------
bool VideoProcessor::extrensic_calibration(std::string &figure_path)
{
    // load calibration parameters
    int n_markers;
    std::map<int, cv::Point3d> marker_pos_world;

    ros::param::param<int>("~num_calib_markers", n_markers, 4);
    for (int i = 0; i < n_markers; i++) {
        std::stringstream ss;
        ss << "~calib_marker" << i;

        int id;
        cv::Point3d pos;
        ros::param::param<int>(ss.str() + "/id", id, i);
        ros::param::param<double>(ss.str() + "/x", pos.x, 0.0);
        ros::param::param<double>(ss.str() + "/y", pos.y, 0.0);
        pos.z = 0.0;

        marker_pos_world.insert({id, pos});

        ROS_INFO("Loaded marker %d, position is (%f, %f)", id, pos.x, pos.y);
    }

    // read in image
    cv::Mat im = cv::imread(figure_path);

    // detect the markers
    std::vector<aruco::Marker> markers = robot_tracker_.detect(im, cam_param_, marker_size_);

    // match with calibration
    std::map<int, cv::Point2d> marker_pos_im;
    for (auto &marker: markers) {
        if (marker_pos_world.count(marker.id)) {
            marker_pos_im.insert({marker.id, marker.getCenter()});
        }
    }

    // need all markers to be detected
    if (marker_pos_im.size() != marker_pos_world.size()) {
        ROS_ERROR("Not all calibration markers were detected!");
        return false;
    }

    // convert map to vector
    std::vector<cv::Point3d> pos_world;
    std::vector<cv::Point2d> pos_im;

    for (auto &it : marker_pos_world)
        pos_world.push_back(it.second);

    for (auto &it : marker_pos_im)
        pos_im.push_back(it.second);

    // all markers detected, then solve for camera pose using PNP solver
    cv::solvePnP(pos_world, pos_im, cam_param_.CameraMatrix, cam_param_.Distorsion, cam_rvec_, cam_tvec_);

    // convert to rotation matrix
    cv::Rodrigues(cam_rvec_, cam_rmat_);

    return true;
}

//----------------------------------------------------------------------------------
void VideoProcessor::process(std::string &video_path, std::string &save_path)
{
    // open video
    cv::VideoCapture cap(video_path);

    if (!cap.isOpened()) {
        ROS_ERROR("Cannot open video file!");
        return;
    }

    // save result to file
    std::ofstream res(save_path + "trajectories.txt");

    if (!res.is_open()) {
        ROS_ERROR("Cannot open text file to save result!");
        return;
    }

    // process the entire video
    cv::Mat frame;
    for (;;) {
        cap >> frame;

        if (frame.empty())
            break;

        // track the humans/hats
        human_tracker_.track(frame, true);

        // get result
        std::vector<cv::Mat> poses;
        std::vector<cv::Mat> vels;
        std::vector<int> ids;

        human_tracker_.get_tracking(poses, vels, ids);

        // convert to world frame
        std::map<int, cv::Mat> human_poses;
        for (int i = 0; i < poses.size(); i++) {
            cv::Mat pose_world;
            calculate_pose_world(poses[i], human_heights_[ids[i]], pose_world);

            human_poses.insert({ids[i], pose_world});
        }

        // TODO: report error if number of tracked human is wrong?

        // track robot
        
    }
}

//----------------------------------------------------------------------------------
void VideoProcessor::process_all(std::string &path)
{

}

//----------------------------------------------------------------------------------
void VideoProcessor::calculate_pose_world(const cv::Mat &pose_im, const double z0, cv::Mat &pose_world)
{
    // first undistort the image coordinate
    cv::Mat pos_undistort(1, 2, CV_64F);
    cv::undistortPoints(pose_im.colRange(0, 2), pos_undistort, cam_param_.CameraMatrix, cam_param_.Distorsion);

    // obtain the truncated transformation matrix [R|t]
    cv::Mat T(3, 4, CV_64F);
    T.rowRange(0, 3).colRange(0, 3) = cam_rmat_;
    T.col(3) = cam_tvec_.t();

    // form the linear equation to solve for (x, y) in world frame
    cv::Mat M(2, 4, CV_64F);
    M.row(0) = T.row(0) - pos_undistort.at<double>(0) * T.row(2);
    M.row(1) = T.row(1) - pos_undistort.at<double>(1) * T.row(2);

    cv::Mat M1 = M.colRange(0, 2);
    cv::Mat M2 = M.colRange(2, 4);

    cv::Mat b(2, 1, CV_64F);
    b.at<double>(0) = z0;
    b.at<double>(1) = 1.0;

    // calculate position
    cv::Mat pos = -M1.inv() * M2 * b;

    // calculate orientation
    cv::Mat rvec(3, 1, CV_64F);
    rvec.at<double>(0) = std::cos(pose_im.at<double>(2));
    rvec.at<double>(1) = std::sin(pose_im.at<double>(2));
    rvec.at<double>(2) = 0.0;

    cv::Mat rvec_world = cam_rmat_ * rvec;
    const double th_world = std::atan2(rvec_world.at<double>(1), rvec_world.at<double>(0));

    // write to result
    pose_world = cv::Mat(1, 3, CV_64F);
    pose_world.at<double>(0) = pos.at<double>(0);
    pose_world.at<double>(1) = pos.at<double>(1);
    pose_world.at<double>(2) = th_world;
}

} // namespace


void test_hat_tracker(const std::string &test_path)
{
    // create a hat tracker object
    tracking::HatTracker tracker;

    // load configuration file
    std::stringstream ss;
    ss << test_path << "/test_config.json";

    tracker.load_config(ss.str());

    // load the video file
    ss.str("");
    ss << test_path << "/test_video.mp4";

    cv::VideoCapture cap(ss.str());
    if (!cap.isOpened()) {
        std::cout << "Video not opened!!!!!" << std::endl;
        return;
    }

    cv::Mat frame;
    int counter = 0;
    for (;;) {
        // get video frame
        cap >> frame;

        if (frame.empty())
            break;

        counter++;
        tracker.track(frame);

        //quit on ESC button
        if(cv::waitKey(1)==27)break;
    }
}

void save_one_frame(const std::string &test_path)
{
    // load the video file
    std::stringstream ss;
    ss.clear();
    ss << test_path << "/test_video.mp4";

    cv::VideoCapture cap(ss.str());
    if (!cap.isOpened()) {
        std::cout << "Video not opened!!!!!" << std::endl;
        return;
    }

    // load the first frame
    cv::Mat frame;
    cap >> frame;

    cv::imshow("test", frame);
    cv::waitKey(0);

    // write to file
    ss.str("");
    ss << test_path << "/first_frame.jpg";
    std::cout << ss.str() << std::endl;
    try {
        cv::imwrite(ss.str(), frame);
    }
    catch (std::runtime_error& ex) {
        fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
        return;
    }
}

int main(int argc, char** argv)
{
    std::string test_path = argv[1];
    test_hat_tracker(test_path);
//    save_one_frame(test_path);

    return 0;
}
