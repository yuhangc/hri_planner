//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 12/14/2017
// Last revision: 01/31/2017
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

    // load human heights
    int n_human;
    ros::param::param<int>("~num_human", n_human, 2);
    for (int i = 0; i < n_human; i++) {
        std::stringstream ss;
        ss << "~human" << i;

        double height;
        ros::param::param<double>(ss.str(), height, 1.70);
        human_heights_.insert({i, height});

        ROS_INFO("Loaded human height %d: %f", i, height);
    }

    // initialize the trackers
    human_tracker_.load_config(hat_tracker_config_file);
    human_tracker_.set_disp_scale(0.5);

    robot_tracker_.setDictionary(dict);

    cam_param_.readFromXMLFile(camera_info_file);

    // initialize the Kalman filter for robot pose tracking
    int state_size = 6;
    int meas_size = 3;
    int ctrl_size = 0;

    robot_pose_filter_ = new cv::KalmanFilter(state_size, meas_size, ctrl_size, CV_64F);

    // set transition matrix
    cv::setIdentity(robot_pose_filter_->transitionMatrix);
    robot_pose_filter_->transitionMatrix.at<double>(0, 3) = 1.0 / fps_;
    robot_pose_filter_->transitionMatrix.at<double>(1, 4) = 1.0 / fps_;
    robot_pose_filter_->transitionMatrix.at<double>(2, 5) = 1.0 / fps_;

    // set measurement matrix
    robot_pose_filter_->measurementMatrix = cv::Mat::zeros(meas_size, state_size, CV_64F);
    robot_pose_filter_->measurementMatrix.at<double>(0, 0) = 1.0;
    robot_pose_filter_->measurementMatrix.at<double>(1, 1) = 1.0;
    robot_pose_filter_->measurementMatrix.at<double>(2, 2) = 1.0;

    // process noise covariance matrix
    cv::setIdentity(robot_pose_filter_->processNoiseCov.rowRange(0, 2).colRange(0, 2), 1e-6);
    cv::setIdentity(robot_pose_filter_->processNoiseCov.rowRange(3, 5).colRange(3, 5), 1e-2);
    robot_pose_filter_->processNoiseCov.at<double>(2, 2) = 1e-6;
    robot_pose_filter_->processNoiseCov.at<double>(5, 5) = 1e-2;

    //! use fixed measurement covariance matrix for now
    cv::setIdentity(robot_pose_filter_->measurementNoiseCov, 1e-4);
    flag_filter_initialized_ = false;
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
            ROS_INFO("Marker %d center position: (%f, %f)",
                     marker.id, marker.getCenter().x, marker.getCenter().y);
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

    // solvePnP gives us world->cam, we want cam->world
    cam_rmat_ = cam_rmat_.t();
    cam_tvec_ = -cam_rmat_ * cam_tvec_;
    cam_rvec_ = -cam_rvec_;

    // for debugging
    std::cout << "camera 3D position: " << std::endl << cam_tvec_ << std::endl;
    std::cout << "camera 3D rotation: " << std::endl;
    std::cout << cam_rmat_ << std::endl;

    return true;
}

//----------------------------------------------------------------------------------
void VideoProcessor::process(std::string &video_path, std::string &save_path)
{
    // open video
    ROS_INFO("Openning video %s", video_path.c_str());
    cv::VideoCapture cap(video_path);

    if (!cap.isOpened()) {
        ROS_ERROR("Cannot open video file!");
        return;
    }

    // save result to file
    std::ofstream res(save_path + "/trajectories.txt");

    if (!res.is_open()) {
        ROS_ERROR("Cannot open text file to save result!");
        return;
    }

    // process the entire video
    cv::Mat frame;
    const double dt = 1.0 / fps_;
    double tstamp = 0.0;
    int counter = 0;

//    for (;;) {
    for (int k = 0; k < 5000; k++) {
        cap >> frame;

        if (frame.empty())
            break;

        if (counter % 10 == 0)
            ROS_INFO("Processing frame %d...", counter);

        // track the humans/hats
        human_tracker_.track(frame, false);

        // get result
        std::vector<cv::Mat> poses;
        std::vector<cv::Mat> vels;
        std::vector<int> ids;

        human_tracker_.get_tracking(poses, vels, ids);

        // convert to world frame
        std::map<int, cv::Mat> human_poses;
        for (int i = 0; i < poses.size(); i++) {
//            cv::Mat pose_world;
//            calculate_pose_world(poses[i], human_heights_[ids[i]], pose_world);
//
//            human_poses.insert({ids[i], pose_world});

            cv::Mat pose_vel_world;
            calculate_pose_vel_world(poses[i], vels[i], human_heights_[ids[i]], pose_vel_world);
            human_poses.insert({ids[i], pose_vel_world});

//            ROS_INFO("Detected pose for human %d: (%f, %f, %f)", ids[i],
//                     pose_world.at<double>(0), pose_world.at<double>(1), pose_world.at<double>(2));
        }

        // TODO: report error if number of tracked human is wrong?

        // track robot
        // detect markers
        std::vector<aruco::Marker> markers = robot_tracker_.detect(frame, cam_param_, marker_size_);

        // find the robot marker
        cv::Mat rvec;
        cv::Mat tvec;
        cv::Point2f marker_center;
        bool flag_robot_detected = false;
        for (auto &marker : markers) {
            if (marker.id == marker_id_robot_) {
                rvec = marker.Rvec;
                tvec = marker.Tvec;
                marker_center = marker.getCenter();

                flag_robot_detected = true;
                break;
            }
        }

        // convert to 2D pose
        cv::Mat pose_meas(3, 1, CV_64F);
        if (!flag_robot_detected) {
            // simply do nothing
            ROS_WARN("Robot not detected in current frame!");
            pose_meas.at<double>(0) = -1;
            pose_meas.at<double>(1) = -1;
            pose_meas.at<double>(2) = -1;
        } else {
            // convert type
            tvec.convertTo(tvec, CV_64F);
            rvec.convertTo(rvec, CV_64F);

            // transform to world coordinate
            cv::Mat rmat;
            cv::Rodrigues(rvec, rmat);
//
//            cv::Mat t_world = cam_rmat_ * tvec + cam_tvec_;
            cv::Mat r_world = cam_rmat_ * rmat;

            // transform to world coordinate assuming a fixed height
            cv::Mat pose_im(3, 1, CV_64F);
            cv::Mat t_world;

            pose_im.at<double>(0) = marker_center.x;
            pose_im.at<double>(1) = marker_center.y;

            calculate_pose_world(pose_im, 0.4, t_world);

            // output info for debugging
//            ROS_INFO("Detected robot pose: (%f, %f, %f)",
//                     t_world.at<double>(0), t_world.at<double>(1), t_world.at<double>(2));

            // record pose
            if (!flag_filter_initialized_) {
                // initialize the filter
                robot_pose_filter_->statePost.at<double>(0) = t_world.at<double>(0);
                robot_pose_filter_->statePost.at<double>(1) = t_world.at<double>(1);
                robot_pose_filter_->statePost.at<double>(2) =
                        std::atan2(r_world.at<double>(1, 1), r_world.at<double>(0, 1));
                robot_pose_filter_->statePost.at<double>(3) = 0.0;
                robot_pose_filter_->statePost.at<double>(4) = 0.0;
                robot_pose_filter_->statePost.at<double>(5) = 0.0;

                cv::setIdentity(robot_pose_filter_->errorCovPost.rowRange(0, 3).colRange(0, 3), 1e-4);
                cv::setIdentity(robot_pose_filter_->errorCovPost.rowRange(3, 6).colRange(3, 6), 1e2);

                // set flag
                flag_filter_initialized_ = true;
            } else {
                // obtain measurement
                pose_meas.at<double>(0) = t_world.at<double>(0);
                pose_meas.at<double>(1) = t_world.at<double>(1);
                pose_meas.at<double>(2) = std::atan2(r_world.at<double>(1, 1), r_world.at<double>(0, 1));

                // filter out outliers
                cv::Mat diff = robot_pose_filter_->statePost.rowRange(0, 3) - pose_meas;
                wrap_to_pi(diff.at<double>(2));
                if (cv::norm(diff) > 0.5) {
                    ROS_WARN("Measurement is an outlier!");
                    pose_meas.at<double>(0) = -1;
                    pose_meas.at<double>(1) = -1;
                    pose_meas.at<double>(2) = -1;
                } else {
                    // correct the orientation measurement range
                    correct_rot_meas_range(robot_pose_filter_->statePre.at<double>(2), pose_meas.at<double>(2));

                    // update Kalman Filter
                    robot_pose_filter_->correct(pose_meas);

                    // wrap orientation to [-pi, pi]
                    wrap_to_pi(robot_pose_filter_->statePost.at<double>(2));
                }
            }
        }

        // write to file
        // time stamp
        res << tstamp << ", ";

        // human poses
        for (auto &it : human_poses) {
            res << it.second.at<double>(0) << ", ";
            res << it.second.at<double>(1) << ", ";
            res << it.second.at<double>(2) << ", ";
            res << it.second.at<double>(3) << ", ";
            res << it.second.at<double>(4) << ", ";
            res << it.second.at<double>(5) << ", ";
        }

        // robot pose
        res << robot_pose_filter_->statePost.at<double>(0) << ", ";
        res << robot_pose_filter_->statePost.at<double>(1) << ", ";
//        res << robot_pose_filter_->statePost.at<double>(2) << std::endl;
//        res << robot_pose_filter_->statePost.at<double>(2) << ", ";
        res << robot_pose_filter_->statePost.at<double>(3) << ", ";
        res << robot_pose_filter_->statePost.at<double>(4) << ", ";
        res << robot_pose_filter_->statePost.at<double>(5) << std::endl;

        // increase counters
        tstamp += dt;
        counter++;

        // prediction step of the robot pose filter
        robot_pose_filter_->predict();

        //quit on ESC button
        if(cv::waitKey(1)==27)break;
    }

    // close file
    res.close();
}

//----------------------------------------------------------------------------------
void VideoProcessor::process_all(std::string &path)
{

}

//----------------------------------------------------------------------------------
void VideoProcessor::calculate_pose_world(const cv::Mat &pose_im, const double z0, cv::Mat &pose_world)
{
    // first undistort the image coordinate
    std::vector<cv::Point2d> src;
    std::vector<cv::Point2d> pos_undistort;

    cv::Point2d pos1;
    pos1.x = pose_im.at<double>(0);
    pos1.y = pose_im.at<double>(1);
    src.push_back(pos1);

    cv::undistortPoints(src, pos_undistort, cam_param_.CameraMatrix, cam_param_.Distorsion);

    // obtain the truncated transformation matrix [R|t]
    cv::Mat T(3, 4, CV_64F);
    T.rowRange(0, 3).colRange(0, 3) = cam_rmat_.t();
    T.col(3) = -cam_rmat_.t() * cam_tvec_;
//    std::cout << "cam rotation matrix:" << std::endl << cam_rmat_ << std::endl;
//    std::cout << "Tranformation matrix:" << std::endl << T << std::endl;

    // form the linear equation to solve for (x, y) in world frame
    cv::Mat M(2, 4, CV_64F);
    M.row(0) = T.row(0) - pos_undistort[0].x * T.row(2);
    M.row(1) = T.row(1) - pos_undistort[0].y * T.row(2);

    cv::Mat M1 = M.colRange(0, 2);
    cv::Mat M2 = M.colRange(2, 4);

    cv::Mat b(2, 1, CV_64F);
    b.at<double>(0) = z0;
    b.at<double>(1) = 1.0;
//    std::cout << "M matrix: " << std::endl << M << std::endl;
//    std::cout << "b vector:" << std::endl << b << std::endl;

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
    pose_world = cv::Mat(3, 1, CV_64F);
    pose_world.at<double>(0) = pos.at<double>(0);
    pose_world.at<double>(1) = pos.at<double>(1);
    pose_world.at<double>(2) = th_world;
}

//----------------------------------------------------------------------------------
void VideoProcessor::calculate_pose_vel_world(const cv::Mat &pose_im, const cv::Mat &vel_im, const double z0,
                                              cv::Mat &pose_vel_world)
{
    pose_vel_world = cv::Mat(6, 1, CV_64F);

    // convert position first
    cv::Mat pose_world;
    calculate_pose_world(pose_im, z0, pose_world);

    pose_world.copyTo(pose_vel_world.rowRange(0, 3));

    // calculate a "secondary" point based on velocity
    const double dt = 0.02;
    cv::Mat pose_im2(3, 1, CV_64F);
    pose_im2 = pose_im + vel_im * dt;

    // transform secondary point position to world frame
    cv::Mat pose_world2;
    calculate_pose_world(pose_im2, z0, pose_world2);

    // calculate velocity
    pose_vel_world.rowRange(3, 6) = (pose_world2 - pose_world) / dt;
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
    tracker.set_disp_scale(0.5);

    // load the video file
    ss.str("");
    ss << test_path << "/human_priority.MP4";

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
//    std::string test_path = argv[1];
//    test_hat_tracker(test_path);
//    save_one_frame(test_path);

    ros::init(argc, argv, "video_process");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    tracking::VideoProcessor processor(nh, pnh);

    // get path parameter
    std::string calibration_file;
    std::string video_file;
    std::string save_path;
    ros::param::param<std::string>("~calibration_file", calibration_file, "calibration.jpg");
    ros::param::param<std::string>("~video_file", video_file, "exp.mp4");
    ros::param::param<std::string>("~save_path", save_path, "processed_data");

    // first do extrinsic calibration
    processor.extrensic_calibration(calibration_file);

    // then process the video
    processor.process(video_file, save_path);

    return 0;
}
