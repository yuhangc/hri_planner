//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 12/14/2017
// Last revision: 12/18/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#include <fstream>
#include <sstream>

#include "top_view_tracker/hat_tracker.h"

namespace tracking {

//----------------------------------------------------------------------------------
HatTracker::HatTracker()
{

}

//----------------------------------------------------------------------------------
void HatTracker::load_config(const std::string &path)
{
    // load configure file (JSON)
    std::ifstream config_file(path, std::ifstream::binary);
    Json::Value root;

    config_file >> root;

    // read number of hats to track
    n_hats_ = root["num_hats"].asInt();

    // load hat templates
    for (int id = 0; id < n_hats_; id++) {
        HatTemplate new_hat;

        std::stringstream ss;
        ss << "hat" << id;

        new_hat.hat_hsv_low = cv::Scalar(root[ss.str()]["hat_hsv_low"][0].asDouble(),
                                         root[ss.str()]["hat_hsv_low"][1].asDouble(),
                                         root[ss.str()]["hat_hsv_low"][2].asDouble());

        new_hat.hat_hsv_high = cv::Scalar(root[ss.str()]["hat_hsv_high"][0].asDouble(),
                                          root[ss.str()]["hat_hsv_high"][1].asDouble(),
                                          root[ss.str()]["hat_hsv_high"][2].asDouble());

        new_hat.cap_hsv_low = cv::Scalar(root[ss.str()]["cap_hsv_low"][0].asDouble(),
                                         root[ss.str()]["cap_hsv_low"][1].asDouble(),
                                         root[ss.str()]["cap_hsv_low"][2].asDouble());

        new_hat.cap_hsv_high = cv::Scalar(root[ss.str()]["cap_hsv_high"][0].asDouble(),
                                          root[ss.str()]["cap_hsv_high"][1].asDouble(),
                                          root[ss.str()]["cap_hsv_high"][2].asDouble());

        new_hat.hat_size = root[ss.str()]["hat_size"].asInt();
        new_hat.cap_size = root[ss.str()]["cap_size"].asInt();

        new_hat.hat_area = (int) (new_hat.hat_size * new_hat.hat_size * 3.14 / 4.0);
        new_hat.cap_area = new_hat.cap_size * new_hat.cap_size >> 1;

        hat_temps_.push_back(new_hat);
    }

    // frequency
    double fps = root["FPS"].asDouble();
    dt_ = 1.0 / fps;

    // detection ratio thresholds
    ratio_th_high_ = root["ratio_threshold_high"].asDouble();
    ratio_th_low_ = root["ratio_threshold_low"].asDouble();

    // TODO: read in process noise and measurement noise matrices from config file
    double process_noise_pos = root["process_noise_pos"].asDouble();
    double process_noise_vel = root["process_noise_vel"].asDouble();
    meas_noise_base_ = root["measurement_noise_base"].asDouble();

    // create tracker instances
    int state_size = 4;
    int meas_size = 2;
    int ctrl_size = 2;

    state_trackers_.clear();
    for (int id = 0; id < n_hats_; id++) {
        // create filter
        cv::Ptr<cv::KalmanFilter> new_tracker = new cv::KalmanFilter(state_size, meas_size, ctrl_size, CV_64F);

        // set transition matrix
        cv::setIdentity(new_tracker->transitionMatrix);
        new_tracker->transitionMatrix.at<double>(0, 2) = dt_;
        new_tracker->transitionMatrix.at<double>(1, 3) = dt_;

        // set measurement matrix
        new_tracker->measurementMatrix = cv::Mat::zeros(meas_size, state_size, CV_64F);
        new_tracker->measurementMatrix.at<double>(0, 0) = 1.0;
        new_tracker->measurementMatrix.at<double>(1, 1) = 1.0;

        // process noise covariance matrix
        //! measurement noise will be updated dynamically
        cv::setIdentity(new_tracker->processNoiseCov.rowRange(0, 2).colRange(0, 2), process_noise_pos);
        cv::setIdentity(new_tracker->processNoiseCov.rowRange(2, 4).colRange(2, 4), process_noise_vel);

        state_trackers_.push_back(new_tracker);
    }

    // initialize flags
    for (int id = 0; id < n_hats_; id++) {
        flag_hat_initialized_.push_back(false);
        flag_hat_lost_.push_back(false);
    }
}

//----------------------------------------------------------------------------------
void HatTracker::track(const cv::Mat im_in, bool flag_vis)
{
    // update frame
    frame_ = im_in;

    // show detection
    cv::Mat im_out = im_in.clone();

    // track all hats
    for (int id = 0; id < n_hats_; id++) {
        if (!flag_hat_initialized_[id] || flag_hat_lost_[id]) {
            if (detect_and_init_hat(id, im_out)) {
                flag_hat_initialized_[id] = true;
                flag_hat_lost_[id] = false;
            }
        } else {
            // obtain prediction with Kalman filter
            cv::Mat hat_state = state_trackers_[id]->predict();

            // define ROI based on prediction
            // FIXME: for now use a constant ROI size
            cv::Rect roi;
            roi.x = (int) hat_state.at<double>(0);
            roi.y = (int) hat_state.at<double>(1);
            roi.width = hat_temps_[id].hat_size << 1;
            roi.height = hat_temps_[id].hat_size << 1;

            // try to detect the hat within roi
            cv::Rect hat_detection;
            double hat_qual = detect_hat_top(hat_temps_[id], roi, hat_detection);

            if (hat_qual < 0) {
                std::cout << "Cannot find hat " << id << " !!!!!" << std::endl;
                continue;
            }

            // transform to global image coordinates
            hat_detection.x += roi.x;
            hat_detection.y += roi.y;

            // try to detect hat cap
            cv::Rect cap_detection;
            get_cap_roi(hat_detection, hat_qual, roi);
            double cap_qual = detect_hat_cap(hat_temps_[id], roi, cap_detection);

            if (cap_qual < 0) {
                std::cout << "Cannot find cap " << id << " !!!!!" << std::endl;
                continue;
            }

            // transform to global image coordinates
            cap_detection.x += roi.x;
            cap_detection.y += roi.y;

            // draw the detections
            if (flag_vis) {
                cv::rectangle(im_out, hat_detection, CV_RGB(255,0,0), 2);
                cv::rectangle(im_out, cap_detection, CV_RGB(0, 0, 255), 2);
            }

            // dynamically update the measurement covariance matrix
            set_kf_cov(hat_qual, state_trackers_[id]->measurementNoiseCov);

            // update Kalman filter
            cv::Vec2d hat_center = rect_center(hat_detection);
            state_trackers_[id]->correct(cv::Mat(hat_center));

            // TODO: update cap and orientation tracking
            cv::Vec2d cap_center = rect_center(cap_detection);
        }
    }

    // display detection result if applicable
    if (flag_vis) {
        cv::imshow("detection", im_out);
        cv::waitKey(1);
    }
}

//----------------------------------------------------------------------------------
bool HatTracker::detect_and_init_hat(const int &id, cv::Mat im_out)
{
    // detect hat top from the entire image
    cv::Rect hat_detection;
    cv::Rect roi(0, 0, frame_.cols, frame_.rows);
    const double hat_qual = detect_hat_top(hat_temps_[id], roi, hat_detection);

    if (hat_qual < 0) {
        std::cout << "Cannot find hat!" << std::endl;
        return false;
    }

    // detect hat cap from the surrounding of the detect hat
    cv::Rect cap_detection;
    get_cap_roi(hat_detection, hat_qual, roi);
    const double cap_qual = detect_hat_cap(hat_temps_[id], roi, cap_detection);

    if (cap_qual) {
        std::cout << "Cannot find cap!" << std::endl;
        return false;
    }

    // calculate the centers
    cv::Vec2d hat_center = rect_center(hat_detection);
    cv::Vec2d cap_center = rect_center(cap_detection);

    // initialize the Kalman Filter
    state_trackers_[id]->statePost = hat_center;
    set_kf_cov(hat_qual, state_trackers_[id]->errorCovPost);

    // TODO: initialize orientation/cap tracking

    // draw the bounding boxes
    if (!im_out.empty()) {
        cv::rectangle(im_out, hat_detection, CV_RGB(255,0,0), 2);
        cv::rectangle(im_out, cap_detection, CV_RGB(0, 0, 255), 2);
    }

    return true;
}

////----------------------------------------------------------------------------------
//void HatTracker::detect_and_init_hats()
//{
//    // detect all hats
//    for (int id = 0; id < n_hats_; id++)
//        detect_and_init_hat(id);
//}

//----------------------------------------------------------------------------------
double HatTracker::detect_hat_top(const HatTemplate &hat_temp, const cv::Rect &roi, cv::Rect &detection)
{
    cv::Mat im = frame_(roi);
    std::vector<std::vector<cv::Point>> contours;

    detect_hat_preprocess(im, hat_temp.hat_hsv_low, hat_temp.hat_hsv_high, contours);

    // report lost if no contours found
    if (contours.empty()) {
        return -1;
    }

    // find the contour that is closest to the size of the cap, and within threshold
    double size_th_low = ratio_th_low_ * hat_temp.hat_area;
    double size_th_high = ratio_th_high_ * hat_temp.hat_area;
    double min_diff = 1e6;
    double area_detection = 0;

    for (auto &ct : contours) {
        // check size
        const double area = cv::contourArea(ct);
        if (area < size_th_low || area > size_th_high)
            continue;

        const double diff = std::abs(area - hat_temp.hat_area);
        if (diff < min_diff) {
            min_diff = diff;
            area_detection = area;
            detection = cv::boundingRect(ct);
        }
    }

    // if not found
    if (min_diff < 1e6)
        return area_detection / hat_temp.hat_area;
    else
        return -1;
}

//----------------------------------------------------------------------------------
double HatTracker::detect_hat_cap(const HatTemplate &hat_temp, const cv::Rect &roi, cv::Rect &detection)
{
    cv::Mat im = frame_(roi);
    std::vector<std::vector<cv::Point>> contours;

    detect_hat_preprocess(im, hat_temp.cap_hsv_low, hat_temp.cap_hsv_high, contours);

    // report lost if no contours found
    if (contours.empty()) {
        return false;
    }

    // report lost if no contours found
    if (contours.empty()) {
        return -1;
    }

    // find the contour that is closest to the size of the cap, and within threshold
    double size_th_low = ratio_th_low_ * hat_temp.cap_area;
    double size_th_high = ratio_th_high_ * hat_temp.cap_area;
    double min_diff = 1e6;
    double area_detection = 0;

    for (auto &ct : contours) {
        // check size
        const double area = cv::contourArea(ct);
        if (area < size_th_low || area > size_th_high)
            continue;

        const double diff = std::abs(area - hat_temp.cap_area);
        if (diff < min_diff) {
            min_diff = diff;
            area_detection = area;
            detection = cv::boundingRect(ct);
        }
    }

    // if not found
    if (min_diff < 1e6)
        return area_detection / hat_temp.cap_area;
    else
        return -1;
}

//----------------------------------------------------------------------------------
void HatTracker::detect_hat_preprocess(const cv::Mat &im, const cv::Scalar &lb, const cv::Scalar &ub,
                                       std::vector<std::vector<cv::Point>> &contours)
{
    cv::Mat img_hsv;

    // convert to hsv color space
    cv::cvtColor(im, img_hsv, CV_BGR2HSV);

    // create mask based on hat template
    cv::Mat color_mask;
    cv::inRange(img_hsv, lb, ub, color_mask);

    // filtering
    // opening filter to remove small objects (false positives)
    cv::erode(color_mask, color_mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    cv::dilate(color_mask, color_mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

    // closing to remove small holes (false negatives)
    cv::dilate(color_mask, color_mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    cv::erode(color_mask, color_mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));

    // find contours in the image
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(color_mask, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
}

//----------------------------------------------------------------------------------
void HatTracker::get_cap_roi(const cv::Rect &hat_detection, const double qual, cv::Rect &cap_roi)
{
    cap_roi.width = (int) ((hat_detection.width << 1) / qual);
    cap_roi.height = (int) ((hat_detection.height << 1) / qual);

    cap_roi.x = std::max(0, hat_detection.x + (hat_detection.width - cap_roi.width) >> 1);
    cap_roi.y = std::max(0, hat_detection.y + (hat_detection.height - cap_roi.height) >> 1);
}

//----------------------------------------------------------------------------------
void HatTracker::set_kf_cov(const double qual, cv::Mat &cov)
{
    cv::setIdentity(cov, meas_noise_base_ / (qual * qual));
}

}