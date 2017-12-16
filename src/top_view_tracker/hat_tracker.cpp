//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 12/14/2017
// Last revision: 12/14/2017
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
void HatTracker::load_object_templates(const std::string &path)
{
    // load configure file (JSON)
    std::ifstream config_file(path, std::ifstream::binary);
    Json::Value root;

    config_file >> root;

    // read number of hats to track
    n_hats_ = root["num_hats"].asInt();

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

        new_hat.hat_top_radius = root[ss.str()]["top_radius"].asInt();
        new_hat.hat_cap_length = root[ss.str()]["cap_length"].asInt();

        hat_temps_.push_back(new_hat);
    }
}

//----------------------------------------------------------------------------------
void HatTracker::track(const cv::Mat img_in)
{

}

//----------------------------------------------------------------------------------
void HatTracker::detect_and_init_hat_single(const int &id, cv::Mat im_out)
{
    if (id >= n_hats_) {
        std::cerr << "Invalid hat id!" << std::endl;
        return;
    }

    // detect hat top and cap from the entire image
    cv::Rect hat_detection;
    cv::Rect cap_detection;
    cv::Rect roi(0, 0, frame_.cols, frame_.rows);
    detect_hat_top(hat_temps_[id], roi, hat_detection);
    detect_hat_cap(hat_temps_[id], roi, cap_detection);

    // calculate the centers

}

//----------------------------------------------------------------------------------
void HatTracker::detect_and_init_hats()
{
    // detect all hats
    for (int id = 0; id < n_hats_; id++)
        detect_and_init_hat_single(id);
}

//----------------------------------------------------------------------------------
bool HatTracker::detect_hat_top(const HatTemplate &hat_temp, const cv::Rect &roi, cv::Rect &detection)
{
    cv::Mat im = frame_(roi);
    std::vector<std::vector<cv::Point>> contours;

    detect_hat_preprocess(im, hat_temp.hat_hsv_low, hat_temp.hat_hsv_high, contours);

    // report lost if no contours found
    if (contours.empty()) {
        return false;
    }

    // find the contour that is closest to the size of the cap, and within threshold
    int size_th_low = (int) (factor_th_low_ * hat_temp.hat_top_radius);
    int size_th_high = (int) (factor_th_high_ * hat_temp.hat_top_radius);
    int min_dist = 0 << 15;

    for (auto &ct : contours) {
        cv::Rect bound_rect = cv::boundingRect(ct);

        // check both height and width
        if (bound_rect.height < size_th_low || bound_rect.height > size_th_high)
            continue;
        if (bound_rect.width < size_th_low || bound_rect.width > size_th_high)
            continue;

        int dist = std::max(std::abs(bound_rect.height - hat_temp.hat_top_radius),
                            std::abs(bound_rect.width - hat_temp.hat_top_radius));

        if (dist < min_dist) {
            min_dist = dist;
            detection = bound_rect;
        }
    }

    // if not found
    return min_dist < 0 << 15;
}

//----------------------------------------------------------------------------------
bool HatTracker::detect_hat_cap(const HatTemplate &hat_temp, const cv::Rect &roi, cv::Rect &detection)
{
    cv::Mat im = frame_(roi);
    std::vector<std::vector<cv::Point>> contours;

    detect_hat_preprocess(im, hat_temp.cap_hsv_low, hat_temp.cap_hsv_high, contours);

    // report lost if no contours found
    if (contours.empty()) {
        return false;
    }

    // find the contour that is closest to the size of the cap, and within threshold
    int size_th_low = (int) (factor_th_low_ * hat_temp.hat_top_radius);
    int size_th_high = (int) (factor_th_high_ * hat_temp.hat_top_radius);
    int min_dist = 0 << 15;

    for (auto &ct : contours) {
        cv::Rect bound_rect = cv::boundingRect(ct);

        // check both height and width
        if (bound_rect.height < size_th_low || bound_rect.height > size_th_high)
            continue;
        if (bound_rect.width < size_th_low || bound_rect.width > size_th_high)
            continue;

        int dist = std::max(std::abs(bound_rect.height - hat_temp.hat_top_radius),
                            std::abs(bound_rect.width - hat_temp.hat_top_radius));

        if (dist < min_dist) {
            min_dist = dist;
            detection = bound_rect;
        }
    }

    // if not found
    return min_dist < 0 << 15;
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
static cv::Vec2d HatTracker::rect_center(const cv::Rect &rect)
{
    return cv::Vec2d(rect.x + rect.width << 1, rect.y + rect.height << 1);
}

}