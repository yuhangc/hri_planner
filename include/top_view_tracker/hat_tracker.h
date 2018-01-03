//----------------------------------------------------------------------------------
//
// Human Robot Interaction Planning Framework
//
// Created on   : 12/14/2017
// Last revision: 12/20/2017
// Author       : Che, Yuhang <yuhangc@stanford.edu>
// Contact      : Che, Yuhang <yuhangc@stanford.edu>
//
//----------------------------------------------------------------------------------

#ifndef HAT_TRACKER_H
#define HAT_TRACKER_H

#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>

#include <Eigen/Dense>
#include <json/json.h>

#define PI 3.14159265359
#define PI2 6.28318530718

namespace tracking {

typedef struct {
    cv::Scalar hat_hsv_low;
    cv::Scalar hat_hsv_high;
    cv::Scalar cap_hsv_low;
    cv::Scalar cap_hsv_high;

    // sizes in terms of number of pixels
    int hat_size;
    int cap_size;
    int hat_area;
    int cap_area;
} HatTemplate;

class HatTracker {
public:
    // potential initialization functions
    void load_config(const std::string &path);

    // main tracking function
    void track(const cv::Mat im_in, bool flag_vis=true);

    // get methods
//    bool get_tracking(std::vector<cv::Mat> &pose, std::vector<cv::Mat> &vel);

private:
    // initial detection of objects
    bool detect_and_init_hat(const int &id, cv::Mat im_out=cv::Mat());
//    void detect_and_init_hats();

    // detection of hat top and cap
    double detect_hat_top(const HatTemplate &hat_temp, cv::Rect &roi, cv::Rect &detection);
    double detect_hat_cap(const HatTemplate &hat_temp, cv::Rect &roi, cv::Rect &detection);
    void detect_hat_preprocess(const cv::Mat &im, const cv::Scalar &lb, const cv::Scalar &ub,
                               std::vector<std::vector<cv::Point>> &contours);

    void get_cap_roi(const cv::Rect &hat_detection, const double qual, cv::Rect &cap_roi);

    void correct_rot_meas_range(const double ref, double &meas);

    // helper functions
    void set_kf_cov(const double qual, cv::Mat &cov);
    void set_rot_kf_cov(const double cap_qual, const cv::Mat &vel, const cv::Mat &vel_cov, cv::Mat &rot_cov);
    void clip_roi(cv::Rect &roi);

    static cv::Vec2d rect_center(const cv::Rect &rect)
    {
        return cv::Vec2d(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0);
    }

    static void wrap_to_pi(double &ang)
    {
        while (ang >= PI)
            ang -= PI2;
        while (ang < -PI)
            ang += PI2;
    }

    // setting variables
    int n_hats_;
    std::vector<HatTemplate> hat_temps_;

    double dt_;

    double meas_noise_base_;
    double rot_meas_noise_base_;
    double rot_vel_noise_base_;

    // detection size factor thresholds
    double ratio_th_low_;
    double ratio_th_high_;

    // tracking variables
    std::vector<cv::Ptr<cv::KalmanFilter> > pos_trackers_;
    std::vector<cv::Ptr<cv::KalmanFilter> > rot_trackers_;

    // other variables
    cv::Mat frame_;
    cv::Mat global_mask_;

    // flags
    std::vector<bool> flag_hat_initialized_;
    std::vector<bool> flag_hat_lost_;
};

}

#endif //HAT_TRACKER_H
