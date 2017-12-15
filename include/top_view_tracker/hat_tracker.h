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

#ifndef HAT_TRACKER_H
#define HAT_TRACKER_H

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Dense>

namespace tracking {

class HatTracker {
public:
    // constructor
    HatTracker();

    // potential initialization functions

    // main tracking function
    void track(const cv::Mat img_in);

private:
    // initial detection of objects
    void detect_hats(const bool &flag_with_color_template);
    void detect_hats_template();
    void detect_hats_auto();

    // setting variables

    // tracking variables

    // other variables

    // flags
    bool flag_auto_detect_;
    bool flag_initialized_;
};

}

#endif //HAT_TRACKER_H
