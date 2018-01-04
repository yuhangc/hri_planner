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

#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "top_view_tracker/video_processor.h"

namespace tracking {



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
