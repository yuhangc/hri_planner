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

#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "top_view_tracker/video_processor.h"
#include "top_view_tracker/hat_tracker.h"

void test_hat_tracker(const std::string &test_path)
{
    // create a hat tracker object
    tracking::HatTracker tracker;

    // load configuration file
    std::stringstream ss;
    ss << test_path << "test_config.json";

    tracker.load_config(ss.str());

    // load the video file
    ss.clear();
    ss << test_path << "test_video.mp4";

    cv::VideoCapture cap(ss.str());
    if (!cap.isOpened()) {
        std::cout << "Video not opened!!!!!" << std::endl;
        return;
    }

    cv::Mat frame;
    for (;;) {
        // get video frame
        cap >> frame;

        if (frame.empty())
            break;

        tracker.track(frame);

        //quit on ESC button
        if(cv::waitKey(1)==27)break;
    }
}

int main(int argc, char** argv)
{
    return 0;
}

//#include <opencv2/core/utility.hpp>
//#include <opencv2/tracking.hpp>
//#include <opencv2/videoio.hpp>
//#include <opencv2/highgui.hpp>
//#include <iostream>
//#include <cstring>
//using namespace std;
//using namespace cv;
//int main( int argc, char** argv ){
//    // show help
//    if(argc<2){
//        cout<<
//            " Usage: tracker <video_name>\n"
//                    " examples:\n"
//                    " example_tracking_kcf Bolt/img/%04d.jpg\n"
//                    " example_tracking_kcf faceocc2.webm\n"
//            << endl;
//        return 0;
//    }
//    // declares all required variables
//    Rect2d roi;
//    Mat frame;
//    // create a tracker object
//    Ptr<Tracker> tracker = TrackerKCF::create();
//    // set input video
//    std::string video = argv[1];
//    VideoCapture cap(video);
//    if (!cap.isOpened()) {
//        std::cout << "Video not opened!!!!!" << std::endl;
//        return -1;
//    }
//    // get bounding box
//    cap >> frame;
//    roi=selectROI("tracker",frame);
//    //quit if ROI was not selected
//    if(roi.width==0 || roi.height==0)
//        return 0;
//    // initialize the tracker
//    tracker->init(frame,roi);
//    // perform the tracking process
//    printf("Start the tracking process, press ESC to quit.\n");
//    for ( ;; ){
//        // get frame from the video
//        cap >> frame;
//        // stop the program if no more images
//        if(frame.rows==0 || frame.cols==0)
//            break;
//        // update the tracking result
//        tracker->update(frame,roi);
//        // draw the tracked object
//        rectangle( frame, roi, Scalar( 255, 0, 0 ), 2, 1 );
//        // show image with the tracked object
//        imshow("tracker",frame);
//        //quit on ESC button
//        if(waitKey(1)==27)break;
//    }
//    return 0;
//}