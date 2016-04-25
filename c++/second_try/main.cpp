#include <iostream>
#include <chrono>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
//#include <opencv2/xfeatures2d.hpp>


typedef cv::Point3_<uchar> rgb_point;


class case_runner {
public:
    virtual int call() = 0;
};

namespace kernel {
    namespace edges {
        cv::Mat d = (cv::Mat_<double>(3, 3) <<
                1, 0, -1,
                0, 0, 0,
                -1, 0, 1
        );
        cv::Mat hw = (cv::Mat_<double>(3, 3) <<
                0, 1, 0,
                1, -4, 1,
                0, 1, 0
        );
        cv::Mat dhw = (cv::Mat_<double>(3, 3) <<
                1, 1, 1,
                1, -8, 1,
                1, 1, 1
        );
    }

    cv::Mat sharpen = (cv::Mat_<double>(3, 3) <<
            0, -1, 0,
            -1, 5, -1,
            0, -1, 0
    );

}

uchar gamma_correction(uchar in, double lambd){
    return std::pow(in/255.0,lambd)*255;
}

void gamma_correction(cv::Mat& in, cv::Mat& out, double lambd){
    out = cv::Mat(in.size(),CV_8UC3);
    for (size_t i = 0; i < in.rows; i++)
        for (size_t j = 0; j < in.cols; j++) {
            rgb_point p = in.at<rgb_point>(i, j);
            p.x = gamma_correction(p.x,lambd);
            p.y = gamma_correction(p.y,lambd);
            p.z = gamma_correction(p.z,lambd);
            out.at<rgb_point>(i, j) = p;
        }
}

namespace sec_case {

    std::string first_window = "first";

    class second_case : public case_runner {
    public:
        int call() {
            cv::VideoCapture cap(0);

            cv::Mat frame;

            for (; ;) {
                cap >> frame;

                //cv::Mat g_corrected;

                //gamma_correction(frame,g_corrected,0.5);


                cv::imshow(first_window,frame);

                char key = cv::waitKey(30);

                if (key == 'q')
                    break;

            }
            return 0;
        };

    };
}

int main() {
    case_runner* def_case = new sec_case::second_case();
    def_case->call();

    return 0;
}

/*
 *                 cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create();

                // Detect blobs.
                std::vector<cv::KeyPoint> keypoints;
                detector->detect(frame, keypoints);

                // Draw detected blobs as red circles.
                // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
                cv::Mat im_with_keypoints;
                drawKeypoints(frame, keypoints, im_with_keypoints, cv::Scalar(255, 255, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                // Show blobs
                cv::imshow("keypoints", im_with_keypoints);
 */