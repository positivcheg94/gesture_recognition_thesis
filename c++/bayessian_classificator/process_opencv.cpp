#include <fstream>
#include <chrono>

#include <experimental/filesystem>

#include <opencv2/opencv.hpp>

#include "classificator.hpp"

namespace fs = std::experimental::filesystem;

std::string color_window = "color";
std::string classified_window = "color";

int main(int argc, char* argv[]) {
    assert(argc>3);

    std::string filepath = argv[1];
    size_t camera = std::stoll(argv[2]);
    double treshold = std::stod(argv[3]);


    std::ifstream file(filepath);

    BayesianModel b_m = std::move(BayesianModel::load_from_file(file));


    cv::VideoCapture cap(camera);
    cap.set(CV_CAP_PROP_FPS, 30);


    cv::Mat color_frame;
    cv::Mat hsv_frame;
    cv::Mat classified_frame;
    cv::Mat result;
    cv::Mat struct_element = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(10,10));

    size_t n = 0;
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::duration<double> duration;
    while(cap.isOpened()){
        start_time = std::chrono::high_resolution_clock::now();
        cap >> color_frame;
        cv::cvtColor(color_frame,hsv_frame,cv::COLOR_BGR2HSV);
        duration = std::chrono::high_resolution_clock::now() - start_time;
        std::cout << duration.count() << std::endl;

        start_time = std::chrono::high_resolution_clock::now();
        classified_frame = b_m.classify<0,1>(hsv_frame,treshold);
        duration = std::chrono::high_resolution_clock::now() - start_time;
        std::cout << duration.count() << std::endl<< std::endl<< std::endl;

        //cv::GaussianBlur(classified,result,cv::Size(5,5),0.2,0.2);
        //cv::medianBlur(result,result,5);


        //cv::dilate(result,result,struct_element);
        //cv::erode(result,result,struct_element);
        //cv::imshow("result",result);

        cv::imshow(color_window,color_frame);
        cv::imshow(classified_window,classified_frame);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 27)
            break;
        n++;
    }
    duration = std::chrono::system_clock::now() - start;
    std::cout << "Average FPS " << n/duration.count() << std::endl;

    return 0;
}