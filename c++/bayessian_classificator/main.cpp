#include <fstream>
#include <chrono>

#include <experimental/filesystem>

#include <opencv2/opencv.hpp>

#include "classificator.hpp"

namespace fs = std::experimental::filesystem;

int main(int argc, char* argv[]) {
    assert(argc>2);

    std::string output_folder = argv[1];
    std::string filename = argv[2];

    fs::path path_to_file(output_folder);
    path_to_file/=filename;

    std::ifstream file(path_to_file);
    BayesianResult b_res = std::move(BayesianResult::load_from_file(file));

    cv::VideoCapture cap(0);
    cv::Mat frame;
    cv::Mat classified;
    cv::Mat result;
    cv::Mat struct_element = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(10,10));

    size_t n = 0;
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    while(cap.isOpened()){
        cap.read(frame);
        cv::cvtColor(frame,frame,cv::COLOR_BGR2HSV);
        classified = b_res.classify<0,1>(frame,0.1);

        cv::GaussianBlur(classified,result,cv::Size(5,5),0.2,0.2);
        //cv::medianBlur(result,result,5);


        cv::dilate(result,result,struct_element);
        //cv::erode(result,result,struct_element);

        cv::imshow("classified",classified);
        cv::imshow("result",result);

        int key = cv::waitKey(30) & 0xFF;
        if (key == 27)
            break;
        n++;
    }
    std::chrono::duration<double> duration = std::chrono::system_clock::now() - start;
    std::cout << "Average FPS " << n/duration.count() << std::endl;

    return 0;
}