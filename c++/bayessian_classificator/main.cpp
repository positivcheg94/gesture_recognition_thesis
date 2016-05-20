#include <fstream>
#include <chrono>

#include <experimental/filesystem>

#include <opencv2/opencv.hpp>

#include "classificator.hpp"

namespace fs = std::experimental::filesystem;

int main(int argc, char* argv[]) {
    assert(argc>4);

    std::string output_folder = argv[1];
    std::string filename = argv[2];
    size_t camera = std::stoll(argv[3]);
    double treshold = std::stod(argv[4]);

    fs::path path_to_file(output_folder);
    path_to_file/=filename;

    std::ifstream file(path_to_file.string()+".clr");

    BayesianResult b_res = std::move(BayesianResult::load_from_file(file));


    cv::VideoCapture cap(camera);
    cap.set(CV_CAP_PROP_FPS, 30);


    cv::Mat frame;
    cv::Mat hsv_frame;
    cv::Mat classified;
    cv::Mat result;
    cv::Mat struct_element = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(10,10));

    size_t n = 0;
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::duration<double> duration;
    while(cap.isOpened()){
        start_time = std::chrono::high_resolution_clock::now();
        cap.read(frame);
        cv::cvtColor(frame,hsv_frame,cv::COLOR_BGR2HSV);
        duration = std::chrono::high_resolution_clock::now() - start_time;
        std::cout << duration.count() << std::endl;

        start_time = std::chrono::high_resolution_clock::now();
        classified = b_res.classify<0,1>(hsv_frame,treshold);
        duration = std::chrono::high_resolution_clock::now() - start_time;
        std::cout << duration.count() << std::endl<< std::endl<< std::endl;

        //cv::GaussianBlur(classified,result,cv::Size(5,5),0.2,0.2);
        //cv::medianBlur(result,result,5);


        //cv::dilate(result,result,struct_element);
        //cv::erode(result,result,struct_element);
        //cv::imshow("result",result);

        cv::imshow("frame",frame);
        cv::imshow("classified",classified);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 27)
            break;
        n++;
    }
    duration = std::chrono::system_clock::now() - start;
    std::cout << "Average FPS " << n/duration.count() << std::endl;

    return 0;
}