#include <string>

#include "classificator.hpp"

#define DEBUG

const uint16_t height = 480;
const uint16_t width = 640;

std::string color_window = "color";
std::string classified_window = "color";

auto struct_element = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(4, 4));

size_t n = 0;

// in : cv::Mat of uint16_t
std::pair<double, double> find_min_max(cv::Mat in, bool ignoreZero = true) {
    double min = std::numeric_limits<double>::max(), max = std::numeric_limits<double>::min();
    if (ignoreZero) {
        for (int column = 0; column < in.cols; column++)
            for (int row = 0; row < in.rows; row++) {
                double elem = in.at<uint16_t>(row, column);
                if (elem > max)
                    max = elem;
                if (elem > 0 && elem < min)
                    min = elem;
            }
        // if all elements were ignored ( all zeros )
        if (min == std::numeric_limits<double>::max() && max == std::numeric_limits<double>::min()){
            min = 0;
            max = 0;
        }
    }
    else {
        for (int column = 0; column < in.cols; column++)
            for (int row = 0; row < in.rows; row++) {
                double elem = in.at<uint16_t>(row, column);
                if (elem > max)
                    max = elem;
                if (elem < min)
                    min = elem;
            }
    }
    return std::make_pair(min, max);
};

void filter_color_image_by_depth(cv::Mat &color, cv::Mat &depth, uint16_t max_distance) {
    assert(color.rows == depth.rows && color.cols == depth.cols);
    for (int column = 0; column < color.cols; column++)
        for (int row = 0; row < color.rows; row++) {
            uint16_t current_depth = depth.at<uint16_t>(row, column);
            if (current_depth > max_distance | current_depth == 0)
                color.at<cv::Vec3b>(row, column) = cv::Vec3b(0, 0, 0);
        }
}

int main(int argc, char* argv[]) {
    assert(argc>2);

    std::string filepath = argv[1];
    std::cout << argv[2];
    double treshold = std::stod(argv[2]);

    std::ifstream file(filepath);

    BayesianModel b_m = std::move(BayesianModel::load_from_file(file));

    cv::imshow("repr",b_m.representation());
    cv::imshow("gen repr",b_m.general_representation());


    cv::namedWindow(color_window, 1);
    cv::namedWindow(classified_window, 1);

    rs::device *dev = nullptr;

    try {
        rs::context ctx;

        auto devices_count = ctx.get_device_count();
        if (devices_count == 0) {
            std::cout << "No devices found" << std::endl;
            return -1;
        }

        dev = ctx.get_device(0);

        dev->enable_stream(rs::stream::color, width, height, rs::format::bgr8, 30);
        dev->start();

        cv::Mat color_frame;
        cv::Mat hsv_frame;
        cv::Mat classified_frame;
        cv::Mat kernel = cv::getStructuringElement(CV_SHAPE_ELLIPSE,cv::Size(5,5));

        while (true) {
            dev->wait_for_frames();
#ifdef DEBUG
            std::chrono::duration<double> duration;
            std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
            color_frame = cv::Mat(height, width, CV_8UC3, const_cast<void *>(dev->get_frame_data(rs::stream::color)));
#ifdef DEBUG
            duration = std::chrono::system_clock::now() - start;
            std::cout << "time elapsed for getting frames " << duration.count() << std::endl;
            start = std::chrono::system_clock::now();
#endif

            cv::cvtColor(color_frame,hsv_frame,cv::COLOR_BGR2HSV);
            classified_frame = std::move(b_m.classify<0,1>(hsv_frame,treshold));
            cv::dilate(classified_frame,classified_frame,kernel);
            cv::erode(classified_frame,classified_frame,kernel);

#ifdef DEBUG
            duration = std::chrono::system_clock::now() - start;
            std::cout << "time elapsed for image processing " << duration.count() << std::endl;
#endif


            cv::imshow(color_window, color_frame);
            cv::imshow(classified_window, classified_frame);

            // i'm rly mad cuz this function returns every run different codes just wtf O_O
            int key = cv::waitKey(30) & 0xFF;
            if (key == 27)
                break;
            else if (key == 32) {
                ++n;
                cv::imwrite("color" + std::to_string(n) + ".png", color_frame);
                cv::imwrite("mask" + std::to_string(n) + ".png", classified_frame);
            }
        }
        dev->stop();
    }
    catch (rs::error error) {
        std::cout << error.what() << std::endl;
        return -1;
    }
    return 0;
}

