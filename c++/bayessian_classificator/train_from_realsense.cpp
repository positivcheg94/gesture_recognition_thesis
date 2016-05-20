#include <string>

#include "classificator.hpp"


const uint16_t height = 480;
const uint16_t width = 640;

std::string color_window = "color";

auto struct_element = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(4, 4));


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
    assert(argc>1);
    std::string file_name = argv[1];

    cv::namedWindow(color_window, 1);

    Bayesian b(256,256);

    rs::device *dev = nullptr;

    try {
        rs::context ctx;

        auto devices_count = ctx.get_device_count();
        if (devices_count == 0) {
            std::cout << "No devices found" << std::endl;
            return -1;
        }

        dev = ctx.get_device(0);

        dev->set_option(rs::option::f200_laser_power, 15);
        dev->set_option(rs::option::f200_motion_range, 50);
        dev->set_option(rs::option::f200_confidence_threshold, 15);
        dev->set_option(rs::option::f200_filter_option, 6);
        dev->set_option(rs::option::f200_accuracy, 3);

        dev->enable_stream(rs::stream::depth, width, height, rs::format::z16, 30);
        dev->enable_stream(rs::stream::color, width, height, rs::format::bgr8, 30);
        dev->start();

        cv::Mat depth_frame;
        cv::Mat color_frame;
        cv::Mat hsv_frame;

        while (true) {
            dev->wait_for_frames();
#ifdef DEBUG
            std::chrono::duration<double> duration;
            std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
            depth_frame = cv::Mat(height, width, CV_16UC1, const_cast<void *>(dev->get_frame_data(rs::stream::depth)));
            color_frame = cv::Mat(height, width, CV_8UC3, const_cast<void *>(dev->get_frame_data(rs::stream::color_aligned_to_depth)));
#ifdef DEBUG
            duration = std::chrono::system_clock::now() - start;
            std::cout << "time elapsed for getting frames " << duration.count() << std::endl;
            start = std::chrono::system_clock::now();
#endif
//            cv::erode(depth_frame, depth_frame, struct_element);
//            cv::dilate(depth_frame, depth_frame, struct_element);

#ifdef DIST_DEBUG
            auto minmax = find_min_max(depth_frame);
                std::cout << "min=" << minmax.first << " | max=" << minmax.second << std::endl;
#endif

            filter_color_image_by_depth(color_frame, depth_frame, 20000);


#ifdef DEBUG
            duration = std::chrono::system_clock::now() - start;
            std::cout << "time elapsed for image processing frames " << duration.count() << std::endl;
#endif

            cv::imshow(color_window, color_frame);

            cv::cvtColor(color_frame,hsv_frame,cv::COLOR_BGR2HSV);

            b.train_from_image<0,1>(hsv_frame,hsv_pixel::black);

            // i'm rly mad cuz this function returns every run different codes just wtf O_O
            int key = cv::waitKey(30) & 0xFF;
            if (key == 27)
                break;
            else if (key != 255)
                std::cout << key << std::endl;

        }
        auto res = b.make_result();
        std::ofstream out(file_name);
        res.save_to_file(out);
        dev->stop();
    }
    catch (rs::error error) {
        std::cout << error.what() << std::endl;
        return -1;
    }
    return 0;
}

