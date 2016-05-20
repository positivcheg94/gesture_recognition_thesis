#include <fstream>
#include <iostream>

#include <experimental/filesystem>

#include <opencv2/opencv.hpp>

#include "classificator.hpp"


namespace fs = std::experimental::filesystem;

int main(int argc, char* argv[]) {
    assert(argc>3);

    Bayesian bayesian(256, 256);

    std::string output_folder = argv[1];
    std::string train_folder = argv[2];
    std::string filename = argv[3];

    fs::path path_to_output_folder(output_folder);
    path_to_output_folder/=filename;
    fs::path path_to_train_folder(train_folder);

    bayesian.train_from_folder<0,1>(path_to_train_folder);

    auto bayesian_m = bayesian.model();

    std::ofstream file(path_to_output_folder.string());
    bayesian_m.save_to_file(file);
    bayesian_m.blur(5,1,1);
    cv::Mat color_map = bayesian_m.representation();
    cv::imwrite(path_to_output_folder.string() + ".jpg",color_map);

    cv::imshow("color map",color_map);
    cv::waitKey(0);

    return 0;
}

