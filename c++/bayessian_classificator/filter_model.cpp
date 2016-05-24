#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "classificator.hpp"

int main(int argc, char* argv[]){
    assert(argc>4);
    std::string in_filepath = argv[1];
    std::string out_filepath = argv[2];
    std::string command = argv[3];
    std::vector<std::string> params;
    for(size_t i = 4; i < argc; ++i)
        params.push_back(std::string(argv[i]));

    std::ifstream in_file(in_filepath);
    BayesianModel b_m = std::move(BayesianModel::load_from_file(in_file));
    in_file.close();
    cv::imshow("before repr", b_m.representation());
    cv::imshow("before gen_repr", b_m.general_representation());

    if(command == "normalize") {
        b_m.normalize_probabilities();
    }
    else if(command == "threshold"){
        b_m.threshold_small_probabilities_blobs(std::stoull(params[0]),std::stoull(params[1]));
    }
    else if(command == "filter"){
        b_m.filter_random_probabilities(std::stoull(params[0]),std::stoull(params[1]));
    }
    else if(command == "erode"){
        b_m.erode(std::stoull(params[0]),std::stoull(params[1]));
    }
    else if(command == "dilate"){
        b_m.dilate(std::stoull(params[0]),std::stoull(params[1]));
    }
    else if(command == "blur"){
        b_m.blur(std::stoull(params[0]),std::stoull(params[1]));
    }
    else if(command == "gblur") {
        b_m.gaussian_blur(std::stoull(params[0]),std::stoull(params[1]),std::stod(params[2]),std::stod(params[3]));
    }

    std::ofstream out_file(out_filepath);
    b_m.save_to_file(out_file);

    cv::imshow("after repr", b_m.representation());
    cv::imshow("after gen_repr", b_m.general_representation());
    cv::waitKey(0);
}
