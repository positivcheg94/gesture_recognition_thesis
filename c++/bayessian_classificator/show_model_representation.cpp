#include <fstream>

#include <opencv2/opencv.hpp>

#include "classificator.hpp"

int main(int argc, char* argv[]){
    assert(argc>1);
    std::string filepath = argv[1];
    BayesianModel b_m = std::move(BayesianModel::load_from_file(filepath));
    cv::imshow("repr",b_m.representation());
    auto reprep = std::move(b_m.general_representation());
    cv::imshow("gen repr",reprep);
    cv::waitKey(0);
}

