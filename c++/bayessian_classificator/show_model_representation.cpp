#include <fstream>

#include <opencv2/opencv.hpp>

#include "classificator.hpp"

int main(int argc, char* argv[]){
    assert(argc>1);
    std::string filepath = argv[1];
    std::ifstream file(filepath);
    BayesianModel b_m = std::move(BayesianModel::load_from_file(file));
    b_m.gaussian_blur(3,3,0.9,0.9);
    cv::imshow("repr",b_m.representation());
    auto reprep = std::move(b_m.general_representation());
    cv::imshow("gen repr",reprep);
/*
    auto params = cv::SimpleBlobDetector::Params();
    params.minDistBetweenBlobs = 2.0;
    params.filterByInertia = false;
    params.filterByConvexity = false;
    params.filterByColor = false;
    params.filterByCircularity = false;
    params.filterByArea = true;
    params.minArea = 1.0;
    params.maxArea = std::numeric_limits<double>::max();
    auto blob_detector = cv::SimpleBlobDetector::create(params);
    std::vector< cv::KeyPoint > keypoints;
    blob_detector->detect(reprep,keypoints);
    std::cout << keypoints.size();

    cv::drawKeypoints(reprep,keypoints,reprep, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );


    cv::imshow("rep blobbed",reprep);
*/
    cv::waitKey(0);
}

