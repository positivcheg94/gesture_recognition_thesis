#include "classificator.hpp"


BayesianModel BayesianModel::load_from_file(std::ifstream &stream) {
    boost::archive::text_iarchive in_archive(stream);
    BayesianModel b_m;
    in_archive >> b_m;
    return b_m;
}


BayesianModel::BayesianModel(const size_t first,const size_t second, const smatrix& counts) : first_dim(first), second_dim(second), probs(first,second) {
    size_t m = *std::max_element(counts.cbegin(), counts.cend());
    double multiplier = 1.0 / m;
    for (size_t i = 0; i < first_dim; ++i)
        for (size_t j = 0; j < second_dim; ++j)
            probs(i,j) = counts.get(i,j) * multiplier;

}

void BayesianModel::save_to_file(std::ofstream &stream) {
    boost::archive::text_oarchive out_archive(stream);
    out_archive << *this;
}

cv::Mat BayesianModel::representation() {
    cv::Mat result(cv::Size(first_dim,second_dim),CV_8UC1);
    for(size_t i = 0; i<first_dim; ++i)
        for(size_t j = 0; j<second_dim; ++j)
            result.at<uint8_t>(i,j) = probs(i,j)*255;
    return result;
}


Bayesian Bayesian::load_from_file(std::ifstream &stream) {
    boost::archive::text_iarchive in_archive(stream);
    Bayesian b;
    in_archive >> b;
    return b;
}

void Bayesian::save_to_file(std::ofstream &stream) {
    boost::archive::text_oarchive out_archive(stream);
    out_archive << *this;
}
