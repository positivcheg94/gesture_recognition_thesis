#include "classificator.hpp"


BayesianResult BayesianResult::load_from_file(std::ifstream &stream) {
    boost::archive::text_iarchive in_archive(stream);
    BayesianResult b_res;
    in_archive >> b_res;
    return b_res;
}

BayesianResult::BayesianResult() { }

BayesianResult::BayesianResult(const size_t first,const size_t second, const umatrix& counts)
        : first_dim(first), second_dim(second), probs(first,second) {

    size_t m = *std::max_element(counts.cbegin(), counts.cend());
    double multiplier = 1.0 / m;
    for (size_t i = 0; i < first_dim; i++)
        for (size_t j = 0; j < second_dim; j++)
            probs(i,j) = counts.get(i,j) * multiplier;

}

void BayesianResult::save_to_file(std::ofstream &stream) {
    boost::archive::text_oarchive out_archive(stream);
    out_archive << *this;
}

cv::Mat BayesianResult::representation() {
    cv::Mat result(cv::Size(first_dim,second_dim),CV_8UC1);
    for(size_t i = 0; i<first_dim; i++)
        for(size_t j = 0; j<second_dim; j++)
            result.at<uint8_t>(i,j) = probs(i,j)*255;
    return result;
}


Bayesian Bayesian::load_from_file(std::ifstream &stream) {
    boost::archive::text_iarchive in_archive(stream);
    Bayesian b;
    in_archive >> b;
    return b;
}

Bayesian::Bayesian() { }

Bayesian::Bayesian(const size_t first, const size_t second) : first_dim(first), second_dim(second), counts(first,second) {
}


void Bayesian::save_to_file(std::ofstream &stream) {
    boost::archive::text_oarchive out_archive(stream);
    out_archive << *this;
}

BayesianResult Bayesian::make_result() {
    return BayesianResult(first_dim, second_dim, counts);
}
