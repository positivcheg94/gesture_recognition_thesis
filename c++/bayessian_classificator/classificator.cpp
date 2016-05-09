#include "classificator.hpp"


BayesianResult BayesianResult::load_from_file(std::ifstream &stream) {
    boost::archive::text_iarchive in_archive(stream);
    BayesianResult b_res;
    in_archive >> b_res;
    return b_res;
}

BayesianResult::BayesianResult() { }

BayesianResult::BayesianResult(const size_t first, const size_t second, const std::vector<std::vector<size_t>> counts)
        : first_dim(first), second_dim(second) {
    probs.resize(first_dim);
    for (auto &i : probs)
        i.resize(second_dim);
    size_t m = *std::max_element(counts[0].begin(), counts[0].end());
    auto end = counts.end();
    for (auto i = counts.begin() + 1; i != end; i++)
        m = std::max(m, *std::max_element(i->begin(), i->end()));
    double multiplier = 1.0 / m;
    for (size_t i = 0; i < first_dim; i++)
        for (size_t j = 0; j < second_dim; j++)
            probs[i][j] = counts[i][j] * multiplier;

}

void BayesianResult::save_to_file(std::ofstream &stream) {
    boost::archive::text_oarchive out_archive(stream);
    out_archive << *this;
}

cv::Mat BayesianResult::representation() {
    cv::Mat result(cv::Size(first_dim,second_dim),CV_8UC1);
    for(size_t i = 0; i<first_dim; i++)
        for(size_t j = 0; j<second_dim; j++)
            result.at<uint8_t>(i,j) = probs[i][j]*255;
    return result;
}


Bayesian Bayesian::load_from_file(std::ifstream &stream) {
    boost::archive::text_iarchive in_archive(stream);
    Bayesian b;
    in_archive >> b;
    return b;
}

Bayesian::Bayesian() { }

Bayesian::Bayesian(const size_t first, const size_t second) : first_dim(first), second_dim(second) {
    counts.resize(first_dim);
    for (auto &i : counts)
        i.resize(second_dim);
}


void Bayesian::save_to_file(std::ofstream &stream) {
    boost::archive::text_oarchive out_archive(stream);
    out_archive << *this;
}

BayesianResult Bayesian::make_result() {
    return BayesianResult(first_dim, second_dim, counts);
}
