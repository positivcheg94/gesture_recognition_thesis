#include "classificator.hpp"


spoint push_wave(const dmatrix& probs, bmatrix& visited, dmatrix& out, spoint start){
    std::queue<spoint> curr_q;
    auto height = probs.rows()-1, width = probs.cols()-1;
    curr_q.push(start);
    spoint max_prob = start;
    double max_prob_value = probs(start.x, start.y);
    while (!curr_q.empty()) {
        spoint current_point = std::move(curr_q.front());
        curr_q.pop();
        auto& x = current_point.x, y = current_point.y;
        visited(x, y) = true;

        auto prob = probs(x,y);
        out(x, y) = prob;

        if (x > 1 && !visited(x - 1, y) && probs(x - 1, y) > 0)
            curr_q.push(std::move(spoint(x - 1, y)));
        if (x < width && !visited(x + 1, y) && probs(x + 1, y) > 0)
            curr_q.push(std::move(spoint(x + 1, y)));
        if (y > 1 && !visited(x, y - 1) && probs(x, y - 1) > 0)
            curr_q.push(std::move(spoint(x, y - 1)));
        if (y < height && !visited(x, y + 1) && probs(x, y + 1) > 0)
            curr_q.push(std::move(spoint(x, y + 1)));

        if (prob > max_prob_value)
            max_prob = std::move(current_point);
    }
    return max_prob;
}

BayesianModel BayesianModel::load_from_file(std::ifstream &stream) {
    boost::archive::text_iarchive in_archive(stream);
    BayesianModel b_m;
    in_archive >> b_m;
    return b_m;
}


BayesianModel::BayesianModel(const smatrix& counts) : first_dim(counts.rows()), second_dim(counts.cols()), probs(first_dim,second_dim) {
    size_t m = *std::max_element(counts.cbegin(), counts.cend());
    double multiplier = 1.0 / m;
    for (size_t i = 0; i < first_dim; ++i)
        for (size_t j = 0; j < second_dim; ++j)
            probs(i,j) = counts.get(i,j) * multiplier;
}

BayesianModel::BayesianModel(dmatrix&& prob_in) : first_dim(prob_in.rows()), second_dim(prob_in.cols()), probs(prob_in) { }

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

cv::Mat BayesianModel::general_representation() {
    cv::Mat result(cv::Size(first_dim,second_dim),CV_8UC1);
    for(size_t i = 0; i<first_dim; ++i)
        for(size_t j = 0; j<second_dim; ++j)
            if (probs(i,j)> 0)
                result.at<uint8_t>(i,j) = 255;
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

template <class BinaryFunction>
BayesianModel model_union(const BayesianModel& bm1, const BayesianModel& bm2, BinaryFunction func = std::max){
    assert(bm1.first_dim==bm2.first_dim && bm1.second_dim == bm2.second_dim);
    BayesianModel result(bm1.first_dim,bm1.second_dim);

    auto bm1_b = bm1.probs.cbegin(),bm1_e = bm1.probs.cend();
    auto bm2_b = bm2.probs.cbegin(), res_b = result.probs.begin();

    for(;bm1_b!=bm1_e;++bm1_b,++bm2_b,++res_b)
        *res_b == func(*bm1_b,*bm2_b);
    return result;
}