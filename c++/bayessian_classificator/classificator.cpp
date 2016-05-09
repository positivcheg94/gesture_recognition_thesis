#include "classificator.hpp"


const cv::Vec3b BayesianResult::black(0, 0, 0);
const cv::Vec3b BayesianResult::white(255, 255, 255);


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

void Bayesian::train_from_folder(const fs::path &path, uint8_t color_space_conversion) {
    std::vector<std::pair<fs::path, fs::path>> train_set;
    {
        fs::directory_iterator entries(path);
        train_set.resize(std::distance(fs::begin(entries), fs::end(entries)) / 2);
    }
    for (auto &entry : fs::directory_iterator(path)) {
        std::stringstream str_stream(entry.path().filename());
        std::string temp;
        std::getline(str_stream, temp, '_');
        size_t n = std::stoi(temp);
        char c = str_stream.get();
        if (c == 'i')
            train_set[n - 1].first = std::move(const_cast<fs::path &>(entry.path()));
        else
            train_set[n - 1].second = std::move(const_cast<fs::path &>(entry.path()));
    }
    for (auto &set : train_set) {
        cv::Mat img = cv::imread(set.first.string());
        cv::Mat mask = cv::imread(set.second.string());
        cv::cvtColor(img, img, color_space_conversion);
        auto i_b = img.begin<cv::Vec3b>(), i_e = img.end<cv::Vec3b>();
        auto m_b = mask.begin<cv::Vec3b>(), m_e = mask.end<cv::Vec3b>();
        for (auto i = i_b, m = m_b; i != i_e; i++, m++) {
            cv::Vec3b &i_pixel = *i, m_pixel = *m;
            if (m_pixel != cv::Vec3b(255, 255, 255))
                counts[i_pixel[0]][i_pixel[1]] += 1;
        }
    }
}

BayesianResult Bayesian::make_result() {
    return BayesianResult(first_dim, second_dim, counts);
}
