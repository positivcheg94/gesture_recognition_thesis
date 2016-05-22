#ifndef CLASSIFICATOR
#define CLASSIFICATOR

#include <vector>
#include <algorithm>
#include <string>
#include <utility>
#include <fstream>

#include <experimental/filesystem>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include "continuous_matrix.hpp"
#include "general_defines.hpp"


namespace fs = std::experimental::filesystem;


class BayesianModel{
    friend boost::serialization::access;

    size_t first_dim;
    size_t second_dim;
    dmatrix probs;

    // used for serialization
    template<class Archive>
    inline void serialize(Archive & ar, const unsigned int version){
        ar & first_dim;
        ar & second_dim;
        ar & probs;
    }

    BayesianModel() {};
public:

    static BayesianModel load_from_file(std::ifstream & stream);

    BayesianModel(BayesianModel&&) = default;
    BayesianModel(const size_t first,const size_t second, const smatrix& counts);

    void save_to_file(std::ofstream &stream);

    cv::Mat representation();

    /*
     *
     */
    template <uint first,uint second>
    cv::Mat classify(const cv::Mat& in, double threshold){
        cv::Mat out(in.size(),CV_8UC1);
        auto i = in.begin<vec3b>(), end = in.end<vec3b>();
        auto o = out.begin<uint8_t>();
        for (; i != end; ++i,++o) {
            const vec3b& pixel = *i;
            if (probs(pixel[first],pixel[second]) > threshold)
                *o = gray_pixel::white;
            else
                *o = gray_pixel::black;
        }
        return out;
    };

    void threshold_low_probabilities(double threshold, size_t size){
        cv::Mat process(first_dim,second_dim,CV_64F,probs.data());

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size,size));
        kernel/= cv::countNonZero(kernel);

        cv::filter2D(process,process,-1,kernel);

        auto current = probs.begin(), last = probs.end();
        for(; current!=last; ++current)
            if (*current < threshold)
                *current = 0;
    }

    /*
     * size - uint8_t size of elliptic kernel with a=b=size
     *
     * percents - double in range of [0,1] part of the kernel area to accept as not noise probability
     * adequate tip is to use percents not less than 0.25 ( 1/4 of the circle ) because 0.5 may erase
     * edge probilities of big probabilities blobs
     */
    void filter_random_probabilities(uint8_t size, double percents){

        //prepare kernel and area sizes
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(size,size));
        size_t area_size = cv::countNonZero(kernel);
        size_t area_lower_bound_size = area_size*percents;

        // prepare main double matrix and mask uint8_t matrix
        cv::Mat process(first_dim,second_dim,CV_64F,probs.data());
        continuous_matrix<uint8_t> cmm(first_dim,second_dim);
        auto p_b = probs.cbegin(),p_e = probs.cend();
        auto cmm_b = cmm.begin();
        for(;p_b!=p_e;++p_b,++cmm_b)
            if(*p_b>0)
                *cmm_b = 1;

        cv::Mat mask(first_dim,second_dim,CV_8U,cmm.data());
        cv::filter2D(mask,mask,-1,kernel);

        process.copyTo(process,mask);
    }

    void median_blur(){

    }

    void gaussian_blur(size_t ksize_width ,size_t ksize_height , double sigmax, double sigmay){
        cv::Mat process(first_dim,second_dim,CV_64F,probs.data());
        cv::GaussianBlur(process,process,cv::Size(ksize_width,ksize_height),sigmax,sigmay);
    }
};

class Bayesian{
    friend boost::serialization::access;

    size_t first_dim;
    size_t second_dim;
    smatrix counts;
    
    // used for serialization
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){
        ar & first_dim;
        ar & second_dim;
        ar & counts;
    }

    Bayesian() {};
public:

    static Bayesian load_from_file(std::ifstream & stream);

    Bayesian(Bayesian&&) = default;
    Bayesian(const size_t first, const size_t second) : first_dim(first), second_dim(second), counts(first,second) {};

    void save_to_file(std::ofstream & stream);

    template <uint first,uint second>
    void train_from_folder(const fs::path &path, uchar color_space_conversion = cv::COLOR_BGR2HSV) {
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
            auto i = img.begin<vec3b>(), end = img.end<vec3b>();
            auto m = mask.begin<vec3b>();
            for (; i != end; ++i, ++m) {
                vec3b &i_pixel = *i, m_pixel = *m;
                if (m_pixel != rgb_pixel::white)
                    counts.increment(i_pixel[first],i_pixel[second]);
            }
        }
    };

    template <uint first,uint second>
    void train_from_image(const cv::Mat& image, const vec3b& ignore_color) {
        auto i = image.begin<vec3b>(), end = image.end<vec3b>();
        for (; i != end; ++i) {
            const vec3b& i_pixel = *i;
            if (i_pixel != ignore_color)
                counts.increment(i_pixel[first], i_pixel[second]);
        }
    };

    BayesianModel model() const {
        return BayesianModel(first_dim, second_dim, counts);
    }
};


#endif