#ifndef CLASSIFICATOR
#define CLASSIFICATOR

#include <vector>
#include <algorithm>
#include <string>
#include <exception>
#include <utility>
#include <fstream>
#include <iostream>

#include <experimental/filesystem>

#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <opencv2/opencv.hpp>

namespace fs = std::experimental::filesystem;

typedef std::tuple<uint8_t,uint8_t,uint8_t> vec3b;


struct bound{
    const size_t _left;
    const size_t _right;
    bound(size_t left, size_t right):_left(left),_right(right){
    }
};

class BayesianResult{
    friend boost::serialization::access;

    static const cv::Vec3b black;
    static const cv::Vec3b white;

    size_t first_dim;
    size_t second_dim;
    std::vector<std::vector<double>> probs;

    // used for serialization
    template<class Archive>
    inline void serialize(Archive & ar, const unsigned int version){
        ar & first_dim;
        ar & second_dim;
        ar & probs;
    }

    BayesianResult();
public:
    static BayesianResult load_from_file(std::ifstream & stream);
    BayesianResult(const size_t first,const size_t second, const std::vector<std::vector<size_t>> counts);
    void save_to_file(std::ofstream & stream);
};

class Bayesian{
    friend boost::serialization::access;

    size_t first_dim;
    size_t second_dim;
    std::vector<std::vector<size_t>> counts;

    Bayesian();

    // used for serialization
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version){
        ar & first_dim;
        ar & second_dim;
        ar & counts;
    }
public:

    Bayesian(const size_t first,const size_t second);

    static Bayesian load_from_file(std::ifstream & stream);

    void save_to_file(std::ofstream & stream);

    void train_from_folder(const fs::path & path, uint8_t color_space_conversion = cv::COLOR_BGR2HSV);

    BayesianResult make_result();
};


#endif