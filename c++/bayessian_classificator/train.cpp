#include <fstream>
#include <iostream>

#include <experimental/filesystem>

#include "classificator.hpp"

#define file_name "trained.clr"

namespace fs = std::experimental::filesystem;

int main() {
    Bayesian bayesian(256, 256);
    fs::path p = "./train/";
    bayesian.train_from_folder(p);
    auto bayesian_result = bayesian.make_result();
    std::ofstream file(file_name);
    bayesian_result.save_to_file(file);
    return 0;
}

