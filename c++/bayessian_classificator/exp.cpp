#include <iostream>

#include "continuous_matrix.hpp"

int main(){
    continuous_matrix<int,uint8_t> m(6,7);
    size_t counter = 1;
    for (auto j = m.begin(); j != m.end(); j++)
        *j = counter++;

    for (size_t i = 0; i < m.rows(); i++) {
        for (size_t j = 0; j < m.cols(); j++) {
            std::cout << m(i,j) << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}