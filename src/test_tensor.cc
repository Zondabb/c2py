#include <stdio.h>
#include <vector>
#include <iostream>

#include "tensor.h"
#include "config.hpp"

int main() {
    Tensor t1;
    Vector<int8_t> vec(10, 100);
    t1.Init(vec);

    Tensor t2;
    // Mat2D<float> mat_2d(10, Vector<float>(10, 150.88f));
    Mat2D<float> mat_2d = {{1.1f}, {1.4f}, {2.1f}, {150.0f}};
    t2.Init(mat_2d);
    t2.at<float>(1) = 111.111f;

    std::cout << t2 << std::endl;
    return 0;
}