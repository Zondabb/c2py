#include "mat.h"

Mat::Mat() : dims(0), data(nullptr) {}

Mat::Mat(size_t rows, size_t cols, int type) : dims(2), data(nullptr), size({rows, cols}), step({rows, cols}){
  data.reset(new unsigned char[rows * cols]);
}

template<typename T> Mat::Mat(const std::vector<T>& vec) : dims(2), data(nullptr) {
  size = std::vector<size_t>({vec.size(), 1});
  step = std::vector<size_t>({vec.size(), 1});
  data.reset(vec.data());
}