#ifndef __MAT_H__
#define __MAT_H__

#include <vector>
#include <memory>

#include "c2py.hpp"

class Mat {
public:
  Mat();

  Mat(size_t rows, size_t cols, int type);

  template<typename T> explicit Mat(const std::vector<T>& vec);

  // the matrix dimensionality, >= 2
  int dims;
  // pointer to the data
  std::shared_ptr<unsigned char> data;

  std::vector<size_t> size;
  std::vector<size_t> step;
};

#include "mat.inl.h"

#endif  // __MAT_H__