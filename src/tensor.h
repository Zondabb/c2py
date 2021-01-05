#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <vector>
#include <memory>
#include <ostream>

#include "c2py.h"
#include <cstring>
#include <iostream>

namespace c2py {
template<typename T>
using Vector = std::vector<T>;

template<typename T>
using Mat2D = std::vector<Vector<T>>;

template<typename T>
using Mat3D = std::vector<Mat2D<T>>;

struct CV_EXPORTS_W TensorType {
public:
  enum Type : uint8_t {
    UNKNOWN = 0,
    BOOL,
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    UINT32,
    INT64,
    UINT64,
    FLOAT16,
    FLOAT32,
    FLOAT64,
    NUM_OF_TYPES
  };

  CV_WRAP TensorType() : _type(UNKNOWN) {}

  TensorType(Type d) : _type(d) {}

  ~TensorType() = default;

  Type Value() const { return _type; }

  uint8_t SizeInBytes() const;

private:
  Type _type;
};

class BasicAllocator {
 public:
  template <typename T>
  void operator ()(T *ptr){
    delete ptr;
  }
};

struct CV_EXPORTS_W Tensor {
public:
  CV_WRAP Tensor();

  CV_WRAP Tensor(std::vector<size_t> shape, TensorType type);

  CV_WRAP Tensor(const std::vector<size_t> &shape, uint8_t type) {
    new(this) Tensor(shape, TensorType(static_cast<TensorType::Type>(type)));
  }

  Tensor(std::vector<size_t> shape, std::vector<size_t> step, TensorType type);

  // Tensor(int8_t *data, std::vector<size_t> shape, TensorType type);

  void Reset(std::shared_ptr<int8_t> data, std::vector<size_t> shape, TensorType type);

  template<typename T> void Init(const Vector<T>& vec);

  template<typename T> void Init(const Mat2D<T>& mat_2d);

  template<typename T> void Init(const Mat3D<T>& mat_3d);

  template<typename T> T& at(size_t idx=0);

  int getDims() {
    return dims_;
  }

  uint8_t* getPtr() {
    return (uint8_t*)data_.get();
  }

  size_t getSize(size_t dim=0) const {
    size_t s = 1;
    for (; dim < shape_.size(); dim++) {
      s *= shape_[dim];
    }
    return s;
  }

  size_t getAllocSize(size_t dim=0) const {
    size_t s = 1;
    for (; dim < step_.size(); dim++) {
      s *= step_[dim];
    }
    return s;
  }

  CV_WRAP std::string print();

  void Print(std::ostream &out) const;

  friend std::ostream &operator<<(std::ostream &out, const Tensor &so) {
    so.Print(out);
    return out;
  }

private:
  int dims_;
  std::shared_ptr<int8_t> data_;
  std::vector<size_t> shape_;
  std::vector<size_t> step_;
  TensorType type_;
};

} // namespace c2py
#endif  // __TENSOR_H__