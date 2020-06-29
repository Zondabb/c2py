#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <vector>
#include <memory>
#include <ostream>

#include "c2py.hpp"

template<typename T>
using Vector = std::vector<T>;

template<typename T>
using Mat2D = std::vector<Vector<T>>;

template<typename T>
using Mat3D = std::vector<Mat2D<T>>;

struct TensorType {
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

  TensorType() : _type(UNKNOWN) {}

  TensorType(Type d) : _type(d) {}

  ~TensorType() = default;

  Type value() const { return _type; }

  uint8_t SizeInBytes() const {
    if (_type < TensorType::NUM_OF_TYPES)
      return SIZE_IN_BYTES[_type];
    else
      return 0;
  }

private:
  static inline const uint8_t SIZE_IN_BYTES[] = {
    0,  // UNKNOWN
    1,  // BOOL
    1,  // INT8
    1,  // UINT8
    2,  // INT16
    2,  // UINT16
    4,  // INT32
    4,  // UINT32
    8,  // INT64
    8,  // UINT64
    2,  // FLOAT16
    4,  // FLOAT32
    8,  // FLOAT64
  };

  Type _type;
};

class Tensor {
public:
  Tensor();

  Tensor(std::vector<size_t> shape, TensorType type);

  Tensor(std::vector<size_t> shape, std::vector<size_t> step, TensorType type);

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

  void Print(std::ostream &out) const;

  friend std::ostream &operator<<(std::ostream &out, const Tensor &so) {
    so.Print(out);
    return out;
  }

private:
  int dims_;
  std::unique_ptr<int8_t[]> data_;
  std::vector<size_t> shape_;
  std::vector<size_t> step_;
  TensorType type_;
};

#include "tensor.inl.h"

#endif  // __TENSOR_H__