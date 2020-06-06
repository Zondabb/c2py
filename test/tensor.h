#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <vector>
#include <memory>

#include "c2py.hpp"

struct TensorType {
public:
  enum Type : uint8_t {
    DE_UNKNOWN = 0,
    DE_BOOL,
    DE_INT8,
    DE_UINT8,
    DE_INT16,
    DE_UINT16,
    DE_INT32,
    DE_UINT32,
    DE_INT64,
    DE_UINT64,
    DE_FLOAT16,
    DE_FLOAT32,
    DE_FLOAT64,
    NUM_OF_TYPES
  };

  TensorType() : _type(DE_UNKNOWN) {}

  constexpr explicit TensorType(Type d) : _type(d) {}

  ~TensorType() = default;

  Type value() const { return _type; }

  uint8_t SizeInBytes() const {
    if (_type < TensorType::NUM_OF_TYPES)
      return SIZE_IN_BYTES[_type];
    else
      return 0;
  }

private:
  static constexpr uint8_t SIZE_IN_BYTES[] = {
    0,  // DE_UNKNOWN
    1,  // DE_BOOL
    1,  // DE_INT8
    1,  // DE_UINT8
    2,  // DE_INT16
    2,  // DE_UINT16
    4,  // DE_INT32
    4,  // DE_UINT32
    8,  // DE_INT64
    8,  // DE_UINT64
    2,  // DE_FLOAT16
    4,  // DE_FLOAT32
    8,  // DE_FLOAT64
  };

  Type _type;
};

class Tensor {
public:
  Tensor();

  Tensor(std::vector<size_t> shape, TensorType type);

  Tensor(std::vector<size_t> shape, std::vector<size_t> step, TensorType type);

  // template<typename T> explicit Tensor(const std::vector<T>& vec);

private:
  int _dims;
  std::shared_ptr<uint8_t> _data;
  std::vector<size_t> _shape;
  std::vector<size_t> _step;
  TensorType _type;
};

#include "tensor.inl.h"

#endif  // __TENSOR_H__