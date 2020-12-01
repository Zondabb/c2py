#include "tensor.h"

namespace c2py {

static const uint8_t SIZE_IN_BYTES[] = {
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

uint8_t TensorType::SizeInBytes() const {
  if (_type < TensorType::NUM_OF_TYPES)
      return SIZE_IN_BYTES[_type];
    else
      return 0;
}

Tensor::Tensor()
  : dims_(0), data_(nullptr), shape_({0}), step_({0}),
    type_(TensorType::UNKNOWN) {}

Tensor::Tensor(std::vector<size_t> shape, TensorType type)
    : dims_(shape.size()), data_(nullptr), shape_(shape), step_(shape), type_(type) {
  size_t size = 1;
  for (int i = 0; i < shape.size(); i++) {
    size *= shape[i];
  }
  data_ = std::shared_ptr<int8_t>(new int8_t[size * type.SizeInBytes()](), std::default_delete<int8_t[]>());
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<size_t> step, TensorType type)
    : dims_(shape.size()), data_(nullptr), shape_(shape), step_(step), type_(type) {
  size_t size = 1;
  for (int i = 0; i < step.size(); i++) {
    size *= step[i];
  }
  data_ = std::shared_ptr<int8_t>(new int8_t[size * type.SizeInBytes()](), std::default_delete<int8_t[]>());
}

void Tensor::Reset(std::shared_ptr<int8_t> data, std::vector<size_t> shape, TensorType type) {
  dims_ = shape.size();
  data_ = std::move(data);
  shape_ = shape;
  step_ = shape;
  type_ = type;
}

void Tensor::Print(std::ostream &out) const {
  int sz = getSize();
  out << "Tensor (shape: ";
  out << "<";
  for (int i = 0; i < shape_.size(); i++) {
    out << shape_[i];
    if (i != shape_.size() - 1) {
      out << ",";
    }
  }
  out << ">";
  out << " Size: " << sz;
  if (sz < 10) {
    out << "," <<  " Value: {";
    if (type_.Value() == TensorType::INT8) {
      int8_t* ptr = (int8_t*)data_.get();
      for (int i = 0; i < sz; i++) {
        out << *ptr << (i == sz - 1 ? "" : ", ");
        ptr++;
      }
    } else if (type_.Value() == TensorType::INT32) {
      int32_t* ptr = (int32_t*)data_.get();
      for (int i = 0; i < sz; i++) {
        out << *ptr << (i == sz - 1 ? "" : ", ");
        ptr++;
      }
    } else if (type_.Value() == TensorType::FLOAT32) {
      float* ptr = (float*)data_.get();
      for (int i = 0; i < sz; i++) {
        out << *ptr << (i == sz - 1 ? "" : ", ");
        ptr++;
      }
    }
    out << "}";
  }
  out << ")\n";
}

template<typename T>
void Tensor::Init(const Vector<T>& vec) {
  if (std::is_same<T, int8_t>::Value) {
    type_ = TensorType::INT8;
  } else if (std::is_same<T, int32_t>::Value) {
    type_ = TensorType::INT32;
  } else if (std::is_same<T, float>::Value) {
    type_ = TensorType::FLOAT32;
  } else {
    INFO_LOG("type is not support.");
    return;
  }

  int size = vec.size();
  int buff_size = size * type_.SizeInBytes();

  shape_.resize(1);
  shape_[0] = vec.size();

  step_.resize(1);
  step_[0] = vec.size();

  data_ = data_ = std::shared_ptr<int8_t>(new int8_t[buff_size](), std::default_delete<int8_t[]>());
  memcpy(data_.get(), vec.data(), buff_size);
}

template<typename T>
void Tensor::Init(const Mat2D<T>& mat_2d) {
  if (std::is_same<T, int8_t>::Value) {
    type_ = TensorType::INT8;
  } else if (std::is_same<T, int32_t>::Value) {
    type_ = TensorType::INT32;
  } else if (std::is_same<T, float>::Value) {
    type_ = TensorType::FLOAT32;
  } else {
    INFO_LOG("type is not support.");
    return;
  }

  size_t size = mat_2d.size() * mat_2d[0].size();
  size_t buff_size = size * type_.SizeInBytes();
  size_t last_dim_buff_size = mat_2d[0].size() * type_.SizeInBytes();

  shape_ = {mat_2d.size(), mat_2d[0].size()};
  step_ = {mat_2d.size(), mat_2d[0].size()};

  data_ = std::shared_ptr<int8_t>(new int8_t[buff_size](), std::default_delete<int8_t[]>());
  int8_t* data_ptr = data_.get();
  for (int i = 0; i < step_[0]; i++) {
    memcpy(data_ptr, mat_2d[i].data(), last_dim_buff_size);
    data_ptr += last_dim_buff_size;
  }
}

template<typename T>
void Tensor::Init(const Mat3D<T>& mat_3d) {
  if (std::is_same<T, int8_t>::Value) {
    type_ = TensorType::INT8;
  } else if (std::is_same<T, int32_t>::Value) {
    type_ = TensorType::INT32;
  } else if (std::is_same<T, float>::Value) {
    type_ = TensorType::FLOAT32;
  } else {
    INFO_LOG("type is not support.");
    return;
  }

  size_t size = mat_3d.size() * mat_3d[0].size() * mat_3d[0][0].size();
  size_t buff_size = size * type_.SizeInBytes();
  size_t last_dim_buff_size = mat_3d[0][0].size() * type_.SizeInBytes();

  shape_ = {mat_3d.size(), mat_3d[0].size(), mat_3d[0][0].size()};
  step_ = {mat_3d.size(), mat_3d[0].size(), mat_3d[0][0].size()};

  data_ = std::shared_ptr<int8_t>(new int8_t[buff_size](), std::default_delete<int8_t[]>());
  int8_t* data_ptr = data_.get();
  for (int i = 0; i < step_[0]; i++) {
    for (int j = 0; j < step_[1]; j++) {
      memcpy(data_ptr, mat_3d[i][j].data(), last_dim_buff_size);
      data_ptr += last_dim_buff_size;
    }
  }
}

template<typename T> T& Tensor::at(size_t idx) {
  if (idx < getSize()) {
    size_t cursor = 0;
    for (int i = 0; i < shape_.size(); i++) {
      size_t sz = getSize(i+1);
      size_t sub_row = idx / sz;
      idx -= sub_row * sz;
      cursor += sub_row * getAllocSize(i+1);
    }
    return *((T*)data_.get() + cursor + idx);
  } else {
    // INFO_LOG("The index: %d is out of range.", idx);
    return ((T*)data_.get())[0];
  }
}

} // namespace c2py
