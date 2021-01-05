#include "tensor_ex.h"
#include <memory>

namespace c2py {
std::shared_ptr<Tensor> empty(int a, int b) {
  return std::make_shared<Tensor>();
}

std::string __repr__() {
  return "hello! This is my demo.";
}

std::string TensorEx::__repr__() {
  return "hello! This is TensorEx.";
}

}