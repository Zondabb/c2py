#ifndef __TENSOR_EX_H__
#define __TENSOR_EX_H__

#include "c2py.h"
#include "tensor.h"
#include <string>

namespace c2py {

CV_WRAP std::shared_ptr<Tensor> empty(int a, int b);

CV_WRAP std::string __repr__();

class CV_EXPORTS_W TensorEx {
public:
  CV_WRAP TensorEx() {}
  virtual ~TensorEx() {}

  CV_WRAP std::string __repr__();
};

} // namespace c2py

#endif