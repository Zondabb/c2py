#ifndef __FUNC_1_H__
#define __FUNC_1_H__

#include "c2py.hpp"

namespace c2py {

CV_WRAP int sub_func(int a, int b) {
  return a - b;
}

} // namespace c2py

#endif