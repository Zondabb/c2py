#ifndef __FUNC_2_H__
#define __FUNC_2_H__

#include "c2py.h"

namespace c2py {

CV_WRAP int add_func(int a, int b) {
  return a + b;
}

} // namespace c2py

#endif