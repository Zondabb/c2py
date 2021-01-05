import sys
sys.path.append('build')

import c2py
from c2py import Tensor
from c2py import TensorType

print(c2py.add_func(1, 2))
print(c2py.sub_func(1, 2))

t = Tensor()
print(t)
print(c2py.TENSOR_TYPE_FLOAT64)
# print(TensorType.BOOL)
