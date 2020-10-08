import numpy as np
import inspect
import c2py
from c2py import Tensor
from c2py import TensorType
from c2py import dnn_inference

print(dir(c2py))
a = dnn_inference.Model()
b = dnn_inference.sub.SubModel()
print(b)

module_list = inspect.getmembers(c2py, inspect.ismodule)
print(module_list)

# print(dnn_inference.sub.sub_add(1,2))
# a = dnn_inference_Model()
# a = dnn_inference.Model()
a.open('ddd', 'aaa')
t = Tensor((3,3), c2py.TENSOR_TYPE_FLOAT16)
# A = np.ones((9)).astype(np.float32)
# A = A * 0.7
# a.compute(A, np.zeros((90, 7, 88)).astype(np.int32))
# print ('done...')
# a.compute((1.0, 2.0, 3.3), (1, 2, 3, 44))