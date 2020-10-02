import numpy as np
import c2py
from c2py import dnn_inference_Model
from c2py import Tensor
from c2py import TensorType


a = dnn_inference_Model()
a.open('ddd', 'aaa')
print(dir(c2py))
t = Tensor((3,3), c2py.TENSOR_TYPE_FLOAT16)
# A = np.ones((9)).astype(np.float32)
# A = A * 0.7
# a.compute(A, np.zeros((90, 7, 88)).astype(np.int32))
# print ('done...')
# a.compute((1.0, 2.0, 3.3), (1, 2, 3, 44))