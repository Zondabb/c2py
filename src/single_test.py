import numpy as np
from Extest import c2py_Model

a = c2py_Model()
a.open('ddd', 'aaa')
A = np.ones((9)).astype(np.float32)
A = A * 0.7
a.compute(A, np.zeros((90, 7, 88)).astype(np.int32))
print ('done...')
# a.compute((1.0, 2.0, 3.3), (1, 2, 3, 44))