import numpy as np
from Extest import c2py_Model

a = c2py_Model()
a.open('ddd', 'aaa')
# a.compute(np.ones((3,3)), np.zeros((3,3)))
a.compute((1.0, 2.0, 3.3), (1, 2, 3, 44))