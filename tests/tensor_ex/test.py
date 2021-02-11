import sys
sys.path.append('build')

import c2py
from c2py import Tensor
from c2py import TensorEx

tx = TensorEx()
print(tx.__repr__())
