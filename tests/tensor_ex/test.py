import sys
sys.path.append('build')

import c2py
from c2py import Tensor
from c2py import TensorEx

class Person():
    def __repr__(self):
        return 'This is a Person.'

# print(Person())
# print(c2py.empty(4, 2))
# print(c2py.__repr__())
# print('aaa')

tx = TensorEx()
print(tx.__repr__)

t = Tensor()
print(dir(Tensor))
print(t.__repr__)
# print(t.print())

# class Test:
#     def __repr__(self):
#         return "Test()"

#     # def __str__(self):
#     #     return "member of Test"

# t = Test()
# print(t)

