import sys
sys.path.append('build')

import c2py
m = c2py.Mod()
print(m.Sum((1, 2, 3, 4)))
m.Print('hello')
